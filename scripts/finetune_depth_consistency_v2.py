#!/usr/bin/env python3
"""Frozen-backbone consistency distillation for Depth Anything V2.

Freezes the DINOv2 backbone (89% of params) and only fine-tunes the
DPT neck + head. This preserves learned depth representations while
teaching equivariance under D4 transforms.

Usage:
    python scripts/finetune_depth_consistency_v2.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats

# Reuse utilities from v1
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.finetune_depth_consistency import (
    download_training_images,
    apply_d4_torch, inverse_d4_torch, normalize_depth,
    DepthAnythingTrainable,
    evaluate_warrant_gap,
)

MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
DEVICE = "cuda"
OUT_DIR = Path("out_depth_consistency")


def freeze_backbone(model_wrapper: DepthAnythingTrainable):
    """Freeze backbone, keep neck+head trainable. Returns param counts."""
    model = model_wrapper.model
    
    total = sum(p.numel() for p in model.parameters())
    
    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False
    
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Parameters: total={total:,}, frozen={frozen:,}, trainable={trainable:,}")
    print(f"  Backbone (frozen): {sum(p.numel() for p in model.backbone.parameters()):,}")
    print(f"  Neck (trainable):  {sum(p.numel() for p in model.neck.parameters()):,}")
    print(f"  Head (trainable):  {sum(p.numel() for p in model.head.parameters()):,}")
    
    return {"total": total, "frozen": frozen, "trainable": trainable}


def train_frozen(
    student: DepthAnythingTrainable,
    teacher: DepthAnythingTrainable,
    train_imgs: list[np.ndarray],
    steps: int = 500,
    batch_size: int = 2,
    lr: float = 1e-4,
    alpha: float = 0.5,
    curriculum: tuple[float, float] | None = None,  # (start_alpha, end_alpha)
    log_every: int = 25,
) -> list[dict]:
    """Train with frozen backbone."""
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student.train()
    # Only optimize trainable params
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    rng = np.random.default_rng(42)
    logs = []
    t0 = time.time()

    for step in range(1, steps + 1):
        # Curriculum alpha
        if curriculum:
            current_alpha = curriculum[0] + (curriculum[1] - curriculum[0]) * (step - 1) / max(steps - 1, 1)
        else:
            current_alpha = alpha

        idxs = rng.integers(0, len(train_imgs), size=batch_size)
        batch_imgs = [train_imgs[i] for i in idxs]
        d4_idxs = rng.integers(1, 8, size=batch_size)

        from mezzanine.symmetries.depth_geometric import apply_d4
        transformed_imgs = [apply_d4(img, int(d4_idx)) for img, d4_idx in zip(batch_imgs, d4_idxs)]

        pv_orig = student.preprocess_numpy(batch_imgs)
        pv_trans = student.preprocess_numpy(transformed_imgs)

        depth_orig = student(pv_orig)
        depth_trans = student(pv_trans)

        depth_trans_aligned = torch.stack([
            inverse_d4_torch(depth_trans[i:i+1], int(d4_idxs[i]))[0]
            for i in range(batch_size)
        ])

        L_consistency = F.mse_loss(depth_orig, depth_trans_aligned)

        with torch.no_grad():
            pv_orig_t = teacher.preprocess_numpy(batch_imgs)
            depth_teacher = teacher(pv_orig_t)

        L_anchor = F.mse_loss(depth_orig, depth_teacher)

        loss = current_alpha * L_consistency + (1 - current_alpha) * L_anchor

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == 1:
            log = {
                "step": step, "loss": loss.item(),
                "L_consistency": L_consistency.item(),
                "L_anchor": L_anchor.item(),
                "alpha": current_alpha,
                "elapsed": time.time() - t0,
            }
            logs.append(log)
            print(f"  Step {step:4d} | α={current_alpha:.3f} | loss={loss.item():.6f} | "
                  f"consist={L_consistency.item():.6f} | anchor={L_anchor.item():.6f}")

    return logs


def evaluate_depth_quality(
    student: DepthAnythingTrainable,
    teacher: DepthAnythingTrainable,
    test_images: list[tuple[str, np.ndarray]],
) -> list[dict]:
    """Measure correlation between student and teacher depth predictions."""
    student.eval()
    teacher.eval()
    results = []
    
    for name, img in test_images:
        pv_s = student.preprocess_numpy([img])
        pv_t = teacher.preprocess_numpy([img])
        
        with torch.no_grad():
            ds = student(pv_s)[0].cpu().numpy().flatten()
            dt = teacher(pv_t)[0].cpu().numpy().flatten()
        
        pearson_r, _ = stats.pearsonr(ds, dt)
        spearman_r, _ = stats.spearmanr(ds, dt)
        
        results.append({
            "image": name,
            "pearson": float(pearson_r),
            "spearman": float(spearman_r),
        })
        print(f"  {name}: Pearson={pearson_r:.4f}, Spearman={spearman_r:.4f}")
    
    return results


def run_ablation(
    alpha: float,
    train_imgs: list[np.ndarray],
    test_images: list[tuple[str, np.ndarray]],
    curriculum: tuple[float, float] | None = None,
    steps: int = 500,
) -> dict:
    """Run one ablation config and return results."""
    label = f"curriculum_{curriculum[0]}-{curriculum[1]}" if curriculum else f"alpha_{alpha}"
    print(f"\n{'='*60}")
    print(f"CONFIG: {label} | steps={steps}")
    print(f"{'='*60}")

    student = DepthAnythingTrainable(MODEL_NAME, DEVICE)
    teacher = DepthAnythingTrainable(MODEL_NAME, DEVICE)
    
    param_counts = freeze_backbone(student)

    logs = train_frozen(student, teacher, train_imgs, steps=steps,
                        alpha=alpha, curriculum=curriculum)

    print("\n  Evaluating warrant gap...")
    wg_results = evaluate_warrant_gap(student, test_images)
    mean_cv = float(np.mean([r["mean_cv"] for r in wg_results]))

    print("\n  Evaluating depth quality...")
    dq_results = evaluate_depth_quality(student, teacher, test_images)
    mean_pearson = float(np.mean([r["pearson"] for r in dq_results]))
    mean_spearman = float(np.mean([r["spearman"] for r in dq_results]))

    result = {
        "alpha": alpha if not curriculum else f"{curriculum[0]}->{curriculum[1]}",
        "curriculum": curriculum,
        "mean_cv": mean_cv,
        "mean_pearson": mean_pearson,
        "mean_spearman": mean_spearman,
        "param_counts": param_counts,
        "training_log": logs,
        "warrant_gap_detail": wg_results,
        "depth_quality_detail": dq_results,
    }
    
    print(f"\n  RESULT: CV={mean_cv:.4f}, Pearson={mean_pearson:.4f}, Spearman={mean_spearman:.4f}")
    
    # Clean up GPU memory
    del student, teacher
    torch.cuda.empty_cache()
    
    return result


def make_pareto_plot(v2_results: list[dict], out_path: Path):
    """Generate Pareto plot comparing v1 (full fine-tune) and v2 (frozen backbone)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # V1 results (from prior experiments)
    v1_points = [
        {"label": "v1 α=0.95", "cv": 0.035, "pearson": 0.048},
        {"label": "v1 α=0.8", "cv": 0.051, "pearson": 0.120},
        {"label": "v1 α=0.5", "cv": 0.116, "pearson": 0.259},
    ]

    # Plot v1
    for p in v1_points:
        ax.scatter(p["cv"], p["pearson"], c="red", s=100, zorder=5, marker="x", linewidths=2)
        ax.annotate(p["label"], (p["cv"], p["pearson"]), textcoords="offset points",
                    xytext=(8, 5), fontsize=8, color="red")

    # Plot v2
    for r in v2_results:
        label = f"v2 α={r['alpha']}"
        ax.scatter(r["mean_cv"], r["mean_pearson"], c="blue", s=100, zorder=5, marker="o")
        ax.annotate(label, (r["mean_cv"], r["mean_pearson"]), textcoords="offset points",
                    xytext=(8, 5), fontsize=8, color="blue")

    # Ideal region
    ax.axhline(y=0.85, color="green", linestyle="--", alpha=0.5, label="Quality threshold (Pearson=0.85)")
    ax.axvline(x=0.15, color="orange", linestyle="--", alpha=0.5, label="Equivariance threshold (CV=0.15)")

    # Shade the "good" quadrant
    ax.axhspan(0.85, 1.0, xmin=0, xmax=0.15/ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 0.5,
               alpha=0.1, color="green")

    ax.set_xlabel("Mean CV (lower = better equivariance)", fontsize=12)
    ax.set_ylabel("Mean Pearson (higher = better depth quality)", fontsize=12)
    ax.set_title("Pareto: Frozen Backbone (v2) vs Full Fine-Tune (v1)", fontsize=14)
    ax.legend(loc="lower right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Add baseline point
    ax.scatter([0.186], [1.0], c="gray", s=150, zorder=5, marker="*")
    ax.annotate("Baseline (no FT)", (0.186, 1.0), textcoords="offset points",
                xytext=(8, -10), fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Pareto plot saved to {out_path}")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load test images
    print("Loading test images...")
    test_dir = Path("test_images")
    test_images = []
    for i in range(1, 6):
        p = test_dir / f"topdown_robot_0{i}.jpg"
        if p.exists():
            img = np.asarray(Image.open(p).convert("RGB"))
            test_images.append((p.stem, img))
    print(f"  {len(test_images)} test images")

    # Download training images
    print("\nDownloading training images...")
    train_imgs = download_training_images(60)

    # Measure baseline depth quality (teacher vs teacher = perfect correlation)
    print("\n" + "="*60)
    print("BASELINE evaluation")
    print("="*60)
    baseline_model = DepthAnythingTrainable(MODEL_NAME, DEVICE)
    baseline_wg = evaluate_warrant_gap(baseline_model, test_images)
    baseline_cv = float(np.mean([r["mean_cv"] for r in baseline_wg]))
    print(f"  Baseline mean CV: {baseline_cv:.4f}")
    del baseline_model
    torch.cuda.empty_cache()

    # Run ablations
    all_results = []
    
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        result = run_ablation(alpha, train_imgs, test_images)
        all_results.append(result)

    # Curriculum: α ramps 0.2 → 0.8
    curriculum_result = run_ablation(0.0, train_imgs, test_images, curriculum=(0.2, 0.8))

    # Find best config
    best = None
    for r in all_results + [curriculum_result]:
        if best is None or (r["mean_cv"] < 0.15 and r["mean_pearson"] > (best.get("mean_pearson", 0))):
            best = r
    if best is None or best["mean_cv"] >= 0.15:
        # Just pick the one with best Pearson among those with CV < 0.2
        candidates = [r for r in all_results + [curriculum_result] if r["mean_cv"] < 0.2]
        if candidates:
            best = max(candidates, key=lambda r: r["mean_pearson"])
        else:
            best = max(all_results + [curriculum_result], key=lambda r: r["mean_pearson"])

    # Save results
    output = {
        "approach": "frozen_backbone",
        "trainable_params": all_results[0]["param_counts"]["trainable"],
        "frozen_params": all_results[0]["param_counts"]["frozen"],
        "baseline_cv": baseline_cv,
        "ablations": [
            {
                "alpha": r["alpha"],
                "mean_cv": r["mean_cv"],
                "mean_pearson": r["mean_pearson"],
                "mean_spearman": r["mean_spearman"],
            }
            for r in all_results
        ],
        "curriculum": {
            "schedule": "0.2->0.8 linear over 500 steps",
            "mean_cv": curriculum_result["mean_cv"],
            "mean_pearson": curriculum_result["mean_pearson"],
            "mean_spearman": curriculum_result["mean_spearman"],
        },
        "best_config": f"alpha={best['alpha']}, CV={best['mean_cv']:.4f}, Pearson={best['mean_pearson']:.4f}",
    }

    results_path = OUT_DIR / "v2_frozen_backbone_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate Pareto plot
    make_pareto_plot(all_results + [curriculum_result], OUT_DIR / "v2_pareto.png")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline CV: {baseline_cv:.4f}")
    for r in all_results:
        meets = "✓" if r["mean_cv"] < 0.15 and r["mean_pearson"] > 0.85 else "✗"
        print(f"  α={r['alpha']}: CV={r['mean_cv']:.4f}, Pearson={r['mean_pearson']:.4f}, Spearman={r['mean_spearman']:.4f} {meets}")
    r = curriculum_result
    meets = "✓" if r["mean_cv"] < 0.15 and r["mean_pearson"] > 0.85 else "✗"
    print(f"  Curriculum 0.2→0.8: CV={r['mean_cv']:.4f}, Pearson={r['mean_pearson']:.4f}, Spearman={r['mean_spearman']:.4f} {meets}")
    print(f"\nBest: {output['best_config']}")


if __name__ == "__main__":
    main()
