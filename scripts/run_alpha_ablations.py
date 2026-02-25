#!/usr/bin/env python3
"""Run α ablations for consistency distillation quality-equivariance tradeoff.

Tests α ∈ {0.1, 0.3, 0.5, 0.8} plus a curriculum schedule.
For each: train 500 steps from fresh pretrained weights, measure warrant gap + depth correlation.
"""

import copy, json, os, sys, time, gc
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.finetune_depth_consistency import (
    DepthAnythingTrainable, normalize_depth, download_training_images,
    evaluate_warrant_gap, train,
    apply_d4_torch, inverse_d4_torch,
)

DEVICE = "cuda"
MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
OUT_DIR = Path("out_depth_consistency")


def measure_correlation(teacher, student, test_images):
    """Measure teacher-student depth correlation on test images."""
    teacher.eval()
    student.eval()
    results = []
    for name, img in test_images:
        pv_t = teacher.preprocess_numpy([img])
        pv_s = student.preprocess_numpy([img])
        with torch.no_grad():
            d_t = teacher(pv_t)[0].cpu().numpy()
            d_s = student(pv_s)[0].cpu().numpy()
        # Resize to same shape
        if d_t.shape != d_s.shape:
            h, w = min(d_t.shape[0], d_s.shape[0]), min(d_t.shape[1], d_s.shape[1])
            d_t = np.array(Image.fromarray(d_t).resize((w, h), Image.BILINEAR))
            d_s = np.array(Image.fromarray(d_s).resize((w, h), Image.BILINEAR))
        # Normalize to [0,1]
        d_t = (d_t - d_t.min()) / (d_t.max() - d_t.min() + 1e-8)
        d_s = (d_s - d_s.min()) / (d_s.max() - d_s.min() + 1e-8)
        pearson_r, _ = stats.pearsonr(d_t.flatten(), d_s.flatten())
        spearman_r, _ = stats.spearmanr(d_t.flatten(), d_s.flatten())
        results.append({"image": name, "pearson": float(pearson_r), "spearman": float(spearman_r)})
    mean_p = float(np.mean([r["pearson"] for r in results]))
    mean_s = float(np.mean([r["spearman"] for r in results]))
    return mean_p, mean_s, results


def train_curriculum(student, teacher, train_imgs, test_images, device="cuda"):
    """Two-phase curriculum: α=0.3 for steps 1-250, α=0.7 for 251-500."""
    import torch.nn.functional as F
    from mezzanine.symmetries.depth_geometric import apply_d4

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-4)
    
    rng = np.random.default_rng(42)
    batch_size = 2
    logs = []
    t0 = time.time()

    for step in range(1, 501):
        alpha = 0.3 if step <= 250 else 0.7

        idxs = rng.integers(0, len(train_imgs), size=batch_size)
        batch_imgs = [train_imgs[i] for i in idxs]
        d4_idxs = rng.integers(1, 8, size=batch_size)

        transformed_imgs = []
        for img, d4_idx in zip(batch_imgs, d4_idxs):
            t_img = apply_d4(img, int(d4_idx))
            transformed_imgs.append(t_img)

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

        loss = alpha * L_consistency + (1 - alpha) * L_anchor
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(f"  [curriculum] Step {step:4d} α={alpha:.1f} | loss={loss.item():.6f} | "
                  f"consist={L_consistency.item():.6f} | anchor={L_anchor.item():.6f}")

    return logs


def run_single_ablation(alpha, train_imgs, test_images):
    """Train with given α from scratch, return metrics."""
    print(f"\n{'='*60}")
    print(f"ABLATION α={alpha}")
    print(f"{'='*60}")

    # Fresh models each time
    student = DepthAnythingTrainable(MODEL_NAME, DEVICE)
    teacher = DepthAnythingTrainable(MODEL_NAME, DEVICE)

    # Train
    train_logs = train(
        student, teacher, train_imgs,
        steps=500, batch_size=2, lr=1e-4, alpha=alpha, device=DEVICE,
    )

    # Measure warrant gap
    print(f"\nEvaluating warrant gap (α={alpha})...")
    wg_results = evaluate_warrant_gap(student, test_images)
    mean_cv = float(np.mean([r["mean_cv"] for r in wg_results]))

    # Measure correlation
    print(f"Evaluating teacher-student correlation (α={alpha})...")
    mean_p, mean_s, corr_results = measure_correlation(teacher, student, test_images)
    print(f"  → mean_cv={mean_cv:.4f}, pearson={mean_p:.4f}, spearman={mean_s:.4f}")

    # Save checkpoint
    ckpt_path = OUT_DIR / f"student_alpha_{alpha}.pt"
    torch.save(student.model.state_dict(), ckpt_path)

    # Cleanup
    del student, teacher
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "alpha": alpha,
        "steps": 500,
        "mean_cv": mean_cv,
        "mean_pearson": mean_p,
        "mean_spearman": mean_s,
        "per_image_wg": wg_results,
        "per_image_corr": corr_results,
    }


def run_curriculum_ablation(train_imgs, test_images):
    """Curriculum: α=0.3 (steps 1-250) → α=0.7 (steps 251-500)."""
    print(f"\n{'='*60}")
    print(f"CURRICULUM: α=0.3→0.7")
    print(f"{'='*60}")

    student = DepthAnythingTrainable(MODEL_NAME, DEVICE)
    teacher = DepthAnythingTrainable(MODEL_NAME, DEVICE)

    train_curriculum(student, teacher, train_imgs, test_images, DEVICE)

    print(f"\nEvaluating curriculum...")
    wg_results = evaluate_warrant_gap(student, test_images)
    mean_cv = float(np.mean([r["mean_cv"] for r in wg_results]))
    mean_p, mean_s, corr_results = measure_correlation(teacher, student, test_images)
    print(f"  → mean_cv={mean_cv:.4f}, pearson={mean_p:.4f}, spearman={mean_s:.4f}")

    ckpt_path = OUT_DIR / "student_curriculum.pt"
    torch.save(student.model.state_dict(), ckpt_path)

    del student, teacher
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "schedule": "α=0.3 (1-250) → α=0.7 (251-500)",
        "steps": 500,
        "mean_cv": mean_cv,
        "mean_pearson": mean_p,
        "mean_spearman": mean_s,
        "per_image_wg": wg_results,
        "per_image_corr": corr_results,
    }


def make_pareto_plot(ablations, curriculum, out_path):
    """Pareto plot: warrant gap (x) vs depth quality (y=Pearson)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    alphas = [a["alpha"] for a in ablations]
    cvs = [a["mean_cv"] for a in ablations]
    pearsons = [a["mean_pearson"] for a in ablations]

    ax.scatter(cvs, pearsons, s=120, c="steelblue", zorder=5)
    for a, cv, p in zip(alphas, cvs, pearsons):
        ax.annotate(f"α={a}", (cv, p), textcoords="offset points",
                    xytext=(8, 8), fontsize=11, fontweight="bold")

    # Curriculum point
    ax.scatter([curriculum["mean_cv"]], [curriculum["mean_pearson"]],
               s=150, c="red", marker="*", zorder=6)
    ax.annotate("curriculum", (curriculum["mean_cv"], curriculum["mean_pearson"]),
                textcoords="offset points", xytext=(8, -12), fontsize=11, color="red")

    # Reference lines
    ax.axhline(y=0.9, color="green", linestyle="--", alpha=0.5, label="Quality threshold (Pearson=0.9)")
    ax.axvline(x=0.1, color="orange", linestyle="--", alpha=0.5, label="Equivariance threshold (CV=0.1)")

    ax.set_xlabel("Warrant Gap (Mean CV) — lower is better →", fontsize=12)
    ax.set_ylabel("Depth Quality (Pearson with Teacher) — higher is better →", fontsize=12)
    ax.set_title("Quality vs Equivariance Pareto Tradeoff", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Pareto plot saved to {out_path}")


def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load test images
    print("Loading test images...")
    test_dir = Path("test_images")
    test_images = [(p.stem, np.asarray(Image.open(p).convert("RGB")))
                   for p in sorted(test_dir.glob("*.jpg"))]
    print(f"  {len(test_images)} test images")

    # Download training images once
    print("\nDownloading training images...")
    train_imgs = download_training_images(60)

    # Run ablations
    ablation_results = []
    for alpha in [0.1, 0.3, 0.5, 0.8]:
        result = run_single_ablation(alpha, train_imgs, test_images)
        ablation_results.append(result)
        # Save intermediate results
        with open(OUT_DIR / "ablation_results_partial.json", "w") as f:
            json.dump({"ablations": ablation_results}, f, indent=2)

    # Curriculum
    curriculum_result = run_curriculum_ablation(train_imgs, test_images)

    # Determine recommendation
    # Find configs meeting both thresholds
    good = [a for a in ablation_results if a["mean_cv"] < 0.1 and a["mean_pearson"] > 0.9]
    if good:
        best = max(good, key=lambda x: x["mean_pearson"] - x["mean_cv"])
        rec = f"α={best['alpha']} achieves both targets (CV={best['mean_cv']:.4f}, Pearson={best['mean_pearson']:.4f})"
    else:
        # Check curriculum
        if curriculum_result["mean_cv"] < 0.1 and curriculum_result["mean_pearson"] > 0.9:
            rec = f"Curriculum approach achieves both targets (CV={curriculum_result['mean_cv']:.4f}, Pearson={curriculum_result['mean_pearson']:.4f})"
        else:
            # Find best tradeoff
            all_opts = ablation_results + [curriculum_result]
            # Rank by sum of normalized metrics
            best = max(all_opts, key=lambda x: x["mean_pearson"] - x["mean_cv"] * 5)
            label = f"α={best.get('alpha', 'curriculum')}" if 'alpha' in best else "curriculum"
            rec = f"No config meets both thresholds. Best tradeoff: {label} (CV={best['mean_cv']:.4f}, Pearson={best['mean_pearson']:.4f}). Consider more steps or lower α."

    # Save full results
    output = {
        "ablations": ablation_results,
        "curriculum": curriculum_result,
        "recommendation": rec,
    }
    with open(OUT_DIR / "ablation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUT_DIR / 'ablation_results.json'}")
    print(f"Recommendation: {rec}")

    # Pareto plot
    make_pareto_plot(ablation_results, curriculum_result,
                     OUT_DIR / "pareto_quality_vs_equivariance.png")


if __name__ == "__main__":
    main()
