#!/usr/bin/env python3
"""Task 2: Ablation experiments - vary steps and alpha."""

import json, os, sys, time, copy
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.finetune_depth_consistency import (
    DepthAnythingTrainable, download_training_images, train, evaluate_warrant_gap,
)

DEVICE = "cuda"
MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
OUT_DIR = Path("out_depth_consistency")


def compute_quality(teacher, student, test_images):
    """Compute Pearson/Spearman correlation between teacher and student."""
    correlations = []
    for name, img in test_images:
        pv_t = teacher.preprocess_numpy([img])
        pv_s = student.preprocess_numpy([img])
        with torch.no_grad():
            d_t = teacher(pv_t)[0].cpu().numpy().flatten()
            d_s = student(pv_s)[0].cpu().numpy().flatten()
        pr, _ = stats.pearsonr(d_t, d_s)
        sr, _ = stats.spearmanr(d_t, d_s)
        correlations.append({"image": name, "pearson": float(pr), "spearman": float(sr)})
    return {
        "mean_pearson": float(np.mean([c["pearson"] for c in correlations])),
        "mean_spearman": float(np.mean([c["spearman"] for c in correlations])),
        "per_image": correlations,
    }


def run_ablation(teacher, train_imgs, test_images, steps, alpha, lr=1e-4, bs=2):
    """Run one ablation experiment."""
    print(f"\n{'='*60}")
    print(f"ABLATION: steps={steps}, α={alpha}")
    print(f"{'='*60}")

    # Fresh student each time
    student = DepthAnythingTrainable(MODEL_NAME, DEVICE)
    
    train_logs = train(student, teacher, train_imgs,
                       steps=steps, batch_size=bs, lr=lr, alpha=alpha,
                       device=DEVICE, log_every=50)
    
    # Evaluate warrant gap
    print("\nEvaluating warrant gap...")
    wg_results = evaluate_warrant_gap(student, test_images)
    wg_summary = {
        "mean_cv": float(np.mean([r["mean_cv"] for r in wg_results])),
        "mean_abs_vbias_avg": float(np.mean([abs(r["vertical_bias_averaged"]) for r in wg_results])),
    }

    # Evaluate quality
    print("Evaluating depth quality...")
    quality = compute_quality(teacher, student, test_images)

    # Save checkpoint
    ckpt_name = f"ablation_s{steps}_a{alpha}.pt"
    torch.save(student.model.state_dict(), OUT_DIR / ckpt_name)

    del student
    torch.cuda.empty_cache()

    return {
        "config": {"steps": steps, "alpha": alpha, "lr": lr, "batch_size": bs},
        "warrant_gap": wg_summary,
        "quality": {"mean_pearson": quality["mean_pearson"], "mean_spearman": quality["mean_spearman"]},
        "final_loss": train_logs[-1] if train_logs else None,
        "checkpoint": ckpt_name,
    }


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    # Load test images
    test_dir = Path("test_images")
    test_images = [(p.stem, np.asarray(Image.open(p).convert("RGB")))
                   for p in sorted(test_dir.glob("*.jpg"))]

    # Download training images
    print("Downloading training images...")
    train_imgs = download_training_images(60)

    # Load teacher (frozen, shared)
    print("Loading teacher model...")
    teacher = DepthAnythingTrainable(MODEL_NAME, DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Run ablations
    ablations = [
        {"steps": 1000, "alpha": 0.8},   # More steps
        {"steps": 500, "alpha": 0.5},    # More anchor weight
        {"steps": 500, "alpha": 0.95},   # Almost pure consistency
    ]

    results = []
    for config in ablations:
        result = run_ablation(teacher, train_imgs, test_images, **config)
        results.append(result)
        print(f"\n  → CV={result['warrant_gap']['mean_cv']:.4f}, "
              f"Pearson={result['quality']['mean_pearson']:.4f}")

    # Save all results
    with open(OUT_DIR / "ablation_results.json", "w") as f:
        json.dump({"ablations": results}, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Config':>25s} | {'Mean CV':>8s} | {'|VBias|':>8s} | {'Pearson':>8s}")
    print(f"{'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    # Include baseline from results.json
    with open(OUT_DIR / "results.json") as f:
        orig = json.load(f)
    print(f"{'baseline':>25s} | {orig['baseline']['summary']['mean_cv']:>8.4f} | {orig['baseline']['summary']['mean_abs_vbias_avg']:>8.4f} | {'1.0000':>8s}")
    print(f"{'s500 α=0.8 (original)':>25s} | {orig['finetuned']['summary']['mean_cv']:>8.4f} | {orig['finetuned']['summary']['mean_abs_vbias_avg']:>8.4f} | {'N/A':>8s}")
    for r in results:
        label = f"s{r['config']['steps']} α={r['config']['alpha']}"
        print(f"{label:>25s} | {r['warrant_gap']['mean_cv']:>8.4f} | {r['warrant_gap']['mean_abs_vbias_avg']:>8.4f} | {r['quality']['mean_pearson']:>8.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
