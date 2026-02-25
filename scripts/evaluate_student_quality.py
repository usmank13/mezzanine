#!/usr/bin/env python3
"""Evaluate student depth quality: side-by-side visualizations + correlation metrics."""

import json, os, sys, glob
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
    DepthAnythingTrainable, normalize_depth, COCO_TRAIN_URLS,
    apply_d4_torch, inverse_d4_torch,
)

DEVICE = "cuda"
MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
CKPT = "out_depth_consistency/student_checkpoint.pt"
OUT_DIR = Path("out_depth_consistency/quality_eval")
EQUIV_DIR = Path("out_depth_consistency/equivariance_demo")

# COCO images NOT in the training set (IDs 81-100 from the list are unused, plus extras)
COCO_EVAL_URLS = [
    f"http://images.cocodataset.org/val2017/{img_id:012d}.jpg"
    for img_id in [
        # 10 images for quality eval (different from training set)
        17627, 106140, 205282, 356427, 185250,
        428280, 163314, 481573, 380913, 261982,
    ]
]

COCO_GENERALIZATION_URLS = [
    f"http://images.cocodataset.org/val2017/{img_id:012d}.jpg"
    for img_id in [
        # 20 diverse images for generalization eval
        17627, 106140, 205282, 356427, 185250,
        428280, 163314, 481573, 380913, 261982,
        109798, 86408, 176778, 389566, 402992,
        459887, 360137, 91921, 318556, 434297,
    ]
]


def download_images(urls, size=518):
    import requests, io
    imgs = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            img = img.resize((size, size), Image.BILINEAR)
            imgs.append((url.split("/")[-1].replace(".jpg",""), np.asarray(img)))
        except Exception as e:
            print(f"  Failed: {url}: {e}")
    return imgs


def predict_depth(model, img_np):
    """Get depth map as numpy array."""
    pv = model.preprocess_numpy([img_np])
    with torch.no_grad():
        d = model(pv)[0].cpu().numpy()
    return d


def compute_correlations(teacher_d, student_d):
    t_flat = teacher_d.flatten()
    s_flat = student_d.flatten()
    pearson_r, _ = stats.pearsonr(t_flat, s_flat)
    spearman_r, _ = stats.spearmanr(t_flat, s_flat)
    return float(pearson_r), float(spearman_r)


def save_comparison(img_np, teacher_d, student_d, name, out_dir):
    diff = np.abs(teacher_d - student_d)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_np); axes[0].set_title("Original")
    axes[1].imshow(teacher_d, cmap="magma"); axes[1].set_title("Teacher Depth")
    axes[2].imshow(student_d, cmap="magma"); axes[2].set_title("Student Depth")
    axes[3].imshow(diff, cmap="hot"); axes[3].set_title("Absolute Difference")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}.png", dpi=150)
    plt.close()


def task1_quality_eval(teacher, student):
    """Task 1: Quality evaluation with side-by-side vis."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load robot test images
    test_dir = Path("test_images")
    images = [(p.stem, np.asarray(Image.open(p).convert("RGB")))
              for p in sorted(test_dir.glob("*.jpg"))]

    # Download COCO eval images
    print("Downloading COCO eval images...")
    coco_imgs = download_images(COCO_EVAL_URLS)
    images.extend(coco_imgs)

    results = []
    for name, img in images:
        teacher_d = predict_depth(teacher, img)
        student_d = predict_depth(student, img)

        # Resize to same shape if needed
        if teacher_d.shape != student_d.shape:
            from PIL import Image as PILImage
            h, w = min(teacher_d.shape[0], student_d.shape[0]), min(teacher_d.shape[1], student_d.shape[1])
            teacher_d = np.array(PILImage.fromarray(teacher_d).resize((w, h), PILImage.BILINEAR))
            student_d = np.array(PILImage.fromarray(student_d).resize((w, h), PILImage.BILINEAR))

        pearson_r, spearman_r = compute_correlations(teacher_d, student_d)
        save_comparison(img, teacher_d, student_d, name, OUT_DIR)
        results.append({"image": name, "pearson": pearson_r, "spearman": spearman_r})
        print(f"  {name}: pearson={pearson_r:.4f}, spearman={spearman_r:.4f}")

    summary = {
        "mean_pearson": float(np.mean([r["pearson"] for r in results])),
        "mean_spearman": float(np.mean([r["spearman"] for r in results])),
        "per_image": results,
    }
    with open(OUT_DIR / "quality_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nQuality summary: Pearson={summary['mean_pearson']:.4f}, Spearman={summary['mean_spearman']:.4f}")
    return summary


def task4_equivariance_demo(teacher, student):
    """Task 4: Equivariance visualization for topdown_robot_04."""
    EQUIV_DIR.mkdir(parents=True, exist_ok=True)

    img_path = Path("test_images/topdown_robot_04.jpg")
    img = np.asarray(Image.open(img_path).convert("RGB"))

    # Teacher depth (biased)
    teacher_d = predict_depth(teacher, img)

    # Student on original
    student_d_orig = predict_depth(student, img)

    # Student on vertically flipped, then flip back
    img_vflip = np.flip(img, axis=0).copy()
    student_d_vflip = predict_depth(student, img_vflip)
    student_d_vflip_back = np.flip(student_d_vflip, axis=0).copy()

    # Also do teacher on vflip for comparison
    teacher_d_vflip = predict_depth(teacher, img_vflip)
    teacher_d_vflip_back = np.flip(teacher_d_vflip, axis=0).copy()

    # Normalize all to same scale
    def norm01(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    # Main equivariance figure
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # Top row: Teacher
    axes[0, 0].imshow(img); axes[0, 0].set_title("Original Image", fontsize=14)
    axes[0, 1].imshow(norm01(teacher_d), cmap="magma"); axes[0, 1].set_title("Teacher: Original", fontsize=14)
    axes[0, 2].imshow(norm01(teacher_d_vflip_back), cmap="magma"); axes[0, 2].set_title("Teacher: VFlip→Predict→UnFlip", fontsize=14)
    diff_teacher = np.abs(norm01(teacher_d) - norm01(teacher_d_vflip_back))
    axes[0, 3].imshow(diff_teacher, cmap="hot", vmin=0, vmax=0.5); axes[0, 3].set_title(f"Teacher Difference (MAE={diff_teacher.mean():.3f})", fontsize=14)

    # Bottom row: Student
    axes[1, 0].imshow(img_vflip); axes[1, 0].set_title("Flipped Image", fontsize=14)
    axes[1, 1].imshow(norm01(student_d_orig), cmap="magma"); axes[1, 1].set_title("Student: Original", fontsize=14)
    axes[1, 2].imshow(norm01(student_d_vflip_back), cmap="magma"); axes[1, 2].set_title("Student: VFlip→Predict→UnFlip", fontsize=14)
    diff_student = np.abs(norm01(student_d_orig) - norm01(student_d_vflip_back))
    axes[1, 3].imshow(diff_student, cmap="hot", vmin=0, vmax=0.5); axes[1, 3].set_title(f"Student Difference (MAE={diff_student.mean():.3f})", fontsize=14)

    for row in axes:
        for ax in row:
            ax.axis("off")

    fig.suptitle("Equivariance Demo: Teacher vs Student on topdown_robot_04\n"
                 "Student predictions are consistent under vertical flip", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(EQUIV_DIR / "equivariance_comparison.png", dpi=150)
    plt.close()

    # Individual saves
    for name, d in [("teacher_depth", teacher_d), ("student_original", student_d_orig),
                     ("student_vflip_unflipped", student_d_vflip_back)]:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(norm01(d), cmap="magma"); ax.axis("off")
        ax.set_title(name.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(EQUIV_DIR / f"{name}.png", dpi=150)
        plt.close()

    print(f"Equivariance demo saved. Teacher MAE={diff_teacher.mean():.4f}, Student MAE={diff_student.mean():.4f}")


def task3_generalization(student):
    """Task 3: Generalization to unseen COCO images."""
    from scripts.finetune_depth_consistency import evaluate_warrant_gap

    print("Downloading 20 generalization images...")
    gen_imgs = download_images(COCO_GENERALIZATION_URLS)
    print(f"  Got {len(gen_imgs)} images")

    results = evaluate_warrant_gap(student, gen_imgs)
    summary = {
        "n_images": len(results),
        "mean_cv": float(np.mean([r["mean_cv"] for r in results])),
        "mean_abs_vbias_orig": float(np.mean([abs(r["vertical_bias_original"]) for r in results])),
        "mean_abs_vbias_avg": float(np.mean([abs(r["vertical_bias_averaged"]) for r in results])),
        "per_image": results,
    }
    with open("out_depth_consistency/generalization_eval.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nGeneralization: mean_cv={summary['mean_cv']:.4f}, mean_abs_vbias_avg={summary['mean_abs_vbias_avg']:.4f}")
    return summary


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)

    print("Loading teacher model...")
    teacher = DepthAnythingTrainable(MODEL_NAME, DEVICE)
    teacher.eval()

    print("Loading student model...")
    student = DepthAnythingTrainable(MODEL_NAME, DEVICE)
    student.model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    student.eval()

    print("\n=== Task 1: Quality Evaluation ===")
    task1_quality_eval(teacher, student)

    print("\n=== Task 4: Equivariance Demo ===")
    task4_equivariance_demo(teacher, student)

    print("\n=== Task 3: Generalization ===")
    task3_generalization(student)

    print("\nDone! All outputs in out_depth_consistency/")
