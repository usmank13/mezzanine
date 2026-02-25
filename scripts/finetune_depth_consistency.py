#!/usr/bin/env python3
"""Consistency distillation for Depth Anything V2 under D4 symmetry.

Fine-tunes Depth Anything so that its predictions are equivariant under
D4 geometric transforms (rotations + reflections).  Uses two losses:

  L_consistency = MSE(depth(img), inverse_g(depth(g(img))))
  L_anchor      = MSE(depth_student(img), depth_teacher(img))
  L_total       = α * L_consistency + (1-α) * L_anchor

The teacher is a frozen copy of the original model.  Training data is
fetched from COCO val URLs (no labels needed — this is self-supervised).

Usage:
    python scripts/finetune_depth_consistency.py \
        --steps 500 --alpha 0.8 --lr 1e-4 --batch-size 2 \
        --out out_depth_consistency
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ── COCO training images ────────────────────────────────────────────────

COCO_TRAIN_URLS = [
    f"http://images.cocodataset.org/val2017/{img_id:012d}.jpg"
    for img_id in [
        # 80 diverse COCO val images
        39769, 397133, 37777, 80340, 227765, 146457, 183648, 460347,
        309391, 178028, 296649, 386912, 502136, 448365, 312421, 388056,
        297343, 84477, 511599, 458054, 360661, 360943, 15335, 106563,
        429281, 404568, 283520, 370677, 153343, 218439, 428454, 493286,
        414034, 292456, 356387, 64574, 370999, 309391, 571313, 579321,
        127517, 487583, 46252, 394940, 182611, 226111, 308531, 456394,
        61584, 383386, 558840, 462565, 153299, 301135, 400082, 506656,
        210273, 169076, 414795, 532058, 540414, 22192, 197388, 79144,
        335177, 544565, 174482, 131386, 226903, 101068, 162092, 323355,
        84270, 455555, 321214, 239274, 443303, 284991, 404922, 459467,
    ]
]


def download_training_images(n: int = 60) -> list[np.ndarray]:
    """Download n COCO images for training."""
    import requests
    import io

    imgs = []
    for url in COCO_TRAIN_URLS[:n + 20]:  # fetch extra in case some fail
        if len(imgs) >= n:
            break
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            # Resize to 518x518 (Depth Anything native resolution)
            img = img.resize((518, 518), Image.BILINEAR)
            imgs.append(np.asarray(img))
        except Exception:
            continue
    print(f"Downloaded {len(imgs)} training images")
    return imgs


# ── D4 transforms (torch versions) ──────────────────────────────────────

def apply_d4_torch(x: torch.Tensor, idx: int) -> torch.Tensor:
    """Apply D4 transform to a tensor. x: (B,C,H,W) or (B,H,W)."""
    if idx == 0:
        return x
    elif idx == 1:  # rot90 CW
        return torch.rot90(x, -1, [-2, -1])
    elif idx == 2:
        return torch.rot90(x, 2, [-2, -1])
    elif idx == 3:  # rot270 CW
        return torch.rot90(x, 1, [-2, -1])
    elif idx == 4:  # vflip
        return torch.flip(x, [-2])
    elif idx == 5:  # hflip
        return torch.flip(x, [-1])
    elif idx == 6:  # transpose
        return x.transpose(-2, -1)
    elif idx == 7:  # anti-transpose = rot90 + vflip
        return torch.flip(torch.rot90(x, 1, [-2, -1]), [-2])
    raise ValueError(f"Invalid D4 index: {idx}")


_D4_INV = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}


def inverse_d4_torch(x: torch.Tensor, idx: int) -> torch.Tensor:
    return apply_d4_torch(x, _D4_INV[idx])


def normalize_depth(d: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max normalization. d: (B, H, W)."""
    b = d.shape[0]
    d_flat = d.view(b, -1)
    dmin = d_flat.min(dim=1, keepdim=True).values
    dmax = d_flat.max(dim=1, keepdim=True).values
    scale = (dmax - dmin).clamp(min=1e-8)
    return ((d_flat - dmin) / scale).view_as(d)


# ── Model wrapper for training ──────────────────────────────────────────

class DepthAnythingTrainable(nn.Module):
    """Wraps HF Depth Anything for gradient-based training.

    Returns normalized (B, H, W) depth at the model's native resolution.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(device)
        self.device = device

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: (B, 3, H, W) preprocessed. Returns (B, H, W) depth."""
        out = self.model(pixel_values=pixel_values)
        depth = out.predicted_depth  # (B, 1, h, w) or (B, h, w)
        if depth.ndim == 4:
            depth = depth.squeeze(1)
        return normalize_depth(depth)

    def preprocess_numpy(self, imgs: list[np.ndarray]) -> torch.Tensor:
        """Preprocess numpy images to model input tensor."""
        pil_imgs = [Image.fromarray(img) for img in imgs]
        inputs = self.processor(images=pil_imgs, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)


# ── Training loop ────────────────────────────────────────────────────────

def train(
    student: DepthAnythingTrainable,
    teacher: DepthAnythingTrainable,
    train_imgs: list[np.ndarray],
    steps: int = 500,
    batch_size: int = 2,
    lr: float = 1e-4,
    alpha: float = 0.8,
    device: str = "cuda",
    log_every: int = 25,
) -> list[dict]:
    """Fine-tune student with consistency + anchor loss."""
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)

    rng = np.random.default_rng(42)
    logs = []
    t0 = time.time()

    for step in range(1, steps + 1):
        # Sample a batch
        idxs = rng.integers(0, len(train_imgs), size=batch_size)
        batch_imgs = [train_imgs[i] for i in idxs]

        # Random D4 transform per image
        d4_idxs = rng.integers(1, 8, size=batch_size)  # exclude identity

        # Create transformed versions
        transformed_imgs = []
        for img, d4_idx in zip(batch_imgs, d4_idxs):
            from mezzanine.symmetries.depth_geometric import apply_d4
            t_img = apply_d4(img, int(d4_idx))
            transformed_imgs.append(t_img)

        # Preprocess both original and transformed
        pv_orig = student.preprocess_numpy(batch_imgs)
        pv_trans = student.preprocess_numpy(transformed_imgs)

        # Forward pass - student on both
        depth_orig = student(pv_orig)  # (B, H, W)
        depth_trans = student(pv_trans)  # (B, H, W)

        # Inverse-transform depth_trans back to original frame
        depth_trans_aligned = torch.stack([
            inverse_d4_torch(depth_trans[i:i+1], int(d4_idxs[i]))[0]
            for i in range(batch_size)
        ])

        # Note: depth maps may be different spatial sizes after transform
        # Since we resize to square (518x518), D4 transforms preserve dimensions
        L_consistency = F.mse_loss(depth_orig, depth_trans_aligned)

        # Anchor loss: stay close to teacher predictions
        with torch.no_grad():
            pv_orig_t = teacher.preprocess_numpy(batch_imgs)
            depth_teacher = teacher(pv_orig_t)

        L_anchor = F.mse_loss(depth_orig, depth_teacher)

        loss = alpha * L_consistency + (1 - alpha) * L_anchor

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == 1:
            log = {
                "step": step,
                "loss": loss.item(),
                "L_consistency": L_consistency.item(),
                "L_anchor": L_anchor.item(),
                "elapsed": time.time() - t0,
            }
            logs.append(log)
            print(f"  Step {step:4d} | loss={loss.item():.6f} | "
                  f"consist={L_consistency.item():.6f} | anchor={L_anchor.item():.6f}")

    return logs


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate_warrant_gap(
    model: DepthAnythingTrainable,
    test_images: list[tuple[str, np.ndarray]],
) -> list[dict]:
    """Measure warrant gap on test images using full D4."""
    from mezzanine.symmetries.depth_geometric import (
        DepthGeometricSymmetry, DepthGeometricSymmetryConfig,
    )

    model.eval()
    sym = DepthGeometricSymmetry(DepthGeometricSymmetryConfig(subgroup="d4"))

    results = []
    for name, img in test_images:
        views = sym.batch(img)
        # Predict depth for each view
        depths = []
        target_size = 256  # Use fixed square size for evaluation
        for v in views:
            pv = model.preprocess_numpy([v])
            with torch.no_grad():
                d = model(pv)[0].cpu().numpy()
            # Resize to fixed square for consistent alignment
            from PIL import Image as PILImage
            d_resized = np.array(PILImage.fromarray(d).resize(
                (target_size, target_size), PILImage.BILINEAR))
            depths.append(d_resized)

        # Align back
        depths_aligned = [sym.inverse(d, i) for i, d in enumerate(depths)]

        # Compute metrics
        stack = np.stack(depths_aligned, axis=0)
        pixel_mean = stack.mean(axis=0)
        pixel_std = stack.std(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(pixel_mean > 1e-8, pixel_std / pixel_mean, 0.0)

        h = pixel_mean.shape[0]
        row_indices = np.arange(h, dtype=np.float64)

        def vg_strength(dm):
            rm = dm.mean(axis=1).astype(np.float64)
            if rm.std() < 1e-10:
                return 0.0
            return float(np.corrcoef(row_indices, rm)[0, 1])

        results.append({
            "image": name,
            "mean_std": float(pixel_std.mean()),
            "mean_cv": float(cv.mean()),
            "max_std": float(pixel_std.max()),
            "vertical_bias_original": vg_strength(depths_aligned[0]),
            "vertical_bias_averaged": vg_strength(pixel_mean),
        })
        print(f"  {name}: CV={results[-1]['mean_cv']:.4f}, "
              f"vbias_orig={results[-1]['vertical_bias_original']:.4f}, "
              f"vbias_avg={results[-1]['vertical_bias_averaged']:.4f}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Consistency distillation for Depth Anything")
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--n-train", type=int, default=60, help="Number of training images")
    parser.add_argument("--out", default="out_depth_consistency")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test images
    print("Loading test images...")
    test_dir = Path("test_images")
    test_images = []
    for p in sorted(test_dir.glob("*.jpg")):
        img = np.asarray(Image.open(p).convert("RGB"))
        test_images.append((p.stem, img))
    print(f"  {len(test_images)} test images")

    # Download training images
    print("\nDownloading training images...")
    train_imgs = download_training_images(args.n_train)

    # Initialize models
    print(f"\nLoading models on {args.device}...")
    student = DepthAnythingTrainable(args.model, args.device)
    teacher = DepthAnythingTrainable(args.model, args.device)

    # Baseline evaluation
    print("\n" + "=" * 60)
    print("BASELINE (before fine-tuning)")
    print("=" * 60)
    baseline_results = evaluate_warrant_gap(student, test_images)

    # Fine-tune
    print("\n" + "=" * 60)
    print(f"FINE-TUNING: {args.steps} steps, α={args.alpha}, lr={args.lr}, bs={args.batch_size}")
    print("=" * 60)
    train_logs = train(
        student, teacher, train_imgs,
        steps=args.steps, batch_size=args.batch_size,
        lr=args.lr, alpha=args.alpha, device=args.device,
    )

    # Post-training evaluation
    print("\n" + "=" * 60)
    print("AFTER FINE-TUNING")
    print("=" * 60)
    finetuned_results = evaluate_warrant_gap(student, test_images)

    # Save results
    def summarize(results):
        return {
            "mean_cv": float(np.mean([r["mean_cv"] for r in results])),
            "mean_abs_vbias_orig": float(np.mean([abs(r["vertical_bias_original"]) for r in results])),
            "mean_abs_vbias_avg": float(np.mean([abs(r["vertical_bias_averaged"]) for r in results])),
        }

    baseline_summary = summarize(baseline_results)
    finetuned_summary = summarize(finetuned_results)

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  Mean CV:         {baseline_summary['mean_cv']:.4f} → {finetuned_summary['mean_cv']:.4f}")
    print(f"  Mean |vbias| orig: {baseline_summary['mean_abs_vbias_orig']:.4f} → {finetuned_summary['mean_abs_vbias_orig']:.4f}")
    print(f"  Mean |vbias| avg:  {baseline_summary['mean_abs_vbias_avg']:.4f} → {finetuned_summary['mean_abs_vbias_avg']:.4f}")

    output = {
        "config": {
            "model": args.model,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "alpha": args.alpha,
            "n_train_images": len(train_imgs),
            "n_test_images": len(test_images),
        },
        "baseline": {"per_image": baseline_results, "summary": baseline_summary},
        "finetuned": {"per_image": finetuned_results, "summary": finetuned_summary},
        "training_log": train_logs,
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save model checkpoint
    ckpt_path = out_dir / "student_checkpoint.pt"
    torch.save(student.model.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
