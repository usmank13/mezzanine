#!/usr/bin/env python3
"""Generate side-by-side depth comparisons: Original | Head (debiased) | Orbit Average.

Usage:
    python scripts/visualize_depth_comparison.py --checkpoint out_distill_frozen_200/depth_head.pt --device cuda
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

# Reuse classes from distill script
sys.path.insert(0, str(Path(__file__).parent))
from distill_frozen_head import FrozenEncoder, DepthConvHead

# ── Symmetry helpers (minimal) ──
def d4_transforms():
    """Return list of (transform_fn, inverse_fn) for D4 group."""
    fns = [
        (lambda x: x,                        lambda x: x),                        # identity
        (lambda x: np.rot90(x, 1),           lambda x: np.rot90(x, -1)),          # rot90CW
        (lambda x: np.rot90(x, 2),           lambda x: np.rot90(x, 2)),           # rot180
        (lambda x: np.rot90(x, -1),          lambda x: np.rot90(x, 1)),           # rot270CW
        (lambda x: np.flipud(x),             lambda x: np.flipud(x)),             # vflip
        (lambda x: np.fliplr(x),             lambda x: np.fliplr(x)),             # hflip
        (lambda x: np.transpose(x, (1,0,*range(2,x.ndim))) if x.ndim>2 else x.T,
         lambda x: np.transpose(x, (1,0,*range(2,x.ndim))) if x.ndim>2 else x.T),  # transpose
        (lambda x: np.rot90(np.flipud(x), 0) if False else np.fliplr(np.transpose(x, (1,0,*range(2,x.ndim))) if x.ndim>2 else np.fliplr(x.T)),
         lambda x: np.fliplr(np.transpose(x, (1,0,*range(2,x.ndim))) if x.ndim>2 else np.fliplr(x.T))),  # anti-transpose
    ]
    return fns

def orbit_average(encoder, img_np):
    """Compute D4 orbit-averaged depth for an image."""
    transforms = d4_transforms()
    aligned = []
    for tfwd, tinv in transforms:
        view = np.ascontiguousarray(tfwd(img_np))
        depth = encoder.predict_depth(view)
        aligned.append(tinv(depth))
    return np.stack(aligned).mean(axis=0)

def vertical_gradient_strength(depth_map):
    h = depth_map.shape[0]
    rows = np.arange(h, dtype=np.float64) / max(h - 1, 1)
    return float(np.corrcoef(rows, depth_map.mean(axis=1))[0, 1])

def normalize_depth(d):
    dmin, dmax = d.min(), d.max()
    if dmax - dmin > 1e-8:
        return (d - dmin) / (dmax - dmin)
    return d - dmin

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="out_distill_frozen_200/depth_head.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="out_distill_frozen_200/visualizations")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading encoder...")
    encoder = FrozenEncoder("depth-anything/Depth-Anything-V2-Small-hf", device=args.device)
    
    print("Loading head...")
    head = DepthConvHead(in_channels=384)
    head.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    head = head.to(args.device).eval()

    # Eval images
    test_dir = Path("test_images")
    eval_names = [
        ("robot_01", "topdown_robot_01.jpg"),
        ("robot_02", "topdown_robot_02.jpg"),
        ("robot_04", "topdown_robot_04.jpg"),
        ("robot_05", "topdown_robot_05.jpg"),
        ("fruit_td", "coco_fruit_td.jpg"),
        ("remote_td", "coco_remote_td.jpg"),
        ("hotdog_td", "coco_hotdog_td.jpg"),
        ("cake_td", "coco_cake_td.jpg"),
    ]
    
    eval_imgs = []
    for name, fname in eval_names:
        p = test_dir / fname
        if p.exists():
            eval_imgs.append((name, np.array(Image.open(p).convert("RGB"))))
        else:
            print(f"  Skipping {fname} (not found)")

    # Generate comparisons
    print(f"\nGenerating visualizations for {len(eval_imgs)} images...")
    
    all_results = []
    for name, img_np in eval_imgs:
        print(f"  Processing {name}...")
        
        # Original depth
        orig_depth = encoder.predict_depth(img_np)
        orig_vbias = vertical_gradient_strength(orig_depth)
        
        # Head prediction
        feats = encoder.extract_features(img_np)
        with torch.no_grad():
            head_pred = head(feats.to(args.device)).cpu().numpy().squeeze()
        head_vbias = vertical_gradient_strength(head_pred)
        
        # Orbit average
        orb = orbit_average(encoder, img_np)
        orb_vbias = vertical_gradient_strength(orb)
        
        all_results.append((name, img_np, orig_depth, head_pred, orb, orig_vbias, head_vbias, orb_vbias))
    
    # ── Big comparison grid ──
    n = len(all_results)
    fig = plt.figure(figsize=(16, 4 * n), dpi=100)
    gs = GridSpec(n, 4, figure=fig, wspace=0.05, hspace=0.3)
    
    for i, (name, img_np, orig, head_d, orb, ov, hv, orbv) in enumerate(all_results):
        # Input image
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img_np)
        ax.set_title(f"Input: {name}", fontsize=11)
        ax.axis("off")
        
        # Original depth
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(normalize_depth(orig), cmap="inferno")
        ax.set_title(f"Original (vbias={ov:+.2f})", fontsize=11,
                     color="red" if abs(ov) > 0.5 else "black")
        ax.axis("off")
        
        # Head prediction
        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(normalize_depth(head_d), cmap="inferno")
        delta = abs(ov) - abs(hv)
        color = "green" if delta > 0.1 else ("red" if delta < -0.1 else "black")
        ax.set_title(f"Debiased Head (vbias={hv:+.2f})", fontsize=11, color=color)
        ax.axis("off")
        
        # Orbit average
        ax = fig.add_subplot(gs[i, 3])
        ax.imshow(normalize_depth(orb), cmap="inferno")
        ax.set_title(f"D4 Orbit Avg (vbias={orbv:+.2f})", fontsize=11)
        ax.axis("off")
    
    fig.suptitle("Depth Estimation: Original vs Debiased Head vs D4 Orbit Average",
                 fontsize=14, fontweight="bold", y=1.0)
    
    grid_path = out_dir / "comparison_grid.png"
    fig.savefig(grid_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved grid: {grid_path}")
    
    # ── Summary bar chart ──
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    names = [r[0] for r in all_results]
    orig_abs = [abs(r[5]) for r in all_results]
    head_abs = [abs(r[6]) for r in all_results]
    orb_abs = [abs(r[7]) for r in all_results]
    
    x = np.arange(len(names))
    w = 0.25
    ax2.bar(x - w, orig_abs, w, label="Original", color="#d62728", alpha=0.85)
    ax2.bar(x,     head_abs, w, label="Debiased Head", color="#2ca02c", alpha=0.85)
    ax2.bar(x + w, orb_abs,  w, label="D4 Orbit Avg", color="#1f77b4", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.set_ylabel("|Vertical Bias|")
    ax2.set_title("Vertical Bias Reduction: Frozen Head vs Original vs Orbit Average")
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="~unbiased")
    
    bar_path = out_dir / "bias_comparison.png"
    fig2.savefig(bar_path, bbox_inches="tight", facecolor="white", dpi=120)
    plt.close(fig2)
    print(f"Saved bar chart: {bar_path}")


if __name__ == "__main__":
    main()
