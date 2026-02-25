#!/usr/bin/env python3
"""Test weighted orbit averaging strategies for depth bias reduction.

Instead of uniform averaging over D4 transforms, weight each transform's
contribution by how much it agrees with the others (consistency weighting).

Hypothesis: transforms that produce out-of-distribution garbage will disagree
with the consensus and get downweighted, improving the orbit average.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def load_images(paths):
    pairs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        pairs.append((Path(p).stem, np.asarray(img)))
    return pairs


def vertical_gradient_strength(depth_map):
    """Pearson correlation between row index and row-mean depth."""
    h = depth_map.shape[0]
    row_indices = np.arange(h, dtype=np.float64)
    row_means = depth_map.mean(axis=1).astype(np.float64)
    if row_means.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(row_indices, row_means)[0, 1])


def compute_weights_consistency(depths_aligned, temperature=1.0):
    """Weight each transform by inverse disagreement with the others.
    
    For each transform k, compute mean L2 distance to all other transforms.
    Then softmax(-temperature * distance) to get weights.
    """
    K = len(depths_aligned)
    stack = np.stack(depths_aligned, axis=0)  # (K, H, W)
    
    # Mean pairwise distance for each transform
    distances = np.zeros(K)
    for k in range(K):
        others = np.concatenate([stack[:k], stack[k+1:]], axis=0)  # (K-1, H, W)
        distances[k] = np.mean((stack[k] - others.mean(axis=0)) ** 2)
    
    # Softmax weighting
    logits = -temperature * distances
    logits -= logits.max()  # numerical stability
    weights = np.exp(logits)
    weights /= weights.sum()
    
    return weights


def compute_weights_variance(depths_aligned, temperature=1.0):
    """Weight each transform by inverse per-pixel variance contribution.
    
    For each transform k, measure how much removing it reduces variance.
    Transforms that increase variance (outliers) get lower weight.
    """
    K = len(depths_aligned)
    stack = np.stack(depths_aligned, axis=0)  # (K, H, W)
    
    full_var = stack.var(axis=0).mean()
    
    importance = np.zeros(K)
    for k in range(K):
        # Variance without this transform
        reduced = np.concatenate([stack[:k], stack[k+1:]], axis=0)
        reduced_var = reduced.var(axis=0).mean()
        # How much does removing k reduce variance?
        importance[k] = full_var - reduced_var  # positive = k is an outlier
    
    # Lower importance (outlier) → lower weight
    logits = -temperature * importance
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    
    return weights


def compute_weights_median_distance(depths_aligned, temperature=1.0):
    """Weight by distance to pixel-wise median (robust to outliers)."""
    K = len(depths_aligned)
    stack = np.stack(depths_aligned, axis=0)  # (K, H, W)
    
    median = np.median(stack, axis=0)  # (H, W)
    
    distances = np.zeros(K)
    for k in range(K):
        distances[k] = np.mean((stack[k] - median) ** 2)
    
    logits = -temperature * distances
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    
    return weights


WEIGHT_STRATEGIES = {
    "uniform": lambda depths, t: np.ones(len(depths)) / len(depths),
    "consistency": compute_weights_consistency,
    "variance": compute_weights_variance,
    "median": compute_weights_median_distance,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--subgroup", default="d4", choices=["d4", "vflip", "flips"])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.5, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--out", default="out_depth_weighted")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = load_images(args.images)

    print(f"Loading {args.model}...")
    from mezzanine.encoders.depth_anything import DepthAnythingEncoder, DepthAnythingEncoderConfig
    from mezzanine.symmetries.depth_geometric import DepthGeometricSymmetry, DepthGeometricSymmetryConfig

    encoder = DepthAnythingEncoder(DepthAnythingEncoderConfig(model_name=args.model), device=args.device)
    sym = DepthGeometricSymmetry(DepthGeometricSymmetryConfig(subgroup=args.subgroup))

    D4_NAMES = ["identity", "rot90", "rot180", "rot270", "vflip", "hflip", "transpose", "anti-transpose"]
    element_names = [D4_NAMES[i] for i in sym.elements]
    K = len(sym.elements)
    print(f"Subgroup: {args.subgroup} ({K} elements: {element_names})")

    all_results = []

    for img_name, img_np in images:
        print(f"\n{'='*60}")
        print(f"Image: {img_name}")

        # Get aligned depth maps
        views = sym.batch(img_np)
        depth_maps = [encoder.predict(v) for v in views]
        depths_aligned = [sym.inverse(d, i) for i, d in enumerate(depth_maps)]

        original_vbias = vertical_gradient_strength(depths_aligned[0])
        
        result = {
            "image": img_name,
            "original_vbias": original_vbias,
            "strategies": {},
        }

        print(f"  Original vbias: {original_vbias:.4f}")

        # Test each strategy at each temperature
        for strategy_name, weight_fn in WEIGHT_STRATEGIES.items():
            temps = [1.0] if strategy_name == "uniform" else args.temperatures
            
            for temp in temps:
                if strategy_name == "uniform":
                    weights = weight_fn(depths_aligned, temp)
                    key = "uniform"
                else:
                    weights = weight_fn(depths_aligned, temp)
                    key = f"{strategy_name}_t{temp}"

                # Weighted average
                stack = np.stack(depths_aligned, axis=0)
                weighted_avg = np.einsum("k,khw->hw", weights, stack)

                avg_vbias = vertical_gradient_strength(weighted_avg)
                reduction = abs(original_vbias) - abs(avg_vbias)

                result["strategies"][key] = {
                    "weights": weights.tolist(),
                    "vbias": avg_vbias,
                    "reduction": reduction,
                }

                w_str = ", ".join(f"{w:.3f}" for w in weights)
                marker = "✓" if reduction > 0.05 else ("✗" if reduction < -0.05 else "~")
                print(f"  {marker} {key:<25} vbias={avg_vbias:+.4f}  Δ={reduction:+.4f}  w=[{w_str}]")

        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Mean bias reduction by strategy")
    print(f"{'='*60}")

    strategies = set()
    for r in all_results:
        strategies.update(r["strategies"].keys())
    
    for s in sorted(strategies):
        reds = [r["strategies"][s]["reduction"] for r in all_results if s in r["strategies"]]
        if reds:
            print(f"  {s:<25} mean={np.mean(reds):+.4f}  median={np.median(reds):+.4f}  "
                  f"helps={sum(1 for r in reds if r > 0.05)}/{len(reds)}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
