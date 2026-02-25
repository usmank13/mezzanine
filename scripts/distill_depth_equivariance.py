#!/usr/bin/env python3
"""Mezzanine-style distillation: train a lightweight head on frozen DINOv2 features
to predict D4 orbit-averaged depth maps in a single forward pass.

The Mezzanine pattern:
  1. Freeze encoder → extract features
  2. Compute orbit-averaged predictions (expensive but equivariant)
  3. Train small head: frozen_features → orbit_average (one pass)

Usage:
    python scripts/distill_depth_equivariance.py --device cuda --out out_depth_distilled
"""

from __future__ import annotations
import argparse, json, time, sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Reuse COCO download from finetune script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_depth_consistency import download_training_images, COCO_TRAIN_URLS


# ── Convolutional head ───────────────────────────────────────────────────

class DepthHead(nn.Module):
    """Small conv head: (B, 384, 37, 37) → (B, 1, 37, 37)."""
    def __init__(self, in_channels=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )
    
    def forward(self, x):
        # x: (B, 384, 37, 37)
        return self.net(x).squeeze(1)  # (B, 37, 37)


# ── Feature extraction ───────────────────────────────────────────────────

class FeatureExtractor:
    """Extract frozen DINOv2 backbone features from Depth Anything."""
    
    def __init__(self, model_name, device):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False
    
    def extract_features(self, imgs_np: list[np.ndarray]) -> torch.Tensor:
        """Extract last hidden state features. Returns (B, 384, 37, 37).
        Images are resized to 518x518 square to ensure consistent 37x37 patch grid."""
        # Force square resize to 518x518 for consistent patch count
        pil_imgs = []
        for img in imgs_np:
            pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img
            pil = pil.resize((518, 518), Image.BILINEAR)
            pil_imgs.append(pil)
        inputs = self.processor(images=pil_imgs, return_tensors="pt", do_resize=False)
        pixel_values = inputs["pixel_values"].to(self.device)
        
        with torch.no_grad():
            out = self.model.backbone(pixel_values, output_hidden_states=True)
        
        # Last hidden state: (B, 1370, 384) — drop CLS token, reshape to spatial
        hidden = out.hidden_states[-1]  # (B, 1370, 384)
        spatial = hidden[:, 1:, :]  # (B, 1369, 384) = (B, 37*37, 384)
        B, N, C = spatial.shape
        h = w = int(N ** 0.5)
        return spatial.permute(0, 2, 1).reshape(B, C, h, w)  # (B, 384, 37, 37)
    
    def predict_depth(self, img_np: np.ndarray) -> np.ndarray:
        """Full model depth prediction → (H, W) numpy."""
        pil = Image.fromarray(img_np) if isinstance(img_np, np.ndarray) else img_np
        inputs = self.processor(images=[pil], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = [(img_np.shape[0], img_np.shape[1])]
        post = self.processor.post_process_depth_estimation(outputs, target_sizes=target_sizes)
        return post[0]["predicted_depth"].cpu().float().numpy()


# ── Orbit averaging ──────────────────────────────────────────────────────

def compute_orbit_average(extractor: FeatureExtractor, img_np: np.ndarray) -> np.ndarray:
    """Compute D4 orbit-averaged depth map for an image. Returns (H, W) numpy."""
    from mezzanine.symmetries.depth_geometric import (
        DepthGeometricSymmetry, DepthGeometricSymmetryConfig
    )
    sym = DepthGeometricSymmetry(DepthGeometricSymmetryConfig(subgroup="d4"))
    
    views = sym.batch(img_np)
    depths = [extractor.predict_depth(v) for v in views]
    return sym.orbit_average(depths)


# ── Training ─────────────────────────────────────────────────────────────

def prepare_dataset(
    extractor: FeatureExtractor,
    images: list[np.ndarray],
    target_size: int = 37,
    batch_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute features and orbit-averaged targets for all images.
    
    Returns:
        features: (N, 384, 37, 37) tensor
        targets: (N, 37, 37) tensor (orbit-averaged depth, downsampled)
    """
    all_features = []
    all_targets = []
    
    for i, img in enumerate(images):
        print(f"  [{i+1}/{len(images)}] Computing orbit average + features...")
        
        # Compute orbit average (expensive: 8 forward passes)
        orbit_avg = compute_orbit_average(extractor, img)
        
        # Downsample orbit average to target_size
        orbit_t = torch.from_numpy(orbit_avg).float().unsqueeze(0).unsqueeze(0)
        orbit_down = F.interpolate(orbit_t, size=(target_size, target_size), mode="bilinear", align_corners=False)
        # Normalize to [0, 1]
        omin, omax = orbit_down.min(), orbit_down.max()
        if omax - omin > 1e-8:
            orbit_down = (orbit_down - omin) / (omax - omin)
        all_targets.append(orbit_down.squeeze(0).squeeze(0))
        
        # Extract features (1 forward pass)
        feats = extractor.extract_features([img])  # (1, 384, 37, 37)
        all_features.append(feats.squeeze(0).cpu())
    
    return torch.stack(all_features), torch.stack(all_targets)


def train_head(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    device: str = "cuda",
) -> tuple[DepthHead, list[dict]]:
    """Train the depth head on frozen features."""
    N = features.shape[0]
    head = DepthHead(in_channels=features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    features = features.to(device)
    targets = targets.to(device)
    
    logs = []
    for epoch in range(1, epochs + 1):
        head.train()
        pred = head(features)  # (N, 37, 37)
        loss = F.mse_loss(pred, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 50 == 0 or epoch == 1:
            log = {"epoch": epoch, "mse": loss.item()}
            logs.append(log)
            print(f"    Epoch {epoch:4d} | MSE={loss.item():.6f}")
    
    head.eval()
    return head, logs


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate(
    extractor: FeatureExtractor,
    head: DepthHead,
    test_images: list[tuple[str, np.ndarray]],
    out_dir: Path,
    device: str = "cuda",
):
    """Evaluate the distilled student vs original model vs orbit average."""
    from mezzanine.symmetries.depth_geometric import (
        DepthGeometricSymmetry, DepthGeometricSymmetryConfig
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr
    
    sym = DepthGeometricSymmetry(DepthGeometricSymmetryConfig(subgroup="d4"))
    head.eval()
    
    results = []
    
    for name, img in test_images:
        print(f"\n  Evaluating: {name}")
        H, W = img.shape[:2]
        
        # 1. Original model depth
        original_depth = extractor.predict_depth(img)
        
        # 2. Orbit-averaged depth (teacher)
        orbit_avg = compute_orbit_average(extractor, img)
        
        # Use square size to avoid shape mismatch from D4 rotations on non-square images
        S = max(H, W)
        
        # 3. Student prediction
        with torch.no_grad():
            feats = extractor.extract_features([img])  # (1, 384, 37, 37)
            student_small = head(feats.to(device))  # (1, 37, 37)
            # Upsample to square
            student_up = F.interpolate(
                student_small.unsqueeze(1), size=(S, S), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0).cpu().numpy()
        
        # 4. Student warrant gap: run student on all D4 transforms
        student_depths_aligned = []
        for view_idx, view_img in enumerate(sym.batch(img)):
            with torch.no_grad():
                vf = extractor.extract_features([view_img])
                vd = head(vf.to(device))
                vd_up = F.interpolate(
                    vd.unsqueeze(1), size=(S, S), mode="bilinear", align_corners=False
                ).squeeze(0).squeeze(0).cpu().numpy()
            student_depths_aligned.append(sym.inverse(vd_up, view_idx))
        
        student_stack = np.stack(student_depths_aligned)
        student_orbit_avg = student_stack.mean(axis=0)
        student_std = student_stack.std(axis=0)
        student_mean = student_stack.mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            student_cv = np.where(student_mean > 1e-8, student_std / np.abs(student_mean), 0.0)
        
        # Also measure original model warrant gap (resize to SxS for consistency)
        orig_depths_aligned = []
        for view_idx, view_img in enumerate(sym.batch(img)):
            d = extractor.predict_depth(view_img)
            # Resize to SxS square before inverse transform
            d_t = torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0)
            d_resized = F.interpolate(d_t, size=(S, S), mode="bilinear", align_corners=False).squeeze().numpy()
            orig_depths_aligned.append(sym.inverse(d_resized, view_idx))
        orig_stack = np.stack(orig_depths_aligned)
        orig_std = orig_stack.std(axis=0)
        orig_mean = orig_stack.mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            orig_cv = np.where(orig_mean > 1e-8, orig_std / np.abs(orig_mean), 0.0)
        
        # Normalize all to [0,1] for fair correlation
        def norm(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-8)
        
        # Resize original_depth and orbit_avg to SxS for fair comparison with student
        def to_square(d, size):
            t = torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0)
            return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False).squeeze().numpy()
        
        on = norm(to_square(original_depth, S)).ravel()
        oa = norm(to_square(orbit_avg, S)).ravel()
        sn = norm(student_up).ravel()
        
        corr_student_teacher = pearsonr(sn, oa)[0]
        corr_student_original = pearsonr(sn, on)[0]
        corr_original_teacher = pearsonr(on, oa)[0]
        spearman_student_teacher = spearmanr(sn, oa)[0]
        
        metrics = {
            "image": name,
            "corr_student_vs_orbit_avg": float(corr_student_teacher),
            "corr_student_vs_original": float(corr_student_original),
            "corr_original_vs_orbit_avg": float(corr_original_teacher),
            "spearman_student_vs_orbit_avg": float(spearman_student_teacher),
            "warrant_gap_cv_original": float(orig_cv.mean()),
            "warrant_gap_cv_student": float(student_cv.mean()),
            "warrant_gap_reduction": float(orig_cv.mean() - student_cv.mean()),
        }
        results.append(metrics)
        
        print(f"    Student↔Teacher corr:  {corr_student_teacher:.4f}")
        print(f"    Student↔Original corr: {corr_student_original:.4f}")
        print(f"    Original↔Teacher corr: {corr_original_teacher:.4f}")
        print(f"    Warrant gap CV: original={orig_cv.mean():.4f} → student={student_cv.mean():.4f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 5, figsize=(25, 4))
        axes[0].imshow(img)
        axes[0].set_title("Input")
        axes[0].axis("off")
        
        vmin = min(norm(original_depth).min(), norm(orbit_avg).min(), norm(student_up).min())
        vmax = max(norm(original_depth).max(), norm(orbit_avg).max(), norm(student_up).max())
        
        axes[1].imshow(norm(original_depth), cmap="inferno", vmin=vmin, vmax=vmax)
        axes[1].set_title("Original model")
        axes[1].axis("off")
        
        axes[2].imshow(norm(orbit_avg), cmap="inferno", vmin=vmin, vmax=vmax)
        axes[2].set_title("D4 orbit average (teacher)")
        axes[2].axis("off")
        
        axes[3].imshow(norm(student_up), cmap="inferno", vmin=vmin, vmax=vmax)
        axes[3].set_title(f"Student (r={corr_student_teacher:.3f})")
        axes[3].axis("off")
        
        # Warrant gap comparison
        cv_max = max(orig_cv.max(), student_cv.max(), 0.01)
        ax4 = axes[4]
        ax4_split = fig.add_axes(ax4.get_position(), frameon=False)
        ax4.imshow(orig_cv, cmap="hot", vmin=0, vmax=cv_max)
        ax4.set_title(f"Orig CV={orig_cv.mean():.3f}")
        ax4.axis("off")
        ax4_split.axis("off")
        
        fig.suptitle(f"{name}: Mezzanine Distillation", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / f"comparison_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Also save a warrant gap comparison figure
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        cv_max = max(orig_cv.max(), student_cv.max(), 0.01)
        ax1.imshow(orig_cv, cmap="hot", vmin=0, vmax=cv_max)
        ax1.set_title(f"Original warrant gap\nCV={orig_cv.mean():.4f}")
        ax1.axis("off")
        ax2.imshow(student_cv, cmap="hot", vmin=0, vmax=cv_max)
        ax2.set_title(f"Student warrant gap\nCV={student_cv.mean():.4f}")
        ax2.axis("off")
        fig2.suptitle(f"{name}: Warrant Gap Reduction", fontweight="bold")
        fig2.tight_layout()
        fig2.savefig(out_dir / f"warrant_gap_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
    
    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--n-train", type=int, default=40, help="COCO training images")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="out_depth_distilled")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    
    # Load test images
    print("Loading test images...")
    test_images = []
    for p in sorted(Path("test_images").glob("topdown_robot_*.jpg")):
        img = np.asarray(Image.open(p).convert("RGB"))
        test_images.append((p.stem, img))
    print(f"  {len(test_images)} test images")
    
    # Initialize extractor
    print(f"\nLoading {args.model}...")
    extractor = FeatureExtractor(args.model, args.device)
    
    # Download training images
    print("\nDownloading COCO training images...")
    train_imgs = download_training_images(args.n_train)
    
    # Add robot test images to training (they're our domain)
    for name, img in test_images:
        resized = np.asarray(Image.fromarray(img).resize((518, 518), Image.BILINEAR))
        train_imgs.append(resized)
    print(f"Total training images: {len(train_imgs)}")
    
    # Prepare dataset
    print("\nPreparing dataset (computing orbit averages + features)...")
    features, targets = prepare_dataset(extractor, train_imgs)
    print(f"  Features: {features.shape}, Targets: {targets.shape}")
    
    # Train
    print("\nTraining depth head...")
    head, train_logs = train_head(features, targets, epochs=args.epochs, lr=args.lr, device=args.device)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    eval_results = evaluate(extractor, head, test_images, out_dir, args.device)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    mean_corr_st = np.mean([r["corr_student_vs_orbit_avg"] for r in eval_results])
    mean_corr_so = np.mean([r["corr_student_vs_original"] for r in eval_results])
    mean_wg_orig = np.mean([r["warrant_gap_cv_original"] for r in eval_results])
    mean_wg_student = np.mean([r["warrant_gap_cv_student"] for r in eval_results])
    
    print(f"  Mean student↔teacher corr:     {mean_corr_st:.4f}")
    print(f"  Mean student↔original corr:    {mean_corr_so:.4f}")
    print(f"  Warrant gap CV: {mean_wg_orig:.4f} → {mean_wg_student:.4f} ({(1 - mean_wg_student/mean_wg_orig)*100:.1f}% reduction)")
    print(f"  Total time: {time.time() - t0:.1f}s")
    
    # Save results
    output = {
        "config": {
            "model": args.model,
            "n_train": len(train_imgs),
            "epochs": args.epochs,
            "lr": args.lr,
        },
        "training_logs": train_logs,
        "evaluation": eval_results,
        "summary": {
            "mean_corr_student_vs_orbit_avg": float(mean_corr_st),
            "mean_corr_student_vs_original": float(mean_corr_so),
            "mean_warrant_gap_cv_original": float(mean_wg_orig),
            "mean_warrant_gap_cv_student": float(mean_wg_student),
            "warrant_gap_reduction_pct": float((1 - mean_wg_student / mean_wg_orig) * 100),
        },
    }
    
    with open(out_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_dir / 'results.json'}")
    
    # Save model
    torch.save(head.state_dict(), out_dir / "depth_head.pt")
    print(f"Model saved to {out_dir / 'depth_head.pt'}")


if __name__ == "__main__":
    main()
