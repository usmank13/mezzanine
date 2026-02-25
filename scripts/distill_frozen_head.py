#!/usr/bin/env python3
"""Mezzanine-style distillation for depth: frozen encoder + learned head.

Following the actual Mezzanine pattern:
  1. Freeze the encoder (DINOv2 backbone from Depth Anything)
  2. Compute D4 orbit-averaged depth targets (expensive, offline)
  3. Train a small conv head: frozen_features → orbit_averaged_depth

The head learns to predict equivariant depth from non-equivariant features.
This is cheap (small head, few params) and preserves encoder quality.

Usage:
    python scripts/distill_frozen_head.py --device cuda --n-train 200
"""

from __future__ import annotations
import argparse, json, time, os, sys, io
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ── Head architecture ────────────────────────────────────────────────────

class DepthConvHead(nn.Module):
    """Conv head: (B, C, H, W) → (B, 1, H, W) depth prediction.
    
    Deeper than previous attempt, with skip connection.
    """
    def __init__(self, in_channels=384, hidden=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, hidden // 2, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden // 2, 1, 1)
        self.act = nn.GELU()
        # Skip projection
        self.skip = nn.Conv2d(in_channels, hidden, 1) if in_channels != hidden else nn.Identity()
    
    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h)) + self.skip(x)  # residual
        h = self.act(self.conv3(h))
        return self.conv4(h).squeeze(1)  # (B, H, W)


# ── Feature extractor (frozen) ──────────────────────────────────────────

class FrozenEncoder:
    """Extract DINOv2 features from Depth Anything, frozen."""
    
    def __init__(self, model_name, device):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
    
    def extract_features(self, img_np):
        """Extract backbone features → (1, 384, 37, 37)."""
        pil = Image.fromarray(img_np).resize((518, 518), Image.BILINEAR)
        inputs = self.processor(images=[pil], return_tensors="pt", do_resize=False)
        pixel_values = inputs["pixel_values"].to(self.device)
        
        with torch.no_grad():
            out = self.model.backbone(pixel_values, output_hidden_states=True)
        
        hidden = out.hidden_states[-1]  # (1, 1370, 384)
        spatial = hidden[:, 1:, :]  # (1, 1369, 384)
        B, N, C = spatial.shape
        h = w = int(N ** 0.5)  # 37
        return spatial.permute(0, 2, 1).reshape(B, C, h, w)  # (1, 384, 37, 37)
    
    def predict_depth(self, img_np):
        """Full model prediction → (H, W) numpy."""
        pil = Image.fromarray(img_np)
        inputs = self.processor(images=[pil], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        target_sizes = [(img_np.shape[0], img_np.shape[1])]
        post = self.processor.post_process_depth_estimation(out, target_sizes=target_sizes)
        return post[0]["predicted_depth"].cpu().float().numpy()


# ── Data ─────────────────────────────────────────────────────────────────

def download_coco_batch(n_images, cache_dir, seed=42):
    """Download n random COCO val2017 images."""
    import requests
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO val2017 image IDs (a large subset of known valid ones)
    # We'll use a deterministic list
    rng = np.random.RandomState(seed)
    
    # Known valid COCO val2017 IDs (verified working)
    known_ids = [
        39769, 397133, 37777, 87470, 131131, 181859, 87875, 146457,
        459467, 284991, 579635, 226111, 348881, 80340, 227765,
        295316, 360661, 2587, 174004, 279541, 180135, 84270,
        226592, 349860, 434230, 532493, 173091, 153529, 176446, 87038,
        56350, 404484, 289393,
        # More COCO val2017 images
        581929, 578489, 574769, 572678, 565778, 562150, 558854,
        554291, 551215, 547816, 545129, 544565, 542145, 540763,
        539715, 537991, 536947, 534827, 531968, 530854,
        528399, 527029, 525155, 522713, 520264, 518770, 517687,
        515579, 514508, 512194, 511321, 509735, 508602, 506656,
        504711, 502737, 501523, 499313, 498286, 496861,
        494913, 493286, 491757, 490171, 488592, 487583, 486479,
        484792, 483108, 481404, 479248, 477118, 474854, 473237,
        471087, 469067, 466567, 464522, 462565, 460347,
        458790, 456559, 455483, 453860, 451478, 449406, 447313,
        445365, 443303, 441286, 439180, 437313, 435081, 433068,
        430961, 428454, 426329, 424551, 422886, 421023,
        419096, 417779, 416343, 414673, 412894, 410650, 409475,
        407614, 405249, 403385, 401244, 399462, 397327, 395633,
        393569, 391895, 390755, 389109, 387655, 386134,
        384350, 382734, 380913, 379800, 378454, 376442, 374369,
        372577, 370818, 369370, 367680, 365484, 363942, 361919,
        360137, 358525, 357081, 354533, 352900, 351810,
        349837, 348708, 347693, 345027, 343218, 341681, 340175,
        338428, 336587, 334767, 332845, 331352, 329827, 328238,
        326248, 324266, 322563, 321214, 319607, 318219,
        316654, 315257, 313034, 311303, 309391, 307658, 305695,
        303566, 302165, 300659, 299552, 297343, 295713, 294163,
        292060, 290768, 289516, 287545, 285664, 284282,
        282296, 280891, 279806, 278705, 276707, 275198, 273198,
        271997, 270244, 268831, 267434, 265989, 264411, 262682,
        261061, 259571, 257478, 255401, 253742, 252219,
        250282, 248752, 247131, 245026, 243204, 241677, 240023,
        238410, 236412, 234607, 232684, 230831, 228981, 227187,
        225532, 223959, 222455, 220581, 218439, 216739, 215245,
        213086, 211674, 210230, 208805, 207477, 205834, 204186,
        202408, 200961, 199310, 197796, 196246, 194875, 193271,
    ]
    
    rng.shuffle(known_ids)
    selected = known_ids[:n_images]
    
    pairs = []
    for img_id in selected:
        fname = f"{img_id:012d}.jpg"
        cache_path = cache_dir / fname
        
        if cache_path.exists():
            try:
                img = Image.open(cache_path).convert("RGB")
                pairs.append((f"coco_{img_id}", np.asarray(img)))
                continue
            except:
                pass
        
        url = f"http://images.cocodataset.org/val2017/{fname}"
        try:
            import requests
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.save(cache_path)
            pairs.append((f"coco_{img_id}", np.asarray(img)))
        except Exception as e:
            pass  # skip failed downloads silently
    
    return pairs


# ── Metrics ──────────────────────────────────────────────────────────────

def vertical_gradient_strength(depth_map):
    h = depth_map.shape[0]
    row_indices = np.arange(h, dtype=np.float64)
    row_means = depth_map.mean(axis=1).astype(np.float64)
    if row_means.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(row_indices, row_means)[0, 1])


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--out", default="out_distill_frozen")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--subgroup", default="d4")
    parser.add_argument("--target-size", type=int, default=37, help="Spatial size of feature/target maps")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "image_cache"

    # Download training images
    print(f"Downloading {args.n_train} COCO images for training...")
    train_imgs = download_coco_batch(args.n_train, cache_dir)
    print(f"  Got {len(train_imgs)} images")

    # Also load local test images for eval
    eval_imgs = []
    robot_dir = Path("test_images")
    for i in range(1, 6):
        p = robot_dir / f"topdown_robot_{i:02d}.jpg"
        if p.exists():
            eval_imgs.append((f"robot_{i:02d}", np.asarray(Image.open(p).convert("RGB"))))
    # Add a few COCO top-down for eval
    td_names = ["coco_fruit_td.jpg", "coco_remote_td.jpg", "coco_hotdog_td.jpg", "coco_cake_td.jpg"]
    for name in td_names:
        p = robot_dir / name
        if p.exists():
            eval_imgs.append((Path(name).stem, np.asarray(Image.open(p).convert("RGB"))))
    print(f"  Eval: {len(eval_imgs)} images")

    # Load frozen encoder
    print(f"\nLoading {args.model} (frozen)...")
    encoder = FrozenEncoder(args.model, args.device)

    from mezzanine.symmetries.depth_geometric import DepthGeometricSymmetry, DepthGeometricSymmetryConfig
    sym = DepthGeometricSymmetry(DepthGeometricSymmetryConfig(subgroup=args.subgroup))
    n_elements = len(sym.elements)
    print(f"Subgroup: {args.subgroup} ({n_elements} elements)")

    # Pre-compute features and orbit-averaged targets
    print(f"\nPre-computing features + D4 orbit targets for {len(train_imgs)} images...")
    all_features = []
    all_targets = []
    t0 = time.time()
    
    for i, (name, img_np) in enumerate(train_imgs):
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(train_imgs) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(train_imgs)}] {rate:.1f} img/s, ETA {eta:.0f}s")
        
        # Features (1 forward pass through backbone)
        feats = encoder.extract_features(img_np)  # (1, 384, 37, 37)
        all_features.append(feats.squeeze(0).cpu())
        
        # Orbit-averaged depth target (n_elements forward passes)
        views = sym.batch(img_np)
        depths = [encoder.predict_depth(v) for v in views]
        aligned = [sym.inverse(d, i_elem) for i_elem, d in enumerate(depths)]
        orbit_avg = np.stack(aligned).mean(axis=0)
        
        # Downsample to feature map size
        orbit_t = torch.from_numpy(orbit_avg).float().unsqueeze(0).unsqueeze(0)
        orbit_down = F.interpolate(orbit_t, size=(args.target_size, args.target_size), 
                                    mode="bilinear", align_corners=False)
        # Normalize per-image to [0, 1]
        omin, omax = orbit_down.min(), orbit_down.max()
        if omax - omin > 1e-8:
            orbit_down = (orbit_down - omin) / (omax - omin)
        all_targets.append(orbit_down.squeeze())
    
    features = torch.stack(all_features)  # (N, 384, 37, 37)
    targets = torch.stack(all_targets)    # (N, 37, 37)
    print(f"  Done in {time.time()-t0:.1f}s. Features: {features.shape}, Targets: {targets.shape}")

    # Also pre-compute features for original (non-orbit) predictions for comparison
    print("\nPre-computing original depth targets (for baseline)...")
    all_orig_targets = []
    for name, img_np in train_imgs:
        orig_depth = encoder.predict_depth(img_np)
        orig_t = torch.from_numpy(orig_depth).float().unsqueeze(0).unsqueeze(0)
        orig_down = F.interpolate(orig_t, size=(args.target_size, args.target_size),
                                   mode="bilinear", align_corners=False)
        omin, omax = orig_down.min(), orig_down.max()
        if omax - omin > 1e-8:
            orig_down = (orig_down - omin) / (omax - omin)
        all_orig_targets.append(orig_down.squeeze())
    orig_targets = torch.stack(all_orig_targets)

    # Measure how different orbit targets are from original
    diffs = (targets - orig_targets).abs().mean(dim=(1, 2))
    print(f"  Mean |orbit_avg - original| per image: {diffs.mean():.4f} (std: {diffs.std():.4f})")
    print(f"  Max diff images: {diffs.topk(5).values.tolist()}")

    # Train head
    print(f"\nTraining head: {args.epochs} epochs, lr={args.lr}, batch={args.batch_size}")
    head = DepthConvHead(in_channels=features.shape[1]).to(args.device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"  Head params: {n_params:,}")
    
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    features_gpu = features.to(args.device)
    targets_gpu = targets.to(args.device)
    N = features.shape[0]
    
    logs = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        indices = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0
        
        for start in range(0, N, args.batch_size):
            idx = indices[start:start + args.batch_size]
            feat_batch = features_gpu[idx]
            tgt_batch = targets_gpu[idx]
            
            pred_raw = head(feat_batch)
            # Normalize pred per-image to [0,1] for scale-invariant comparison
            pred_norm = torch.zeros_like(pred_raw)
            for b in range(pred_raw.shape[0]):
                pmin, pmax = pred_raw[b].min(), pred_raw[b].max()
                if pmax - pmin > 1e-8:
                    pred_norm[b] = (pred_raw[b] - pmin) / (pmax - pmin)
                else:
                    pred_norm[b] = pred_raw[b]
            
            loss = F.mse_loss(pred_norm, tgt_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}: loss={avg_loss:.6f}")
            logs.append({"epoch": epoch, "loss": avg_loss})

    # Evaluate: for each eval image, compare head prediction vs original model
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    
    head.eval()
    results = []
    for name, img_np in eval_imgs:
        # Original model depth
        orig_depth = encoder.predict_depth(img_np)
        orig_vbias = vertical_gradient_strength(orig_depth)
        
        # Head prediction from frozen features
        feats = encoder.extract_features(img_np)
        with torch.no_grad():
            head_pred = head(feats.to(args.device)).cpu().numpy().squeeze()
        head_vbias = vertical_gradient_strength(head_pred)
        
        # Orbit average (ground truth for equivariance)
        views = sym.batch(img_np)
        depths = [encoder.predict_depth(v) for v in views]
        aligned = [sym.inverse(d, i_elem) for i_elem, d in enumerate(depths)]
        orbit_avg = np.stack(aligned).mean(axis=0)
        orbit_vbias = vertical_gradient_strength(orbit_avg)
        
        delta = abs(orig_vbias) - abs(head_vbias)
        marker = "✓" if delta > 0.05 else ("✗" if delta < -0.05 else "~")
        td = "TD" if ("robot" in name or "_td" in name) else "  "
        print(f"  {marker} {td} {name:<20} orig={orig_vbias:+.4f}  head={head_vbias:+.4f}  orbit={orbit_vbias:+.4f}  Δ={delta:+.4f}")
        
        results.append({
            "image": name,
            "orig_vbias": orig_vbias,
            "head_vbias": head_vbias,
            "orbit_vbias": orbit_vbias,
            "improvement": delta,
        })
    
    # Summary
    if results:
        mean_orig = np.mean([abs(r["orig_vbias"]) for r in results])
        mean_head = np.mean([abs(r["head_vbias"]) for r in results])
        mean_orbit = np.mean([abs(r["orbit_vbias"]) for r in results])
        print(f"\n  Mean |vbias|: original={mean_orig:.4f} → head={mean_head:.4f} (orbit target={mean_orbit:.4f})")
        print(f"  Improvement: {mean_orig - mean_head:+.4f}")

    # Save
    torch.save(head.state_dict(), out_dir / "depth_head.pt")
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "config": vars(args),
            "head_params": n_params,
            "n_train_actual": len(train_imgs),
            "training_log": logs,
            "eval_results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
