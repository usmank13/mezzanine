#!/usr/bin/env python3
"""Mezzanine distillation: fine-tune Depth Anything to predict orbit-averaged depth.

Proper approach:
  1. For each training image, compute D4 orbit-averaged depth (teacher signal)
  2. Fine-tune the full DA model to predict this orbit average
  3. On images where model is already equivariant → target ≈ original → no change
  4. On images where bias exists → target is corrected → model learns the fix

Usage:
    python scripts/distill_orbit_average.py --device cuda --out out_distill_orbit
"""

from __future__ import annotations
import argparse, json, time, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Data ─────────────────────────────────────────────────────────────────

# Mix of regular + top-down COCO images for diverse training
TRAIN_URLS = [
    # Regular perspective scenes
    ("coco_cats", "http://images.cocodataset.org/val2017/000000039769.jpg"),
    ("coco_baseball", "http://images.cocodataset.org/val2017/000000397133.jpg"),
    ("coco_kitchen", "http://images.cocodataset.org/val2017/000000037777.jpg"),
    ("coco_bus", "http://images.cocodataset.org/val2017/000000087470.jpg"),
    ("coco_train", "http://images.cocodataset.org/val2017/000000131131.jpg"),
    ("coco_bathroom", "http://images.cocodataset.org/val2017/000000181859.jpg"),
    ("coco_living", "http://images.cocodataset.org/val2017/000000087875.jpg"),
    ("coco_plane", "http://images.cocodataset.org/val2017/000000146457.jpg"),
    ("coco_bear", "http://images.cocodataset.org/val2017/000000459467.jpg"),
    ("coco_boat", "http://images.cocodataset.org/val2017/000000284991.jpg"),
    ("coco_giraffe", "http://images.cocodataset.org/val2017/000000579635.jpg"),
    ("coco_mountain", "http://images.cocodataset.org/val2017/000000226111.jpg"),
    ("coco_clock", "http://images.cocodataset.org/val2017/000000348881.jpg"),
    # Top-down / overhead views (where bias is worst)
    ("coco_food_td", "http://images.cocodataset.org/val2017/000000080340.jpg"),
    ("coco_pizza_td", "http://images.cocodataset.org/val2017/000000227765.jpg"),
    ("coco_donuts_td", "http://images.cocodataset.org/val2017/000000295316.jpg"),
    ("coco_cake_td", "http://images.cocodataset.org/val2017/000000360661.jpg"),
    ("coco_fruit_td", "http://images.cocodataset.org/val2017/000000002587.jpg"),
    ("coco_hotdog_td", "http://images.cocodataset.org/val2017/000000174004.jpg"),
    ("coco_sandwich_td", "http://images.cocodataset.org/val2017/000000279541.jpg"),
    ("coco_broccoli_td", "http://images.cocodataset.org/val2017/000000180135.jpg"),
    ("coco_plates_td", "http://images.cocodataset.org/val2017/000000084270.jpg"),
    ("coco_bread_td", "http://images.cocodataset.org/val2017/000000226592.jpg"),
    ("coco_salad_td", "http://images.cocodataset.org/val2017/000000349860.jpg"),
    ("coco_bowl_td", "http://images.cocodataset.org/val2017/000000434230.jpg"),
    ("coco_books_td", "http://images.cocodataset.org/val2017/000000532493.jpg"),
    ("coco_remote_td", "http://images.cocodataset.org/val2017/000000173091.jpg"),
    ("coco_laptop_td", "http://images.cocodataset.org/val2017/000000153529.jpg"),
    # More regular scenes for balance
    ("coco_street", "http://images.cocodataset.org/val2017/000000176446.jpg"),
    ("coco_skateboard", "http://images.cocodataset.org/val2017/000000087038.jpg"),
]

# Held-out eval images (not in training)
EVAL_URLS = [
    ("eval_desk_td", "http://images.cocodataset.org/val2017/000000056350.jpg"),
    ("eval_dog", "http://images.cocodataset.org/val2017/000000222564.jpg"),
    ("eval_surfboard", "http://images.cocodataset.org/val2017/000000404484.jpg"),
    ("eval_bench", "http://images.cocodataset.org/val2017/000000289393.jpg"),
]


def download_images(url_list, cache_dir):
    """Download images, return list of (name, np.ndarray)."""
    import requests, io
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    pairs = []
    for name, url in url_list:
        cache_path = cache_dir / f"{name}.jpg"
        if cache_path.exists():
            img = Image.open(cache_path).convert("RGB")
        else:
            try:
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                img.save(cache_path)
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                continue
        pairs.append((name, np.asarray(img)))
        print(f"  ✓ {name} ({img.size[0]}×{img.size[1]})")
    return pairs


# ── Orbit averaging ─────────────────────────────────────────────────────

def compute_orbit_target(encoder, sym, img_np):
    """Compute orbit-averaged depth for a single image. Returns (H, W) numpy."""
    views = sym.batch(img_np)
    depths = [encoder.predict(v) for v in views]
    aligned = [sym.inverse(d, i) for i, d in enumerate(depths)]
    return np.stack(aligned).mean(axis=0)


# ── Metrics ──────────────────────────────────────────────────────────────

def vertical_gradient_strength(depth_map):
    h = depth_map.shape[0]
    row_indices = np.arange(h, dtype=np.float64)
    row_means = depth_map.mean(axis=1).astype(np.float64)
    if row_means.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(row_indices, row_means)[0, 1])


def evaluate_model(model, processor, sym, images, device, label=""):
    """Evaluate warrant gap and vertical bias."""
    from mezzanine.encoders.depth_anything import DepthAnythingEncoder, DepthAnythingEncoderConfig
    
    results = []
    for name, img_np in images:
        # Student prediction
        pil = Image.fromarray(img_np)
        inputs = processor(images=[pil], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        target_sizes = [(img_np.shape[0], img_np.shape[1])]
        post = processor.post_process_depth_estimation(out, target_sizes=target_sizes)
        student_depth = post[0]["predicted_depth"].cpu().float().numpy()
        
        # Measure vbias
        vbias = vertical_gradient_strength(student_depth)
        
        # Measure warrant gap (vflip only for speed)
        flipped = np.flipud(img_np).copy()
        pil_f = Image.fromarray(flipped)
        inputs_f = processor(images=[pil_f], return_tensors="pt")
        inputs_f = {k: v.to(device) for k, v in inputs_f.items()}
        with torch.no_grad():
            out_f = model(**inputs_f)
        post_f = processor.post_process_depth_estimation(out_f, target_sizes=target_sizes)
        flipped_depth = np.flipud(post_f[0]["predicted_depth"].cpu().float().numpy())
        
        # CV between original and flipped-back
        stack = np.stack([student_depth, flipped_depth])
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(mean > 1e-8, std / mean, 0.0)
        mean_cv = float(cv.mean())
        
        results.append({
            "image": name,
            "vbias": vbias,
            "mean_cv": mean_cv,
        })
    
    if results:
        mean_vbias = np.mean([abs(r["vbias"]) for r in results])
        mean_cv = np.mean([r["mean_cv"] for r in results])
        print(f"  {label}: mean |vbias|={mean_vbias:.4f}, mean CV={mean_cv:.4f}")
        for r in results:
            td = "_td" in r["image"] or "robot" in r["image"]
            marker = "TD" if td else "  "
            print(f"    {marker} {r['image']:<20} vbias={r['vbias']:+.4f}  CV={r['mean_cv']:.4f}")
    
    return results


# ── Training ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--out", default="out_distill_orbit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--subgroup", default="d4", choices=["d4", "vflip", "flips"],
                        help="Symmetry subgroup for orbit average target")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "image_cache"

    # Download images
    print("Downloading training images...")
    train_imgs = download_images(TRAIN_URLS, cache_dir)
    print(f"\nDownloading eval images...")
    eval_imgs = download_images(EVAL_URLS, cache_dir)
    
    # Also include local robot images for eval
    robot_dir = Path("test_images")
    for i in range(1, 6):
        p = robot_dir / f"topdown_robot_{i:02d}.jpg"
        if p.exists():
            img = np.asarray(Image.open(p).convert("RGB"))
            eval_imgs.append((f"robot_{i:02d}", img))
            print(f"  ✓ robot_{i:02d} ({img.shape[1]}×{img.shape[0]})")

    print(f"\nTrain: {len(train_imgs)} images, Eval: {len(eval_imgs)} images")

    # Load model
    print(f"\nLoading {args.model}...")
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).to(args.device)
    
    # Also need the encoder wrapper for orbit averaging
    from mezzanine.encoders.depth_anything import DepthAnythingEncoder, DepthAnythingEncoderConfig
    from mezzanine.symmetries.depth_geometric import DepthGeometricSymmetry, DepthGeometricSymmetryConfig
    
    teacher = DepthAnythingEncoder(DepthAnythingEncoderConfig(model_name=args.model), device=args.device)
    sym = DepthGeometricSymmetry(DepthGeometricSymmetryConfig(subgroup=args.subgroup))
    n_elements = len(sym.elements)
    print(f"Subgroup: {args.subgroup} ({n_elements} elements)")

    # Pre-compute orbit-averaged targets
    print(f"\nPre-computing orbit-averaged targets ({n_elements} passes per image)...")
    targets = {}
    for i, (name, img_np) in enumerate(train_imgs):
        print(f"  [{i+1}/{len(train_imgs)}] {name}...", end="", flush=True)
        t0 = time.time()
        orbit_avg = compute_orbit_target(teacher, sym, img_np)
        targets[name] = orbit_avg
        print(f" ({time.time()-t0:.1f}s)")

    # Free teacher model to save GPU memory
    del teacher
    torch.cuda.empty_cache()

    # Evaluate before training
    print("\n--- Before training ---")
    model.eval()
    pre_results = evaluate_model(model, processor, sym, eval_imgs, args.device, "Pre-train")

    # Fine-tune
    print(f"\nFine-tuning: {args.epochs} epochs, lr={args.lr}, batch_size={args.batch_size}")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_names = [name for name, _ in train_imgs]
    train_images = {name: img for name, img in train_imgs}
    
    logs = []
    for epoch in range(1, args.epochs + 1):
        # Shuffle
        indices = np.random.permutation(len(train_names))
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_start in range(0, len(indices), args.batch_size):
            batch_idx = indices[batch_start:batch_start + args.batch_size]
            
            # Prepare batch — resize all to 518x518 for uniform batching
            batch_pils = []
            batch_targets = []
            for idx in batch_idx:
                name = train_names[idx]
                img_np = train_images[name]
                target = targets[name]
                
                pil = Image.fromarray(img_np).resize((518, 518), Image.BILINEAR)
                batch_pils.append(pil)
                batch_targets.append(target)
            
            # Forward pass
            inputs = processor(images=batch_pils, return_tensors="pt", do_resize=False)
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            pred_depth = outputs.predicted_depth  # (B, H, W)
            
            # Compute loss against orbit-averaged targets
            loss = torch.tensor(0.0, device=args.device, requires_grad=True)
            for b in range(pred_depth.shape[0]):
                pred = pred_depth[b]  # (H, W)
                tgt_np = batch_targets[b]
                
                # Resize target to match prediction size
                tgt_t = torch.from_numpy(tgt_np).float().unsqueeze(0).unsqueeze(0).to(args.device)
                tgt_resized = F.interpolate(tgt_t, size=pred.shape, mode="bilinear", align_corners=False)
                tgt_resized = tgt_resized.squeeze()
                
                # Direct MSE — both pred and target come from same model, compatible scales
                loss = loss + F.mse_loss(pred, tgt_resized)
            
            loss = loss / pred_depth.shape[0]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}: loss={avg_loss:.6f}")
            logs.append({"epoch": epoch, "loss": avg_loss})
        
        # Evaluate periodically
        if epoch % 25 == 0 or epoch == args.epochs:
            model.eval()
            mid_results = evaluate_model(model, processor, sym, eval_imgs, args.device, f"Epoch {epoch}")
            logs[-1]["eval"] = mid_results
            model.train()

    # Final evaluation
    print("\n--- After training ---")
    model.eval()
    post_results = evaluate_model(model, processor, sym, eval_imgs, args.device, "Post-train")

    # Save
    # Checkpoint
    model.save_pretrained(out_dir / "student_model")
    processor.save_pretrained(out_dir / "student_model")
    print(f"\nModel saved to {out_dir / 'student_model'}")

    # Results
    results = {
        "config": {
            "model": args.model,
            "subgroup": args.subgroup,
            "epochs": args.epochs,
            "lr": args.lr,
            "n_train": len(train_imgs),
            "n_eval": len(eval_imgs),
        },
        "pre_train": pre_results,
        "post_train": post_results,
        "training_log": logs,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_dir / 'results.json'}")

    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    for pre, post in zip(pre_results, post_results):
        name = pre["image"]
        delta_vbias = abs(post["vbias"]) - abs(pre["vbias"])
        delta_cv = post["mean_cv"] - pre["mean_cv"]
        marker = "✓" if delta_vbias < -0.05 else ("✗" if delta_vbias > 0.05 else "~")
        print(f"  {marker} {name:<20} |vbias| {abs(pre['vbias']):.3f}→{abs(post['vbias']):.3f} ({delta_vbias:+.3f})  "
              f"CV {pre['mean_cv']:.3f}→{post['mean_cv']:.3f} ({delta_cv:+.3f})")


if __name__ == "__main__":
    main()
