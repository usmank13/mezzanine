from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:  # pragma: no cover
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Pillow is required for GIF visualisation. Install with: pip install pillow"
    ) from e


def _wrap_phi(phi: np.ndarray) -> np.ndarray:
    return (phi + np.pi) % (2 * np.pi) - np.pi


def _apply_palette(v: np.ndarray) -> np.ndarray:
    """Map v in [0,1] to RGB uint8 image via a simple 5-point palette."""
    v = np.clip(v, 0.0, 1.0)
    # Palette anchors (t, (r,g,b))
    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    r = np.array([0, 25, 120, 220, 255], dtype=np.float32)
    g = np.array([0, 0, 20, 120, 255], dtype=np.float32)
    b = np.array([0, 50, 60, 20, 255], dtype=np.float32)

    flat = v.reshape(-1)
    rr = np.interp(flat, t, r)
    gg = np.interp(flat, t, g)
    bb = np.interp(flat, t, b)
    rgb = np.stack([rr, gg, bb], axis=-1).reshape(v.shape + (3,))
    return rgb.astype(np.uint8)


def render_jet_image(
    particles: np.ndarray,
    *,
    extent: float = 0.8,
    bins: int = 160,
    pt_normalize: bool = True,
    log_scale: bool = True,
    gamma: float = 0.6,
    bg: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Render a jet as a calorimeter-style image in the (y,phi) plane.

    particles: [P,4] array (pt, y, phi, pid). Padding rows have pt=0.
    """
    P = np.asarray(particles, dtype=np.float32)
    if P.ndim != 2 or P.shape[1] < 3:
        raise ValueError(f"Expected particles [P,4], got {P.shape}")

    pt = P[:, 0]
    mask = pt > 0
    if not np.any(mask):
        img = Image.new("RGB", (bins, bins), color=bg)
        return img

    pt = pt[mask]
    y = P[mask, 1]
    phi = _wrap_phi(P[mask, 2])

    if pt_normalize:
        s = float(pt.sum())
        if s > 0:
            pt = pt / s

    # 2D histogram
    H, _, _ = np.histogram2d(
        y,
        phi,
        bins=bins,
        range=[(-extent, extent), (-extent, extent)],
        weights=pt,
    )

    H = H.astype(np.float32)

    if log_scale:
        H = np.log1p(H)

    m = float(H.max())
    if m > 0:
        H = H / m

    if gamma != 1.0:
        H = np.power(H, float(gamma), dtype=np.float32)

    rgb = _apply_palette(H)
    img = Image.fromarray(rgb, mode="RGB")
    return img


@dataclass
class JetGifConfig:
    extent: float = 0.8
    bins: int = 160
    duration_ms: int = 90
    loop: int = 0


def write_jet_nuisance_gif(
    *,
    particles_views: List[np.ndarray],
    p_base_views: np.ndarray,
    p_student_views: np.ndarray,
    label: int,
    out_path: Path,
    cfg: JetGifConfig | None = None,
    class_names: Tuple[str, str] = ("gluon", "quark"),
    highlight_class: int = 1,
) -> None:
    """Create a GIF showing nuisance views + prediction instability.

    particles_views: list length K, each [P,4]
    p_*_views: [K,C] probabilities for the same K views
    label: true class index
    """
    cfg = cfg or JetGifConfig()

    K = len(particles_views)
    if K == 0:
        raise ValueError("particles_views must be non-empty")

    p_base_views = np.asarray(p_base_views, dtype=np.float32)
    p_student_views = np.asarray(p_student_views, dtype=np.float32)
    if p_base_views.shape[0] != K or p_student_views.shape[0] != K:
        raise ValueError("prob arrays must have length K")

    font = ImageFont.load_default()

    frames: List[Image.Image] = []
    for j in range(K):
        img = render_jet_image(particles_views[j], extent=cfg.extent, bins=cfg.bins)
        draw = ImageDraw.Draw(img)

        base_p = float(p_base_views[j, highlight_class])
        stud_p = float(p_student_views[j, highlight_class])

        # Text overlay
        title = f"Jet nuisance view {j + 1}/{K}"
        truth = f"truth: {class_names[int(label)]}"
        pb = f"baseline P({class_names[highlight_class]}) = {base_p:.2f}"
        ps = f"student  P({class_names[highlight_class]}) = {stud_p:.2f}"

        # Slight shadow for readability
        for dx, dy in [(1, 1), (1, 0), (0, 1)]:
            draw.text((8 + dx, 8 + dy), title, font=font, fill=(0, 0, 0))
            draw.text((8 + dx, 22 + dy), truth, font=font, fill=(0, 0, 0))
            draw.text((8 + dx, 36 + dy), pb, font=font, fill=(0, 0, 0))
            draw.text((8 + dx, 50 + dy), ps, font=font, fill=(0, 0, 0))
        draw.text((8, 8), title, font=font, fill=(255, 255, 255))
        draw.text((8, 22), truth, font=font, fill=(255, 255, 255))
        draw.text((8, 36), pb, font=font, fill=(255, 255, 255))
        draw.text((8, 50), ps, font=font, fill=(255, 255, 255))

        frames.append(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(cfg.duration_ms),
        loop=int(cfg.loop),
        optimize=False,
    )
