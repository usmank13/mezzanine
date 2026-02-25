from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def load_json(path: str | Path) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_metric(x: float, *, digits: int = 3) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(v):
        return str(v)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:.{digits}f}"
    # small numbers: scientific looks better
    return f"{v:.{digits}e}"


def footer_from_results(
    results: dict[str, Any], *, prefix: str = "Run snapshot"
) -> str:
    teacher_gap = float(results["teacher"]["metrics"].get("gap_mse", float("nan")))
    student_gap = float(results["student"]["metrics"].get("gap_mse", float("nan")))
    verdict = str(results["make_break"]["verdict"])
    verdict = verdict.replace("✅", "OK").replace("❌", "X")
    return f"{prefix}: gap_mse {fmt_metric(teacher_gap)} \u2192 {fmt_metric(student_gap)}  |  verdict: {verdict}"


def fig_to_image(fig) -> Image.Image:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # Matplotlib 3.10 removed `tostring_rgb`; use `buffer_rgba` when available.
    if hasattr(fig.canvas, "buffer_rgba"):
        buf = np.asarray(fig.canvas.buffer_rgba())
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return Image.fromarray(arr, mode="RGBA").convert("RGB")
    if hasattr(fig.canvas, "tostring_rgb"):
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        return Image.fromarray(buf, mode="RGB")
    if hasattr(fig.canvas, "tostring_argb"):
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        # ARGB -> RGBA
        buf = buf[:, :, [1, 2, 3, 0]]
        return Image.fromarray(buf, mode="RGBA").convert("RGB")
    raise RuntimeError("Unsupported matplotlib canvas: cannot extract pixels")


def quantize_frames(
    frames_rgb: list[Image.Image], *, colors: int = 256
) -> list[Image.Image]:
    if not frames_rgb:
        return []
    pal = frames_rgb[0].convert("P", palette=Image.ADAPTIVE, colors=int(colors))
    out = [pal]
    for fr in frames_rgb[1:]:
        out.append(fr.quantize(palette=pal))
    return out


def save_gif(
    frames: Iterable[Image.Image],
    out_path: str | Path,
    *,
    duration_ms: int = 80,
    loop: int = 0,
    optimize: bool = False,
) -> None:
    frames_list = list(frames)
    if not frames_list:
        raise ValueError("No frames to write")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_list[0].save(
        out,
        save_all=True,
        append_images=frames_list[1:],
        duration=int(duration_ms),
        loop=int(loop),
        disposal=2,
        optimize=bool(optimize),
    )
