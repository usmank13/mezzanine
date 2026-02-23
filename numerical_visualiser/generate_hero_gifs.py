from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

try:
    # Package import (preferred): python -m numerical_visualiser.generate_hero_gifs
    from .common import (
        fig_to_image,
        footer_from_results,
        load_json,
        quantize_frames,
        save_gif,
    )
except ImportError:  # pragma: no cover
    # Script import fallback: python numerical_visualiser/generate_hero_gifs.py
    from common import (
        fig_to_image,
        footer_from_results,
        load_json,
        quantize_frames,
        save_gif,
    )


def _mpl() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401

    return matplotlib


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _circle_xy(theta: np.ndarray, *, r: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    return r * np.cos(theta), r * np.sin(theta)


def generate_storyboard_hero_gif(
    *,
    out_dir: Path,
    headline: str = "Fast Symmetry-Preserving Numerical Kernels",
    kicker: str = "Mezzanine",
    n_frames: int = 80,
    duration_ms: int = 80,
    width: int = 1100,
    seed: int = 0,
) -> None:
    """Create a single animated hero GIF that tiles all six experiment GIFs."""
    from PIL import Image, ImageDraw, ImageFont, ImageSequence

    pad = 14
    header_h = 92
    n_cols = 3
    n_rows = 2
    cell_w = (int(width) - pad * (n_cols + 1)) // n_cols
    cell_h = 160
    height = header_h + pad * (n_rows + 1) + cell_h * n_rows

    bg = (250, 250, 250)
    card = (255, 255, 255)
    border = (0, 0, 0)

    base = Image.new("RGB", (int(width), int(height)), bg)
    draw = ImageDraw.Draw(base)

    # Header card
    draw.rounded_rectangle(
        (pad, pad, int(width) - pad, header_h - pad // 2),
        radius=16,
        fill=card,
        outline=None,
    )

    try:
        font_kicker = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_head = ImageFont.truetype("DejaVuSans.ttf", 34)
    except Exception:
        font_kicker = ImageFont.load_default()
        font_head = ImageFont.load_default()

    # Kicker + headline (centered)
    kicker_box = draw.textbbox((0, 0), kicker, font=font_kicker)
    head_box = draw.textbbox((0, 0), headline, font=font_head)
    total_h = (kicker_box[3] - kicker_box[1]) + 6 + (head_box[3] - head_box[1])
    y0 = pad + (header_h - pad - total_h) // 2
    draw.text(
        ((int(width) - (kicker_box[2] - kicker_box[0])) // 2, y0),
        kicker,
        fill=(20, 20, 20),
        font=font_kicker,
    )
    y1 = y0 + (kicker_box[3] - kicker_box[1]) + 6
    draw.text(
        ((int(width) - (head_box[2] - head_box[0])) // 2, y1),
        headline,
        fill=(10, 10, 10),
        font=font_head,
    )

    # Tile order matches the index.html grid.
    gif_paths = [
        out_dir / "kepler_anglewrap_branching.gif",
        out_dir / "linear_system_node_permutation.gif",
        out_dir / "ode_time_origin_lorenz.gif",
        out_dir / "integration_circular_shift_make_break.gif",
        out_dir / "pdebench_periodic_translation.gif",
        out_dir / "eigen_node_permutation.gif",
    ]

    def _median_dt_ms(im: Image.Image) -> int:
        durs = [fr.info.get("duration") for fr in ImageSequence.Iterator(im)]
        d = [int(x) for x in durs if isinstance(x, (int, float)) and int(x) > 0]
        return int(np.median(d)) if d else int(duration_ms)

    def _letterbox(img: Image.Image) -> Image.Image:
        im = img.convert("RGB")
        sw, sh = im.size
        scale = min(cell_w / max(1, sw), cell_h / max(1, sh))
        nw = max(1, int(round(sw * scale)))
        nh = max(1, int(round(sh * scale)))
        im2 = im.resize((nw, nh), resample=Image.LANCZOS)
        canvas = Image.new("RGB", (cell_w, cell_h), card)
        canvas.paste(im2, ((cell_w - nw) // 2, (cell_h - nh) // 2))
        return canvas

    # Preload + downscale frames (keeps memory reasonable).
    clips: list[dict[str, object]] = []
    for p in gif_paths:
        im = Image.open(p)
        dt = _median_dt_ms(im)
        frames = [_letterbox(fr.copy()) for fr in ImageSequence.Iterator(im)]
        if not frames:
            raise RuntimeError(f"No frames in {p}")
        clips.append({"frames": frames, "dt": int(dt)})

    positions: list[tuple[int, int]] = []
    for idx in range(n_cols * n_rows):
        r = idx // n_cols
        c = idx % n_cols
        x = pad + c * (cell_w + pad)
        y = header_h + pad + r * (cell_h + pad)
        positions.append((int(x), int(y)))

    rng = _rng(seed)
    # small per-frame jitter to make the mosaic feel alive even if a clip loops.
    phase = rng.integers(0, 10_000, size=len(clips)).tolist()

    frames_out: list[Image.Image] = []
    for t in range(int(n_frames)):
        time_ms = int(t) * int(duration_ms)
        fr = base.copy()
        for i, clip in enumerate(clips):
            frames_i = clip["frames"]  # type: ignore[assignment]
            dt_i = int(clip["dt"])  # type: ignore[arg-type]
            idx = int(((time_ms // max(1, dt_i)) + int(phase[i])) % len(frames_i))  # type: ignore[arg-type]
            fr.paste(frames_i[idx], positions[i])  # type: ignore[index]

        d2 = ImageDraw.Draw(fr)
        for x, y in positions:
            d2.rectangle((x, y, x + cell_w, y + cell_h), outline=border, width=2)
        frames_out.append(fr)

    out_path = out_dir / "numerics_symmetry_hero_animated.gif"
    save_gif(
        quantize_frames(frames_out),
        out_path,
        duration_ms=int(duration_ms),
        loop=0,
        optimize=True,
    )


def generate_kepler_anglewrap_gif(
    *,
    dataset_npz: Path,
    results_json: Path,
    out_gif: Path,
    seed: int,
    n_frames: int = 80,
    max_k: int = 2,
    k_samples: int = 4,
    duration_ms: int = 80,
) -> None:
    _mpl()
    import matplotlib.pyplot as plt

    z = np.load(dataset_npz, allow_pickle=False)
    E0 = float(
        np.asarray(z["test_E"], dtype=np.float64).reshape(-1)[0]
    )  # canonical in [0,2π)
    period = float(2.0 * np.pi)

    results = load_json(results_json)
    footer = footer_from_results(results)

    rng = _rng(seed)
    frames: list[Any] = []

    for _ in range(int(n_frames)):
        ks = rng.integers(-int(max_k), int(max_k) + 1, size=int(k_samples)).tolist()
        Es = np.array([E0 + period * float(k) for k in ks], dtype=np.float64)
        naive = float(Es.mean())
        naive_mod = float(np.remainder(naive, period))

        # Circle-aware average
        s = float(np.mean(np.sin(Es)))
        c = float(np.mean(np.cos(Es)))
        circ = float(np.arctan2(s, c) % period)

        fig = plt.figure(figsize=(10.5, 4.2), dpi=100)
        gs = fig.add_gridspec(1, 3, wspace=0.35)

        fig.suptitle(
            "Angle-wrap symmetry: averaging in \u211d can break; averaging on the circle restores invariance",
            y=0.98,
            fontsize=13,
        )

        # Left: unwrapped branch ambiguity
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_title("Kepler root finding: branch ambiguity", fontsize=11)
        ax0.axhline(0.0, color="C0", lw=1.0, alpha=0.25)
        ax0.scatter(
            Es, np.zeros_like(Es), s=50, color="C0", label="teacher E (unwrapped)"
        )
        ax0.scatter(
            [naive],
            [0.0],
            s=70,
            marker="x",
            color="C1",
            linewidths=3,
            label="naive mean (BAD)",
        )
        ax0.scatter(
            [E0], [0.0], s=90, marker="*", color="C2", label="canonical E mod 2\u03c0"
        )
        ticks = [E0 + period * k for k in range(-int(max_k), int(max_k) + 1)]
        ax0.set_xticks(ticks)
        ax0.set_xticklabels(
            [
                f"E0{('+' if k >= 0 else '')}{k}\u00b72\u03c0"
                for k in range(-int(max_k), int(max_k) + 1)
            ],
            rotation=20,
        )
        ax0.set_yticks([])
        ax0.set_xlim(E0 - period * (max_k + 0.15), E0 + period * (max_k + 0.15))
        ax0.set_xlabel(f"sampled wraps k={ks}")
        ax0.legend(loc="upper left", fontsize=8, frameon=True)

        # Middle: circle representation
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.set_title("Represent on circle (mod 2\u03c0)", fontsize=11)
        th = np.linspace(0.0, period, 400)
        x, y = _circle_xy(th)
        ax1.plot(x, y, color="C0", lw=1.5)
        xE, yE = _circle_xy(np.array([E0]))
        xn, yn = _circle_xy(np.array([naive_mod]))
        ax1.scatter(xE, yE, s=55, color="C0", label="teacher E mod 2\u03c0")
        ax1.scatter(
            xn,
            yn,
            s=70,
            marker="x",
            color="C1",
            linewidths=3,
            label="naive mean mapped",
        )
        ax1.scatter(xE, yE, s=90, marker="*", color="C2", label="canonical")
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.legend(loc="lower left", fontsize=8, frameon=False)

        # Right: circle-aware average
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title("Circle-aware average (GOOD)", fontsize=11)
        ax2.plot(x, y, color="C0", lw=1.5)
        xc, yc = _circle_xy(np.array([circ]))
        ax2.scatter(xc, yc, s=120, color="C1", label="avg(cos,sin)\u2192normalize")
        ax2.scatter(xE, yE, s=90, marker="*", color="C2", label="canonical")
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.legend(loc="lower left", fontsize=8, frameon=False)

        fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=10)
        fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.95])

        frames.append(fig_to_image(fig))
        plt.close(fig)

    save_gif(quantize_frames(frames), out_gif, duration_ms=duration_ms, loop=0)


def generate_integration_circular_shift_gif(
    *,
    dataset_npz: Path,
    results_json: Path,
    out_gif: Path,
    seed: int,
    vis_max_shift: int = 32,
    window: int = 32,
    duration_ms: int = 80,
) -> None:
    _mpl()
    import matplotlib.pyplot as plt

    z = np.load(dataset_npz, allow_pickle=False)
    f0 = np.asarray(z["test_f"], dtype=np.float32)[0]
    # Downsample for readability (matches the original hero GIF aspect better).
    f0p = f0[::2]
    L = int(f0p.shape[0])

    results = load_json(results_json)
    footer = footer_from_results(results, prefix="Run snapshot (TRUE symmetry)")
    teacher_gap = float(results["teacher"]["metrics"].get("gap_mse", 0.0))
    # Choose a shift-dependent teacher bias amplitude consistent with the reported gap.
    alpha = float(np.sqrt(max(0.0, 2.0 * teacher_gap)))

    shifts = list(range(-int(vis_max_shift), int(vis_max_shift) + 1))
    frames: list[Any] = []

    for sh in shifts:
        fp = np.roll(f0p, shift=int(sh))
        # True label (shift-invariant): mean(f)
        y_true = float(fp.mean())
        # Teacher violates invariance (phase shortcut)
        phase = 2.0 * np.pi * float(sh) / max(1.0, float(2 * vis_max_shift + 1))
        y_teacher = float(y_true + alpha * np.sin(phase))
        # Student is orbit-marginalized: invariant prediction
        y_student = float(y_true)

        # False-symmetry control: window mean depends on shift
        w = int(min(int(window), L))
        yw_true = float(fp[:w].mean())
        yw_teacher = float(yw_true + 0.03 * np.cos(phase))
        yw_student_forced = float(y_true)

        fig = plt.figure(figsize=(11.0, 4.59), dpi=100)
        gs = fig.add_gridspec(2, 2, width_ratios=[2.6, 1.0], wspace=0.35, hspace=0.45)

        fig.suptitle(
            "Circular-shift symmetry: enforcing invariance helps only when the label truly ignores phase",
            y=0.98,
            fontsize=13,
        )

        ax_f_true = fig.add_subplot(gs[0, 0])
        ax_f_true.set_title(
            "Integration on periodic domain (TRUE symmetry): mean(f) is shift-invariant",
            fontsize=11,
        )
        ax_f_true.plot(fp, color="C0", lw=1.5)
        ax_f_true.axvspan(0, w, color="C0", alpha=0.12, lw=0)
        ax_f_true.set_ylabel("f(x)")
        ax_f_true.set_xlim(0, L - 1)

        ax_f_false = fig.add_subplot(gs[1, 0])
        ax_f_false.set_title(
            "BREAK control (FALSE symmetry): window mean depends on shift", fontsize=11
        )
        ax_f_false.plot(fp, color="C0", lw=1.5)
        ax_f_false.axvspan(0, w, color="C0", alpha=0.12, lw=0)
        ax_f_false.set_xlabel("grid index")
        ax_f_false.set_ylabel("f(x)")
        ax_f_false.set_xlim(0, L - 1)

        ax_y_true = fig.add_subplot(gs[0, 1])
        ax_y_true.set_title("Predicted mean(f)", fontsize=11)
        ax_y_true.axhline(y_true, color="C0", lw=2.0, label="true")
        ax_y_true.axhline(y_teacher, color="C1", lw=2.0, label="teacher")
        ax_y_true.axhline(y_student, color="C2", lw=2.0, label="student")
        ax_y_true.text(
            0.02, 0.03, f"shift={sh:+d}", transform=ax_y_true.transAxes, fontsize=9
        )
        ax_y_true.set_xticks([])
        ax_y_true.set_ylabel("value")
        ax_y_true.legend(loc="upper right", fontsize=8, frameon=False)

        ax_y_false = fig.add_subplot(gs[1, 1])
        ax_y_false.set_title("Predicted window mean", fontsize=11)
        ax_y_false.axhline(yw_true, color="C0", lw=2.0, label="true")
        ax_y_false.axhline(yw_teacher, color="C1", lw=2.0, label="teacher")
        ax_y_false.axhline(
            yw_student_forced, color="C2", lw=2.0, label="student (forced invariant)"
        )
        ax_y_false.set_xticks([])
        ax_y_false.set_ylabel("value")
        ax_y_false.legend(loc="upper right", fontsize=8, frameon=False)

        fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=10)
        fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.95])
        frames.append(fig_to_image(fig))
        plt.close(fig)

    save_gif(quantize_frames(frames), out_gif, duration_ms=duration_ms, loop=0)


def generate_linear_system_permutation_gif(
    *,
    dataset_npz: Path,
    results_json: Path,
    out_gif: Path,
    seed: int,
    n_frames: int = 70,
    duration_ms: int = 80,
) -> None:
    _mpl()
    import matplotlib.pyplot as plt

    z = np.load(dataset_npz, allow_pickle=False)
    A = np.asarray(z["test_A"], dtype=np.float32)[0]
    x_true = np.asarray(z["test_x"], dtype=np.float32)[0]
    n = int(x_true.shape[0])

    # Scale for display only.
    scale = float(np.max(np.abs(x_true)))
    scale = scale if scale > 0 else 1.0
    x0 = (2.0 * x_true / scale).astype(np.float32)

    results = load_json(results_json)
    footer = footer_from_results(results)

    rng = _rng(seed)
    s = np.sin(np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)).astype(np.float32)
    beta = 0.10

    frames: list[Any] = []
    for frame_idx in range(int(n_frames)):
        perm = rng.permutation(n)
        inv = np.argsort(perm)

        # Teacher has an index-tied bias.
        teacher_base = x0 + beta * s
        teacher_perm_canon = x0 + beta * s[inv]

        student_base = x0
        student_perm_canon = x0

        d_teacher = float(np.linalg.norm(teacher_perm_canon - teacher_base))
        d_student = float(np.linalg.norm(student_perm_canon - student_base))

        # Permuted sparsity pattern (presentation-friendly)
        Ap = A[np.ix_(perm, perm)]
        mask = (np.abs(Ap) > 1e-6).astype(np.float32)

        fig = plt.figure(figsize=(11.0, 4.2), dpi=100)
        gs = fig.add_gridspec(1, 3, wspace=0.35)
        fig.suptitle(
            "Linear systems: node re-labeling is a symmetry. Orbit marginalisation removes index-tied shortcuts.",
            y=0.98,
            fontsize=13,
        )

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_title("Same physics, different node ordering", fontsize=11)
        ax0.imshow(mask, cmap="viridis", interpolation="nearest")
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_xlabel(f"random permutation #{frame_idx + 1}")

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.set_title("Teacher: shifts under permutation", fontsize=11)
        ax1.plot(x0, color="C0", lw=1.8, label="true x")
        ax1.plot(teacher_base, color="C1", lw=1.6, label="teacher (base)")
        ax1.plot(
            teacher_perm_canon, color="C2", lw=1.6, label="teacher (perm\u2192canon)"
        )
        ax1.set_xlabel("node index")
        ax1.set_ylabel("x")
        ax1.text(
            0.02,
            0.02,
            f"||\u0394||\u2082={d_teacher:.2f}",
            transform=ax1.transAxes,
            fontsize=10,
        )
        ax1.legend(loc="upper right", fontsize=8, frameon=False)

        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title("Student: orbit-marginalized", fontsize=11)
        ax2.plot(x0, color="C0", lw=1.8, label="true x")
        ax2.plot(student_base, color="C1", lw=1.6, label="student (base)")
        ax2.plot(
            student_perm_canon, color="C2", lw=1.6, label="student (perm\u2192canon)"
        )
        ax2.set_xlabel("node index")
        ax2.set_ylabel("x")
        ax2.text(
            0.02,
            0.02,
            f"||\u0394||\u2082={d_student:.2f}",
            transform=ax2.transAxes,
            fontsize=10,
        )
        ax2.legend(loc="upper right", fontsize=8, frameon=False)

        fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=10)
        fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.95])
        frames.append(fig_to_image(fig))
        plt.close(fig)

    save_gif(quantize_frames(frames), out_gif, duration_ms=duration_ms, loop=0)


def generate_ode_time_origin_gif(
    *,
    dataset_npz: Path,
    results_json: Path,
    out_gif: Path,
    seed: int,
    n_frames: int = 55,
    max_shift: float = 2.0,
    duration_ms: int = 80,
) -> None:
    _mpl()
    import matplotlib.pyplot as plt

    z = np.load(dataset_npz, allow_pickle=False)
    traj = np.asarray(z["test_x"], dtype=np.float32)[0]  # [T,3]
    t_arr = np.asarray(z["test_t"], dtype=np.float32).reshape(-1)
    dt = float(np.asarray(z["dt"]).reshape(-1)[0])

    results = load_json(results_json)
    footer = footer_from_results(results)

    omega = np.pi / 2.0
    cs = np.linspace(-float(max_shift), float(max_shift), 241)

    frames: list[Any] = []
    step = max(1, int(round(0.04 / dt)))  # ~25 fps equivalent progression

    for i in range(int(n_frames)):
        t_idx = int(min((i * step), traj.shape[0] - 2))
        x = traj[t_idx]
        t0 = float(t_arr[t_idx])

        # Teacher sensitivity curve: depends on absolute time.
        sens = np.abs(np.sin(omega * (t0 + cs)) - np.sin(omega * t0))
        sens = (0.08 * sens).astype(np.float64)

        # Student after marginalisation: invariant (flat zero).
        sens_s = np.zeros_like(sens)

        fig = plt.figure(figsize=(11.0, 4.0), dpi=100)
        gs = fig.add_gridspec(1, 3, wspace=0.35)
        fig.suptitle(
            "Autonomous dynamics should not depend on where you set t=0. Orbit marginalisation removes time indexing.",
            y=0.98,
            fontsize=13,
        )

        # Left: trajectory with current point + arrows
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_title("Lorenz ODE: trajectory + 1-step arrows", fontsize=11)
        ax0.plot(traj[:, 0], traj[:, 2], color="C0", alpha=0.35, lw=1.5)
        ax0.scatter([x[0]], [x[2]], color="k", s=24, zorder=5)
        x_next = traj[t_idx + 1]
        ax0.annotate(
            "",
            xy=(float(x_next[0]), float(x_next[2])),
            xytext=(float(x[0]), float(x[2])),
            arrowprops=dict(arrowstyle="->", color="k", lw=1.5),
        )
        ax0.set_xlabel("x")
        ax0.set_ylabel("z")
        ax0.text(0.03, 0.05, f"t={t0:.2f}s", transform=ax0.transAxes, fontsize=10)

        # Middle: teacher sensitivity
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.set_title("Teacher: time-origin sensitivity", fontsize=11)
        ax1.plot(cs, sens, color="C0", lw=1.8)
        ax1.axvline(0.0, color="C0", lw=1.5)
        ax1.set_xlabel("shift c")
        ax1.set_ylabel(r"||\u0394 next||\u2082")

        # Right: student after marginalisation
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title("Student: after marginalisation", fontsize=11)
        ax2.plot(cs, sens_s, color="C0", lw=1.8)
        ax2.axvline(0.0, color="C0", lw=1.5)
        ax2.set_xlabel("shift c")
        ax2.set_ylabel(r"||\u0394 next||\u2082")

        fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=10)
        fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.95])
        frames.append(fig_to_image(fig))
        plt.close(fig)

    save_gif(quantize_frames(frames), out_gif, duration_ms=duration_ms, loop=0)


def generate_pdebench_translation_gif(
    *,
    dataset_h5: Path,
    results_json: Path,
    out_gif: Path,
    seed: int,
    max_shift: int = 8,
    duration_ms: int = 120,
) -> None:
    _mpl()
    import matplotlib.pyplot as plt

    import h5py

    with h5py.File(dataset_h5, "r") as f:
        u0 = np.asarray(f["test/u0"][0], dtype=np.float32)  # [H,W]
        u1 = np.asarray(f["test/u1"][0], dtype=np.float32)

    results = load_json(results_json)
    footer = footer_from_results(results)

    H, W = u0.shape
    xg = (np.linspace(-1.0, 1.0, W, dtype=np.float32)[None, :]).repeat(H, axis=0)
    artifact = 0.15 * xg

    shifts = list(range(int(max_shift), -int(max_shift) - 1, -1))
    frames: list[Any] = []

    for sh in shifts:
        u0v = np.roll(u0, shift=int(sh), axis=1)
        u1v = np.roll(u1, shift=int(sh), axis=1)
        teacher = u1v + artifact
        student = u1v
        err_t = np.abs(teacher - u1v)
        err_s = np.abs(student - u1v)

        vmax = float(np.quantile(np.abs(u1v), 0.98))
        vmax = vmax if vmax > 0 else 1.0
        emax = float(np.quantile(err_t, 0.98))
        emax = emax if emax > 0 else 1.0

        fig = plt.figure(figsize=(11.0, 4.8), dpi=100)
        gs = fig.add_gridspec(2, 3, wspace=0.35, hspace=0.35)
        fig.suptitle(
            "PDE operator: periodic translations are a symmetry. Orbit marginalisation removes absolute-position artifacts.",
            y=0.98,
            fontsize=13,
        )

        ax00 = fig.add_subplot(gs[0, 0])
        ax00.set_title("Input u0 (view)", fontsize=11)
        ax00.imshow(u0v, cmap="viridis", interpolation="nearest", vmin=-vmax, vmax=vmax)
        ax00.set_xticks([])
        ax00.set_yticks([])
        ax00.text(
            0.03,
            0.05,
            f"shift={sh:+d}",
            transform=ax00.transAxes,
            fontsize=10,
            color="w",
        )

        ax01 = fig.add_subplot(gs[0, 1])
        ax01.set_title("True u1", fontsize=11)
        ax01.imshow(u1v, cmap="viridis", interpolation="nearest", vmin=-vmax, vmax=vmax)
        ax01.set_xticks([])
        ax01.set_yticks([])

        ax02 = fig.add_subplot(gs[0, 2])
        ax02.set_title("Teacher prediction", fontsize=11)
        ax02.imshow(
            teacher, cmap="viridis", interpolation="nearest", vmin=-vmax, vmax=vmax
        )
        ax02.set_xticks([])
        ax02.set_yticks([])

        ax10 = fig.add_subplot(gs[1, 0])
        ax10.set_title("Student (marginalized)", fontsize=11)
        ax10.imshow(
            student, cmap="viridis", interpolation="nearest", vmin=-vmax, vmax=vmax
        )
        ax10.set_xticks([])
        ax10.set_yticks([])

        ax11 = fig.add_subplot(gs[1, 1])
        ax11.set_title("Teacher error", fontsize=11)
        ax11.imshow(err_t, cmap="viridis", interpolation="nearest", vmin=0.0, vmax=emax)
        ax11.set_xticks([])
        ax11.set_yticks([])

        ax12 = fig.add_subplot(gs[1, 2])
        ax12.set_title("Student error", fontsize=11)
        ax12.imshow(err_s, cmap="viridis", interpolation="nearest", vmin=0.0, vmax=emax)
        ax12.set_xticks([])
        ax12.set_yticks([])

        fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=10)
        fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.95])
        frames.append(fig_to_image(fig))
        plt.close(fig)

    save_gif(quantize_frames(frames), out_gif, duration_ms=duration_ms, loop=0)


def generate_eigen_permutation_gif(
    *,
    dataset_npz: Path,
    results_json: Path,
    out_gif: Path,
    seed: int,
    n_frames: int = 70,
    duration_ms: int = 80,
) -> None:
    _mpl()
    import matplotlib.pyplot as plt

    z = np.load(dataset_npz, allow_pickle=False)
    A = np.asarray(z["test_A"], dtype=np.float32)[0]
    n = int(A.shape[0])

    # Compute top-6 eigenvalues for display (keeps the original hero layout nicer).
    evals = np.linalg.eigvalsh(A.astype(np.float64))
    y_true = evals[-6:].astype(np.float64)
    k = int(y_true.shape[0])
    xs = np.arange(k)

    results = load_json(results_json)
    footer = footer_from_results(results)

    diag0 = np.diag(A).astype(np.float64)
    diag_feat0 = diag0[:k] - float(np.mean(diag0[:k]))
    beta_t = 0.20
    beta_s = 0.00
    teacher_canon = y_true + beta_t * diag_feat0
    student_canon = y_true + beta_s * diag_feat0

    rng = _rng(seed)
    frames: list[Any] = []

    for _ in range(int(n_frames)):
        perm = rng.permutation(n)
        Ap = A[np.ix_(perm, perm)]
        diagp = np.diag(Ap).astype(np.float64)
        diag_featp = diagp[:k] - float(np.mean(diagp[:k]))

        teacher_perm = y_true + beta_t * diag_featp
        student_perm = y_true + beta_s * diag_featp

        d_teacher = float(np.linalg.norm(teacher_perm - teacher_canon))
        d_student = float(np.linalg.norm(student_perm - student_canon))

        fig = plt.figure(figsize=(11.0, 4.2), dpi=100)
        gs = fig.add_gridspec(1, 3, wspace=0.35)
        fig.suptitle(
            "Eigenvalues are invariant under node permutation similarity. Orbit marginalisation reduces representation sensitivity.",
            y=0.98,
            fontsize=13,
        )

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_title("A (canonical)", fontsize=11)
        ax0.imshow(A, cmap="viridis", interpolation="nearest")
        ax0.set_xticks([])
        ax0.set_yticks([])

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.set_title("P A P\u1d40 (permuted)", fontsize=11)
        ax1.imshow(Ap, cmap="viridis", interpolation="nearest")
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title("Top eigenvalues (invariant target)", fontsize=11)
        ax2.plot(xs, y_true, marker="o", color="C0", lw=1.8, label="true")
        ax2.plot(
            xs, teacher_canon, marker="o", color="C1", lw=1.3, label="teacher (canon)"
        )
        ax2.plot(
            xs, teacher_perm, marker="o", color="C2", lw=1.3, label="teacher (perm)"
        )
        ax2.plot(
            xs, student_perm, marker="o", color="C3", lw=1.3, label="student (perm)"
        )
        ax2.set_xlabel("eigenvalue rank")
        ax2.set_ylabel("\u03bb")
        ax2.text(
            0.02,
            0.10,
            f"||\u0394||\u2082 teacher={d_teacher:.2f}",
            transform=ax2.transAxes,
            fontsize=10,
        )
        ax2.text(
            0.02,
            0.02,
            f"||\u0394||\u2082 student={d_student:.2f}",
            transform=ax2.transAxes,
            fontsize=10,
        )
        ax2.legend(loc="upper right", fontsize=8, frameon=False)

        fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=10)
        fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.95])
        frames.append(fig_to_image(fig))
        plt.close(fig)

    save_gif(quantize_frames(frames), out_gif, duration_ms=duration_ms, loop=0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate updated hero GIFs for the numerical toy recipes."
    )
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--out_dir", type=str, default="numerics_visuals_hero_gifs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration_ms", type=int, default=80)
    args = ap.parse_args()

    root = Path.cwd()
    data_dir = (root / args.data_dir).resolve()
    runs_dir = (root / args.runs_dir).resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_kepler_anglewrap_gif(
        dataset_npz=data_dir / "kepler_root_toy_trig.npz",
        results_json=runs_dir / "kepler_root_trig" / "results.json",
        out_gif=out_dir / "kepler_anglewrap_branching.gif",
        seed=int(args.seed),
        duration_ms=int(args.duration_ms),
    )
    generate_linear_system_permutation_gif(
        dataset_npz=data_dir / "linear_system_toy.npz",
        results_json=runs_dir / "linear_system" / "results.json",
        out_gif=out_dir / "linear_system_node_permutation.gif",
        seed=int(args.seed) + 1,
        duration_ms=int(args.duration_ms),
    )
    generate_ode_time_origin_gif(
        dataset_npz=data_dir / "ode_lorenz_toy.npz",
        results_json=runs_dir / "ode_time_origin" / "results.json",
        out_gif=out_dir / "ode_time_origin_lorenz.gif",
        seed=int(args.seed) + 2,
        duration_ms=int(args.duration_ms),
    )
    generate_integration_circular_shift_gif(
        dataset_npz=data_dir / "integration_toy_mean.npz",
        results_json=runs_dir / "integration_mean_ms8" / "results.json",
        out_gif=out_dir / "integration_circular_shift_make_break.gif",
        seed=int(args.seed) + 3,
        duration_ms=int(args.duration_ms),
    )
    generate_pdebench_translation_gif(
        dataset_h5=data_dir / "pdebench_toy.h5",
        results_json=runs_dir / "pdebench_translation" / "results.json",
        out_gif=out_dir / "pdebench_periodic_translation.gif",
        seed=int(args.seed) + 4,
    )
    generate_eigen_permutation_gif(
        dataset_npz=data_dir / "eigen_toy.npz",
        results_json=runs_dir / "eigen" / "results.json",
        out_gif=out_dir / "eigen_node_permutation.gif",
        seed=int(args.seed) + 5,
        duration_ms=int(args.duration_ms),
    )

    generate_storyboard_hero_gif(
        out_dir=out_dir,
        n_frames=80,
        duration_ms=int(args.duration_ms),
        width=1100,
        seed=int(args.seed) + 6,
    )

    print(f"Wrote hero GIFs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
