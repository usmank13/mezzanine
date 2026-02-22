"""Generate a 1D periodic integration dataset for Mezzanine.

This writes a `.npz` consumed by `integration_circular_shift_distill`.

We sample periodic functions f(x) on a uniform grid and define the target as

    y = (1/2π) ∫_0^{2π} f(x)^2 dx

This is an "energy"-like integral (shift-invariant), with a gold-standard value
computed analytically from the Fourier coefficients.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--L", "--n_grid", dest="L", type=int, default=256, help="Grid points (alias: --n_grid)")
    ap.add_argument("--K", type=int, default=8, help="Fourier modes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp_scale", type=float, default=1.0)
    ap.add_argument("--include_offset", action="store_true", help="Include a constant offset term")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    n_total = int(args.n_train) + int(args.n_test)
    L = int(args.L)
    K = int(args.K)
    dx = float(2.0 * np.pi / L)
    x = (np.arange(L, dtype=np.float64) * dx).astype(np.float64)

    f = np.zeros((n_total, L), dtype=np.float32)
    y = np.zeros((n_total, 1), dtype=np.float32)

    for i in range(n_total):
        a = rng.normal(scale=float(args.amp_scale), size=(K,))
        b = rng.normal(scale=float(args.amp_scale), size=(K,))
        c0 = float(rng.normal(scale=float(args.amp_scale))) if bool(args.include_offset) else 0.0

        # f(x) = c0 + Σ a_k sin(kx) + b_k cos(kx)
        vals = c0 * np.ones_like(x)
        for k in range(1, K + 1):
            vals = vals + a[k - 1] * np.sin(k * x) + b[k - 1] * np.cos(k * x)
        f[i] = vals.astype(np.float32)

        # Analytic mean of f^2 over a full period:
        # mean(sin^2)=mean(cos^2)=1/2, cross terms vanish
        energy = c0 * c0 + 0.5 * float(np.sum(a * a + b * b))
        y[i, 0] = float(energy)

    perm = rng.permutation(n_total)
    f, y = f[perm], y[perm]
    n_train = int(args.n_train)

    np.savez(
        out,
        train_f=f[:n_train],
        train_y=y[:n_train],
        test_f=f[n_train:],
        test_y=y[n_train:],
        dx=np.array([dx], dtype=np.float32),
        source=str({"type": "fourier_energy", "L": L, "K": K, "dx": dx}),
    )

    print(f"Wrote {out} with {n_train} train and {n_total - n_train} test samples. L={L} K={K}")


if __name__ == "__main__":
    main()
