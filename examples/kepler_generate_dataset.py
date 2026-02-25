"""Generate a Kepler root-finding dataset (Kepler's equation) for Mezzanine.

This produces a `.npz` consumed by the `kepler_root_distill` recipe.

Gold standard: Newton-Raphson solve of

    M = E - e sin(E)

where M and E are angles (radians) and 0 <= e < 1.

The symmetry of interest is the 2π periodicity of M.

Optional: use real orbital parameters from a TLE file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np


def parse_tle_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse eccentricity and mean anomaly from a TLE file.

    Supports files where each satellite has either:
      - (name line) + line1 + line2
      - line1 + line2
    """

    lines = [
        ln.rstrip("\n")
        for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if ln.strip()
    ]
    out_e: List[float] = []
    out_M: List[float] = []
    i = 0
    while i < len(lines) - 1:
        # Find a line1 starting with '1 '
        if lines[i].startswith("1 "):
            l2 = lines[i + 1]
            i += 2
        else:
            # assume name line
            if i + 2 >= len(lines):
                break
            if not lines[i + 1].startswith("1 "):
                i += 1
                continue
            l2 = lines[i + 2]
            i += 3

        if not l2.startswith("2 "):
            continue

        # TLE format columns (1-indexed):
        #   eccentricity: line2 col 27-33 (7 digits, implied decimal)
        #   mean anomaly: line2 col 44-51 (degrees)
        try:
            ecc_digits = l2[26:33].strip()
            if len(ecc_digits) != 7 or not ecc_digits.isdigit():
                continue
            e = float("0." + ecc_digits)
            M_deg = float(l2[43:51].strip())
            M = float(np.deg2rad(M_deg))
        except Exception:
            continue

        if not (0.0 <= e < 1.0):
            continue

        out_e.append(e)
        # keep M in [0, 2π)
        out_M.append(float(M % (2.0 * np.pi)))

    if not out_e:
        raise RuntimeError(f"No valid TLE records parsed from {path}")
    return np.array(out_e, dtype=np.float32), np.array(out_M, dtype=np.float32)


def kepler_newton(
    M: np.ndarray, e: np.ndarray, *, max_iter: int = 50, tol: float = 1e-12
) -> np.ndarray:
    """Solve Kepler's equation for E given M and e (vectorized)."""
    M = np.asarray(M, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    # Initial guess
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        d = f / fp
        E = E - d
        if np.max(np.abs(d)) < tol:
            break
    # principal value [0, 2π)
    return (E % (2.0 * np.pi)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--tle_file", type=str, default=None, help="Optional: path to a TLE file"
    )
    ap.add_argument("--min_e", type=float, default=0.0)
    ap.add_argument("--max_e", type=float, default=0.95)

    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    n_total = int(args.n_train) + int(args.n_test)

    if args.tle_file:
        e_all, M_all = parse_tle_file(Path(args.tle_file))
        # sample (with replacement if needed)
        idx = rng.integers(0, len(e_all), size=n_total)
        e = e_all[idx]
        M = M_all[idx]
        source = {
            "type": "tle",
            "path": str(args.tle_file),
            "n_unique": int(len(e_all)),
        }
    else:
        e = rng.uniform(float(args.min_e), float(args.max_e), size=n_total).astype(
            np.float32
        )
        M = rng.uniform(0.0, 2.0 * np.pi, size=n_total).astype(np.float32)
        source = {
            "type": "synthetic",
            "min_e": float(args.min_e),
            "max_e": float(args.max_e),
        }

    E = kepler_newton(M, e)
    y = np.stack([np.sin(E), np.cos(E)], axis=1).astype(np.float32)

    # Shuffle then split
    perm = rng.permutation(n_total)
    e, M, E, y = e[perm], M[perm], E[perm], y[perm]

    n_train = int(args.n_train)
    train_e, test_e = e[:n_train], e[n_train:]
    train_M, test_M = M[:n_train], M[n_train:]
    train_E, test_E = E[:n_train], E[n_train:]
    train_y, test_y = y[:n_train], y[n_train:]

    np.savez(
        out,
        train_e=train_e,
        train_M=train_M,
        train_E=train_E,
        train_y=train_y,
        test_e=test_e,
        test_M=test_M,
        test_E=test_E,
        test_y=test_y,
        source=str(source),
    )
    print(
        f"Wrote {out} with {len(train_e)} train samples and {len(test_e)} test samples. "
        f"source={source['type']}"
    )


if __name__ == "__main__":
    main()
