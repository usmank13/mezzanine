"""Generate a small linear-system dataset for Mezzanine.

This writes a `.npz` consumed by `linear_system_permutation_distill`.

Gold standard: direct solve (numpy.linalg.solve) on small dense subproblems.

If you provide a Matrix Market `.mtx` from SuiteSparse, we extract random
submatrices to obtain *real* coefficient patterns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def load_mtx(path: Path):
    try:
        from scipy.io import mmread  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("This script requires scipy (scipy.io.mmread)") from e
    A = mmread(str(path))
    try:
        import scipy.sparse as sp  # type: ignore
        if sp.issparse(A):
            A = A.tocsr()
    except Exception:
        pass
    return A


def extract_dense_submatrix(A, idx: np.ndarray) -> np.ndarray:
    # works for scipy sparse or numpy
    sub = A[idx][:, idx]
    try:
        sub = sub.toarray()  # type: ignore[attr-defined]
    except Exception:
        sub = np.asarray(sub)
    return np.asarray(sub, dtype=np.float64)


def make_spd(M: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    # Ensure symmetric positive definite
    B = 0.5 * (M + M.T)
    C = B.T @ B
    C = C + lam * np.eye(C.shape[0])
    return C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--mtx", type=str, default=None, help="Optional: Matrix Market .mtx file")
    ap.add_argument("--n", type=int, default=32, help="Subproblem dimension")
    ap.add_argument(
        "--make_spd",
        "--spd",
        dest="make_spd",
        action="store_true",
        help="Convert extracted submatrices to SPD (alias: --spd)",
    )
    ap.add_argument("--diag_shift", type=float, default=1e-3, help="Diagonal regularization")
    ap.add_argument(
        "--density",
        type=float,
        default=None,
        help="Synthetic-only: density of the random factor matrix (0<density<=1).",
    )

    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    n_total = int(args.n_train) + int(args.n_test)
    n = int(args.n)

    if args.density is not None:
        density = float(args.density)
        if not (0.0 < density <= 1.0):
            raise ValueError(f"--density must be in (0, 1], got {density}")

    if args.mtx:
        A_big = load_mtx(Path(args.mtx))
        n_big = int(A_big.shape[0])
        if int(A_big.shape[0]) != int(A_big.shape[1]):
            m = min(int(A_big.shape[0]), int(A_big.shape[1]))
            A_big = A_big[:m, :m]
            n_big = m
        if n_big < n:
            raise ValueError(f"Matrix is too small (n_big={n_big}) for subproblem n={n}")
        source = {"type": "mtx", "path": str(args.mtx), "n_big": n_big}
    else:
        # Synthetic fallback: random SPD
        A_big = None
        source = {"type": "synthetic"}
        if args.density is not None:
            source["density"] = float(args.density)

    As = np.zeros((n_total, n, n), dtype=np.float32)
    bs = np.zeros((n_total, n), dtype=np.float32)
    xs = np.zeros((n_total, n), dtype=np.float32)

    for i in range(n_total):
        if A_big is None:
            M = rng.normal(size=(n, n))
            if args.density is not None:
                mask = rng.random(size=(n, n)) < float(args.density)
                M = M * mask
            A = M.T @ M + float(args.diag_shift) * np.eye(n)
        else:
            idx = rng.choice(int(A_big.shape[0]), size=n, replace=False)
            A = extract_dense_submatrix(A_big, idx)
            if bool(args.make_spd):
                A = make_spd(A, lam=float(args.diag_shift))
            else:
                A = 0.5 * (A + A.T)
                A = A + float(args.diag_shift) * np.eye(n)

        b = rng.normal(size=(n,))
        x = np.linalg.solve(A, b)

        As[i] = A.astype(np.float32)
        bs[i] = b.astype(np.float32)
        xs[i] = x.astype(np.float32)

    # Shuffle then split
    perm = rng.permutation(n_total)
    As, bs, xs = As[perm], bs[perm], xs[perm]
    n_train = int(args.n_train)

    np.savez(
        out,
        train_A=As[:n_train],
        train_b=bs[:n_train],
        train_x=xs[:n_train],
        test_A=As[n_train:],
        test_b=bs[n_train:],
        test_x=xs[n_train:],
        source=str(source),
    )

    print(
        f"Wrote {out} with {n_train} train and {n_total - n_train} test systems. "
        f"n={n} source={source['type']}"
    )


if __name__ == "__main__":
    main()
