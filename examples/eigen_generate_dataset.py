"""Generate an eigenvalue benchmark dataset for Mezzanine.

This writes a `.npz` consumed by `eigen_permutation_distill`.

Gold standard: dense symmetric eigensolver (numpy.linalg.eigvalsh) on small
submatrices.

If you provide a Matrix Market `.mtx` from SuiteSparse, we extract random
submatrices to obtain *real* coefficient patterns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    sub = A[idx][:, idx]
    try:
        sub = sub.toarray()  # type: ignore[attr-defined]
    except Exception:
        sub = np.asarray(sub)
    return np.asarray(sub, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--mtx", type=str, default=None, help="Optional: Matrix Market .mtx file"
    )
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument(
        "--density",
        type=float,
        default=None,
        help="Synthetic-only: density of the random symmetric sparsity pattern (0<density<=1).",
    )
    ap.add_argument(
        "--k", type=int, default=5, help="How many smallest eigenvalues to store"
    )
    ap.add_argument("--diag_shift", type=float, default=1e-3)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    n_total = int(args.n_train) + int(args.n_test)
    n = int(args.n)
    k = int(args.k)

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
            raise ValueError(
                f"Matrix is too small (n_big={n_big}) for subproblem n={n}"
            )
        source = {"type": "mtx", "path": str(args.mtx), "n_big": n_big}
    else:
        A_big = None
        source = {"type": "synthetic"}
        if args.density is not None:
            source["density"] = float(args.density)

    As = np.zeros((n_total, n, n), dtype=np.float32)
    evals = np.zeros((n_total, k), dtype=np.float32)

    for i in range(n_total):
        if A_big is None:
            M = rng.normal(size=(n, n))
            if args.density is not None:
                mask_ut = rng.random(size=(n, n)) < float(args.density)
                mask_ut = np.triu(mask_ut, k=1)
                mask = mask_ut | mask_ut.T
                np.fill_diagonal(mask, True)
                M = M * mask
            A = 0.5 * (M + M.T)
            A = A + float(args.diag_shift) * np.eye(n)
        else:
            idx = rng.choice(int(A_big.shape[0]), size=n, replace=False)
            A = extract_dense_submatrix(A_big, idx)
            A = 0.5 * (A + A.T)
            A = A + float(args.diag_shift) * np.eye(n)

        w = np.linalg.eigvalsh(A)
        w = np.sort(w)[:k]
        As[i] = A.astype(np.float32)
        evals[i] = w.astype(np.float32)

    perm = rng.permutation(n_total)
    As, evals = As[perm], evals[perm]
    n_train = int(args.n_train)

    np.savez(
        out,
        train_A=As[:n_train],
        train_eval=evals[:n_train],
        test_A=As[n_train:],
        test_eval=evals[n_train:],
        source=str(source),
    )

    print(
        f"Wrote {out} with {n_train} train and {n_total - n_train} test matrices. "
        f"n={n} k={k} source={source['type']}"
    )


if __name__ == "__main__":
    main()
