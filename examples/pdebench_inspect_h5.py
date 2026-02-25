"""Inspect an HDF5 file (e.g., PDEBench) to help configure the adapter.

Usage:
  python examples/pdebench_inspect_h5.py --path path/to/file.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True)
    args = ap.parse_args()

    try:
        import h5py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("This script requires h5py") from e

    p = Path(args.path)
    if not p.exists():
        raise FileNotFoundError(p)

    def visit(name, obj):
        if hasattr(obj, "shape"):
            print(f"{name}: shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"{name}/")

    with h5py.File(p, "r") as f:
        print(f"HDF5 file: {p}")
        f.visititems(visit)


if __name__ == "__main__":
    main()
