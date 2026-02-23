from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="run_crystal_spacegroup_suite",
        description="Run crystal_spacegroup_distill for a list of teacher families.",
    )
    p.add_argument("--out_root", type=str, required=True, help="Root output directory.")
    p.add_argument("--dataset_name", type=str, default="matbench_mp_e_form")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument(
        "--teacher_families",
        type=str,
        default="mlp,matgl_megnet,matgl_m3gnet,matgl_chgnet",
        help="Comma-separated list of teacher families.",
    )
    p.add_argument(
        "--teacher_batch_size",
        type=int,
        default=32,
        help="Teacher batch size (used for matgl_* teachers).",
    )

    args, extra = p.parse_known_args(argv)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    families = [s.strip() for s in str(args.teacher_families).split(",") if s.strip()]
    if not families:
        raise SystemExit("No teacher families specified.")

    for fam in families:
        out_dir = out_root / f"{args.dataset_name}_fold{int(args.fold)}_{fam}"
        cmd = [
            "mezzanine",
            "run",
            "crystal_spacegroup_distill",
            "--out",
            str(out_dir),
            "--dataset_name",
            str(args.dataset_name),
            "--fold",
            str(int(args.fold)),
            "--teacher_family",
            str(fam),
        ]
        if str(fam).startswith("matgl_") and int(args.teacher_batch_size) > 0:
            cmd += ["--teacher_batch_size", str(int(args.teacher_batch_size))]
        cmd += list(extra)

        print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
