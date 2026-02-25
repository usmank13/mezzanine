from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt


def plot_finance_bar_offset_distill(summary: Dict[str, Any], out_path: Path) -> None:
    """Simple diagnostic: accuracy + bar-offset warrant gap bars."""
    base_acc = float(summary["baseline"]["acc"])
    stud_acc = float(summary["student"]["acc"])
    base_gap = float(summary["baseline"]["gap_mean_tv_to_mean"])
    stud_gap = float(summary["student"]["gap_mean_tv_to_mean"])

    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.bar(["baseline", "student"], [base_acc, stud_acc])
    ax1.set_ylim(0, 1)
    ax1.set_title("Accuracy (canonical)")
    ax1.set_ylabel("acc")

    ax2.bar(["baseline", "student"], [base_gap, stud_gap])
    ax2.set_title("Warrant gap (bar offset)")
    ax2.set_ylabel("mean TV to mixture")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
