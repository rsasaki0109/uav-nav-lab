"""Side-by-side comparison of multiple run directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .metrics import evaluate_run


def compare_runs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    return [evaluate_run(Path(d)) for d in run_dirs]


def _hw(ci: list[float]) -> float:
    """95% CI half-width (the ± term)."""
    return (ci[1] - ci[0]) / 2.0


def format_comparison_text(summaries: list[dict[str, Any]]) -> str:
    headers = ("name", "succ%", "±", "coll%", "±", "avg_v", "ATE")
    widths = (28, 6, 5, 6, 5, 9, 11)
    lines = []
    head = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines.append(head)
    lines.append("-" * len(head))
    for s in summaries:
        name = Path(s["run_dir"]).name
        succ_hw = _hw(s["success_ci95"]) * 100
        coll_hw = _hw(s["collision_ci95"]) * 100
        avg_hw = (s["avg_speed"]["ci_hi"] - s["avg_speed"]["ci_lo"]) / 2.0
        ate_hw = (s["ate_rms"]["ci_hi"] - s["ate_rms"]["ci_lo"]) / 2.0
        row = (
            name[: widths[0]].ljust(widths[0]),
            f"{s['success_rate'] * 100:5.1f}".rjust(widths[1]),
            f"{succ_hw:4.1f}".rjust(widths[2]),
            f"{s['collision_rate'] * 100:5.1f}".rjust(widths[3]),
            f"{coll_hw:4.1f}".rjust(widths[4]),
            f"{s['avg_speed']['mean']:.2f}±{avg_hw:.2f}".rjust(widths[5]),
            f"{s['ate_rms']['mean']:.3f}±{ate_hw:.3f}".rjust(widths[6]),
        )
        lines.append("  ".join(row))
    return "\n".join(lines)
