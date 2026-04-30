"""Side-by-side comparison of multiple run directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .metrics import evaluate_run


def compare_runs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    return [evaluate_run(Path(d)) for d in run_dirs]


def format_comparison_text(summaries: list[dict[str, Any]]) -> str:
    headers = ("name", "succ%", "coll%", "to%", "avg_v", "path", "replans", "ATE")
    widths = (28, 7, 7, 6, 8, 8, 8, 8)
    lines = []
    head = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines.append(head)
    lines.append("-" * len(head))
    for s in summaries:
        name = Path(s["run_dir"]).name
        row = (
            name[: widths[0]].ljust(widths[0]),
            f"{s['success_rate'] * 100:5.1f}".rjust(widths[1]),
            f"{s['collision_rate'] * 100:5.1f}".rjust(widths[2]),
            f"{s['timeout_rate'] * 100:5.1f}".rjust(widths[3]),
            f"{s['avg_speed']['mean']:.2f}".rjust(widths[4]),
            f"{s['path_length']['mean']:.1f}".rjust(widths[5]),
            f"{s['replans']['mean']:.1f}".rjust(widths[6]),
            f"{s['ate_rms']['mean']:.3f}".rjust(widths[7]),
        )
        lines.append("  ".join(row))
    return "\n".join(lines)
