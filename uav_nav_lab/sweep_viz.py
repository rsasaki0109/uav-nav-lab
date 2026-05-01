"""Sweep-level visualization.

Reads `sweep_manifest.json` + each run's `summary.json` and renders a single
multi-panel figure showing how each metric varies with the swept parameter(s).

  - 1 swept param  → line plot (one subplot per metric)
  - 2 swept params → heatmap   (one subplot per metric)
  - 3+ params      → not supported in MVP; raise with a clear message
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .config import get_dotted
from .eval.metrics import evaluate_run

def _safe_dt(s: dict, key: str) -> float:
    """Read planner-dt block from a summary that may pre-date the column."""
    block = s.get(key, {})
    return float(block.get("mean", 0.0)) if isinstance(block, dict) else 0.0


_METRICS_DEFAULT = [
    ("success_rate", "success rate", "%", lambda s: s["success_rate"] * 100),
    ("collision_rate", "collision rate", "%", lambda s: s["collision_rate"] * 100),
    ("avg_speed_mean", "avg speed (m/s)", "", lambda s: s["avg_speed"]["mean"]),
    ("ate_rms_mean", "ATE rms (m)", "", lambda s: s["ate_rms"]["mean"]),
    ("planner_dt_ms_mean", "planner dt mean (ms)", "",
     lambda s: _safe_dt(s, "planner_dt_ms_mean")),
    ("planner_dt_ms_p95", "planner dt p95 (ms)", "",
     lambda s: _safe_dt(s, "planner_dt_ms_p95")),
]

# Multi-drone runs report a separate `joint_*` block in summary.json; we
# swap the per-drone-episode rates for the joint ones (the metric an
# operator would actually report) and keep the continuous metrics intact.
_METRICS_MULTI = [
    ("joint_success_rate", "joint success rate", "%",
     lambda s: s["joint_success_rate"] * 100),
    ("joint_collision_rate", "joint collision rate", "%",
     lambda s: s["joint_collision_rate"] * 100),
    ("avg_speed_mean", "avg speed (m/s)", "", lambda s: s["avg_speed"]["mean"]),
    ("ate_rms_mean", "ATE rms (m)", "", lambda s: s["ate_rms"]["mean"]),
    ("planner_dt_ms_mean", "planner dt mean (ms)", "",
     lambda s: _safe_dt(s, "planner_dt_ms_mean")),
    ("planner_dt_ms_p95", "planner dt p95 (ms)", "",
     lambda s: _safe_dt(s, "planner_dt_ms_p95")),
]


def _need_mpl() -> Any:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for sweep viz. Install with: "
            "pip install -e '.[viz]'"
        ) from e


def _load_sweep(sweep_root: Path) -> tuple[list[str], list[dict[str, Any]]]:
    manifest_path = sweep_root / "sweep_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found — not a sweep dir")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    keys = [k for k, _ in manifest.get("overrides", [])]
    runs = []
    for entry in manifest["runs"]:
        run_dir = Path(entry["dir"])
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        else:
            summary = evaluate_run(run_dir)  # writes summary.json as a side-effect
        with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        params = {k: get_dotted(cfg, k) for k in keys}
        runs.append({"params": params, "summary": summary, "name": entry["name"]})
    return keys, runs


def _sorted_axis(values: set[Any]) -> list[Any]:
    """Sort numerically when every value can be coerced to float; otherwise
    sort by string representation. Avoids the '12' < '3' lex-sort trap."""
    try:
        return sorted(values, key=float)
    except (TypeError, ValueError):
        pass
    return sorted(values, key=str)


def _line_plot(plt, ax, runs, key: str, metric_label: str, value_fn) -> None:
    raw = [(r["params"][key], value_fn(r["summary"])) for r in runs]
    # Sort numerically when every x is numeric, else fall back to a bar plot
    # over the unique categorical values (preserving first-seen order).
    try:
        numeric = [(float(x), y) for x, y in raw]
        numeric.sort(key=lambda p: p[0])
        xs_num, ys = zip(*numeric)
        ax.plot(xs_num, ys, "o-")
        ax.set_xlabel(key)
    except (TypeError, ValueError):
        seen: list[Any] = []
        ys_by_x: dict[Any, float] = {}
        for x, y in raw:
            if x not in ys_by_x:
                seen.append(x)
            ys_by_x[x] = y
        ax.bar(range(len(seen)), [ys_by_x[x] for x in seen])
        ax.set_xticks(range(len(seen)))
        ax.set_xticklabels([str(x) for x in seen], rotation=30, ha="right")
        ax.set_xlabel(key)
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.3)


def _heatmap(plt, ax, runs, kx: str, ky: str, metric_label: str, value_fn) -> None:
    import numpy as np

    def _hashable(v: Any) -> Any:
        return tuple(v) if isinstance(v, list) else v

    xs_set = _sorted_axis({_hashable(r["params"][kx]) for r in runs})
    ys_set = _sorted_axis({_hashable(r["params"][ky]) for r in runs})
    grid = np.full((len(ys_set), len(xs_set)), np.nan, dtype=float)
    for r in runs:
        i = ys_set.index(_hashable(r["params"][ky]))
        j = xs_set.index(_hashable(r["params"][kx]))
        grid[i, j] = value_fn(r["summary"])
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xs_set)))
    ax.set_xticklabels([str(x) for x in xs_set], rotation=30, ha="right")
    ax.set_yticks(range(len(ys_set)))
    ax.set_yticklabels([str(y) for y in ys_set])
    ax.set_xlabel(kx)
    ax.set_ylabel(ky)
    ax.set_title(metric_label)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # annotate cells
    for i in range(len(ys_set)):
        for j in range(len(xs_set)):
            v = grid[i, j]
            if v != v:  # NaN
                continue
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if v < (im.get_clim()[0] + im.get_clim()[1]) / 2 else "black",
                    fontsize=8)


def sweep_viz(sweep_root: Path) -> Path:
    plt = _need_mpl()
    sweep_root = Path(sweep_root)
    keys, runs = _load_sweep(sweep_root)
    if len(keys) == 0:
        raise ValueError(f"sweep at {sweep_root} has no overrides recorded")
    # Drop axes that turn out to take a single value across the manifest —
    # users routinely pass `--param x=fixed_value` alongside a real sweep,
    # and a 1-value axis is not actually swept.
    def _hashable(v: Any) -> Any:
        return tuple(v) if isinstance(v, list) else v

    keys = [k for k in keys if len({_hashable(r["params"][k]) for r in runs}) > 1]
    if len(keys) == 0:
        raise ValueError(
            f"sweep at {sweep_root} has overrides but no axis varies — "
            "all `--param` values were fixed."
        )
    if len(keys) > 2:
        raise NotImplementedError(
            f"sweep viz only supports 1 or 2 swept params (got {len(keys)}: {keys}). "
            "Filter or marginalize the manifest before viz-ing."
        )

    # Multi-drone runs surface joint-success / joint-collision in addition
    # to the per-drone-episode rates; pick the right metric set off the
    # first run's summary so the sweep figure shows the right view.
    metrics = _METRICS_MULTI if "joint_success_rate" in runs[0]["summary"] else _METRICS_DEFAULT
    fig, axes = plt.subplots(3, 2, figsize=(11, 13))
    for ax, (_, label, _, fn) in zip(axes.flat, metrics):
        if len(keys) == 1:
            _line_plot(plt, ax, runs, keys[0], label, fn)
            ax.set_title(label)
        else:
            _heatmap(plt, ax, runs, keys[0], keys[1], label, fn)
    fig.suptitle(f"sweep: {sweep_root.name}  ({len(runs)} runs)")
    fig.tight_layout()
    out = sweep_root / "sweep_summary.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
