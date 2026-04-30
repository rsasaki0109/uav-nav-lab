"""`uav-nav` command-line interface.

Subcommands:
  run     <exp.yaml>                      run all episodes for a config
  eval    <run_dir>                       compute metrics from existing logs
  compare <run_dir> <run_dir> [...]       tabulate multiple runs
  sweep   <exp.yaml> --param k=spec ...   Cartesian-product sweep
  viz     <run_dir>                       render trajectory PNGs
  list                                    show registered backends
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ExperimentConfig
from .eval import compare_runs, evaluate_run
from .eval.compare import format_comparison_text
from .eval.metrics import format_summary_text
from .planner import PLANNER_REGISTRY
from .runner import run_experiment, run_sweep
from .scenario import SCENARIO_REGISTRY
from .sensor import SENSOR_REGISTRY
from .sim import SIM_REGISTRY


def _default_output_dir(cfg: ExperimentConfig, given: str | None) -> Path:
    if given:
        return Path(given)
    if cfg.output and "dir" in cfg.output:
        return Path(cfg.output["dir"])
    return Path("results") / cfg.name


def cmd_run(args: argparse.Namespace) -> int:
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    out_dir = _default_output_dir(cfg, args.output_dir)
    run_experiment(cfg, out_dir)
    print(f"\n[run] done. results at: {out_dir}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    summary = evaluate_run(Path(args.run_dir))
    print(format_summary_text(summary))
    print(f"\n[eval] summary written: {Path(args.run_dir) / 'summary.json'}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    summaries = compare_runs([Path(d) for d in args.run_dirs])
    print(format_comparison_text(summaries))
    return 0


def _parse_param_arg(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--param expects key=spec, got {s!r}")
    k, v = s.split("=", 1)
    return k.strip(), v.strip()


def cmd_sweep(args: argparse.Namespace) -> int:
    base = ExperimentConfig.from_yaml(args.config)
    overrides = [_parse_param_arg(p) for p in args.param]
    out_root = Path(args.output_dir) if args.output_dir else Path("results") / f"{base.name}_sweep"
    run_sweep(base, overrides, out_root, parallel=int(args.parallel))
    # auto-eval each run and print a comparison table
    run_dirs = sorted(p for p in out_root.iterdir() if p.is_dir() and p.name.startswith("run_"))
    if run_dirs:
        summaries = compare_runs(run_dirs)
        print()
        print(format_comparison_text(summaries))
    print(f"\n[sweep] done. results at: {out_root}")
    return 0


def cmd_viz(args: argparse.Namespace) -> int:
    target = Path(args.run_dir)
    if (target / "sweep_manifest.json").exists():
        from .sweep_viz import sweep_viz

        out = sweep_viz(target)
        print(f"[viz] sweep summary: {out}")
        return 0
    from .viz import viz_run

    saved = viz_run(target, show=bool(args.show))
    for p in saved:
        print(f"  wrote {p}")
    print(f"[viz] {len(saved)} image(s) saved")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    print("planners:  ", ", ".join(PLANNER_REGISTRY.names()))
    print("sensors:   ", ", ".join(SENSOR_REGISTRY.names()))
    print("simulators:", ", ".join(SIM_REGISTRY.names()))
    print("scenarios: ", ", ".join(SCENARIO_REGISTRY.names()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="uav-nav", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run an experiment from YAML")
    pr.add_argument("config", help="path to experiment YAML")
    pr.add_argument("--output-dir", "-o", help="override output directory")
    pr.add_argument("--seed", type=int, help="override base seed")
    pr.set_defaults(func=cmd_run)

    pe = sub.add_parser("eval", help="evaluate an existing run directory")
    pe.add_argument("run_dir")
    pe.set_defaults(func=cmd_eval)

    pc = sub.add_parser("compare", help="tabulate metrics across multiple runs")
    pc.add_argument("run_dirs", nargs="+")
    pc.set_defaults(func=cmd_compare)

    ps = sub.add_parser("sweep", help="parameter sweep")
    ps.add_argument("config", help="path to base experiment YAML")
    ps.add_argument(
        "--param",
        action="append",
        default=[],
        help="dotted-key override, e.g. planner.max_speed=5:30:5 (repeatable)",
    )
    ps.add_argument("--output-dir", "-o", help="override sweep root directory")
    ps.add_argument("--parallel", "-j", type=int, default=1, help="parallel worker count")
    ps.set_defaults(func=cmd_sweep)

    pv = sub.add_parser("viz", help="render trajectory PNGs for a run dir")
    pv.add_argument("run_dir")
    pv.add_argument("--show", action="store_true", help="also open an interactive window")
    pv.set_defaults(func=cmd_viz)

    pl = sub.add_parser("list", help="list registered backends")
    pl.set_defaults(func=cmd_list)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
