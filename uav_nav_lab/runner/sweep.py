"""Parameter sweep — Cartesian product over `--param key=spec` overrides.

Spec syntax:
  - `start:stop`            → integer range [start, stop)
  - `start:stop:step`       → numeric range with step (inclusive of start, exclusive of stop)
  - `a,b,c`                 → explicit list (strings or numbers, parsed when possible)
  - `value`                 → single value (treated as 1-element list)

Example:
  --param planner.max_speed=5:30:5 --param planner.type=astar,straight
"""

from __future__ import annotations

import copy
import itertools
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import yaml

from ..config import ExperimentConfig, set_dotted
from .experiment import run_experiment


def _split_top_level(spec: str, sep: str = ",") -> list[str]:
    """Split on `sep` while respecting `[...]` brackets — so vector values
    like `[3,0]` survive a comma-separated list."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for c in spec:
        if c == "[":
            depth += 1
            cur.append(c)
        elif c == "]":
            depth -= 1
            cur.append(c)
        elif c == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    out.append("".join(cur))
    return out


def _parse_value(s: str) -> Any:
    s = s.strip()
    # vector / list literal: [a, b, c]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(p) for p in _split_top_level(inner)]
    # bool literals (so `--param planner.use_prediction=true,false` works)
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    # try int, then float, else keep as string
    try:
        i = int(s)
        if str(i) == s:
            return i
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_spec(spec: str) -> list[Any]:
    spec = spec.strip()
    if ":" in spec and "," not in spec:
        parts = spec.split(":")
        if len(parts) == 2:
            start = _parse_value(parts[0])
            stop = _parse_value(parts[1])
            step: Any = 1
        elif len(parts) == 3:
            start = _parse_value(parts[0])
            stop = _parse_value(parts[1])
            step = _parse_value(parts[2])
        else:
            raise ValueError(f"bad range spec: {spec!r}")
        if any(isinstance(v, float) for v in (start, stop, step)):
            vals: list[Any] = []
            v = float(start)
            while v < float(stop) - 1e-9:
                vals.append(round(v, 9))
                v += float(step)
            return vals
        return list(range(int(start), int(stop), int(step)))
    if "," in spec or "[" in spec:
        return [_parse_value(p) for p in _split_top_level(spec)]
    return [_parse_value(spec)]


def expand_sweep(
    base_cfg: ExperimentConfig, overrides: list[tuple[str, str]]
) -> list[ExperimentConfig]:
    keys = [k for k, _ in overrides]
    value_lists = [_parse_spec(v) for _, v in overrides]
    out: list[ExperimentConfig] = []
    for combo in itertools.product(*value_lists):
        raw = copy.deepcopy(base_cfg.raw or base_cfg.to_dict())
        run_name = base_cfg.name
        for k, v in zip(keys, combo):
            set_dotted(raw, k, v)
            run_name += f"__{k.replace('.', '_')}={v}"
        raw["name"] = run_name
        out.append(ExperimentConfig.from_dict(raw))
    return out


def _run_one(args: tuple[dict, str]) -> dict:
    """Worker entry for multiprocessing. Top-level so it pickles cleanly."""
    cfg_dict, run_dir = args
    cfg = ExperimentConfig.from_dict(cfg_dict)
    run_experiment(cfg, Path(run_dir))
    return {"name": cfg.name, "dir": run_dir}


def run_sweep(
    base_cfg: ExperimentConfig,
    overrides: list[tuple[str, str]],
    output_root: Path,
    parallel: int = 1,
) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cfgs = expand_sweep(base_cfg, overrides)
    print(f"[sweep] expanded to {len(cfgs)} runs (parallel={parallel})")
    work: list[tuple[dict, str]] = []
    manifest = []
    for i, cfg in enumerate(cfgs):
        run_dir = output_root / f"run_{i:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
        work.append((cfg.to_dict(), str(run_dir)))
        manifest.append({"index": i, "name": cfg.name, "dir": str(run_dir)})

    if parallel <= 1:
        for w in work:
            _run_one(w)
    else:
        # `spawn` keeps things deterministic across platforms and avoids
        # inheriting numpy / matplotlib / module-level state from the parent.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=parallel) as pool:
            for _ in pool.imap_unordered(_run_one, work):
                pass

    with (output_root / "sweep_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"runs": manifest, "overrides": overrides}, f, indent=2)
    return output_root
