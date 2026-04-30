"""Experiment config loading.

Top-level YAML schema (validated loosely; sub-blocks are passed through to
the plugin's `from_config` so backends own their own schema):

    name: <str>
    seed: <int>
    num_episodes: <int>

    scenario:   { type: <name>, ... }
    simulator:  { type: <name>, dt: <float>, max_steps: <int>, ... }
    planner:    { type: <name>, replan_period: <float>, max_speed: <float>, ... }
    sensor:     { type: <name>, ... }

    output:     { dir: <path> }
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class ExperimentConfig:
    name: str
    seed: int = 0
    num_episodes: int = 1
    scenario: dict = field(default_factory=dict)
    simulator: dict = field(default_factory=dict)
    planner: dict = field(default_factory=dict)
    sensor: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExperimentConfig":
        d = dict(d)
        required = ("name", "scenario", "simulator", "planner")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"experiment config missing keys: {missing}")
        return cls(
            name=str(d["name"]),
            seed=int(d.get("seed", 0)),
            num_episodes=int(d.get("num_episodes", 1)),
            scenario=dict(d["scenario"]),
            simulator=dict(d["simulator"]),
            planner=dict(d["planner"]),
            sensor=dict(d.get("sensor", {"type": "perfect"})),
            output=dict(d.get("output", {})),
            raw=copy.deepcopy(dict(d)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top-level must be a mapping")
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return copy.deepcopy(self.raw) if self.raw else {
            "name": self.name,
            "seed": self.seed,
            "num_episodes": self.num_episodes,
            "scenario": self.scenario,
            "simulator": self.simulator,
            "planner": self.planner,
            "sensor": self.sensor,
            "output": self.output,
        }


def set_dotted(d: dict, dotted_key: str, value: Any) -> dict:
    """Set `d['a']['b']['c'] = value` from `'a.b.c'`. Returns d for chaining."""
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    return d


def get_dotted(d: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read `d['a']['b']['c']` from `'a.b.c'`. Returns `default` if any key missing."""
    parts = dotted_key.split(".")
    cur: Any = d
    for p in parts:
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur
