# uav-nav-lab

OSS Python research framework for **high-speed UAV navigation**, built to
let you run controlled ablations in minutes and see whether a proposed
change actually wins. Pluggable simulator / scenario / planner / sensor /
predictor backends; YAML experiments; Cartesian-product parameter sweeps;
Wilson-score / SEM confidence intervals on every metric; multi-drone
scenarios with joint-success aggregation; built-in heatmap and trajectory
visualisation.

The framework deliberately ships small primitives plus a CLI — research
runs through `uav-nav run|sweep|eval|compare|viz|anim`, and every
example YAML carries the validated sweep result it was designed to
produce so a reader sees the conclusion before the config.

## Why

Most UAV planning research either (a) hard-codes a single MPC variant,
single sensor, single scenario, and reports a number, or (b) buries
ablations under stacks of glue code. Neither makes it easy to ask "does
this idea actually beat what I already have, with the CI to back it?".

`uav-nav-lab` is the framework I wanted while doing the research itself:
declare the experiment in YAML, sweep with `--param`, get heatmaps and
Wilson 95% CIs out of the box, and have every config carry its own
research finding so the file is the artifact, not a Notion page.

## Quick start

```bash
git clone https://github.com/rsasaki0109/uav-nav-lab
cd uav-nav-lab
pip install -e '.[dev,viz]'        # numpy + pyyaml + matplotlib + pytest
pytest -q                          # 42 tests, runs in seconds

# basic A* run, 5 episodes, prints success/collision/timeout with
# Wilson 95% CIs and per-call planner compute cost
uav-nav run examples/exp_basic.yaml
uav-nav eval results/basic_astar
uav-nav viz  results/basic_astar
```

A 2D heatmap sweep is one CLI invocation:

```bash
uav-nav sweep examples/exp_predictive.yaml \
  --param planner.horizon=20 --param planner.n_samples=16 \
  --param planner.max_speed=10,15,20,25,30 \
  --param planner.replan_period=0.1,0.2,0.5,1.0,2.0 \
  --param num_episodes=20 -j 4
uav-nav viz <out>     # → 6-panel sweep_summary.png
```

## CLI

| command | what |
|---|---|
| `uav-nav run <yaml>` | run all episodes, write per-episode JSONs + `summary.json` |
| `uav-nav eval <run_dir>` | recompute metrics from logs, print Wilson 95% rates + planner-dt budget |
| `uav-nav compare <a> <b> ...` | side-by-side table with ± half-widths |
| `uav-nav sweep <yaml> --param k=spec` | Cartesian-product over `--param`s; each cell gets its own dir |
| `uav-nav viz <run_or_sweep>` | trajectory PNG per episode, or 1D/2D sweep heatmap |
| `uav-nav anim <run_dir>` | animated GIF replay (2D) |
| `uav-nav list` | enumerate registered planners / sensors / sims / scenarios |

`--param` syntax: `start:stop:step` for ranges, `a,b,c` for explicit lists,
`[3,0]` for vector values, `true`/`false` literals. Three-level dotted
keys work: `planner.predictor.velocity_noise_std=0.0,0.5,1.0`.

## Architecture

```
uav_nav_lab/
├── sim/         dummy_2d / dummy_3d (point-mass), airsim, ros2 (stubs)
├── scenario/    grid_world, voxel_world, multi_drone_grid
├── planner/     astar, straight, mpc           (registry: PLANNER_REGISTRY)
├── sensor/      perfect, delayed, kalman_delayed, lidar
├── predictor/   constant_velocity, noisy_velocity, kalman_velocity
├── runner/      experiment, multi (multi-drone), sweep
├── eval/        metrics (Wilson + SEM CIs), compare
├── viz / anim / sweep_viz   2D + 3D + GIF + 6-panel heatmap
└── cli          run / eval / compare / sweep / viz / anim / list
```

Adding a new backend is one new file with a `@REGISTRY.register("name")`
decorator and a `from_config(cfg)` classmethod — the CLI picks it up via
`type: name` in YAML, no central wiring needed.

## Selected research findings

Each finding lives in the comment header of the YAML that produces it,
along with a one-line `uav-nav sweep` invocation that reproduces it.
Wilson 95% intervals on rates, mean ± 1.96·SEM on continuous metrics.

### Planner head-to-head on dynamic obstacles

`examples/exp_compare_{straight,astar,mpc}.yaml` — same 50×50 world,
same three bouncing obstacles, same perfect sensor, only the planner
changes (n=30 each):

| planner | succ % | coll % | plan_ms mean / p95 |
|---|---|---|---|
| straight     |   0.0 ± 5.7  | 100.0 ± 5.7 |  0.04 /  0.05 |
| astar        |  20.0 ± 13.9 |  80.0 ± 13.9 |  4.75 /  8.97 |
| mpc (Pareto) | **100.0** ± 5.7 |   0.0 ± 5.7 | 52.16 / 56.96 |

A* sees only a snapshot at replan time and walks into where the bouncing
obstacles will be 0.2 s later — 20 % success. MPC at the Pareto config
(n_samples=16, horizon=20) routes around future obstacle positions and
clears every episode. The +80 pp gap is the experimentally-measured
value of having a motion model in this scenario, paid for at ~11× the
per-replan cost of A*.

### MPC compute Pareto

`examples/exp_predictive.yaml` — n_samples × horizon, n=20:

| n_samples \ horizon | 20 | 40 | 60 | 80 | 120 |
|---|---|---|---|---|---|
| 8   | 100 | 90  | 80 | 65 | 45 |
| 16  | **100** | 85  | 80 | 65 | 35 |
| 32  | 100 | 95  | 75 | 60 | 35 |
| 64  | 100 | 100 | 75 | 60 | 45 |
| 128 | 100 | 100 | 95 | 80 | 40 |

Sole Pareto-optimal point: **n_samples=16, horizon=20 → 100% / 51 ms**.
Longer rollouts actively hurt success (the reach-goal bonus fires less
often when the rollout overshoots the goal radius mid-trajectory).

### Pareto config materially rewrites prior conclusions

The previous heatmap on the same scenario at the YAML's old defaults
(n_samples=32, horizon=60) reported a "dynamic-feasibility cliff at
25 m/s". At the Pareto config that cliff disappears (35–65% success at
speed=25-30 m/s), and replan_period — which "barely mattered" before —
now drives a 40–70 pp swing across 0.1–2.0 s. The earlier conclusion was
partly a CPU-saturation artifact: at horizon=60 every replan took ~200 ms,
so even replan_period=0.1 s could not actually keep up.

**Methodological lesson** baked into the YAML header: always validate
ablation conclusions at the planner's Pareto-optimal config — suboptimal
MPC settings can mask both ceilings (max feasible speed) and
sensitivities (replan-period effect, delay tolerance).

### Multi-drone N-scaling and peer-prediction coordination

`examples/exp_multi_drone_{2,3,4}.yaml` — same world, only drone count
changes. n=30, joint metrics with Wilson 95% CIs:

| N | joint succ | joint coll | per-drone succ |
|---|---|---|---|
| 2 | 96.7% [83, 99] | 3.3%  | 98.3% |
| 3 | 70.0% [52, 83] | 30.0% | 87.8% |
| 4 | 73.3% [56, 86] | 26.7% | 87.5% |

Independence-model expectation `joint = per_drone^N`:
- N=4: actual 73.3 % > expected 58.6 %  (Δ = **+14.7 pp**)

The MPC's constant-velocity peer prediction *correlates failures in the
right direction* — when one drone yields, the others see its new
trajectory and react, so the system as a whole degrades less than
independent drones would.

### The perception-latency cliff: a four-step research saga

A single persistent `delay=0.5 s` × `speed=15 m/s` cell on the
predictive-MPC scenario (≤ 25 % success regardless of `inflate` /
`safety_margin` tuning at the Pareto config) drove four progressive
experiments — each documented in `examples/exp_predictive.yaml`:

1. **Predictor-side delay compensation** (negative result). Adding
   `delay_compensation=0.5` to the Kalman *predictor* on the obstacle
   stream actually *hurt* success. The MPC plans against future
   obstacles using a past self.
2. **Sensor-side ego extrapolation** (`sensor.extrapolate=true`). Project
   the stale position forward by `delay` using a 1-sample finite-difference
   velocity. Lifts success in moderate-delay × moderate-speed cells by
   +15 .. +35 pp; *hurts* at delay=0 (1-step lag artifact) and at
   high-speed × high-delay (acceleration noise overshoots).
3. **Velocity-window smoothing** (`sensor.velocity_window=5`). Average
   the FD velocity over multiple sample pairs to suppress acceleration
   noise. The persistent cliff lifts from 26.7 % → 43.3 % (+17 pp) at
   the headline cell, +35-40 pp at high speed. Catch: optimum window
   depends on the speed regime — low speed prefers `=1` (no lag), high
   speed needs `=5`.
4. **Kalman ego sensor** (`sensor.type=kalman_delayed`). Honest negative
   result. Best tuning of process / measurement noise tops out at the
   no-extrap baseline (25 %); the moving-average wins. The KF assumes
   a CV motion model; under MPC's frequent re-planning the assumption
   breaks and the MA's structure-free responsiveness dominates.

Engineering takeaway: simple model-free estimators can dominate more
sophisticated ones when the motion-model assumption breaks. Picking the
estimator that *actually wins* is more useful than picking the one that
sounds fanciest — the framework is built to make that picking trivial.

## Status

* v0.1.0 released, 42 tests, GitHub Actions CI on Python 3.10 / 3.11 / 3.12
  + a CLI smoke job.
* 5 sensor backends (`perfect`, `delayed`, `kalman_delayed`, `lidar`),
  3 predictor backends (`constant_velocity`, `noisy_velocity`,
  `kalman_velocity`), 3 planners (`astar`, `straight`, `mpc`),
  3 scenarios (`grid_world`, `voxel_world`, `multi_drone_grid`).
* All ablation results are reproducible from the example YAMLs by
  copy-pasting one `uav-nav sweep ...` line.

External backends (AirSim / PX4 / ROS 2) ship as lazy-import stubs —
swap them in by registering a real `SimInterface` subclass without
changing the rest of the stack.

## License

Apache-2.0.
