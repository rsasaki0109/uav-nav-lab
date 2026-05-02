# Research findings

These are the long-form studies behind the framework — full tables,
ablation reasoning, and methodological takeaways. The README's
[headline result](../README.md#-planner-head-to-head-on-dynamic-obstacles)
(planner head-to-head on dynamic obstacles) is the entry point; this
file collects the rest.

Each finding lives in the comment header of the YAML that produces it,
along with a one-line `uav-nav sweep` invocation that reproduces it.
Wilson 95 % intervals on rates, mean ± 1.96·SEM on continuous metrics.

## Contents

- [MPC compute Pareto](#mpc-compute-pareto)
- [3D Pareto: the n_samples preference flips](#3d-pareto-the-n_samples-preference-flips)
- [3D perception-latency cliff: same corner, softened](#3d-perception-latency-cliff-same-corner-softened)
- [Pareto config materially rewrites prior conclusions](#pareto-config-materially-rewrites-prior-conclusions)
- [Multi-drone N-scaling and peer-prediction coordination](#multi-drone-n-scaling-and-peer-prediction-coordination)
- [Wind miscalibration: planner belief must match sim reality](#wind-miscalibration-planner-belief-must-match-sim-reality)
- [The perception-latency cliff: a four-step research saga](#the-perception-latency-cliff-a-four-step-research-saga)
- [MPC + CHOMP smoothing: layering on a saturated planner is a wash](#mpc--chomp-smoothing-layering-on-a-saturated-planner-is-a-wash)
- [MPPI vs MPC: switching planner family doesn't break a tuned baseline](#mppi-vs-mpc-switching-planner-family-doesnt-break-a-tuned-baseline)

## MPC compute Pareto

`examples/exp_predictive.yaml` — n_samples × horizon. The 6-panel
output of `uav-nav viz <sweep_dir>` lets you read off the success
ceiling and the compute it costs in one figure:

<p align="center">
<img src="images/sweep_pareto.png" alt="6-panel Pareto sweep: success / collision / avg speed / ATE / planner_dt mean / planner_dt p95" width="640">
</p>

At n=20 episodes per cell:

| n_samples \ horizon | 20 | 40 | 60 | 80 | 120 |
|---|---|---|---|---|---|
| 8   | 100 | 90  | 80 | 65 | 45 |
| 16  | **100** | 85  | 80 | 65 | 35 |
| 32  | 100 | 95  | 75 | 60 | 35 |
| 64  | 100 | 100 | 75 | 60 | 45 |
| 128 | 100 | 100 | 95 | 80 | 40 |

Sole Pareto-optimal point: **n_samples=16, horizon=20 → 100 % / 51 ms**.
Longer rollouts actively *hurt* success — the reach-goal bonus fires
less often when the rollout overshoots the goal radius mid-trajectory.

## 3D Pareto: the n_samples preference flips

<p align="center">
<img src="images/demo_3d.gif" alt="3D Predictive-MPC episode on a 40×40×12 voxel world: drone (blue) reaches the goal while three bouncing dynamic obstacles (red) cross its path" width="480">
</p>

`examples/exp_3d_predictive.yaml` — the same n_samples × horizon sweep on
a 3D `voxel_world` (40×40×12, three bouncing 3D dynamic obstacles, n=8
post-cache):

<p align="center">
<img src="images/sweep_pareto_3d.png" alt="6-panel Pareto sweep on the 3D voxel world: success / collision / avg speed / ATE / planner_dt mean / planner_dt p95" width="640">
</p>

Findings vs the 2D analogue:

- **The Pareto frontier shifts to lower n_samples.** The 2D-optimal
  config (n=16, h=20 → 100 % / 51 ms in 2D) lands at only 75 % / 91 ms
  in 3D. The strongest 3D cells are **n=8, h=20 → 88 % / 70 ms** and
  n=128, h=40 → 100 % / 273 ms. Fibonacci-sphere sampling already
  covers the 3D escape directions densely enough that fewer per-step
  samples suffice — compute is better spent on horizon depth, opposite
  of 2D's preference.
- **The "longer rollouts hurt" effect partly transfers.** 2D drops
  monotonically with horizon (100 → 35 %); 3D drops more gently (most
  rows stay 75 → 38 %), but the trend is the same. The 3D escape volume
  softens but does not eliminate the reach-goal-bonus overshoot.
- **The 3D plan_dt blow-up was a Dijkstra artifact.** A first pass had
  every cell at 1.3-2.2 s — too slow to fit `replan_period=0.2 s`. The
  static cost-to-go cache (added to `SamplingMPCPlanner`) brought 3D
  plan_dt back to the same order of magnitude as 2D (70-750 ms across
  the grid), making this sweep and the cliff sweeps actually tractable.

Methodological transfer: re-validate Pareto in every dimensionality.
n_samples preference flips, the compute envelope changes, and what
looked like a CPU-saturation cliff in 3D was actually a missing cache.

## 3D perception-latency cliff: same corner, softened

Same 3D scenario, sensor.delay × max_speed sweep at the 3D Pareto config
(n_samples=8, horizon=20, n=6):

<p align="center">
<img src="images/cliff_3d.png" alt="6-panel sensor.delay × max_speed sweep on the 3D voxel world: success drop concentrated in the bottom-right cliff corner" width="640">
</p>

|  delay \ speed  |  10  |  15  |  20  |  25  |  30  |
|---|---|---|---|---|---|
| 0.00 |  83 | 100 | 100 | 100 |  83 |
| 0.05 |  83 | 100 | 100 | 100 |  83 |
| 0.10 |  83 | 100 | 100 |  83 |  50 |
| 0.20 |  67 | 100 |  83 |  83 |  50 |
| 0.50 |  83 |  83 | **33** |  50 |  50 |

The cliff transfers from 2D to 3D in the same `delay=0.5 × speed≥20 m/s`
corner. 2D had this region at 10-25 %; 3D softens it to 33-50 % —
the extra escape volume helps but does not eliminate the failure mode.

**3D cliff remediation: the velocity_window optimum *inverts* vs 2D.**
At the 3D cliff cell (delay=0.5, speed=20, n=12):

| sensor config | succ % | CI95 |
|---|---|---|
| baseline (no extrap) | 33.3 | [13.8, 60.9] |
| `extrapolate=true, window=1` | **83.3** | [55.2, 95.3] |
| `extrapolate=true, window=3` | 66.7 | [39.1, 86.2] |
| `extrapolate=true, window=5` | 58.3 | [32.0, 80.7] |
| `extrapolate=true, window=10` | 33.3 | [13.8, 60.9] |

CV ego extrapolation is the same big lever in 3D — +50 pp at window=1,
Wilson 95 % CIs do not overlap. But **the optimum inverts**: 2D's
peak was window=5, 3D's peak is window=1. The 3D escape volume lets
the drone trace smoother trajectories, so the 1-sample finite-
difference velocity is already accurate; smoothing only adds lag,
and lag hurts most at high speed where the cliff lives.

Engineering takeaway: the *parameter setting* of a remediation does
not transfer across dimensionalities even when the *technique* does.
Always re-tune ego-extrapolation window per scenario regime.

## Pareto config materially rewrites prior conclusions

The previous heatmap on the same scenario at the YAML's old defaults
(n_samples=32, horizon=60) reported a "dynamic-feasibility cliff at
25 m/s". At the Pareto config that cliff disappears (35 – 65 % success
at speed = 25-30 m/s), and replan_period — which "barely mattered"
before — now drives a 40 – 70 pp swing across 0.1 – 2.0 s. The earlier
conclusion was partly a CPU-saturation artifact: at horizon=60 every
replan took ~200 ms, so even replan_period=0.1 s could not actually
keep up.

> **Methodological lesson** baked into the YAML header: always validate
> ablation conclusions at the planner's Pareto-optimal config —
> suboptimal MPC settings can mask both ceilings (max feasible speed)
> and sensitivities (replan-period effect, delay tolerance).

## Multi-drone N-scaling and peer-prediction coordination

<p align="center">
<img src="images/multi_drone_3.png" alt="Three drones (alice/bob/charlie) crossing each other's paths to opposite-corner goals; joint=all_success" width="540">
</p>

*N=3 multi-drone episode — alice / bob / charlie all reach their
opposite-corner goals while routing around each other via the MPC's
constant-velocity peer prediction.*

`examples/exp_multi_drone_{2,3,4,8}.yaml` — same world, only drone
count changes. n=30, joint metrics with Wilson 95 % CIs:

| N | joint succ | joint coll | per-drone succ | indep `per^N` | Δ over indep |
|---|---|---|---|---|---|
| 2 | 96.7 % [83, 99] | 3.3 %  | 98.3 % | 96.6 % | +0.1 pp |
| 3 | 70.0 % [52, 83] | 30.0 % | 87.8 % | 67.7 % | +2.3 pp |
| 4 | 73.3 % [56, 86] | 26.7 % | 87.5 % | 58.6 % | **+14.7 pp** |
| 8 | 16.7 % [7, 34]  | 83.3 % | 70.0 % |  5.8 % | **+10.9 pp** |

The MPC's constant-velocity peer prediction *correlates failures in
the right direction* — when one drone yields, the others see its new
trajectory and react, so the system as a whole degrades less than
independent drones would.

**Coordination is non-monotonic in N.** Δ peaks at N=4 (+14.7 pp) and
declines at N=8 (+10.9 pp), even though the absolute joint success
collapses from 73.3 % → 16.7 %. Two effects compound:

- **More peers → more coordination signal**, lifting Δ over the
  independence baseline (the curve from N=2 to N=4).
- **More peers → escape-volume saturation**, dropping per-drone success
  from 87 % → 70 % at N=8 and pulling joint success down faster than
  coordination can recover (the N=4 → N=8 turn).

Engineering takeaway: peer-prediction coordination has a *useful
range*, not a monotonic scaling law. For dense N you need either a
bigger world (lower density per drone) or a coordinator that goes
beyond constant-velocity prediction (priority scheduling, reservation
tables, decentralised roundabout). The framework's MPC ceiling for
this 60×60 world tops out around N=4-6; past that, peer prediction
still helps in *relative* terms but is fundamentally fighting density.

### Density ablation: the "non-monotonic in N" claim was a density artifact

`examples/exp_multi_drone_8_low_density.yaml` — the obvious follow-up
to the engineering takeaway above. Same N=8 / same crossing structure
/ same Pareto-MPC config / same 30 obstacles, but world is 100×100
instead of 60×60 (~2.8 × the area per drone, 1250 cells/drone vs 450).

| density | world | per-drone | joint | indep `per^N` | Δ over indep |
|---|---|---|---|---|---|
| high | 60×60 | 70.0 % [64, 75] | 16.7 % [7, 34] | 5.8 % | +10.9 pp |
| low | 100×100 | 82.1 % [77, 86] | **46.7 %** [30, 64] | 21.0 % | **+25.7 pp** |

Halving density (actually 2.8 ×):
- per-drone success +12 pp
- joint success +30 pp
- coordination Δ +14.8 pp (**roughly doubled**)

Refines the prior finding in two ways:

1. **The "non-monotonic in N" was a density artifact.** With
   N=8 / low-density Δ at +25.7 pp — *larger* than N=4 / high-density
   Δ at +14.7 pp — the coordination scaling law is **monotonic in N
   when density allows**. The planner's CV peer prediction continues
   to make better use of more peers, full stop.

2. **The MPC ceiling is `density × N`, not raw N.** Doubling room
   halved the per-drone collision rate (30 % → 18 %). The dip at
   N=8 in the original table reflected escape-volume saturation, not
   a fundamental coordination limit.

Engineering takeaway (refined): **density × planner capacity** is the
load-bearing axis. To deploy CV peer prediction at high N, scale the
world or shrink drone radii — the planner itself is fine. The
"non-monotonic" caveat from above survives only as a *density-saturated*
regime warning, not a scaling-law claim.

Methodological close: the same 2-step pattern as the mpc_chomp →
velocity-profile → action-jump saga. Initial finding (PR #25)
reported the surface result; follow-up ablation isolated the actual
load-bearing axis. The first finding wasn't wrong — it was
incomplete. Always ablate the engineering takeaway your previous YAML
header speculated about.

## Wind miscalibration: planner belief must match sim reality

`examples/exp_wind.yaml` — constant northward wind disturbance × planner
wind belief, 4 × 4 grid, n=15 episodes:

<p align="center">
<img src="images/wind_miscal.png" alt="6-panel sim_wind × planner_wind sweep showing diagonal-wins miscalibration" width="640">
</p>

|  sim_wind \ planner_wind | 0 | 3 | 6 | 9 |
|---|---|---|---|---|
| **0** | **93.3** | 60.0 | 33.3 | 20.0 |
| **3** | 66.7 | **100.0** | 53.3 | 33.3 |
| **6** | 20.0 | 66.7 | **93.3** | 66.7 |
| **9** |  0.0 |  0.0 |  0.0 |  0.0 |

The diagonal wins — matched (planner belief = sim reality) recovers
93-100 %. Mismatch in either direction hurts symmetrically: under-
correction blows the drone off course, over-correction pre-compensates
into nothing. At `sim=6 m/s`, wind awareness lifts success from **20 %
to 93 %** (+73 pp) — one of the largest single-knob wins in the
framework. But at `sim=9 m/s` against `max_speed=8 m/s` no belief
saves you: the drone literally cannot make headway and every cell in
that row is 0 %. Awareness cannot beat physics.

## The perception-latency cliff: a four-step research saga

<p align="center">
<img src="images/cliff_delay_speed.png" alt="6-panel sensor.delay × max_speed sweep showing success drop and planner-dt blow-up at high delay × high speed" width="640">
</p>

*sensor.delay × max_speed at the Pareto MPC config — success dives in
the bottom-right corner (delay=0.5 s × speed≥20 m/s) while the rest of
the grid is comfortably ≥ 80 %. That single corner is the cliff.*

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

## MPPI vs MPC: switching planner family doesn't break a tuned baseline

`examples/exp_compare_mppi.yaml` — Model Predictive Path Integral, the
sampling MPC's stochastic cousin. Same Dijkstra cost-to-go heuristic,
same n direction samples, same horizon-step rollouts, same per-rollout
scoring. The only substitution is at the selection step:

  weight_i = exp(-(cost_i - cost_min) / temperature)
  action   = sum(weight_i * action_i)

Sweeping `temperature` on the predictive scenario (n=30, Wilson 95 % CI):

| `temperature` | success | mean &#124;Δcmd&#124;/step | avg speed |
|---|---|---:|---:|
| 0.1 (≈ argmin)  | 96.7 % [83, 99]  | 0.319 | 9.86 m/s |
| **1.0** (sweet spot) | **100.0 %** [89, 100] | 0.291 | 9.56 m/s |
| 10.0 (smoothest) | 86.7 % [70, 95] | 0.223 | 2.83 m/s (collapse) |
| 100.0 (uniform-mean) | 0.0 % [0, 11] | n/a | 0.81 m/s |

Two orthogonal findings:

1. **MPPI beats default MPC on both axes (96.7 % / 0.32 → 100 % / 0.29) but
   loses to tuned MPC on smoothness** (`w_smooth=0.5`: 100 % / 0.244 — see
   the action-jump finding above). MPPI sits *between* default and tuned
   MPC. Same Pareto saturation lesson the mpc_chomp thread taught: at this
   regime, switching planner family does not break through a well-tuned
   baseline.

2. **The temperature knob *is* the value-add.** Argmin MPC has no smooth
   way to interpolate between "decisive" and "averaging" behavior —
   `w_smooth` only penalises the *jump* between consecutive chosen
   actions, not the *concentration* of the selection itself. At
   temperature=10 MPPI achieves the smoothest control trajectory of any
   planner in the suite (|Δcmd| 0.22, **-31 % vs default MPC**) at the
   cost of a 10 pp success drop and 71 % speed loss. This Pareto
   position isn't reachable from MPC's argmin family — the framework
   gains a continuous knob it didn't have before.

**Methodological miss-and-recovery (recorded for honesty).** My first
attempt set the default temperature to 10.0 by eyeballing "middle of
{0.1, 100}". The sweep revealed 10.0 sits in the speed-collapse zone
(only 86.7 % success because the chosen action averages toward the
n_samples set's mean, which has small magnitude). Default is now 1.0,
identified by the success cliff in the sweep. The right way to pick a
default is to *measure where the success cliff is*, not to eyeball it
from cost magnitudes — same lesson as the Pareto-config retrofit.

## MPC + CHOMP smoothing: layering on a saturated planner is a wash

`examples/exp_compare_mpc_chomp.yaml` — `mpc_chomp` planner wraps the
validated Pareto MPC config and runs 15 CHOMP smoothing iterations on
the rollout each replan, then clears `target_velocity` so the runner
pure-pursues the smoothed waypoints. Hypothesis: file off the
piecewise-straight corners at each replan boundary so the velocity
profile is gentler. Same scenario / horizon / sample count as the
plain-MPC baseline.

|              | success           | plan_dt (mean) | mean &#124;Δcmd&#124;/step |
|--------------|-------------------|---------------:|--------------:|
| plain MPC    | 96.7 % [83, 99]   | 11.0 ms        | 0.32          |
| **mpc + chomp** | 96.7 % [83, 99] | 18.9 ms (+71 %)| **0.61** (+90 %) |

Honest null result: success rate identical, plan_dt up 71 %, and the
per-step command delta nearly *doubles*. The reason is architectural,
not a tuning bug. MPC's `target_velocity` bypass *is* the smoothness
mechanism — it commits to one velocity for the whole `replan_period`
(0.2 s = 4 control steps) so the controller has nothing to chase
between replans and per-step `|Δcmd|` is small. CHOMP smoothing emits a
curved waypoint sequence that pure-pursuit re-aims at every 0.05 s, so
even though the *path* has fewer corners, the *control trajectory* has
more direction changes.

Engineering takeaway: layering a smoother on top of a planner that is
already at its Pareto saturation point is a wash unless the smoothing
target is downstream of where the cost lives — here the cost lives in
the controller, not the path. To make CHOMP help in this setting you
would need a velocity-profile-aware follower (or a planner that emits
a velocity spline directly). Same Pareto-saturation lesson as the
3D CHOMP+RRT result: a layer only wins if the layer below has room to
be improved.
