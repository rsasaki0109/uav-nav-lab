#!/usr/bin/env python3
"""Train RL baseline (SAC) on grid_world and compare against MPC.

Usage:
    python3 scripts/train_rl_baseline.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from uav_nav_lab.rl import GridNavEnv


def evaluate_agent(model: SAC, env: GridNavEnv, n_episodes: int = 50) -> dict:
    """Evaluate a trained agent and return metrics."""
    success = 0
    collision = 0
    timeout = 0
    times = []
    path_lens = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        pos_prev = obs[:2].copy()
        total_len = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            pos_curr = obs[:2]
            total_len += float(np.linalg.norm(pos_curr - pos_prev))
            pos_prev = pos_curr.copy()
        if info.get("goal_reached"):
            success += 1
        elif info.get("collision"):
            collision += 1
        else:
            timeout += 1
        times.append(env._step_count * env._dt)
        path_lens.append(total_len)
    return {
        "success_rate": success / n_episodes * 100,
        "collision_rate": collision / n_episodes * 100,
        "timeout_rate": timeout / n_episodes * 100,
        "avg_time": float(np.mean(times)),
        "avg_path_len": float(np.mean(path_lens)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="results/rl_baseline")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = GridNavEnv(max_speed=8.0, max_steps=1500, local_occ_size=7)
    env = Monitor(env)

    # Train SAC
    print(f"Training SAC for {args.timesteps} timesteps...")
    t0 = time.perf_counter()
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=0,
    )
    model.learn(total_timesteps=args.timesteps)
    train_time = time.perf_counter() - t0
    print(f"Training completed in {train_time:.0f}s")

    # Save model
    model_path = str(out_dir / "sac_grid_nav")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate
    print(f"\nEvaluating ({args.eval_episodes} episodes)...")
    eval_env = GridNavEnv(max_speed=8.0, max_steps=1500, local_occ_size=7)
    results = evaluate_agent(model, eval_env, args.eval_episodes)
    print(f"  Success:   {results['success_rate']:.0f}%")
    print(f"  Collision: {results['collision_rate']:.0f}%")
    print(f"  Timeout:   {results['timeout_rate']:.0f}%")
    print(f"  Avg time:  {results['avg_time']:.1f}s")
    print(f"  Avg path:  {results['avg_path_len']:.1f}m")

    # Write results
    with (out_dir / "rl_results.json").open("w") as f:
        json.dump({
            "train_timesteps": args.timesteps,
            "train_time_s": train_time,
            "eval_episodes": args.eval_episodes,
            **results,
        }, f, indent=2)
    print(f"Results written to {out_dir / 'rl_results.json'}")


if __name__ == "__main__":
    main()
