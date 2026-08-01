"""Resumable long-horizon, act-balanced whole-run training curriculum."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(command: list[str], root: Path) -> None:
    print("RUN", " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=root, check=True)
    print(f"DONE seconds={time.perf_counter() - started:.1f}", flush=True)


def generation_command(
        *, output: str, labels: int, episodes: int, per_type: int,
        seed: int, min_act: int, max_act: int, policy: str) -> list[str]:
    return [
        sys.executable, "-m", "lightspeed.parallel_generate_whole_run_rollouts",
        "--workers", "6",
        "--out", output,
        "--max-labels", str(labels),
        "--max-episodes", str(episodes),
        "--per-type", str(per_type),
        "--seed", str(seed),
        "--policy", policy,
        "--combat-sims", "300",
        "--rollouts", "8",
        "--rollout-decisions", "96",
        "--policy-temperature", "1.10",
        "--label-temperature", "0.15",
        "--priority-accept-base", "0.08",
        "--max-labels-per-episode", "2",
        "--min-act", str(min_act),
        "--max-act", str(max_act),
        "--save-every", "2",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        default="runs/whole_run_transformer_route_tuned_a20_v24.pt")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--evaluation-runs", type=int, default=500)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    if args.workers != 6:
        raise ValueError(
            "this curriculum is calibrated for six physical CPU cores")

    root = Path(__file__).resolve().parents[1]
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    train = {
        1: "runs/whole_run_long_v26_act1_train_400.pt",
        2: "runs/whole_run_long_v26_act2_train_400.pt",
        3: "runs/whole_run_long_v26_act3_train_240.pt",
    }
    validation = {
        1: "runs/whole_run_long_v26_act1_validation_60.pt",
        2: "runs/whole_run_long_v26_act2_validation_60.pt",
        3: "runs/whole_run_long_v26_act3_validation_60.pt",
    }
    stages = [
        generation_command(
            output=train[1], labels=400, episodes=1_500, per_type=80,
            seed=5_100_000, min_act=1, max_act=1, policy=args.policy),
        generation_command(
            output=validation[1], labels=60, episodes=500, per_type=20,
            seed=5_300_000, min_act=1, max_act=1, policy=args.policy),
        generation_command(
            output=train[2], labels=400, episodes=3_000, per_type=80,
            seed=5_500_000, min_act=2, max_act=2, policy=args.policy),
        generation_command(
            output=validation[2], labels=60, episodes=1_000, per_type=20,
            seed=5_700_000, min_act=2, max_act=2, policy=args.policy),
        generation_command(
            output=train[3], labels=240, episodes=12_000, per_type=60,
            seed=5_900_000, min_act=3, max_act=3, policy=args.policy),
        generation_command(
            output=validation[3], labels=60, episodes=4_000, per_type=20,
            seed=6_100_000, min_act=3, max_act=3, policy=args.policy),
    ]
    manifest = {
        "policy": args.policy,
        "workers": args.workers,
        "combat_sims": 300,
        "rollouts": 8,
        "rollout_decisions": 96,
        "max_labels_per_episode": 2,
        "train": train,
        "validation": validation,
        "output_checkpoint": "runs/whole_run_transformer_long_horizon_a20_v26.pt",
        "evaluation_runs": args.evaluation_runs,
        "created_unix": time.time(),
    }
    (runs / "whole_run_long_v26_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    for command in stages:
        run(command, root)

    checkpoint = "runs/whole_run_transformer_long_horizon_a20_v26.pt"
    run([
        sys.executable, "-m", "lightspeed.train_whole_run_replay",
        "--dataset", train[1], train[2], train[3],
        "--validation-dataset", validation[1], validation[2], validation[3],
        "--load", args.policy,
        "--out", checkpoint,
        "--epochs", "40",
        "--lr", "0.00005",
        "--anchor-weight", "0.15",
        "--train-scope", "act-adapter",
        "--seed", "6_300_000",
    ], root)
    if not args.skip_evaluation:
        run([
            sys.executable, "-m", "lightspeed.eval_whole_run_policy",
            args.policy, checkpoint,
            "--runs", str(args.evaluation_runs),
            "--seed-base", "6_500_000",
            "--sims", "300",
            "--ascension", "20",
            "--out", "runs/whole_run_v24_vs_v26_a20_500seeds_300sims.jsonl",
        ], root)
    print("LONG_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
