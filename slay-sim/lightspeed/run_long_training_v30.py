"""Resumable outcome-supervised v30 curriculum warm-started from v28."""
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
    subprocess.run(command, cwd=root, check=True, env=os.environ.copy())
    print(f"DONE seconds={time.perf_counter() - started:.1f}", flush=True)


def generation_command(
        *, output: str, labels: int, episodes: int, per_type: int,
        seed: int, min_act: int, max_act: int, policy: str,
        workers: int) -> list[str]:
    return [
        sys.executable, "-m", "lightspeed.parallel_generate_whole_run_rollouts",
        "--workers", str(workers),
        "--out", output,
        "--max-labels", str(labels),
        "--max-episodes", str(episodes),
        "--per-type", str(per_type),
        "--seed", str(seed),
        "--policy", policy,
        "--combat-sims", "300",
        "--rollouts", "8",
        "--rollout-decisions", "96",
        "--policy-temperature", "1.05",
        "--label-temperature", "0.15",
        "--priority-accept-base", "0.10",
        "--max-labels-per-episode", "2",
        "--min-act", str(min_act),
        "--max-act", str(max_act),
        "--save-every", "2",
        "--trajectory-auxiliary-targets",
        "--torch-threads", "1",
    ]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    # Ensure subprocesses can find slaythespire and the lightspeed module
    build_dir = str(root.parent / "sts_lightspeed" / "build")
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [p for p in (build_dir, str(root), existing) if p])
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        default="runs/whole_run_transformer_outcome_a20_v28.pt")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--evaluation-runs", type=int, default=500)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    root = Path(__file__).resolve().parents[1]
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    train = {
        1: "runs/whole_run_v30_act1_train_80.pt",
        2: "runs/whole_run_v30_act2_train_100.pt",
        3: "runs/whole_run_v30_act3_train_120.pt",
    }
    validation = {
        1: "runs/whole_run_v30_act1_validation_15.pt",
        2: "runs/whole_run_v30_act2_validation_20.pt",
        3: "runs/whole_run_v30_act3_validation_25.pt",
    }
    stages = [
        generation_command(
            output=train[1], labels=80, episodes=800, per_type=25,
            seed=18_200_000, min_act=1, max_act=1, policy=args.policy,
            workers=args.workers),
        generation_command(
            output=validation[1], labels=15, episodes=300, per_type=8,
            seed=18_300_000, min_act=1, max_act=1, policy=args.policy,
            workers=args.workers),
        generation_command(
            output=train[2], labels=100, episodes=2_500, per_type=30,
            seed=18_400_000, min_act=2, max_act=2, policy=args.policy,
            workers=args.workers),
        generation_command(
            output=validation[2], labels=20, episodes=700, per_type=10,
            seed=18_500_000, min_act=2, max_act=2, policy=args.policy,
            workers=args.workers),
        generation_command(
            output=train[3], labels=120, episodes=10_000, per_type=35,
            seed=18_600_000, min_act=3, max_act=3, policy=args.policy,
            workers=args.workers),
        generation_command(
            output=validation[3], labels=25, episodes=3_000, per_type=12,
            seed=18_700_000, min_act=3, max_act=3, policy=args.policy,
            workers=args.workers),
    ]
    legacy_train = [
        "runs/whole_run_long_v26_act1_train_400.pt",
        "runs/whole_run_long_v26_act2_train_400.pt",
        "runs/whole_run_long_v26_act3_train_240.pt",
    ]
    legacy_validation = [
        "runs/whole_run_long_v26_act1_validation_60.pt",
        "runs/whole_run_long_v26_act2_validation_60.pt",
        "runs/whole_run_long_v26_act3_validation_60.pt",
    ]
    checkpoint = "runs/whole_run_transformer_outcome_a20_v30.pt"
    manifest = {
        "policy": args.policy,
        "workers": args.workers,
        "combat_sims": 300,
        "rollouts": 8,
        "rollout_decisions": 96,
        "trajectory_auxiliary_targets": True,
        "new_train": train,
        "new_validation": validation,
        "legacy_train": legacy_train,
        "legacy_validation": legacy_validation,
        "output_checkpoint": checkpoint,
        "evaluation_runs": args.evaluation_runs,
        "created_unix": time.time(),
    }
    (runs / "whole_run_long_v30_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    for command in stages:
        run(command, root)

    run([
        sys.executable, "-m", "lightspeed.train_whole_run_v27",
        "--dataset", *legacy_train, *train.values(),
        "--validation-dataset", *legacy_validation, *validation.values(),
        "--load", args.policy,
        "--out", checkpoint,
        "--epochs", "30",
        "--lr", "0.00005",
        "--dim", "192",
        "--layers", "3",
        "--heads", "6",
        "--anchor-weight", "0.25",
        "--ensemble-weight", "0.20",
        "--auxiliary-weight", "0.30",
        "--scope", "all-v27",
        "--seed", "18_800_000",
    ], root)
    if not args.skip_evaluation:
        run([
            sys.executable, "-m", "lightspeed.eval_whole_run_policy",
            args.policy, checkpoint,
            "--runs", str(args.evaluation_runs),
            "--seed-base", "18_900_000",
            "--sims", "300",
            "--ascension", "20",
            "--torch-threads", "1",
            "--out", "runs/whole_run_v28_vs_v30_a20_500seeds_300sims.jsonl",
        ], root)
    print("LONG_TRAINING_V30_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
