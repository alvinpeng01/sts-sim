"""Label experiment: is the bottleneck label quality, or label count?

v30 answered "is model capacity the bottleneck?" with a clear no. But its training
log shows it never fit the training set either — train policy loss fell 0.021 nats
over 30 epochs and stopped 0.13 above the 1.272-nat irreducible floor set by the
soft targets' own entropy. v28's log is the same story, only more so: 0.005 nats
over 24 epochs. Neither run learned much from its ~1,300 labels; what v28 knows it
inherited from v27's warm start.

That leaves two candidate bottlenecks, and this script runs one arm for each:

    --arm 300     control     labels from the 300-sim datasets v30 already generated
    --arm 800     quality     labels from 800-sim datasets (Act 1 already generated)
    --arm yield   count       300-sim labels, --label-scale x more of them

All arms train at v28's architecture (dim=96, layers=2, heads=4) so the warm start
from v28 actually lands — v30's did not, which cost it v27's fitted weights. All
arms are evaluated at 300 sims: the treatment is on the labels, so holding the eval
budget fixed is what isolates it.

The 300 arm needs no generation; its datasets are the ones v30 trained on. The 800
arm reuses the existing Act 1 data in runs/v30_comparison_800sims/. The yield arm
raises --max-labels-per-episode so each expensively-played episode gives up more of
its decisions, which also amortizes the main-line play over more labels. Note the
tradeoff it accepts: labels drawn from the same episode are correlated, so effective
sample size grows more slowly than row count.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


# v28's architecture. Every arm uses it so the v28 warm start transfers.
DIM = 96
LAYERS = 2
HEADS = 4

# Held fixed across arms so the treatment is the only difference. rollouts=8
# matches both the existing 800-sim Act 1 data and the 300-sim datasets v30 used.
ROLLOUTS = 8
ROLLOUT_DECISIONS = 96

# Baseline label budgets, as (labels, episodes, per_type), from v28/v30.
TRAIN_SHAPE = {1: (80, 800, 25), 2: (100, 2_500, 30), 3: (120, 10_000, 35)}
VALIDATION_SHAPE = {1: (15, 300, 8), 2: (20, 700, 10), 3: (25, 3_000, 12)}

# Baseline yield throttles. The yield arm relaxes MAX_LABELS_PER_EPISODE; both
# arms keep PRIORITY_ACCEPT_BASE so the label *distribution* stays comparable and
# only the count changes.
MAX_LABELS_PER_EPISODE = 2
YIELD_MAX_LABELS_PER_EPISODE = 12
PRIORITY_ACCEPT_BASE = "0.10"

SEEDS = {
    ("train", 1): 18_200_000,
    ("validation", 1): 18_300_000,
    ("train", 2): 18_400_000,
    ("validation", 2): 18_500_000,
    ("train", 3): 18_600_000,
    ("validation", 3): 18_700_000,
}

LEGACY_TRAIN = [
    "runs/whole_run_long_v26_act1_train_400.pt",
    "runs/whole_run_long_v26_act2_train_400.pt",
    "runs/whole_run_long_v26_act3_train_240.pt",
]
LEGACY_VALIDATION = [
    "runs/whole_run_long_v26_act1_validation_60.pt",
    "runs/whole_run_long_v26_act2_validation_60.pt",
    "runs/whole_run_long_v26_act3_validation_60.pt",
]


def run(command: list[str], root: Path) -> None:
    print("RUN", " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=root, check=True, env=os.environ.copy())
    print(f"DONE seconds={time.perf_counter() - started:.1f}", flush=True)


def generation_command(
        *, output: str, labels: int, episodes: int, per_type: int, seed: int,
        act: int, policy: str, workers: int, combat_sims: int,
        max_labels_per_episode: int, rollouts: int,
        priority_accept_base: str, harvest_rate: str = "0.0",
        truncate_after: str = "0") -> list[str]:
    return [
        sys.executable, "-m", "lightspeed.parallel_generate_whole_run_rollouts",
        "--workers", str(workers),
        "--out", output,
        "--max-labels", str(labels),
        "--max-episodes", str(episodes),
        "--per-type", str(per_type),
        "--seed", str(seed),
        "--policy", policy,
        "--combat-sims", str(combat_sims),
        "--rollouts", str(rollouts),
        "--rollout-decisions", str(ROLLOUT_DECISIONS),
        "--policy-temperature", "1.05",
        "--label-temperature", "0.15",
        "--priority-accept-base", priority_accept_base,
        "--harvest-rate", harvest_rate,
        "--truncate-after", truncate_after,
        "--max-labels-per-episode", str(max_labels_per_episode),
        "--min-act", str(act),
        "--max-act", str(act),
        "--save-every", "2",
        "--trajectory-auxiliary-targets",
        "--torch-threads", "1",
    ]


def arm_plan(arm: str, label_scale: int) -> dict:
    """Resolve an arm into its sim budget, label budgets, and dataset directory.

    The 300 arm deliberately points at runs/ — those datasets already exist, so
    the control costs no generation at all.
    """
    if arm == "300":
        return {
            "combat_sims": 300, "scale": 1, "directory": "runs",
            "max_labels_per_episode": MAX_LABELS_PER_EPISODE,
            "tag": "labelq300"}
    if arm == "800":
        return {
            "combat_sims": 800, "scale": 1,
            "directory": "runs/v30_comparison_800sims",
            "max_labels_per_episode": MAX_LABELS_PER_EPISODE,
            "tag": "labelq800"}
    return {
        "combat_sims": 300, "scale": label_scale,
        "directory": f"runs/v31_yield{label_scale}x",
        "max_labels_per_episode": YIELD_MAX_LABELS_PER_EPISODE,
        "tag": f"yield{label_scale}x"}


def dataset_paths(plan: dict) -> tuple[dict[int, str], dict[int, str]]:
    scale, directory = plan["scale"], plan["directory"]
    train = {
        act: (f"{directory}/whole_run_v31_act{act}_train_"
              f"{TRAIN_SHAPE[act][0] * scale}.pt")
        for act in (1, 2, 3)}
    validation = {
        act: (f"{directory}/whole_run_v31_act{act}_validation_"
              f"{VALIDATION_SHAPE[act][0] * scale}.pt")
        for act in (1, 2, 3)}
    if scale == 1:
        # The scale-1 arms reuse datasets already on disk under their v30 names.
        train = {
            act: f"{directory}/whole_run_v30_act{act}_train_{TRAIN_SHAPE[act][0]}.pt"
            for act in (1, 2, 3)}
        validation = {
            act: (f"{directory}/whole_run_v30_act{act}_validation_"
                  f"{VALIDATION_SHAPE[act][0]}.pt")
            for act in (1, 2, 3)}
    return train, validation


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    build_dir = str(root.parent / "sts_lightspeed" / "build")
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [p for p in (build_dir, str(root), existing) if p])

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm", choices=("300", "800", "yield"), required=True,
        help="300=control, 800=label quality, yield=label count")
    parser.add_argument(
        "--label-scale", type=int, default=10,
        help="yield arm only: multiple of the baseline label budget")
    parser.add_argument(
        "--policy",
        default="runs/whole_run_transformer_outcome_a20_v28.pt",
        help="rollout policy for generation and warm-start source for training")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--rollouts", type=int, default=ROLLOUTS,
        help="counterfactual continuations per label; halving roughly doubles "
             "labels per compute-hour at the cost of noisier soft targets")
    parser.add_argument(
        "--priority-accept-base", default=PRIORITY_ACCEPT_BASE,
        help="acceptance probability for non-priority decisions. 0.10 keeps "
             "v31's curation bias; 1.0 keeps every eligible decision and "
             "changes the label distribution")
    parser.add_argument(
        "--max-labels-per-episode", type=int, default=None,
        help="0 for unlimited. Defaults to the arm's value (yield arm: 12)")
    parser.add_argument("--evaluation-runs", type=int, default=200)
    parser.add_argument("--eval-sims", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument(
        "--checkpoint-every", default="0",
        help="keep intermediate epoch checkpoints so the epoch curve can be "
             "selected on paired floors rather than validation NLL")
    parser.add_argument(
        "--scope", default="all-v27",
        choices=("experts", "experts-structure", "all-v27", "full"),
        help="all-v27 freezes the trunk (76%% of the model); full trains it too")
    parser.add_argument("--lr", default="0.001")
    parser.add_argument(
        "--anchor-weight", default="0.25",
        help="KL pull toward the warm-start checkpoint. Binds harder once the "
             "trunk is unfrozen; 0 removes it entirely")
    parser.add_argument(
        "--batch", default="32",
        help="rows per optimizer step; 1 reproduces the pre-v31 behaviour")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument(
        "--tag", default=None,
        help="override the checkpoint/eval descriptor. Required when re-running "
             "an arm whose data already exists, since the default name is "
             "derived from the arm and would overwrite that arm's checkpoint")
    parser.add_argument(
        "--version", default="v31",
        help="lineage version in the checkpoint filename")
    parser.add_argument(
        "--data-dir", default=None,
        help="override the dataset directory. Required when regenerating an "
             "arm whose directory already holds data, since existing files are "
             "skipped and would be silently reused")
    parser.add_argument(
        "--truncate-after", default="0",
        help="stop continuations after N decisions and bootstrap the rest from "
             "the terminal_floor head. Measured at N=20: 2.3x cheaper per label "
             "at unchanged paired SNR, so the saving buys more rollouts")
    parser.add_argument(
        "--harvest-rate", default="0.0",
        help="fraction of continuation decisions kept as (state, action, "
             "return) rows in a sibling .harvest.pt; already simulated, so free")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the plan and the datasets still to generate, then stop")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.label_scale < 1:
        raise ValueError("--label-scale must be positive")

    plan = arm_plan(args.arm, args.label_scale)
    if args.tag:
        plan["tag"] = args.tag
    if args.max_labels_per_episode is not None:
        plan["max_labels_per_episode"] = args.max_labels_per_episode
    if args.data_dir:
        plan["directory"] = args.data_dir
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (root / plan["directory"]).mkdir(parents=True, exist_ok=True)
    train, validation = dataset_paths(plan)
    checkpoint = (
        f"runs/whole_run_transformer_{plan['tag']}_a20_{args.version}.pt")

    # Generation is a no-op wherever the dataset already exists: the 300 arm is
    # fully generated and the 800 arm is missing only Acts 2 and 3.
    stages = []
    for act in (1, 2, 3):
        for kind, paths, shape in (
                ("train", train, TRAIN_SHAPE),
                ("validation", validation, VALIDATION_SHAPE)):
            path = paths[act]
            if (root / path).exists():
                print(f"HAVE {path}", flush=True)
                continue
            labels, episodes, per_type = shape[act]
            stages.append(generation_command(
                output=path, labels=labels * plan["scale"],
                episodes=episodes * plan["scale"],
                per_type=per_type * plan["scale"],
                seed=SEEDS[(kind, act)], act=act, policy=args.policy,
                workers=args.workers, combat_sims=plan["combat_sims"],
                max_labels_per_episode=plan["max_labels_per_episode"],
                rollouts=args.rollouts,
                priority_accept_base=args.priority_accept_base,
                harvest_rate=args.harvest_rate,
                truncate_after=args.truncate_after))

    manifest = {
        "experiment": "label-quality-vs-count",
        "arm": args.arm,
        "combat_sims": plan["combat_sims"],
        "label_scale": plan["scale"],
        "max_labels_per_episode": plan["max_labels_per_episode"],
        "priority_accept_base": float(args.priority_accept_base),
        "policy": args.policy,
        "workers": args.workers,
        "rollouts": args.rollouts,
        "rollout_decisions": ROLLOUT_DECISIONS,
        "trajectory_auxiliary_targets": True,
        "dim": DIM, "layers": LAYERS, "heads": HEADS,
        "epochs": args.epochs,
        "train": train, "validation": validation,
        "legacy_train": LEGACY_TRAIN, "legacy_validation": LEGACY_VALIDATION,
        "output_checkpoint": checkpoint,
        "evaluation_runs": args.evaluation_runs,
        "eval_sims": args.eval_sims,
        "stages_to_generate": len(stages),
        "created_unix": time.time(),
    }
    (runs / f"whole_run_{plan['tag']}_{args.version}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        for command in stages:
            print("WOULD RUN", " ".join(command), flush=True)
        return

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    for command in stages:
        run(command, root)

    # v28's training hyperparameters, unchanged — the labels are the variable.
    run([
        sys.executable, "-m", "lightspeed.train_whole_run_v27",
        "--dataset", *LEGACY_TRAIN, *train.values(),
        "--validation-dataset", *LEGACY_VALIDATION, *validation.values(),
        "--load", args.policy,
        "--out", checkpoint,
        "--epochs", str(args.epochs),
        "--checkpoint-every", args.checkpoint_every,
        "--lr", args.lr,
        "--batch", args.batch,
        "--dim", str(DIM),
        "--layers", str(LAYERS),
        "--heads", str(HEADS),
        "--anchor-weight", args.anchor_weight,
        "--ensemble-weight", "0.20",
        "--auxiliary-weight", "0.30",
        "--scope", args.scope,
        "--seed", "18_800_000",
    ], root)

    if not args.skip_evaluation:
        run([
            sys.executable, "-m", "lightspeed.eval_whole_run_policy",
            args.policy, checkpoint,
            "--runs", str(args.evaluation_runs),
            "--seed-base", "18_900_000",
            "--sims", str(args.eval_sims),
            "--ascension", "20",
            "--torch-threads", "1",
            "--out", (f"runs/whole_run_v28_vs_{plan['tag']}_a20_"
                      f"{args.evaluation_runs}seeds_{args.eval_sims}sims.jsonl"),
        ], root)
    print(
        f"LABEL_EXPERIMENT_{args.version.upper()}_{plan['tag'].upper()}_COMPLETE",
        flush=True)


if __name__ == "__main__":
    main()
