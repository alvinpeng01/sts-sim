"""What is each decision type actually worth?

`_routing_audit.py` establishes that routing is real: replacing every path choice
with a uniform-random legal one costs 4.87 mean floors.  Nobody has asked the
same question of the other eight decision types, and there is now a reason to.
The policy's median top-1/top-2 logit gap is 0.129 nats and 95% of decisions are
settled by under half a logit (07-known-issues.md), so on some screens the argmax
is arbitrary but *deterministic* -- which looks exactly like a policy while
contributing nothing.

This measures the difference.  One arm per decision type: that type is played by
a uniform-random choice among the legal actions, every other decision stays with
the policy, and the seeds and combat are shared with the baseline.  The cost in
floors is what the network's preference on that screen is worth.

A type that measures ~0 is not a failure -- it is an opportunity.  It means a
cheap hand-written rule or a small fitted prior competes with the network there,
without training anything.

Run from slay-sim/:
    python -m lightspeed._decision_ablation --runs 240 --workers 6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

DEFAULT_CHECKPOINT = "runs/whole_run_transformer_postfix-trunc_a20_v37.pt"

SCREEN_TYPES = {
    1: "event", 2: "rewards", 3: "boss_relic", 4: "card_select",
    5: "map", 6: "treasure", 7: "rest", 8: "shop",
}
ARMS = ["baseline", "neow", "event", "map", "rewards", "shop", "rest",
        "boss_relic", "card_select", "treasure"]

_STATE: dict = {}


def _worker_init(checkpoint: str, sims: int, ascension: int) -> None:
    import torch

    from .eval_whole_run_policy import load_policy
    from .search_config import DEFAULT_SEARCH_CONFIG_PATH

    torch.set_num_threads(1)
    _STATE["torch"] = torch
    _STATE["policy"] = load_policy(checkpoint, torch.device("cpu"))
    _STATE["sims"] = sims
    _STATE["ascension"] = ascension
    _STATE["search_config"] = DEFAULT_SEARCH_CONFIG_PATH


def decision_type(obs, floor: int) -> str:
    """The generator's own nine types, recovered from the observation.

    Neow is an EVENT_SCREEN like any other; what distinguishes it is that it is
    the one on floor 0, and it is worth separating because it is a single
    high-leverage choice made before any information exists.
    """
    screen = int(obs["screen"])
    if screen == 1 and floor == 0:
        return "neow"
    return SCREEN_TYPES.get(screen, "other")


def _play(job: tuple[str, int]) -> dict:
    import random

    torch = _STATE["torch"]
    policy = _STATE["policy"]

    from .whole_run_env import RunConfig, WholeRunEnv

    arm, seed = job
    rng = random.Random((seed << 8) ^ hash(arm) & 0xFFFFFFFF)
    env = WholeRunEnv(RunConfig(
        ascension=_STATE["ascension"], combat_sims=_STATE["sims"],
        deterministic_combat=True, search_config_path=_STATE["search_config"]))
    obs = env.reset(seed)
    randomized = 0
    counts: dict[str, int] = {}
    with torch.inference_mode():
        while (env.gc.outcome.name == "UNDECIDED"
               and env.steps < env.config.max_decisions):
            kind = decision_type(obs, int(env.gc.floor_num))
            counts[kind] = counts.get(kind, 0) + 1
            candidates = int(obs["action_features"].shape[0])
            if arm == kind and candidates > 1:
                action = rng.randrange(candidates)
                randomized += 1
            else:
                action, _, _, _ = policy.act(obs, sample=False)
            obs, _, done, _ = env.step(action)
            if done:
                break
    return {"arm": arm, "seed": seed, "floor": int(env.gc.floor_num),
            "act": int(env.gc.act), "outcome": str(env.gc.outcome),
            "randomized": randomized, "decision_counts": counts}


def summarize(rows: list[dict]) -> str:
    by_arm: dict[str, dict[int, dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], {})[row["seed"]] = row
    base = by_arm["baseline"]
    lines = [f"{'arm':>12}  {'n':>4}  {'floor':>6}  {'cost vs baseline':>18}  "
             f"{'t':>6}  {'decisions/run':>13}"]
    order = sorted(
        (a for a in by_arm if a != "baseline"),
        key=lambda a: statistics.mean(r["floor"] for r in by_arm[a].values()))
    floors = [r["floor"] for r in base.values()]
    lines.append(f"{'baseline':>12}  {len(floors):>4}  "
                 f"{statistics.mean(floors):>6.2f}  {'--':>18}  {'--':>6}  "
                 f"{'--':>13}")
    for arm in order:
        seeds = by_arm[arm]
        shared = sorted(set(seeds) & set(base))
        diffs = [seeds[s]["floor"] - base[s]["floor"] for s in shared]
        mean = statistics.mean(diffs)
        sem = (statistics.stdev(diffs) / math.sqrt(len(diffs))
               if len(diffs) > 1 else float("nan"))
        touched = statistics.mean(r["randomized"] for r in seeds.values())
        lines.append(
            f"{arm:>12}  {len(seeds):>4}  "
            f"{statistics.mean(r['floor'] for r in seeds.values()):>6.2f}  "
            f"{mean:>+10.2f} +/-{sem:.2f}  {mean / sem if sem else 0:>6.2f}  "
            f"{touched:>13.1f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--runs", type=int, default=240)
    parser.add_argument("--seed-base", type=int, default=1_003_000)
    parser.add_argument("--sims", type=int, default=300)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--arms", default=None)
    parser.add_argument("--out", default="runs/decision_ablation.jsonl")
    args = parser.parse_args()

    arms = ARMS
    if args.arms:
        wanted = {a.strip() for a in args.arms.split(",")} | {"baseline"}
        arms = [a for a in ARMS if a in wanted]

    jobs = [(arm, args.seed_base + offset)
            for arm in arms for offset in range(args.runs)]
    print(f"{len(arms)} arms x {args.runs} seeds = {len(jobs)} runs "
          f"at {args.sims} sims on {args.workers} workers", flush=True)

    rows: list[dict] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_worker_init,
            initargs=(args.checkpoint, args.sims, args.ascension)) as pool:
        for done, row in enumerate(pool.map(_play, jobs, chunksize=1), start=1):
            rows.append(row)
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"\nwrote {args.out} ({time.perf_counter() - started:.0f}s)\n")

    totals: dict[str, float] = {}
    for row in rows:
        if row["arm"] != "baseline":
            continue
        for kind, count in row["decision_counts"].items():
            totals[kind] = totals.get(kind, 0.0) + count
    baseline_runs = sum(1 for r in rows if r["arm"] == "baseline")
    print("decisions per run, baseline: " + ", ".join(
        f"{k} {v / baseline_runs:.1f}"
        for k, v in sorted(totals.items(), key=lambda kv: -kv[1])) + "\n")
    print(summarize(rows), flush=True)


if __name__ == "__main__":
    main()
