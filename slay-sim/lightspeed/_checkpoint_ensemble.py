"""Average several checkpoints' logits at inference and measure the result.

v28, v31 and v37 do not separate on 600 paired seeds (05-model-lineage.md), and
the policy's decisions turn on a median 0.129-nat margin.  When the deciding
quantity is that small, much of it is estimation noise rather than preference --
so averaging independently-trained estimates of the same logit is the cheapest
available variance reduction, and it needs no training at all.

The three share an architecture (96/2/4) and a candidate ordering, since the
action list comes from the environment rather than from any checkpoint, so their
logits are directly averageable.

Evaluated on the SAME seeds as `runs/sharp_rebaseline_600seeds.jsonl`, so each
single-checkpoint arm is read from that file rather than replayed, and only the
ensemble arms cost anything.

Run from slay-sim/:
    python -m lightspeed._checkpoint_ensemble --runs 600 --workers 6
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

CHECKPOINTS = {
    "v28": "runs/whole_run_transformer_outcome_a20_v28.pt",
    "v31": "runs/whole_run_transformer_yield10x_a20_v31.pt",
    "v37": "runs/whole_run_transformer_postfix-trunc_a20_v37.pt",
}
REBASELINE = "runs/sharp_rebaseline_600seeds.jsonl"

_STATE: dict = {}


def _worker_init(paths_json: str, sims: int, ascension: int) -> None:
    import torch

    from .eval_whole_run_policy import load_policy
    from .search_config import DEFAULT_SEARCH_CONFIG_PATH

    torch.set_num_threads(1)
    _STATE["torch"] = torch
    _STATE["members"] = {
        name: load_policy(path, torch.device("cpu"))
        for name, path in json.loads(paths_json).items()
    }
    _STATE["sims"] = sims
    _STATE["ascension"] = ascension
    _STATE["search_config"] = DEFAULT_SEARCH_CONFIG_PATH


def _play(job: tuple[str, int]) -> dict:
    torch = _STATE["torch"]
    from .whole_run_env import RunConfig, WholeRunEnv

    arm, seed = job
    members = [_STATE["members"][name] for name in arm.split("+")]
    env = WholeRunEnv(RunConfig(
        ascension=_STATE["ascension"], combat_sims=_STATE["sims"],
        deterministic_combat=True, search_config_path=_STATE["search_config"]))
    obs = env.reset(seed)
    started = time.perf_counter()
    with torch.inference_mode():
        while (env.gc.outcome.name == "UNDECIDED"
               and env.steps < env.config.max_decisions):
            # Mean of log-probabilities, not of raw logits: each member's logits
            # carry its own arbitrary additive offset per state, which cancels
            # under log_softmax and would otherwise weight members unequally.
            total = None
            for member in members:
                logits, _ = member(obs)
                normalized = logits - logits.logsumexp(dim=-1)
                total = normalized if total is None else total + normalized
            action = int(torch.argmax(total))
            obs, _, done, _ = env.step(action)
            if done:
                break
    return {"arm": arm, "seed": seed, "floor": int(env.gc.floor_num),
            "act": int(env.gc.act), "outcome": str(env.gc.outcome),
            "seconds": round(time.perf_counter() - started, 3)}


def load_rebaseline(path: str) -> dict[str, dict[int, int]]:
    names = {os.path.basename(p): n for n, p in CHECKPOINTS.items()}
    out: dict[str, dict[int, int]] = collections.defaultdict(dict)
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if "seed" not in row or "floor" not in row:
                continue
            name = names.get(os.path.basename(row.get("checkpoint", "")))
            if name:
                out[name][row["seed"]] = row["floor"]
    return out


def paired(a: dict[int, int], b: dict[int, int]) -> tuple[float, float, int, str]:
    shared = sorted(set(a) & set(b))
    diffs = [b[s] - a[s] for s in shared]
    mean = statistics.mean(diffs)
    sem = statistics.stdev(diffs) / math.sqrt(len(diffs))
    wtl = (f"{sum(1 for d in diffs if d > 0)}/"
           f"{sum(1 for d in diffs if d == 0)}/"
           f"{sum(1 for d in diffs if d < 0)}")
    return mean, sem, len(diffs), wtl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=600)
    parser.add_argument("--seed-base", type=int, default=1_003_000)
    parser.add_argument("--sims", type=int, default=300)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--arms", default="v28+v31+v37,v31+v37")
    parser.add_argument("--no-rebaseline", action="store_true",
                        help="do not merge the stored 600-seed arms; "
                             "required when running on fresh seeds, or "
                             "a checkpoint would carry floors from two "
                             "different seed sets under one name")
    parser.add_argument("--out", default="runs/checkpoint_ensemble.jsonl")
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    jobs = [(arm, args.seed_base + offset)
            for arm in arms for offset in range(args.runs)]
    print(f"{len(arms)} ensemble arms x {args.runs} seeds = {len(jobs)} runs "
          f"at {args.sims} sims on {args.workers} workers", flush=True)

    rows: list[dict] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_worker_init,
            initargs=(json.dumps(CHECKPOINTS), args.sims,
                      args.ascension)) as pool:
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

    floors: dict[str, dict[int, int]] = (
        {} if args.no_rebaseline else load_rebaseline(REBASELINE))
    for row in rows:
        floors.setdefault(row["arm"], {})[row["seed"]] = row["floor"]

    print(f"{'arm':>14}  {'n':>4}  {'floor':>6}")
    for arm, series in floors.items():
        print(f"{arm:>14}  {len(series):>4}  "
              f"{statistics.mean(series.values()):>6.2f}")

    print(f"\n{'comparison':>26}  {'delta':>16}  {'t':>6}  {'W/T/L':>14}  {'n':>4}")
    for arm in arms:
        for reference in ("v37", "v31", "v28"):
            if reference not in floors or arm not in floors:
                continue
            mean, sem, count, wtl = paired(floors[reference], floors[arm])
            print(f"{arm + ' - ' + reference:>26}  {mean:>+9.2f} +/-{sem:.2f}  "
                  f"{mean / sem:>6.2f}  {wtl:>14}  {count:>4}")


if __name__ == "__main__":
    main()
