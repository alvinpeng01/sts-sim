"""Summarize paired-seed evaluation .jsonl files.

`eval_whole_run_policy.py` writes one row per (checkpoint, seed) plus a trailing
summary object per checkpoint, but it reports only per-arm means. The number that
actually decides a promotion is the *paired* delta: the same seed played by two
checkpoints, differenced per seed, so the seed-to-seed variance — which dominates
at sd ~8 floors — cancels.

Run from slay-sim/:
    python -m lightspeed._eval_summary runs/some_eval.jsonl [more.jsonl ...]

The first checkpoint encountered in a file is the baseline; every other arm is
differenced against it on the seeds they share.
"""
from __future__ import annotations

import collections
import json
import math
import os
import statistics
import sys


def load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            # Trailing {"summary": {...}} objects carry no per-seed floor.
            if "seed" in row and "floor" in row:
                rows.append(row)
    return rows


def summarize(path: str) -> None:
    rows = load(path)
    by_checkpoint: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    for row in rows:
        by_checkpoint[row.get("checkpoint", "?")][row["seed"]] = row

    print("=" * 96)
    print(f"{os.path.basename(path)}  ({len(rows)} rows, {len(by_checkpoint)} arms)")
    print("=" * 96)
    if not by_checkpoint:
        print("  no per-seed rows")
        return

    arms = list(by_checkpoint)
    baseline = arms[0]
    for checkpoint in arms:
        seeds = by_checkpoint[checkpoint]
        floors = [row["floor"] for row in seeds.values()]
        if not floors:
            continue
        wins = sum(1 for row in seeds.values()
                   if "VICTORY" in str(row.get("outcome", "")))
        act3 = sum(1 for row in seeds.values() if int(row.get("act", 0)) >= 3)
        mean = statistics.mean(floors)
        sem = (statistics.stdev(floors) / math.sqrt(len(floors))
               if len(floors) > 1 else 0.0)
        line = (f"  {checkpoint:56s} n={len(floors):4d} floor={mean:6.2f} "
                f"±{sem:4.2f} act3+={act3:3d} wins={wins}")
        if checkpoint != baseline:
            shared = set(by_checkpoint[baseline]) & set(seeds)
            deltas = [seeds[s]["floor"] - by_checkpoint[baseline][s]["floor"]
                      for s in shared]
            if deltas:
                delta_mean = statistics.mean(deltas)
                delta_sem = (statistics.stdev(deltas) / math.sqrt(len(deltas))
                             if len(deltas) > 1 else 0.0)
                better = sum(1 for d in deltas if d > 0)
                tied = sum(1 for d in deltas if d == 0)
                worse = sum(1 for d in deltas if d < 0)
                t = delta_mean / delta_sem if delta_sem else float("nan")
                line += (f"  | paired {delta_mean:+6.2f} ±{delta_sem:4.2f} "
                         f"t={t:5.2f} W/T/L={better}/{tied}/{worse} (n={len(deltas)})")
        print(line)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for path in sys.argv[1:]:
        summarize(path)


if __name__ == "__main__":
    main()
