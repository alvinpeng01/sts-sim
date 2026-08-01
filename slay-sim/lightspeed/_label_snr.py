"""Measure label signal-to-noise, and compare label sets head to head.

The project's central measured problem is that labels cannot reliably separate
the best action from the runner-up: on v31, median paired SNR ~1.0 with ~47% of
labels below 1.0. Three separate attempts to fit those labels harder (v32, v33,
v36) all produced worse policies, which is what a noisy target predicts.

So any change to label generation -- truncation, adaptive allocation, more
rollouts, a combat surrogate -- should be judged on SNR per unit compute, not on
validation NLL or on how well the model fits. This makes that one command.

Requires `per_rollout_scores` on the rows, which generate_whole_run_rollouts.py
stores by default.

    python -m lightspeed._label_snr runs/a.pt runs/b.pt
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch


def summarise(path: str) -> dict | None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows = [r for r in payload["rows"] if "per_rollout_scores" in r]
    if not rows:
        print(f"{path}: no per_rollout_scores (generated before it was stored)")
        return None
    absolute, paired, gaps = [], [], []
    for row in rows:
        scores = np.asarray(row["per_rollout_scores"], dtype=float)
        if scores.ndim != 2 or scores.shape[0] < 2:
            continue
        means = scores.mean(1)
        order = np.argsort(means)[::-1]
        best, runner = order[0], order[1]
        gap = float(means[best] - means[runner])
        gaps.append(gap)
        # Absolute: each arm's own SE, as if the arms were independent.
        se_abs = float((scores.std(1, ddof=1) / math.sqrt(scores.shape[1])).mean())
        absolute.append(gap / max(1e-9, se_abs))
        # Paired: sibling branches share a rollout seed (common random numbers),
        # so the SE of their *difference* is what governs the label's argmax.
        difference = scores[best] - scores[runner]
        se_pair = float(difference.std(ddof=1) / math.sqrt(len(difference)))
        paired.append(gap / max(1e-9, se_pair))
    if not paired:
        return None
    absolute, paired, gaps = map(np.asarray, (absolute, paired, gaps))
    # A merged file written by parallel_generate_whole_run_rollouts.py stores only
    # that wrapper's own args at the top level; the generation parameters live in
    # the per-shard metadata it collected. Reading the top level alone reports
    # every sharded dataset as "rollouts 0 / terminal", which is wrong.
    meta = payload.get("metadata", {})
    shards = meta.get("shard_metadata") or []
    if shards and "rollouts" not in meta:
        meta = {**shards[0], **{k: v for k, v in meta.items() if k != "shard_metadata"}}
    stats = {
        "path": path,
        "n": len(paired),
        "rollouts": int(meta.get("rollouts", 0)),
        "truncate_after": int(meta.get("truncate_after", 0)),
        "seconds": float(meta.get("seconds", 0.0)),
        "median_abs": float(np.median(absolute)),
        "median_paired": float(np.median(paired)),
        "frac_below_1": float((paired < 1.0).mean()),
        "mean_gap": float(gaps.mean()),
    }
    # SNR per unit compute is the quantity that matters: a change that halves
    # noise while tripling cost is not an improvement.
    if stats["seconds"] > 0 and stats["n"]:
        per_label = stats["seconds"] / stats["n"]
        stats["seconds_per_label"] = per_label
        stats["snr_per_sqrt_second"] = stats["median_paired"] / math.sqrt(per_label)
    return stats


def report(stats: dict) -> None:
    print(f"\n{stats['path']}")
    print(f"  labels                {stats['n']}")
    print(f"  rollouts / truncate   {stats['rollouts']} / "
          f"{stats['truncate_after'] or 'terminal'}")
    print(f"  median paired SNR     {stats['median_paired']:.3f}")
    print(f"  median absolute SNR   {stats['median_abs']:.3f}")
    print(f"  labels below SNR 1    {100 * stats['frac_below_1']:.1f}%")
    print(f"  mean gap (best-2nd)   {stats['mean_gap']:.3f}")
    if "seconds_per_label" in stats:
        print(f"  seconds / label       {stats['seconds_per_label']:.1f}")
        print(f"  SNR per sqrt(second)  {stats['snr_per_sqrt_second']:.3f}"
              "   <- the number to compare")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()
    collected = [s for s in (summarise(p) for p in args.datasets) if s]
    for stats in collected:
        report(stats)
    if len(collected) >= 2:
        base = collected[0]
        print("\nrelative to the first dataset:")
        for stats in collected[1:]:
            snr = stats["median_paired"] / max(1e-9, base["median_paired"])
            line = f"  {stats['path']}: paired SNR x{snr:.2f}"
            if "snr_per_sqrt_second" in stats and "snr_per_sqrt_second" in base:
                efficiency = (stats["snr_per_sqrt_second"]
                              / max(1e-9, base["snr_per_sqrt_second"]))
                line += f", SNR per unit compute x{efficiency:.2f}"
            print(line)


if __name__ == "__main__":
    main()
