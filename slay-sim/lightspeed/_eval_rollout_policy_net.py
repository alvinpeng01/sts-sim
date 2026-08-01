"""Measure a distilled rollout-policy net on the human benchmark.

Top-1 accuracy against the search's picks is NOT the deliverable. A loaded net is
evaluated for every legal action at every rollout step, so it buys down the
simulation count directly -- measured 1.93x for a 4-unit hidden layer, 4.97x for
16, 6.75x for 32x32. Against the sim curve on this benchmark (100 sims -1.28,
20 sims -5.36, 5 sims -12.29, 1 sim -31.83) that means a net has to recover
roughly 1.5 HP at width 4, or 4.1 HP at width 16, purely to break even.

So this compares at MATCHED WALL CLOCK, not matched simulations, which is the
only comparison that decides whether to ship one:

    baseline    : no net, `--sims` simulations
    candidate   : net loaded, sims scaled down by the net's measured cost

Both arms play identical fights with identical search seeds (CRN), which is only
meaningful because the Gumbel seeding fix made the search reproducible -- before
that, sibling arms silently diverged. See docs/07-known-issues.md.

    python -m lightspeed._eval_rollout_policy_net --net runs/rollout_policy_net_h4.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import slaythespire as sts

from ._human_deck_combat import build_battle, play
from .search_config import DEFAULT_SEARCH_CONFIG_PATH, ensure_search_config

BENCHMARK_PATH = r"C:\Users\Alvin\grok\sts-project\slay-sim\runs\human_fight_benchmark_100.json"


def run(fights, sims: int) -> list[float]:
    """Objective per fight under whatever search params are currently set."""
    out = []
    for index, rec in enumerate(fights):
        bc, _ = build_battle(rec["deck"], rec["relics"], rec["cur_hp"],
                             rec["max_hp"],
                             getattr(sts.MonsterEncounter, rec["encounter"]),
                             20, rec["act"], rec.get("potions", ()))
        damage, outcome = play(bc, sims, index)
        died = outcome != sts.BattleOutcome.PLAYER_VICTORY
        out.append(rec["human_damage"] - (rec["cur_hp"] if died else damage))
    return out


def summarise(label: str, values, seconds: float, sims: int) -> float:
    mean = statistics.mean(values)
    print(f"{label:28s}{mean:+8.3f}   {sims:5d} sims  {seconds:6.1f}s")
    return mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--weights", default="0.5,1,2,4",
                        help="policy_net_weight values to sweep")
    args = parser.parse_args()

    with open(BENCHMARK_PATH, encoding="utf-8") as handle:
        records = json.load(handle)
    fights = [r for r in records if r["split"] == args.split][: args.limit]
    with open(args.net, encoding="utf-8") as handle:
        net = json.load(handle)
    hidden = [len(layer["b"]) for layer in net["layers"][:-1]]
    print(f"{args.net}  hidden {hidden}  on {len(fights)} {args.split} fights\n")

    ensure_search_config(DEFAULT_SEARCH_CONFIG_PATH)
    sts.set_search_params({"policy_net_weight": 0.0})
    start = time.time()
    baseline = run(fights, args.sims)
    base_seconds = time.time() - start
    base_mean = summarise("baseline (no net)", baseline, base_seconds, args.sims)

    # Measure this net's actual cost rather than assuming it, then give the
    # candidate the simulation budget that costs the SAME wall clock.
    sts.load_policy_net(net)
    sts.set_search_params({"policy_net_weight": 1.0})
    probe_n = max(20, len(fights) // 10)
    probe_start = time.time()
    run(fights[:probe_n], args.sims)
    cost = ((time.time() - probe_start) / probe_n) / (base_seconds / len(fights))
    matched = max(1, int(round(args.sims / cost)))
    print(f"\nmeasured net cost {cost:.2f}x -> matched-wall-clock budget "
          f"{matched} sims (vs {args.sims})\n")

    best = (None, -math.inf)
    for raw in args.weights.split(","):
        weight = float(raw)
        sts.set_search_params({"policy_net_weight": weight})
        start = time.time()
        values = run(fights, matched)
        mean = summarise(f"net w={weight:g} (matched)", values, time.time() - start,
                         matched)
        deltas = [c - b for c, b in zip(values, baseline)]
        delta = statistics.mean(deltas)
        stderr = statistics.stdev(deltas) / math.sqrt(len(deltas))
        print(f"{'':28s}vs baseline {delta:+.2f} +/- {stderr:.2f} HP "
              f"(t={delta/stderr:+.2f})")
        if mean > best[1]:
            best = (weight, mean)

    sts.set_search_params({"policy_net_weight": 0.0})
    print(f"\nbest weight {best[0]} at {best[1]:+.3f} vs baseline {base_mean:+.3f}")
    print("Ship only if this beats the baseline at MATCHED WALL CLOCK -- a net that "
          "wins at matched sims but loses here is a net that costs more than it "
          "returns.")


if __name__ == "__main__":
    main()
