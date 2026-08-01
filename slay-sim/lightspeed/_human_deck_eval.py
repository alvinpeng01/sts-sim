"""Validate a tuned params file on the HELD-OUT split of the human benchmark.

`tune_search_human.py` optimizes against the 388 train fights (14 runs). Its own
reported score is therefore a selection high and cannot say whether the params
generalize -- 35 parameters against a fixed fight set will fit that set. This
scores them on the 172 test fights (6 runs the tuner never saw), paired against
the shipped config on identical fights and identical search RNG.

Report it the way the tuning objective is defined -- mean (human_damage -
our_hp_paid), where 0 is human parity -- plus the plain damage/death numbers,
because a params set that trades deaths for damage (or the reverse) should be
visible rather than folded into one scalar.

    python -m lightspeed._human_deck_eval --params runs/tuned_search_human.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics

import slaythespire as sts

from ._human_deck_combat import build_battle, play
from .search_config import DEFAULT_SEARCH_CONFIG_PATH, ensure_search_config

# The 100-run, potion-inclusive benchmark with a 60/20/20 split by run.
# The 20-run file this used to point at had no potions (worth 4.7 HP) and
# only a 14/6 train/test split.
BENCHMARK_PATH = r"C:\Users\Alvin\grok\sts-project\slay-sim\runs\human_fight_benchmark_100.json"


def run_split(fights, sims: int, ascension: int) -> list[dict]:
    """Play every fight under whatever params are currently set."""
    results = []
    for index, rec in enumerate(fights):
        try:
            bc, _ = build_battle(rec["deck"], rec["relics"], rec["cur_hp"],
                                 rec["max_hp"],
                                 getattr(sts.MonsterEncounter, rec["encounter"]),
                                 ascension, rec["act"], rec.get("potions", ()))
        except Exception as error:  # noqa: BLE001 - reported, not hidden
            results.append({**rec, "error": str(error)})
            continue
        # Same CRN scheme as the tuner, and a FIXED seed_base of 0 for both arms,
        # so the two configs face identical randomness on identical fights.
        damage, outcome = play(bc, sims, index)
        died = outcome != sts.BattleOutcome.PLAYER_VICTORY
        results.append({**rec, "died": died,
                        "hp_paid": rec["cur_hp"] if died else damage,
                        "our_damage": None if died else damage})
    return results


def summarise(results: list[dict]) -> dict:
    played = [r for r in results if "died" in r]
    won = [r for r in played if not r["died"]]
    return {
        "n": len(played),
        "deaths": len(played) - len(won),
        "objective": statistics.mean(r["human_damage"] - r["hp_paid"] for r in played),
        "our_damage_on_wins": statistics.mean(r["our_damage"] for r in won) if won else float("nan"),
        "human_damage_on_wins": statistics.mean(r["human_damage"] for r in won) if won else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True, help="tuned params JSON")
    parser.add_argument("--split", default="test", choices=["test", "train", "all"])
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(BENCHMARK_PATH, encoding="utf-8") as handle:
        records = json.load(handle)
    fights = [r for r in records
              if args.split == "all" or r["split"] == args.split]
    with open(args.params, encoding="utf-8") as handle:
        tuned = json.load(handle)
    print(f"{len(fights)} {args.split} fights | tuned artifact: "
          f"score {tuned.get('score')} at generation {tuned.get('generation')}")

    # Baseline FIRST, so the shipped config is measured before any set_search_params
    # call in this process can perturb it.
    ensure_search_config(DEFAULT_SEARCH_CONFIG_PATH)
    shipped = run_split(fights, args.sims, args.ascension)

    sts.set_search_params(tuned["params"])
    candidate = run_split(fights, args.sims, args.ascension)

    base, cand = summarise(shipped), summarise(candidate)
    print(f"\n{'':22s}{'n':>5s}{'deaths':>8s}{'objective':>11s}{'ourDmg':>9s}{'humanDmg':>10s}")
    for name, stats in (("shipped", base), ("tuned", cand)):
        print(f"{name:22s}{stats['n']:5d}{stats['deaths']:8d}"
              f"{stats['objective']:11.2f}{stats['our_damage_on_wins']:9.1f}"
              f"{stats['human_damage_on_wins']:10.1f}")

    # Paired, because both arms played the same fights with the same RNG.
    pairs = [(c["human_damage"] - c["hp_paid"]) - (s["human_damage"] - s["hp_paid"])
             for s, c in zip(shipped, candidate)
             if "died" in s and "died" in c]
    mean = statistics.mean(pairs)
    stderr = statistics.stdev(pairs) / math.sqrt(len(pairs)) if len(pairs) > 1 else float("nan")
    print(f"\npaired delta (tuned - shipped): {mean:+.2f} +/- {stderr:.2f} HP  "
          f"(t={mean / stderr:+.2f}, n={len(pairs)})")
    print(f"better on {sum(1 for p in pairs if p > 0)} fights, "
          f"worse on {sum(1 for p in pairs if p < 0)}, "
          f"unchanged on {sum(1 for p in pairs if p == 0)}")
    if args.split == "train":
        print("\nNOTE: this is the split the tuner optimized -- not evidence of "
              "generalization. Run with --split test for that.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            for s, c in zip(shipped, candidate):
                s.pop("deck", None)
                c.pop("deck", None)
                handle.write(json.dumps({"shipped": s, "tuned": c}) + "\n")


if __name__ == "__main__":
    main()
