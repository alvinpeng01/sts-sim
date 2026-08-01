"""Paired A/B of a single search parameter on the human benchmark.

The gap this fills: `tune_search_human.py` searches 47 parameters at once and
`_human_deck_eval.py` scores a whole params file, but nothing measures ONE
parameter against the shipped config with a paired standard error. That is the
measurement most single-change hypotheses need, and doing it ad hoc is how the
potion-discard arm came to look like +1.0 HP on a 120-fight val spot-check before
measuring **-0.58 +/- 0.56** on 500 train fights.

So this defaults to the TRAIN split. The project's own rule, from
`docs/03-combat-search.md`: sweep on train or val, keep test for a single
pre-registered setting. Reading test after trying three values manufactures a
~2 HP improvement for free -- with ~100 boss fights per split the per-boss
standard error is ~1.4 HP, and best-of-N does the rest.

Every arm plays the same fights with the same search seeds, so the deltas are
paired and the reported standard error is computed from the observed paired
differences rather than assumed. That is only meaningful because the Gumbel
seeding fix made the search reproducible; before it, sibling arms diverged
silently (see `docs/07-known-issues.md`).

    python -m lightspeed._param_ab --param seq_halving_candidates --values 4,6,8
    python -m lightspeed._param_ab --param mast_weight --values 0.5,1,2 --limit 500
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


def score(fights, sims: int, overrides: dict[str, float]) -> tuple[list[float], int]:
    """Objective per fight under the shipped config plus `overrides`.

    Re-applies the shipped config every call rather than setting only the keys
    that changed: `set_search_params` is a partial update over unlocked global
    state, so an arm would otherwise inherit whatever the previous arm left
    behind. That is the same reason `apply_search_config` resets first.
    """
    ensure_search_config(DEFAULT_SEARCH_CONFIG_PATH)
    if overrides:
        sts.set_search_params(overrides)
    values, deaths = [], 0
    for index, record in enumerate(fights):
        battle, _ = build_battle(
            record["deck"], record["relics"], record["cur_hp"], record["max_hp"],
            getattr(sts.MonsterEncounter, record["encounter"]),
            20, record["act"], record.get("potions", ()))
        damage, outcome = play(battle, sims, index)
        died = outcome != sts.BattleOutcome.PLAYER_VICTORY
        deaths += died
        values.append(record["human_damage"] - (record["cur_hp"] if died else damage))
    return values, deaths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", required=True,
                        help="snake_case key from get_search_params()")
    parser.add_argument("--values", required=True,
                        help="comma-separated values to test against the shipped config")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"],
                        help="train by default; see this module's docstring before "
                             "choosing test")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--sims", type=int, default=100)
    args = parser.parse_args()

    live = sts.get_search_params()
    if args.param not in live:
        raise KeyError(f"{args.param} is not a search parameter; "
                       f"get_search_params() has {len(live)} keys")

    with open(BENCHMARK_PATH, encoding="utf-8") as handle:
        records = json.load(handle)
    fights = [r for r in records if r["split"] == args.split][: args.limit]

    print(f"{args.param}: {len(fights)} {args.split} fights at {args.sims} sims, "
          f"paired (common random numbers)")
    start = time.time()
    baseline, base_deaths = score(fights, args.sims, {})
    print(f"  {'shipped':>24s}  {statistics.mean(baseline):+8.3f}   "
          f"deaths {base_deaths:3d}   ({time.time()-start:.0f}s)")

    for raw in args.values.split(","):
        value = float(raw)
        values, deaths = score(fights, args.sims, {args.param: value})
        deltas = [a - b for a, b in zip(values, baseline)]
        mean = statistics.mean(deltas)
        stderr = statistics.stdev(deltas) / math.sqrt(len(deltas))
        changed = sum(1 for d in deltas if d)
        print(f"  {args.param + '=' + raw:>24s}  {statistics.mean(values):+8.3f}   "
              f"deaths {deaths:3d}   delta {mean:+6.2f} +/- {stderr:.2f} "
              f"(t={mean/stderr if stderr else 0:+5.2f})  changed {changed}/{len(deltas)}")

    print("\nA delta inside one standard error is a null, not a small win. If nothing "
          "clears it, record the null -- the dead-ends table in "
          "docs/03-combat-search.md exists so it is not re-run.")


if __name__ == "__main__":
    main()
