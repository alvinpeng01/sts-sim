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

**Why `--seeds` defaults to 3 rather than 1.** Pairing controls the fight, not
the search. `rollout_temperature` is tuned to 2.489, so the rollout SAMPLES
rather than taking an argmax, and any change that alters a trajectory breaks the
shared randomness from that point on. Measured on 2026-08-01 with the config held
IDENTICAL and only the seed set changed, on 250 paired fights:

    seedset A vs B   +1.60 +/- 0.68   t = +2.34
    seedset A vs C   +1.01 +/- 0.68   t = +1.48
    seedset B vs C   -0.59 +/- 0.69   t = -0.85

Changing nothing produced t = 2.34, and 151 of 250 fights differed. The per-fight
sd of the paired difference is **10.82 HP**, so a single-seed run has SE 0.88 at
n=150 and 0.48 at n=500 before any real effect is measured. Most parameters
tested on this benchmark move 1-2 HP, which is the same size. Averaging k seeds
divides that floor by sqrt(k) for k times the compute, and the compute here is
seconds.

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
from .paths import HUMAN_BENCHMARK

BENCHMARK_PATH = str(HUMAN_BENCHMARK)


def score(fights, sims: int, overrides: dict[str, float],
          seeds: int = 3) -> tuple[list[float], int]:
    """Objective per fight under the shipped config plus `overrides`.

    Re-applies the shipped config every call rather than setting only the keys
    that changed: `set_search_params` is a partial update over unlocked global
    state, so an arm would otherwise inherit whatever the previous arm left
    behind. That is the same reason `apply_search_config` resets first.

    Each fight is averaged over `seeds` search seeds. That is not a refinement,
    it is the difference between a measurement and a coin flip -- see the
    module docstring's noise-floor numbers.
    """
    ensure_search_config(DEFAULT_SEARCH_CONFIG_PATH)
    if overrides:
        sts.set_search_params(overrides)
    values, deaths = [], 0
    for index, record in enumerate(fights):
        per_seed = []
        for seed in range(seeds):
            battle, _ = build_battle(
                record["deck"], record["relics"], record["cur_hp"],
                record["max_hp"],
                getattr(sts.MonsterEncounter, record["encounter"]),
                20, record["act"], record.get("potions", ()))
            # Seeds are offset far apart so two arms sharing `index` still share
            # their seed set -- the pairing survives, only the sampling widens.
            damage, outcome = play(battle, sims, index + seed * 1_000_003)
            died = outcome != sts.BattleOutcome.PLAYER_VICTORY
            deaths += died
            per_seed.append(record["human_damage"]
                            - (record["cur_hp"] if died else damage))
        values.append(statistics.mean(per_seed))
    return values, deaths / max(1, seeds)


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
    parser.add_argument("--seeds", type=int, default=3,
                        help="search seeds averaged per fight; 1 reproduces the "
                             "old single-seed behaviour and its 0.88 HP noise floor")
    args = parser.parse_args()

    live = sts.get_search_params()
    if args.param not in live:
        raise KeyError(f"{args.param} is not a search parameter; "
                       f"get_search_params() has {len(live)} keys")

    with open(BENCHMARK_PATH, encoding="utf-8") as handle:
        records = json.load(handle)
    fights = [r for r in records if r["split"] == args.split][: args.limit]

    print(f"{args.param}: {len(fights)} {args.split} fights at {args.sims} sims, "
          f"{args.seeds} seed(s) per fight, paired")
    start = time.time()
    baseline, base_deaths = score(fights, args.sims, {}, args.seeds)
    print(f"  {'shipped':>24s}  {statistics.mean(baseline):+8.3f}   "
          f"deaths {base_deaths:5.1f}   ({time.time()-start:.0f}s)")

    for raw in args.values.split(","):
        value = float(raw)
        values, deaths = score(fights, args.sims, {args.param: value},
                               args.seeds)
        deltas = [a - b for a, b in zip(values, baseline)]
        mean = statistics.mean(deltas)
        stderr = statistics.stdev(deltas) / math.sqrt(len(deltas))
        changed = sum(1 for d in deltas if d)
        print(f"  {args.param + '=' + raw:>24s}  {statistics.mean(values):+8.3f}   "
              f"deaths {deaths:5.1f}   delta {mean:+6.2f} +/- {stderr:.2f} "
              f"(t={mean/stderr if stderr else 0:+5.2f})  changed {changed}/{len(deltas)}")

    print("\nA delta inside one standard error is a null, not a small win. If nothing "
          "clears it, record the null -- the dead-ends table in "
          "docs/03-combat-search.md exists so it is not re-run.")


if __name__ == "__main__":
    main()
