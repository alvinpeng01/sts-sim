"""Matched whole-run wall-clock benchmark for our hybrid agent and Silverbot.

Run separately for each engine because both expose an incompatible module named
``slaythespire``.  For a fair combat-budget comparison, use the same ``--sims``
and leave ``--boss-multiplier`` at 1; Silverbot's production default is 3.

    PYTHONPATH="../sts_lightspeed/build;." python -m lightspeed.benchmark_full_runs --engine ours
    PYTHONPATH="../silverbot-reference/build;." python -m lightspeed.benchmark_full_runs --engine silver
"""

from __future__ import annotations

import argparse
import json
import time

import slaythespire as sts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("ours", "silver"), required=True)
    parser.add_argument("--sims", type=int, default=1_000,
                        help="combat simulations per decision")
    parser.add_argument("--boss-multiplier", type=float, default=1.0,
                        help="Silverbot-only boss budget multiplier (1=fair match)")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=73_000)
    parser.add_argument("--ascension", type=int, default=20)
    args = parser.parse_args()

    if args.engine == "ours":
        from lightspeed.search_config import apply_search_config
        with open("lightspeed/tuned_search_params.json", encoding="utf-8") as f:
            apply_search_config(json.load(f))

    rows = []
    for run_index in range(args.runs):
        seed = args.seed_base + run_index
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, args.ascension)
        agent = sts.Agent()
        started = time.perf_counter()
        if args.engine == "ours":
            # This is the production hybrid: our native MCTS in combat and
            # the same established out-of-combat policy for both comparisons.
            agent.playout_hybrid(gc, args.sims)
        else:
            agent.simulation_count_base = args.sims
            agent.boss_simulation_multiplier = args.boss_multiplier
            agent.verbosity_level = 0
            agent.playout(gc)
        elapsed = time.perf_counter() - started
        row = {
            "seed": seed,
            "outcome": str(gc.outcome),
            "floor": gc.floor_num,
            "act": gc.act,
            "hp": gc.cur_hp,
            "max_hp": gc.max_hp,
            "elapsed_seconds": round(elapsed, 3),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    wins = sum(row["outcome"].endswith("PLAYER_VICTORY") for row in rows)
    print(json.dumps({
        "summary": args.engine,
        "runs": args.runs,
        "sims": args.sims,
        "boss_multiplier": args.boss_multiplier if args.engine == "silver" else 1.0,
        "wins": wins,
        "mean_floor": sum(row["floor"] for row in rows) / args.runs,
        "mean_seconds": sum(row["elapsed_seconds"] for row in rows) / args.runs,
    }), flush=True)


if __name__ == "__main__":
    main()
