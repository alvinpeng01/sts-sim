"""Paired, encounter-level A20 screen for one native-search parameter.

This keeps parameter research out of the active checkpoint: every candidate
uses identical generated decks, combat seeds, and search seeds.  By default it
tests only hallway and elite fights, where surviving HP is the full-run
resource we are trying to improve.

Example (from ``slay-sim``):

    PYTHONPATH=".;../sts_lightspeed/build" python -m lightspeed.evaluate_search_grid \
        --param silver_card_play_prior_weight --values 1 3 5
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


CHECKPOINT_PATH = Path(__file__).with_name("tuned_search_params.json")
NORMAL_ELITE_ENCOUNTERS = (
    "JAW_WORM", "TWO_LOUSE", "GREMLIN_GANG", "EXORDIUM_THUGS",
    "CHOSEN", "SHELLED_PARASITE_AND_FUNGI", "THREE_DARKLINGS", "ORB_WALKER",
    "GREMLIN_NOB", "THREE_SENTRIES", "CENTURION_AND_HEALER", "SPHERIC_GUARDIAN",
)
ALL_CALIBRATED_ENCOUNTERS = NORMAL_ELITE_ENCOUNTERS + (
    "THE_GUARDIAN", "AUTOMATON", "TIME_EATER",
)
ACT1_EASY_POOL_ENCOUNTERS = (
    "CULTIST", "JAW_WORM", "TWO_LOUSE", "SMALL_SLIMES",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", required=True, help="native set_search_params key")
    parser.add_argument("--values", nargs="+", type=float, required=True)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed-base", type=int, default=60_000)
    parser.add_argument("--include-bosses", action="store_true",
                        help="also screen the three calibrated boss fights")
    parser.add_argument("--act1-easy-pool", action="store_true",
                        help="screen only the first-three-fights Act 1 pool")
    parser.add_argument("--boss-value", type=float,
                        help="hold boss_silver_card_play_prior_weight at this value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import slaythespire as sts
    from lightspeed.cards import weighted_ironclad_deck
    from lightspeed.env import IroncladFightEnv, build_full_encounter_resources
    from lightspeed.search_config import apply_search_config

    with CHECKPOINT_PATH.open(encoding="utf-8") as f:
        active = json.load(f)
    apply_search_config(active)
    # Defensive reset of experimental, process-global research controls.
    sts.set_early_act_card_biases({})
    sts.set_search_params({
        "attack_damage_score_weight": 0.0,
        "direct_block_score_weight": 0.0,
        "self_damage_score_penalty": 0.0,
    })
    resources = build_full_encounter_resources()
    encounter_names = (ACT1_EASY_POOL_ENCOUNTERS if args.act1_easy_pool else
                       ALL_CALIBRATED_ENCOUNTERS if args.include_bosses else
                       NORMAL_ELITE_ENCOUNTERS)
    envs = [
        (name, IroncladFightEnv(
            encounter=getattr(sts.MonsterEncounter, name),
            encounter_resources=resources,
            deck_generator=weighted_ironclad_deck,
            ascension=20,
        ))
        for name in encounter_names
    ]

    for value in args.values:
        sts.set_search_params({args.param: value})
        if args.boss_value is not None:
            sts.set_search_params({"boss_silver_card_play_prior_weight": args.boss_value})
        totals = {"wins": 0, "hp_fraction": 0.0, "score": 0.0, "fights": 0}
        by_encounter: dict[str, dict[str, float]] = defaultdict(
            lambda: {"wins": 0, "hp_fraction": 0.0, "fights": 0}
        )
        for encounter_index, (name, env) in enumerate(envs):
            for episode in range(args.episodes):
                seed = args.seed_base + episode
                env.reset(seed=seed)
                done = False
                step = 0
                info = None
                while not done and step < 150:
                    search_seed = (seed << 32) ^ (encounter_index << 16) ^ step
                    action, _ = sts.run_mcts_search(env.bc, args.sims, None, search_seed)
                    _, _, done, info = env.step(action)
                    step += 1
                won = info["outcome"] == sts.BattleOutcome.PLAYER_VICTORY
                hp_fraction = (info["player_hp"] / env.bc.player_max_hp) if won else 0.0
                row = by_encounter[name]
                row["wins"] += int(won)
                row["hp_fraction"] += hp_fraction
                row["fights"] += 1
                totals["wins"] += int(won)
                totals["hp_fraction"] += hp_fraction
                totals["score"] += int(won) + 2.0 * hp_fraction
                totals["fights"] += 1
        summary = {
            "param": args.param,
            "value": value,
            "sims": args.sims,
            "episodes": args.episodes,
            "wins": f"{totals['wins']}/{totals['fights']}",
            "mean_hp_fraction_all": round(totals["hp_fraction"] / totals["fights"], 4),
            "objective": round(totals["score"] / totals["fights"], 4),
            "encounters": {
                name: {
                    "wins": f"{int(row['wins'])}/{int(row['fights'])}",
                    "mean_hp_fraction_all": round(row["hp_fraction"] / row["fights"], 4),
                }
                for name, row in by_encounter.items()
            },
        }
        print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
