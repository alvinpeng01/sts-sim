"""Matched A20 combat comparison for native expectimax versus Silverbot.

Run this file in two separate interpreters because the projects expose two
incompatible pybind modules with the same ``slaythespire`` name:

  # ours
  PYTHONPATH="../sts_lightspeed/build;." python lightspeed/compare_tier_combat.py --engine ours

  # Silverbot (from sts-project root)
  PYTHONPATH="silverbot-reference/build" python slay-sim/lightspeed/compare_tier_combat.py --engine silver

``--suite focused`` reproduces the original six representative fights.
``--suite broad`` covers every calibrated normal, elite, and boss encounter
across all three acts.  Broad decks are generated deterministically from the
same weighted-Ironclad generator and tier resources used by training/eval, so
both engines see exactly the same deck samples without maintaining another
hand-written fixture list.  Neither side receives relics or potions: this
isolates combat search rather than item generation.
"""

from __future__ import annotations

import argparse
import json
from random import Random
import time
from pathlib import Path

import slaythespire as sts


# (tier name, encounter name, HP, starter removals, extra deck cards as
# ``CARD_ID`` names; a trailing '+' means upgraded). Generated with
# weighted_ironclad_deck(Random(9173 + tier_index), ...) at the calibrated
# ACT_TIER_RESOURCES budgets.
FOCUSED_CASES = [
    ("act1_basic", "JAW_WORM", 50, 0,
     "CORRUPTION+,OFFERING,OFFERING+,SHRUG_IT_OFF+,BATTLE_TRANCE+"),
    ("act1_elite", "GREMLIN_NOB", 70, 0,
     "IMPERVIOUS,FEED+,ANGER,UPPERCUT,INTIMIDATE+,BLOODLETTING+,WHIRLWIND,REAPER,FEED,WHIRLWIND+"),
    ("act2_basic", "CHOSEN", 90, 1,
     "FIEND_FIRE,IMPERVIOUS,WHIRLWIND,REAPER,TWIN_STRIKE+,REAPER,INFLAME+,FIEND_FIRE+,DARK_EMBRACE+,EVOLVE+,METALLICIZE,BLOOD_FOR_BLOOD,DARK_EMBRACE+,CORRUPTION,BLOOD_FOR_BLOOD,OFFERING+,REAPER+,BATTLE_TRANCE,FIEND_FIRE,HEAVY_BLADE,POWER_THROUGH,SHOCKWAVE"),
    ("act2_elite", "CENTURION_AND_HEALER", 110, 1,
     "SECOND_WIND,OFFERING,SHRUG_IT_OFF+,OFFERING,REAPER+,RAMPAGE+,ENTRENCH,FLAME_BARRIER+,BASH,SHOCKWAVE,POMMEL_STRIKE,REAPER,BLOOD_FOR_BLOOD+,CORRUPTION,BATTLE_TRANCE+,PUMMEL,BODY_SLAM+,BRUTALITY,GHOSTLY_ARMOR,PUMMEL,BLOOD_FOR_BLOOD,COMBUST+,WHIRLWIND,BATTLE_TRANCE+,BLOOD_FOR_BLOOD,DOUBLE_TAP,IMPERVIOUS+,DEFEND_RED+"),
    ("act3_basic", "ORB_WALKER", 100, 2,
     "CORRUPTION+,BLOODLETTING+,UPPERCUT+,FEED+,IMPERVIOUS,BATTLE_TRANCE+,LIMIT_BREAK+,RAGE,OFFERING,BASH+,SHOCKWAVE,FEED,POMMEL_STRIKE+,RAGE,FEED+,OFFERING+,RUPTURE,SEARING_BLOW,SECOND_WIND+,IMMOLATE+,FIEND_FIRE+,SHOCKWAVE,FLAME_BARRIER+,BLOOD_FOR_BLOOD,IRON_WAVE+,RAGE+,SENTINEL,CORRUPTION"),
    ("act3_elite", "SPHERIC_GUARDIAN", 115, 2,
     "BLOODLETTING+,FIEND_FIRE+,INTIMIDATE+,INFLAME+,IMMOLATE,DISARM+,DOUBLE_TAP+,FIEND_FIRE,SENTINEL,PUMMEL,BLOOD_FOR_BLOOD+,IMPERVIOUS+,DISARM+,ENTRENCH,PUMMEL+,POMMEL_STRIKE,FEEL_NO_PAIN,FEEL_NO_PAIN+,SHOCKWAVE,OFFERING+,CORRUPTION,PUMMEL+,INFLAME+,FIEND_FIRE+,SEVER_SOUL,BRUTALITY+,BRUTALITY,FLAME_BARRIER+,PERFECTED_STRIKE+,IMPERVIOUS,ANGER,POMMEL_STRIKE+"),
]


PICK_RATE_PATH = Path(__file__).with_name("data") / "ironclad_pick_rates.json"

# Self-contained mirror of env.py's ALL_ACT_TIER_GROUPS and the first four
# ACT_TIER_RESOURCES fields. The Silverbot binding cannot import env.py (it
# deliberately does not expose get_card_color), while this comparison must be
# executable unchanged by both engines. Keep this list in encounter-pool order
# so its rows remain directly comparable with our calibrated evaluation suite.
BROAD_GROUPS = (
    ("act1", "basic", ("JAW_WORM", "CULTIST", "TWO_LOUSE", "SMALL_SLIMES", "BLUE_SLAVER",
                         "RED_SLAVER", "LOOTER", "GREMLIN_GANG", "EXORDIUM_WILDLIFE", "EXORDIUM_THUGS")),
    ("act1", "elite", ("GREMLIN_NOB", "LAGAVULIN", "THREE_SENTRIES")),
    ("act1", "boss", ("THE_GUARDIAN", "HEXAGHOST", "SLIME_BOSS")),
    ("act2", "basic", ("THREE_BYRDS", "CHOSEN", "CHOSEN_AND_BYRDS", "SHELL_PARASITE",
                         "SHELLED_PARASITE_AND_FUNGI", "TWO_FUNGI_BEASTS")),
    ("act2", "elite", ("GREMLIN_LEADER", "BOOK_OF_STABBING", "CENTURION_AND_HEALER", "SNAKE_PLANT", "SNECKO")),
    ("act2", "boss", ("AUTOMATON", "COLLECTOR", "CHAMP")),
    ("act3", "basic", ("THREE_DARKLINGS", "ORB_WALKER", "WRITHING_MASS", "THREE_SHAPES",
                         "SPHERE_AND_TWO_SHAPES", "FOUR_SHAPES")),
    ("act3", "elite", ("NEMESIS", "REPTOMANCER", "SPHERIC_GUARDIAN")),
    ("act3", "boss", ("AWAKENED_ONE", "TIME_EATER", "DONU_AND_DECA")),
)
TIER_COMBAT_RESOURCES = {
    ("act1", "basic"): (50, 5, 0.3, 0), ("act1", "elite"): (70, 10, 0.3, 0),
    ("act1", "boss"): (85, 15, 0.3, 0), ("act2", "basic"): (90, 22, 0.4, 1),
    ("act2", "elite"): (110, 28, 0.4, 1), ("act2", "boss"): (120, 35, 0.4, 1),
    ("act3", "basic"): (100, 28, 0.5, 2), ("act3", "elite"): (115, 32, 0.5, 2),
    ("act3", "boss"): (130, 40, 0.5, 2),
}


def _weighted_deck_fixture(rng: Random, extra_cards: int, upgrade_chance: float) -> str:
    """Generate the shared weighted-Ironclad fixture without engine helpers.

    ``lightspeed.cards`` discovers the Ironclad pool through our native
    binding's ``get_card_color`` function. Silverbot's binding intentionally
    lacks that introspection API, so this small equivalent reads the same
    checked-in pick-rate table and adds its three starter-only cards.  Sorting
    by CardId reproduces the pool order used by ``IRONCLAD_CARD_IDS``; all
    cards in this regular-card pool are upgradable.
    """
    with PICK_RATE_PATH.open(encoding="utf-8") as f:
        pick_rates = json.load(f)
    names = [*pick_rates, "BASH", "DEFEND_RED", "STRIKE_RED"]
    pool = sorted((getattr(sts.CardId, name) for name in names), key=int)
    weights = [pick_rates.get(str(card_id).replace("CardId.", ""), 0.0) + 0.05
               for card_id in pool]
    picked = rng.choices(pool, weights=weights, k=extra_cards)
    return ",".join(
        str(card_id).replace("CardId.", "") + ("+" if rng.random() < upgrade_chance else "")
        for card_id in picked
    )


def broad_cases(decks_per_tier: int) -> list[tuple[str, str, str, int, int, str]]:
    """Every calibrated encounter with deterministic deck samples per tier.

    We intentionally generate the encoded deck before constructing a battle.
    This makes the fixture deterministic under either project's distinct
    ``slaythespire`` pybind module, and lets a row be re-run by its case id.
    """
    cases = []
    for tier_index, (act, tier, encounters) in enumerate(BROAD_GROUPS):
        hp, extra_cards, upgrade_chance, removals = TIER_COMBAT_RESOURCES[(act, tier)]
        tier_name = f"{act}_{tier}"
        for deck_index in range(decks_per_tier):
            deck_seed = 91_730 + 101 * tier_index + deck_index
            deck = _weighted_deck_fixture(Random(deck_seed), extra_cards, upgrade_chance)
            for encounter_name in encounters:
                case_id = f"{tier_name}_{encounter_name.lower()}_d{deck_index}"
                cases.append((case_id, tier_name, encounter_name, hp, removals, deck))
    return cases


def make_context(hp: int, removals: int, encoded_deck: str, seed: int, engine: str):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 20)
    gc.cur_hp = hp
    gc.max_hp = hp
    # The base Ironclad deck begins with Strikes, so removing index 0 is a
    # deterministic, shared approximation of the tier-calibrated removals.
    for _ in range(removals):
        gc.remove_card(0)
    for encoded in encoded_deck.split(","):
        upgraded = encoded.endswith("+")
        card_name = encoded[:-1] if upgraded else encoded
        if engine == "silver":
            card = sts.Card(getattr(sts.CardId, card_name), int(upgraded))
        else:
            card = sts.Card(getattr(sts.CardId, card_name))
        if upgraded and engine == "ours":
            card.upgrade()
        gc.obtain_card(card)
    return gc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("ours", "silver"), required=True)
    parser.add_argument("--suite", choices=("focused", "broad"), default="focused",
                        help="focused=6 legacy cases; broad=all calibrated encounters")
    parser.add_argument("--decks-per-tier", type=int, default=1,
                        help="deterministic deck samples per tier in the broad suite")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=5_000_000)
    parser.add_argument("--case", help="run one printed case id instead of the full suite")
    parser.add_argument("--verbosity", type=int, default=0,
                        help="Silverbot SearchAgent verbosity (0=quiet, 2=action trace)")
    parser.add_argument("--summary-only", action="store_true",
                        help="suppress per-encounter rows and print tier aggregates")
    parser.add_argument(
        "--leaf-mode", choices=("rollout", "truncated", "value"),
        default="rollout", help="ours-only native leaf evaluation mode")
    parser.add_argument(
        "--truncated-steps", type=int, default=3,
        help="heuristic actions before static value in truncated mode")
    args = parser.parse_args()
    if args.decks_per_tier < 1:
        parser.error("--decks-per-tier must be at least 1")

    if args.engine == "ours":
        from lightspeed.search_config import apply_search_config
        with open("lightspeed/tuned_search_params.json", encoding="utf-8") as f:
            apply_search_config(json.load(f))
        sts.set_leaf_eval_mode(args.leaf_mode, args.truncated_steps)

    cases = FOCUSED_CASES if args.suite == "focused" else broad_cases(args.decks_per_tier)
    # The focused suite predates case ids; keep its established tier label as
    # the selector and output id, while broad rows have unique case ids.
    if args.suite == "focused":
        cases = [(tier, tier, encounter, hp, removals, deck)
                 for tier, encounter, hp, removals, deck in cases]
    selected_cases = [case for case in cases if args.case is None or case[0] == args.case]
    if not selected_cases:
        parser.error(f"no case matched {args.case!r} in the {args.suite} suite")

    all_rows = []
    for case_id, tier, encounter_name, hp, removals, deck in selected_cases:
        encounter = getattr(sts.MonsterEncounter, encounter_name)
        rows = []
        started = time.perf_counter()
        for episode in range(args.episodes):
            gc = make_context(hp, removals, deck, args.seed_base + episode, args.engine)
            if args.engine == "ours":
                bc = sts.new_battle(gc, encounter)
                sts.native_playout_battle(bc, args.sims)
                won = bc.outcome == sts.BattleOutcome.PLAYER_VICTORY
                final_hp = bc.player_hp
            else:
                agent = sts.Agent()
                agent.simulation_count_base = args.sims
                agent.boss_simulation_multiplier = 1
                agent.verbosity_level = args.verbosity
                try:
                    agent.playout_battle(gc, encounter)
                except RuntimeError:
                    # Their standalone battle wrapper throws after a win when
                    # it enters the unwired reward callback; combat HP is
                    # already final at that point.
                    pass
                won = gc.cur_hp > 0
                final_hp = gc.cur_hp
            rows.append((won, final_hp))
        wins = sum(won for won, _ in rows)
        mean_hp = sum(final_hp for won, final_hp in rows if won) / max(1, wins)
        elapsed = time.perf_counter() - started
        result = {"case": case_id, "tier": tier, "encounter": encounter_name, "wins": wins,
                  "episodes": args.episodes, "mean_hp_on_win": mean_hp,
                  "ms_per_fight": elapsed * 1000 / args.episodes}
        all_rows.append(result)
        if not args.summary_only:
            print(json.dumps(result), flush=True)

    total_wins = sum(row["wins"] for row in all_rows)
    weighted_hp = sum(row["wins"] * row["mean_hp_on_win"] for row in all_rows)
    total_episodes = sum(row["episodes"] for row in all_rows)
    measured_ms_per_fight = (
        sum(row["ms_per_fight"] * row["episodes"] for row in all_rows) / max(1, total_episodes)
    )
    tier_summaries = {}
    for row in all_rows:
        tier = tier_summaries.setdefault(row["tier"], {"wins": 0, "episodes": 0, "hp_sum": 0.0})
        tier["wins"] += row["wins"]
        tier["episodes"] += row["episodes"]
        tier["hp_sum"] += row["wins"] * row["mean_hp_on_win"]
    tier_summaries = {
        tier: {"wins": f"{row['wins']}/{row['episodes']}",
               "mean_hp_on_win": row["hp_sum"] / max(1, row["wins"])}
        for tier, row in tier_summaries.items()
    }
    print(json.dumps({"summary": args.engine, "suite": args.suite,
                      "decks_per_tier": args.decks_per_tier, "wins": total_wins,
                      "episodes": total_episodes,
                      "mean_hp_on_win": weighted_hp / max(1, total_wins),
                      "measured_ms_per_fight": measured_ms_per_fight,
                      "tiers": tier_summaries}), flush=True)


if __name__ == "__main__":
    main()
