"""How much is our combat search worth WITHOUT knowing the draw order?

`run_mcts_search` roots its tree in a full copy of the live `BattleContext`,
including the concrete ordered draw pile, so every simulation draws exactly what
reality will draw. Silverbot found the same defect in their engine on 2026-06-03
("WE'VE BEEN CHEATING: draw-order clairvoyance ~ +34pp") and removed it.

Two reasons the number matters even though combat is saturated for floors:

- **Live play cannot cheat.** `build_battle_context` takes `draw_pile_cards` as
  externally reported state from CommunicationMod, and the real game does not
  expose the shuffle order. Whatever order the bridge supplies is arbitrary, and
  the search plans around draws that will not happen. Every simulation number in
  the docs therefore overstates live performance by an unmeasured amount.
- It bounds what an honest engine would cost, before anyone spends days building
  canonical-CardPile belief search.

Method: at every decision, re-randomise the order of the draw pile before
searching, so the search's belief about future draws is wrong in the same way a
live bridge's is. The cards are unchanged -- same multiset, same everything else
-- so the ONLY difference is knowledge of order. That makes this a lower bound on
the honest-engine gap rather than a full simulation of one: a real belief search
would also average over orders in-tree, which recovers some of the loss (silverbot
measured in-tree averaging as +21pp over committing to one sampled order).

    python -m lightspeed._clairvoyance_cost --limit 250 --sims 100
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics

import slaythespire as sts

from ._human_deck_combat import build_battle
from .search_config import DEFAULT_SEARCH_CONFIG_PATH, ensure_search_config
from .paths import HUMAN_BENCHMARK

BENCHMARK_PATH = str(HUMAN_BENCHMARK)


def shuffle_draw_pile(bc, rng: random.Random) -> None:
    """Randomise the draw pile's ORDER, leaving its contents identical.

    `bc.draw_pile` hands back a copy, so writing to it does nothing -- the same
    trap that made an earlier harness silently fight with the default starter
    deck. Rebuild through the engine's own move API instead.
    """
    pile = list(bc.draw_pile)
    if len(pile) < 2:
        return
    order = list(range(len(pile)))
    rng.shuffle(order)
    bc.set_draw_pile_order(order)


def play_with(bc, sims: int, seed: int, blind: bool, rng: random.Random):
    """Play out a fight; when `blind`, re-randomise draw order every decision."""
    start = bc.player_hp
    for step in range(600):
        if bc.outcome != sts.BattleOutcome.UNDECIDED:
            break
        if not bc.get_legal_actions():
            break
        if blind:
            shuffle_draw_pile(bc, rng)
        action, _ = sts.run_mcts_search(bc, sims, None, (seed << 20) ^ step)
        action.execute(bc)
    return start - bc.player_hp, bc.outcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--sims", type=int, default=100)
    args = parser.parse_args()
    ensure_search_config(DEFAULT_SEARCH_CONFIG_PATH)

    with open(BENCHMARK_PATH, encoding="utf-8") as handle:
        fights = [r for r in json.load(handle)
                  if r["split"] == args.split][: args.limit]
    print(f"{len(fights)} {args.split} fights at {args.sims} sims\n")

    results = {}
    for label, blind in (("clairvoyant (current)", False), ("blind draw order", True)):
        rng = random.Random(20260731)
        scores, deaths = [], 0
        for index, rec in enumerate(fights):
            bc, _ = build_battle(rec["deck"], rec["relics"], rec["cur_hp"],
                                 rec["max_hp"],
                                 getattr(sts.MonsterEncounter, rec["encounter"]),
                                 20, rec["act"], rec.get("potions", ()))
            damage, outcome = play_with(bc, args.sims, index, blind, rng)
            died = outcome != sts.BattleOutcome.PLAYER_VICTORY
            deaths += died
            scores.append(rec["human_damage"] - (rec["cur_hp"] if died else damage))
        results[label] = scores
        print(f"{label:24s} objective {statistics.mean(scores):+7.3f}   "
              f"deaths {deaths}/{len(fights)}")

    a, b = results["clairvoyant (current)"], results["blind draw order"]
    delta = [y - x for x, y in zip(a, b)]
    mean = statistics.mean(delta)
    stderr = statistics.stdev(delta) / math.sqrt(len(delta))
    print(f"\nblind minus clairvoyant: {mean:+.2f} +/- {stderr:.2f} HP "
          f"(t={mean/stderr:+.2f}, n={len(delta)})")
    print("\nThis is a LOWER bound on what an honest engine costs: a real belief "
          "search averages over draw orders in-tree, recovering part of the loss.")


if __name__ == "__main__":
    main()
