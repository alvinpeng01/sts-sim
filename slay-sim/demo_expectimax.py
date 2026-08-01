"""Compare greedy vs expectimax over many fights, across encounters/decks.

Run:  .venv/bin/python demo_expectimax.py

Search is much slower than the greedy heuristic (it's exploring hundreds to
thousands of cloned states per decision), so it runs on fewer trials -- still
enough to see a clear separation in win rate / HP retained.
"""

import inspect
import random
import time

from sts.combat import CombatState, Result
from sts.creatures import Player
from sts.search import choose_action
from sts.cards import ironclad_starter_deck, varied_ironclad_deck
from sts.encounters import encounter_jaw_worm, encounter_cultist, encounter_louse_pair
from demo import greedy_policy


def expectimax_policy(turns_left=1, draw_samples=1):
    def policy(combat):
        action, _val = choose_action(combat, turns_left=turns_left, draw_samples=draw_samples)
        return action
    return policy


def _build_monsters(encounter_fn, rng):
    if "rng" in inspect.signature(encounter_fn).parameters:
        return encounter_fn(rng)
    return encounter_fn()


def play_fight(policy, deck_fn, encounter_fn, seed, player_hp):
    rng = random.Random(seed)
    player = Player(max_hp=player_hp)
    monsters = _build_monsters(encounter_fn, rng)
    combat = CombatState(player, monsters, deck_fn(), rng=rng)
    while combat.result() == Result.ONGOING:
        combat.start_player_turn()
        while combat.result() == Result.ONGOING:
            action = policy(combat)
            if action[0] == "end":
                break
            combat.play_card(action[1], action[2])
        if combat.result() != Result.ONGOING:
            break
        combat.end_player_turn()
        combat.enemy_turn()
    return combat


def bench(name, policy, deck_fn, encounter_fn, n, player_hp):
    wins = hp_total = 0
    t0 = time.perf_counter()
    for i in range(n):
        c = play_fight(policy, deck_fn, encounter_fn, seed=i, player_hp=player_hp)
        if c.result() == Result.WIN:
            wins += 1
            hp_total += c.player.hp
    dt = time.perf_counter() - t0
    avg_hp = hp_total / max(wins, 1)
    print(f"  {name:24s} win {wins/n:6.1%}  avg HP {avg_hp:5.1f}  "
          f"({n} fights in {dt:.2f}s, {dt/n*1000:.1f} ms/fight)")


def main():
    # (name, encounter, deck, player_hp, trials for turns=1, trials for turns=2)
    # Trial counts are hand-tuned per encounter: search cost is dominated by
    # chance-node branching (how many enemies, how stochastic their AI) and
    # deck size, not just "difficulty". Jaw Worm's 3-way stochastic intent
    # makes turns=2 ~4s/decision; Cultist's fully deterministic AI makes the
    # same depth ~1s/decision despite similar HP. 2x Louse on the bigger
    # varied deck is the extreme case (~120s for a SINGLE decision -- two
    # monsters' intents multiply, and dedup only collapses same-name cards)
    # so turns=2 is skipped there rather than making the demo take forever;
    # that's exactly the cost wall determinization/pruning would need to
    # address for multi-enemy fights.
    encounters = [
        ("Jaw Worm", encounter_jaw_worm, ironclad_starter_deck, 45, 100, 5),
        ("Cultist", encounter_cultist, ironclad_starter_deck, 40, 100, 15),
        ("2x Louse", encounter_louse_pair, varied_ironclad_deck, 40, 50, 0),
    ]
    for enc_name, enc_fn, deck_fn, hp, n1, n2 in encounters:
        print(f"\n=== {enc_name}  (deck: {deck_fn.__name__}, player HP {hp}) ===")
        bench("greedy", greedy_policy, deck_fn, enc_fn, n=500, player_hp=hp)
        bench("expectimax (turns_left=1)", expectimax_policy(turns_left=1, draw_samples=1),
              deck_fn, enc_fn, n=n1, player_hp=hp)
        if n2 > 0:
            bench("expectimax (turns_left=2)", expectimax_policy(turns_left=2, draw_samples=4),
                  deck_fn, enc_fn, n=n2, player_hp=hp)
        else:
            print("  expectimax (turns_left=2)  skipped -- ~120s/decision here, see comment above")


if __name__ == "__main__":
    main()
