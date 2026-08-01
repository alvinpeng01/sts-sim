"""Correctness and sanity checks for MCTS: it must make the same provably-
correct tactical calls expectimax does (reusing the exact crafted scenario
from test_search.py), never return an illegal action, and its answer should
get closer to expectimax's as the simulation budget grows."""

import random
from typing import List, Tuple

from sts.combat import CombatState, Result
from sts.creatures import Player
from sts.cards import make_strike, make_defend, ironclad_starter_deck
from sts.enemies import Monster, IntentType, Intent, JawWorm
from sts.search import choose_action, _distinct_actions
from sts.mcts import mcts_choose_action


class _FixedAttacker(Monster):
    def __init__(self, hp: int, dmg: int):
        super().__init__("Dummy", max_hp=hp)
        self._dmg = dmg
        self.force_intent("Attack")

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(1.0, "Attack")]

    def force_intent(self, move: str) -> None:
        self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(self._dmg), "Attack")
        self._pending_move = move

    def take_turn(self, combat) -> None:
        combat.deal_attack_damage(self, combat.player, self._dmg)


def test_mcts_kills_the_bigger_threat_not_the_lower_hp_target():
    """Same crafted scenario as test_search.py: naive "lowest HP" targeting
    is wrong here; the correct play kills the higher-HP monster because it
    telegraphs lethal damage, sparing only a minor hit from the other."""
    player = Player(max_hp=30)
    m1 = _FixedAttacker(hp=10, dmg=20)
    m2 = _FixedAttacker(hp=5, dmg=1)

    strike = make_strike()
    strike.play = lambda combat, target: combat.deal_attack_damage(combat.player, target, 15)

    combat = CombatState(player, [m1, m2], [strike], rng=random.Random(0))
    combat.start_player_turn()

    action, _val = mcts_choose_action(combat, simulations=300)
    assert action[0] == "play"
    assert action[2] is m1


def test_mcts_never_returns_illegal_action():
    player = Player(max_hp=40)
    m = _FixedAttacker(hp=15, dmg=6)
    combat = CombatState(player, [m], [make_strike(), make_strike(), make_defend()],
                         rng=random.Random(2))
    combat.start_player_turn()
    action, _val = mcts_choose_action(combat, simulations=150)
    legal = list(_distinct_actions(combat))
    keys = {(a[0], a[1].name if a[0] == "play" else None,
             id(a[2]) if a[0] == "play" and a[2] is not None else None) for a in legal}
    got = (action[0], action[1].name if action[0] == "play" else None,
           id(action[2]) if action[0] == "play" and action[2] is not None else None)
    assert got in keys


def test_mcts_runs_on_all_new_complex_encounters_without_crashing():
    from sts.encounters import (
        encounter_gremlin_gang, encounter_sentries, encounter_byrd,
        encounter_centurion_mystic, encounter_champ,
    )
    from sts.cards import varied_ironclad_deck

    for enc_fn in [encounter_gremlin_gang, encounter_sentries, encounter_byrd,
                   encounter_centurion_mystic, encounter_champ]:
        rng = random.Random(0)
        player = Player(max_hp=60)
        combat = CombatState(player, enc_fn(), varied_ironclad_deck(), rng=rng)
        combat.start_player_turn()
        action, val = mcts_choose_action(combat, simulations=150)
        assert action is not None


def test_mcts_lands_close_to_exact_expectimax_with_enough_simulations():
    """Not bit-identical (MCTS is a budgeted approximation, not guaranteed
    monotonically closer on any single noisy trial -- only asymptotically),
    but with a generous simulation budget it should land within a small
    absolute tolerance of exact expectimax's value on a simple scenario."""
    rng = random.Random(9)
    player = Player(max_hp=45)
    combat = CombatState(player, [JawWorm()], ironclad_starter_deck(), rng=rng)
    combat.start_player_turn()

    _, exact_val = choose_action(combat, turns_left=1, draw_samples=1)
    _, mcts_val = mcts_choose_action(combat.clone(), simulations=800)

    assert abs(exact_val - mcts_val) < 5.0, (
        f"exact={exact_val}, mcts={mcts_val}"
    )


if __name__ == "__main__":
    import sys, traceback
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); passed += 1
            except Exception:
                failed += 1; print(f"FAIL {name}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
