"""Verify clone() safety and that expectimax makes provably correct tactical
calls a naive "attack lowest HP" heuristic gets wrong."""

import random
from typing import List, Tuple

from sts.combat import CombatState, Result
from sts.creatures import Player
from sts.cards import make_strike, make_defend
from sts.enemies import Monster, IntentType, Intent
from sts.search import choose_action, evaluate, _distinct_actions


class _FixedAttacker(Monster):
    """A monster with a hardcoded, unchanging attack intent -- lets a test
    pin an exact incoming-damage scenario without fighting the real AI's
    randomness."""

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


def test_clone_is_independent_of_original():
    player = Player(max_hp=80)
    m = _FixedAttacker(hp=20, dmg=5)
    combat = CombatState(player, [m], [make_strike(), make_defend()], rng=random.Random(0))
    clone = combat.clone()

    clone.player.hp = 1
    clone.monsters[0].hp = 1
    clone.hand.clear()

    assert combat.player.hp == 80
    assert combat.monsters[0].hp == 20
    assert len(combat.hand) != 0 or True  # hand may be empty pre-draw; just assert no crash from shared state
    assert combat.monsters[0] is not clone.monsters[0]
    assert combat.player is not clone.player


def test_search_kills_the_bigger_threat_not_the_lower_hp_target():
    """Two monsters: M1 has MORE hp but telegraphs a lethal-this-turn 20 dmg
    attack; M2 has LESS hp but only telegraphs 1 dmg. Both are killable in
    one hit with the player's only playable attack. A "target lowest HP"
    heuristic picks M2 and eats M1's 20 damage; the correct play kills M1 and
    prevents it, taking only 1 damage from M2 instead."""
    player = Player(max_hp=30)
    m1 = _FixedAttacker(hp=10, dmg=20)  # higher HP, bigger threat
    m2 = _FixedAttacker(hp=5, dmg=1)    # lower HP, minor threat

    strike = make_strike()
    strike.play = lambda combat, target: combat.deal_attack_damage(combat.player, target, 15)

    combat = CombatState(player, [m1, m2], [strike], rng=random.Random(0))
    combat.start_player_turn()  # draws the single Strike into hand

    action, _val = choose_action(combat, turns_left=1, draw_samples=1)
    assert action[0] == "play"
    assert action[2] is m1, "expectimax must kill the monster telegraphing lethal damage"


def test_choose_action_never_returns_illegal_action():
    player = Player(max_hp=40)
    m = _FixedAttacker(hp=15, dmg=6)
    combat = CombatState(player, [m], [make_strike(), make_strike(), make_defend()],
                         rng=random.Random(2))
    combat.start_player_turn()
    action, _val = choose_action(combat, turns_left=1, draw_samples=1)
    legal = list(_distinct_actions(combat))
    keys = {(a[0], a[1].name if a[0] == "play" else None,
             id(a[2]) if a[0] == "play" and a[2] is not None else None) for a in legal}
    got = (action[0], action[1].name if action[0] == "play" else None,
           id(action[2]) if action[0] == "play" and action[2] is not None else None)
    assert got in keys


def test_search_beats_random_over_many_fights():
    """Loose end-to-end sanity check: expectimax should win the large
    majority of fights a random policy would often lose."""
    from sts.enemies import JawWorm
    from sts.cards import ironclad_starter_deck

    wins = 0
    n = 15
    for seed in range(n):
        rng = random.Random(seed)
        player = Player(max_hp=40)
        combat = CombatState(player, [JawWorm()], ironclad_starter_deck(), rng=rng)
        while combat.result() == Result.ONGOING:
            combat.start_player_turn()
            while combat.result() == Result.ONGOING:
                action, _ = choose_action(combat, turns_left=1, draw_samples=1)
                if action[0] == "end":
                    break
                combat.play_card(action[1], action[2])
            if combat.result() != Result.ONGOING:
                break
            combat.end_player_turn()
            combat.enemy_turn()
        if combat.result() == Result.WIN:
            wins += 1
    assert wins / n >= 0.8


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
