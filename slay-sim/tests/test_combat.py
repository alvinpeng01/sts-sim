"""Pin the core mechanics so refactors don't silently break the math."""

import random

from sts.combat import CombatState, Result
from sts.creatures import Player
from sts.enemies import JawWorm
from sts.cards import (
    ironclad_starter_deck, make_strike, make_defend, make_bash,
)
from sts.powers import Vulnerable, Weak, Strength


def fresh(deck=None, monster_hp=44):
    player = Player(max_hp=80)
    m = JawWorm()
    m.hp = m.max_hp = monster_hp
    deck = deck or ironclad_starter_deck()
    return CombatState(player, [m], deck, rng=random.Random(0))


def test_strike_deals_6():
    c = fresh([make_strike()])
    m = c.monsters[0]
    c.deal_attack_damage(c.player, m, 6)
    assert m.hp == 44 - 6


def test_block_absorbs_then_hp():
    c = fresh([make_defend()])
    c.player.gain_block(5)
    assert c.player.block == 5
    lost = c.player.take_damage(8)
    assert lost == 3
    assert c.player.block == 0
    assert c.player.hp == 80 - 3


def test_strength_is_additive():
    c = fresh([make_strike()])
    c.player.add_power(Strength(3))
    assert c.player.calc_attack_damage(6) == 9


def test_vulnerable_is_1_5x_and_floors():
    c = fresh()
    m = c.monsters[0]
    m.add_power(Vulnerable(1))
    # 9 damage -> floor(9 * 1.5) = 13
    lost = m.take_damage(9)
    assert lost == 13


def test_weak_reduces_and_floors():
    c = fresh()
    c.player.add_power(Weak(1))
    # base 6 -> floor(6 * 0.75) = 4
    assert c.player.calc_attack_damage(6) == 4


def test_strength_then_weak_ordering():
    c = fresh()
    c.player.add_power(Strength(3))
    c.player.add_power(Weak(1))
    # (6 + 3) = 9 -> floor(9 * 0.75) = 6
    assert c.player.calc_attack_damage(6) == 6


def test_bash_applies_vulnerable():
    c = fresh([make_bash()])
    m = c.monsters[0]
    c.player.energy = 3
    bash = c.hand and None
    card = make_bash()
    c.hand = [card]
    c.play_card(card, m)
    assert m.hp == 44 - 8
    assert m.get_power_amount("Vulnerable") == 2


def test_duration_power_ticks_on_end_turn():
    c = fresh()
    c.player.add_power(Weak(2))
    c.player.end_turn(c)
    assert c.player.get_power_amount("Weak") == 1
    c.player.end_turn(c)
    assert c.player.has_power("Weak") is False


def test_draw_reshuffles_discard():
    deck = [make_strike() for _ in range(3)]
    c = fresh(deck)
    c.draw_pile = []
    c.discard_pile = deck[:]
    c.draw_cards(3)
    assert len(c.hand) == 3
    assert c.draw_pile == []


def test_fight_reaches_terminal_state():
    from demo import greedy_policy, play_fight
    c = play_fight(greedy_policy, seed=1)
    assert c.result() in (Result.WIN, Result.LOSS)


if __name__ == "__main__":
    import sys
    import traceback
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed += 1
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
