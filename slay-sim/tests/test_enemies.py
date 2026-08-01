"""Pin monster AI: intent distributions sum to 1, match the original nested-rng
behavior statistically, and the new monsters (Cultist, Louse) act correctly."""

import random
from collections import Counter

from sts.enemies import JawWorm, Cultist, Louse, IntentType
from sts.powers import Ritual


def test_jaw_worm_turn0_forces_chomp():
    m = JawWorm()
    assert m.intent_options() == [(1.0, JawWorm.CHOMP)]


def test_jaw_worm_distributions_sum_to_one():
    m = JawWorm()
    m.turn_count = 1
    for last_move, last_twice in [
        (JawWorm.BELLOW, False), (JawWorm.CHOMP, False),
        (JawWorm.THRASH, True), (JawWorm.THRASH, False),
    ]:
        m.last_move = last_move
        m.last_move_twice = last_twice
        total = sum(p for p, _ in m.intent_options())
        assert abs(total - 1.0) < 1e-9


def test_jaw_worm_never_repeats_bellow_or_chomp_or_triple_thrash():
    """These are hard guards in the original logic: verify they still hold
    by sampling many turns and checking the resulting move sequence."""
    rng = random.Random(42)
    m = JawWorm()
    m.roll_intent(rng)  # turn 0 -> forced Chomp
    history = [m._pending_move]
    for _ in range(500):
        m.take_turn(_NullCombat())
        m.roll_intent(rng)
        history.append(m._pending_move)

    for i in range(1, len(history)):
        if history[i] == JawWorm.BELLOW:
            assert history[i - 1] != JawWorm.BELLOW
        if history[i] == JawWorm.CHOMP:
            assert history[i - 1] != JawWorm.CHOMP
        if history[i] == JawWorm.THRASH and i >= 2:
            assert not (history[i - 1] == JawWorm.THRASH and history[i - 2] == JawWorm.THRASH)


def test_jaw_worm_empirical_frequency_matches_table():
    """Sample intent_options() via roll_intent many times from a fixed AI
    state and check the empirical frequency matches the analytical table."""
    rng = random.Random(7)
    counts = Counter()
    trials = 20000
    for _ in range(trials):
        m = JawWorm()
        m.turn_count = 1
        m.last_move = JawWorm.CHOMP
        m.last_move_twice = False
        m.roll_intent(rng)
        counts[m._pending_move] += 1
    # Expected: Bellow .60, Thrash .40
    assert abs(counts[JawWorm.BELLOW] / trials - 0.60) < 0.02
    assert abs(counts[JawWorm.THRASH] / trials - 0.40) < 0.02


def test_cultist_incantation_then_dark_strike():
    c = Cultist(ritual_amount=3)
    assert c.intent_options() == [(1.0, Cultist.INCANTATION)]
    c.force_intent(Cultist.INCANTATION)
    assert c.intent.type == IntentType.BUFF
    c.take_turn(_NullCombat())
    assert c.has_power("Ritual")
    assert c.get_power_amount("Ritual") == 3

    assert c.intent_options() == [(1.0, Cultist.DARK_STRIKE)]
    c.force_intent(Cultist.DARK_STRIKE)
    assert c.intent.type == IntentType.ATTACK
    assert c.intent.damage == 6  # no Strength yet this instant


def test_ritual_grants_strength_at_end_of_cultist_turn():
    c = Cultist(ritual_amount=3)
    c.force_intent(Cultist.INCANTATION)
    combat = _NullCombat()
    c.take_turn(combat)  # adds Ritual(3), doesn't grant Strength yet
    assert c.get_power_amount("Strength") == 0
    c.end_turn(combat)  # Ritual.on_end_turn grants Strength == its amount
    assert c.get_power_amount("Strength") == 3


def test_louse_cannot_grow_twice_in_a_row():
    m = Louse(random.Random(1))
    m.last_move = Louse.GROW
    assert m.intent_options() == [(1.0, Louse.BITE)]


def test_louse_bite_damage_reflects_strength():
    m = Louse(random.Random(3))
    base_dmg = m.bite_damage
    m.force_intent(Louse.BITE)
    assert m.intent.damage == base_dmg
    from sts.powers import Strength
    m.add_power(Strength(4))
    m.force_intent(Louse.BITE)
    assert m.intent.damage == base_dmg + 4


class _NullCombat:
    """Minimal stand-in exposing what Creature/Monster hooks need."""

    def __init__(self):
        self.player = _Dummy()

    def deal_attack_damage(self, attacker, target, base):
        dmg = attacker.calc_attack_damage(base)
        return target.take_damage(dmg)


class _Dummy:
    """A big-HP punching bag so monster attacks in tests have somewhere to land."""

    def __init__(self):
        self.hp = 10_000
        self.block = 0
        self.powers = {}

    def take_damage(self, dmg):
        self.hp -= dmg
        return dmg


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
