"""Card upgrade ("+") correctness: every upgrade-supporting factory should
produce a distinctly-named, functionally different card when upgraded=True,
and unsupported factories should fail loudly (TypeError) rather than
silently no-op. Also verifies the transposition table treats upgraded and
base copies as genuinely different states (they hash by name)."""

import random

from sts.combat import CombatState
from sts.creatures import Player
from sts.cards import (
    make_strike, make_defend, make_bash, make_pommel_strike, make_iron_wave,
    make_twin_strike, make_cleave, make_thunderclap, make_clothesline,
    make_shrug_it_off, make_body_slam, make_flex, make_inflame,
    make_headbutt, make_perfected_strike, make_uppercut, make_whirlwind,
    make_metallicize, make_bludgeon, make_demon_form,
    make_neutralize, make_survivor, make_dualcast, make_eruption, make_vigilance,
    make_hemokinesis,
)
from sts.enemies import Monster, IntentType, Intent
from sts.search import _state_key


class _Dummy(Monster):
    def __init__(self, hp=9999):
        super().__init__("Dummy", max_hp=hp)

    def intent_options(self):
        return [(1.0, "Attack")]

    def force_intent(self, move):
        self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(5), "Attack")
        self._pending_move = move

    def take_turn(self, combat):
        combat.deal_attack_damage(self, combat.player, 5)


UPGRADE_FACTORIES = [
    make_strike, make_defend, make_bash, make_pommel_strike, make_iron_wave,
    make_twin_strike, make_cleave, make_thunderclap, make_clothesline,
    make_shrug_it_off, make_body_slam, make_flex, make_inflame,
    make_headbutt, make_perfected_strike, make_uppercut, make_whirlwind,
    make_metallicize, make_bludgeon, make_demon_form,
    make_neutralize, make_survivor, make_dualcast, make_eruption, make_vigilance,
]


def test_every_upgrade_factory_names_the_card_with_a_plus_suffix():
    for factory in UPGRADE_FACTORIES:
        base = factory(upgraded=False)
        plus = factory(upgraded=True)
        assert plus.name == base.name + "+", factory.__name__
        assert not base.upgraded
        assert plus.upgraded


def test_every_upgrade_factory_produces_a_different_stat_or_cost():
    """Every upgrade should change SOMETHING observable: cost, or the
    numbers baked into its effect. We can't introspect a closure's captured
    constants directly, so play both versions in an identical dummy combat
    and check for at least one measurable difference (damage dealt, block
    gained, cost, or draw-pile/hand size change)."""
    for factory in UPGRADE_FACTORIES:
        base, plus = factory(upgraded=False), factory(upgraded=True)
        if base.cost != plus.cost:
            continue  # cost-reduction upgrades (Eruption, Vigilance, Body Slam) -- already a real difference
        assert _play_and_measure(base) != _play_and_measure(plus), factory.__name__


def _play_and_measure(card):
    rng = random.Random(0)
    player = Player(max_hp=100)
    m = _Dummy()
    combat = CombatState(player, [m], [card], rng=rng)
    combat.player.energy = 5
    combat.player.orbs.append(__import__("sts.orbs", fromlist=["make_lightning_orb"]).make_lightning_orb())
    combat.hand = [card]
    target = m if card.targeted else None
    combat.play_card(card, target)
    return (m.max_hp - m.hp, player.block, len(combat.hand), len(combat.draw_pile),
            player.get_power_amount("Strength"), player.get_power_amount("Metallicize"),
            player.get_power_amount("Demon Form"), m.get_power_amount("Weak"),
            m.get_power_amount("Vulnerable"), player.stance, player.energy)


def test_unsupported_factory_raises_on_upgraded_kwarg():
    """A card outside the upgrade-supporting subset should fail loudly
    rather than silently pretending to upgrade."""
    try:
        make_hemokinesis(upgraded=True)
        assert False, "expected TypeError"
    except TypeError:
        pass


def test_transposition_table_distinguishes_upgraded_from_base():
    rng = random.Random(0)
    player_a = Player(max_hp=80)
    combat_a = CombatState(player_a, [_Dummy()], [make_strike(upgraded=False)], rng=rng)
    combat_a.start_player_turn()

    rng2 = random.Random(0)
    player_b = Player(max_hp=80)
    combat_b = CombatState(player_b, [_Dummy()], [make_strike(upgraded=True)], rng=rng2)
    combat_b.start_player_turn()

    assert _state_key(combat_a, 1) != _state_key(combat_b, 1)


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
