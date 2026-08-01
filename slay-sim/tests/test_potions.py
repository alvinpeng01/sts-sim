"""Sanity checks for the potion_value() heuristic (sts/potions.py). These
aren't correctness tests against a ground truth -- no such ground truth
exists (see the module docstring: no community resource gives one) -- they
check DIRECTIONAL sanity: a potion should score higher in the situation it's
obviously meant for than in a situation where it clearly isn't."""

import random

from sts.combat import CombatState
from sts.creatures import Player
from sts.cards import make_strike
from sts.enemies import Monster, IntentType, Intent
from sts.potions import potion_value


class _Dummy(Monster):
    def __init__(self, hp=30, dmg=10):
        super().__init__("Dummy", max_hp=hp)
        self.intent = Intent(IntentType.ATTACK, dmg, "Attack")
        self._pending_move = "Attack"

    def intent_options(self):
        return [(1.0, "Attack")]

    def force_intent(self, move):
        pass

    def take_turn(self, combat):
        pass


def _combat(player_hp=80, player_block=0, monster_hp=30, monster_dmg=10):
    player = Player(max_hp=player_hp)
    player.hp = player_hp
    player.block = player_block
    combat = CombatState(player, [_Dummy(hp=monster_hp, dmg=monster_dmg)],
                         [make_strike()], rng=random.Random(0))
    return combat


def test_fire_potion_worth_more_against_lower_hp_enemy():
    weak_enemy = _combat(monster_hp=15)
    tough_enemy = _combat(monster_hp=200)
    assert potion_value("Fire Potion", weak_enemy) > potion_value("Fire Potion", tough_enemy)


def test_fire_potion_gets_lethal_bonus():
    # Both enemies chosen well clear of the ratio term's own saturation
    # point, so the comparison isolates the lethal bonus specifically
    # rather than both sides just hitting the same value ceiling.
    lethal = _combat(monster_hp=18)      # 20 dmg kills an 18 HP enemy
    non_lethal = _combat(monster_hp=100)  # 20 dmg doesn't kill a 100 HP enemy
    assert potion_value("Fire Potion", lethal) > potion_value("Fire Potion", non_lethal)


def test_block_potion_worth_more_under_lethal_pressure():
    danger = _combat(player_hp=20, player_block=0, monster_dmg=25)
    safe = _combat(player_hp=80, player_block=50, monster_dmg=5)
    assert potion_value("Block Potion", danger) > potion_value("Block Potion", safe)


def test_block_potion_near_worthless_when_already_overblocked():
    safe = _combat(player_hp=80, player_block=50, monster_dmg=5)
    assert potion_value("Block Potion", safe) < 0.2


def test_fear_potion_worth_more_against_bigger_threat():
    big = _combat(monster_hp=100)
    small = _combat(monster_hp=10)
    assert potion_value("Fear Potion", big) >= potion_value("Fear Potion", small)


def test_fairy_in_a_bottle_spikes_near_death():
    dying = _combat(player_hp=5)
    healthy = _combat(player_hp=80)
    assert potion_value("Fairy in a Bottle", dying) > potion_value("Fairy in a Bottle", healthy)
    assert potion_value("Fairy in a Bottle", dying) > 1.0


def test_energy_potion_worth_more_with_a_fuller_hand():
    combat_full = _combat()
    combat_full.hand = [make_strike() for _ in range(5)]
    combat_empty = _combat()
    combat_empty.hand = []
    assert potion_value("Energy Potion", combat_full) > potion_value("Energy Potion", combat_empty)


def test_all_implemented_potions_return_a_bounded_nonnegative_float():
    combat = _combat()
    combat.hand = [make_strike() for _ in range(3)]
    for name in ["Fire Potion", "Block Potion", "Weak Potion", "Fear Potion",
                 "Strength Potion", "Dexterity Potion", "Speed Potion",
                 "Energy Potion", "Fairy in a Bottle"]:
        v = potion_value(name, combat)
        assert 0.0 <= v <= 1.5, f"{name}: {v}"


def test_unimplemented_potion_raises_key_error():
    combat = _combat()
    try:
        potion_value("Ambrosia", combat)
        assert False, "expected KeyError"
    except KeyError:
        pass


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
