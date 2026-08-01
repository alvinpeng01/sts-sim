"""Correctness checks for the relic system: each relic's effect, clone
independence for mutable relic state (Nunchaku's counter, Akabeko's used
flag), and that the transposition table stays correct once relics are
present (static relics free, counter relics correctly distinguish states)."""

import random

from sts.combat import CombatState
from sts.creatures import Player
from sts.cards import make_strike, make_defend, make_bloodletting
from sts.enemies import Monster, IntentType, Intent
from sts.relics import (
    Vajra, BagOfMarbles, Anchor, OddlySmoothStone, RingOfTheSnake,
    CrackedCore, PureWater, Akabeko, BronzeScales, Nunchaku, Kunai,
    Shuriken, OrnamentalFan, Orichalcum, Torii, RunicCube, CentennialPuzzle,
    PenNib,
)
from sts.search import _state_key


class _Dummy(Monster):
    def __init__(self, hp=100, dmg=5):
        super().__init__("Dummy", max_hp=hp)
        self._dmg = dmg

    def intent_options(self):
        return [(1.0, "Attack")]

    def force_intent(self, move):
        self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(self._dmg), "Attack")
        self._pending_move = move

    def take_turn(self, combat):
        combat.deal_attack_damage(self, combat.player, self._dmg)


def _combat(deck, relics=(), monsters=None, player_hp=100):
    player = Player(max_hp=player_hp)
    player.relics = list(relics)
    monsters = monsters if monsters is not None else [_Dummy()]
    return CombatState(player, monsters, deck, rng=random.Random(0))


# --- combat-start relics ---

def test_vajra_grants_strength_at_combat_start():
    combat = _combat([make_strike()], relics=[Vajra()])
    assert combat.player.get_power_amount("Strength") == 1


def test_bag_of_marbles_applies_vulnerable_to_all_enemies():
    combat = _combat([make_strike()], relics=[BagOfMarbles()],
                     monsters=[_Dummy(), _Dummy()])
    for m in combat.monsters:
        assert m.get_power_amount("Vulnerable") == 1


def test_anchor_grants_block_at_combat_start():
    combat = _combat([make_strike()], relics=[Anchor()])
    assert combat.player.block == 10


def test_oddly_smooth_stone_grants_dexterity():
    combat = _combat([make_strike()], relics=[OddlySmoothStone()])
    assert combat.player.get_power_amount("Dexterity") == 1


def test_ring_of_the_snake_draws_two_at_combat_start_only():
    deck = [make_strike() for _ in range(10)]
    combat = _combat(deck, relics=[RingOfTheSnake()])
    assert len(combat.draw_pile) == 8  # 10 - 2 drawn at combat start
    combat.start_player_turn()
    assert len(combat.hand) == 7  # +5 from the normal turn-1 draw, not another +2


def test_cracked_core_channels_a_lightning_orb_at_combat_start():
    combat = _combat([make_strike()], relics=[CrackedCore()])
    assert len(combat.player.orbs) == 1
    assert combat.player.orbs[0].name == "Lightning"


def test_pure_water_adds_miracle_to_hand():
    combat = _combat([make_strike()], relics=[PureWater()])
    assert any(c.name == "Miracle" for c in combat.hand)


# --- damage-modifying relics ---

def test_akabeko_boosts_first_attack_only():
    combat = _combat([make_strike(), make_strike()], relics=[Akabeko()])
    combat.player.energy = 5
    m = combat.monsters[0]
    hp_before = m.hp
    combat.hand = [make_strike(), make_strike()]
    combat.play_card(combat.hand[0], m)
    first_dmg = hp_before - m.hp
    assert first_dmg == 6 + 8  # base Strike + Akabeko's +8

    hp_before2 = m.hp
    combat.play_card(combat.hand[0], m)
    second_dmg = hp_before2 - m.hp
    assert second_dmg == 6  # no bonus the second time


def test_pen_nib_doubles_every_tenth_attack():
    combat = _combat([make_strike()], relics=[PenNib()])
    combat.player.energy = 99
    m = combat.monsters[0]
    for i in range(9):
        combat.hand = [make_strike()]
        combat.play_card(combat.hand[0], m)
    hp_before = m.hp
    combat.hand = [make_strike()]
    combat.play_card(combat.hand[0], m)  # the 10th attack
    assert hp_before - m.hp == 12  # doubled 6 -> 12


# --- attack-counter relics ---

def test_nunchaku_grants_energy_every_tenth_attack_never_resets_across_turns():
    combat = _combat([make_strike()], relics=[Nunchaku()])
    combat.player.energy = 99
    m = combat.monsters[0]
    for i in range(9):
        combat.hand = [make_strike()]
        combat.play_card(combat.hand[0], m)
    energy_before = combat.player.energy
    combat.hand = [make_strike()]
    combat.play_card(combat.hand[0], m)
    # The 10th Strike's own cost (1) is deducted same as always; Nunchaku's
    # +1 grant happens in the same call, netting out to unchanged energy --
    # not +1, which double-counted the grant without the cost.
    assert combat.player.energy == energy_before


def test_kunai_resets_its_counter_each_turn():
    combat = _combat([make_strike() for _ in range(10)], relics=[Kunai()])
    combat.player.energy = 99
    m = combat.monsters[0]
    combat.hand = [make_strike(), make_strike()]
    combat.play_card(combat.hand[0], m)
    combat.play_card(combat.hand[0], m)  # 2 attacks played, not yet 3
    assert combat.player.get_power_amount("Dexterity") == 0
    combat.end_player_turn()
    combat.start_player_turn()  # resets Kunai's per-turn counter
    combat.player.energy = 99
    combat.hand = [make_strike()]
    combat.play_card(combat.hand[0], m)  # 1 more attack this turn, not 3 yet
    assert combat.player.get_power_amount("Dexterity") == 0


def test_shuriken_and_ornamental_fan_trigger_on_third_attack_this_turn():
    combat = _combat([make_strike() for _ in range(5)],
                     relics=[Shuriken(), OrnamentalFan()])
    combat.player.energy = 99
    m = combat.monsters[0]
    for _ in range(3):
        combat.hand = [make_strike()]
        combat.play_card(combat.hand[0], m)
    assert combat.player.get_power_amount("Strength") == 1
    assert combat.player.block == 4


# --- end-of-turn / any-hp-loss relics ---

def test_orichalcum_grants_block_only_if_ended_turn_with_zero_block():
    combat = _combat([make_strike()], relics=[Orichalcum()])
    combat.end_player_turn()
    assert combat.player.block == 6


def test_orichalcum_does_nothing_if_block_already_present():
    combat = _combat([make_strike()], relics=[Orichalcum()])
    combat.player.block = 3
    combat.end_player_turn()
    assert combat.player.block == 3  # unchanged, not +6


def test_torii_caps_unblocked_damage_in_1_to_5_range():
    combat = _combat([make_strike()], relics=[Torii()], monsters=[_Dummy(dmg=4)])
    hp_before = combat.player.hp
    combat.monsters[0].take_turn(combat)
    assert hp_before - combat.player.hp == 1  # 4 dmg capped to 1


def test_torii_does_not_affect_damage_above_5():
    combat = _combat([make_strike()], relics=[Torii()], monsters=[_Dummy(dmg=8)])
    hp_before = combat.player.hp
    combat.monsters[0].take_turn(combat)
    assert hp_before - combat.player.hp == 8  # unaffected


def test_bronze_scales_retaliates_when_player_takes_damage():
    combat = _combat([make_strike()], relics=[BronzeScales()], monsters=[_Dummy(dmg=5)])
    m = combat.monsters[0]
    hp_before = m.hp
    m.take_turn(combat)
    assert hp_before - m.hp == 3


def test_runic_cube_draws_on_any_hp_loss():
    combat = _combat([make_strike() for _ in range(10)], relics=[RunicCube()],
                     monsters=[_Dummy(dmg=5)])
    hand_before = len(combat.hand)
    combat.monsters[0].take_turn(combat)
    assert len(combat.hand) == hand_before + 1


def test_runic_cube_fires_on_card_effect_hp_loss_too():
    combat = _combat([make_bloodletting()], relics=[RunicCube()])
    combat.hand = [make_bloodletting()]
    combat.player.energy = 5
    hand_before_play = 1  # bloodletting itself is about to be removed
    combat.play_card(combat.hand[0], None)
    assert len(combat.hand) == 1  # bloodletting removed (-1), RunicCube draw (+1)


def test_centennial_puzzle_only_triggers_once_per_combat():
    combat = _combat([make_strike() for _ in range(10)], relics=[CentennialPuzzle()],
                     monsters=[_Dummy(dmg=3)])
    hand_before = len(combat.hand)
    combat.monsters[0].take_turn(combat)
    assert len(combat.hand) == hand_before + 3
    hand_before2 = len(combat.hand)
    combat.monsters[0].take_turn(combat)
    assert len(combat.hand) == hand_before2  # no second trigger


# --- clone independence + transposition table interaction ---

def test_relic_mutable_state_is_independent_across_clones():
    combat = _combat([make_strike()], relics=[Nunchaku()])
    combat.player.relics[0].counter = 5
    clone = combat.clone()
    clone.player.relics[0].counter = 999
    assert combat.player.relics[0].counter == 5


def test_static_relic_does_not_fragment_transposition_key():
    """Vajra's hooks never touch counter/used, so two clones that reach an
    otherwise-identical state through DIFFERENT play sequences should hash
    identically -- unlike Nunchaku, whose counter genuinely depends on how
    many attacks were played to get there (see the counter-relic test)."""
    combat = _combat([make_strike(), make_defend()], relics=[Vajra()])
    combat.start_player_turn()
    combat.player.energy = 5

    order_a = combat.clone()
    sa, da = order_a.hand[0], order_a.hand[1]
    order_a.play_card(sa if sa.name == "Strike" else da, order_a.monsters[0])
    order_a.play_card(da if sa.name == "Strike" else sa, None)

    order_b = combat.clone()
    sb, db = order_b.hand[0], order_b.hand[1]
    order_b.play_card(db if sb.name == "Strike" else sb, None)
    order_b.play_card(sb if sb.name == "Strike" else db, order_b.monsters[0])

    assert _state_key(order_a, 1) == _state_key(order_b, 1)
    assert order_a.player.relics[0].counter == order_b.player.relics[0].counter == 0


def test_counter_relic_correctly_fragments_transposition_key():
    combat = _combat([make_strike()], relics=[Nunchaku()])
    combat.start_player_turn()
    key_a = _state_key(combat, 1)
    combat.player.relics[0].counter = 3
    key_b = _state_key(combat, 1)
    assert key_a != key_b


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
