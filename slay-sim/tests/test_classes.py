"""Precise correctness checks for the non-Ironclad class mechanics: Poison
(Silent), Orbs + Focus (Defect), and Wrath/Calm stances (Watcher)."""

import random

from sts.combat import CombatState
from sts.creatures import Player
from sts.cards import (
    make_deadly_poison, make_strike, make_defend, make_zap, make_dualcast,
    make_focus_card, make_eruption, make_vigilance, make_empty_body,
    make_cold_snap,
)
from sts.classes import CharClass, make_player, STARTER_DECK, CARD_POOL
from sts.orbs import make_lightning_orb, make_frost_orb
from sts.enemies import Monster, IntentType, Intent


class _Dummy(Monster):
    def __init__(self, hp=999):
        super().__init__("Dummy", max_hp=hp)

    def intent_options(self):
        return [(1.0, "Attack")]

    def force_intent(self, move):
        self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(5), "Attack")
        self._pending_move = move

    def take_turn(self, combat):
        combat.deal_attack_damage(self, combat.player, 5)


def _combat(deck, player=None):
    player = player or Player(max_hp=100)
    combat = CombatState(player, [_Dummy()], deck, rng=random.Random(0))
    combat.start_player_turn()
    combat.player.energy = 10
    return combat


# --- All 4 classes' decks are well-formed ---

def test_every_class_starter_deck_and_pool_build_without_error():
    for cc in CharClass:
        deck = STARTER_DECK[cc]()
        pool = CARD_POOL[cc]()
        assert len(deck) >= 8
        assert len(pool) >= 3


# --- Silent: Poison ---

def test_poison_deals_damage_at_start_of_turn_and_decays():
    combat = _combat([make_deadly_poison()])
    poison = combat.hand[0]
    m = combat.monsters[0]
    combat.play_card(poison, m)
    assert m.get_power_amount("Poison") == 5

    hp_before = m.hp
    m.start_turn(combat)  # simulate the poisoned creature's own turn starting
    assert hp_before - m.hp == 5  # ignores block, direct HP loss
    assert m.get_power_amount("Poison") == 4


def test_poison_ignores_block():
    combat = _combat([make_deadly_poison()])
    m = combat.monsters[0]
    m.block = 100
    m.add_power(__import__("sts.powers", fromlist=["Poison"]).Poison(3))
    hp_before = m.hp
    m.start_turn(combat)
    assert hp_before - m.hp == 3
    assert m.block == 100  # untouched


def test_poison_expires_at_zero():
    combat = _combat([make_deadly_poison()])
    m = combat.monsters[0]
    from sts.powers import Poison
    m.add_power(Poison(1))
    m.start_turn(combat)
    assert not m.has_power("Poison")


# --- Defect: Orbs + Focus ---

def test_channel_orb_passive_fires_at_end_of_turn():
    combat = _combat([make_defend()])
    m = combat.monsters[0]
    combat.channel_orb(make_frost_orb())
    block_before = combat.player.block
    combat.end_player_turn()
    assert combat.player.block == block_before + 2  # Frost passive: 2 block


def test_channeling_past_orb_slots_evokes_the_oldest():
    combat = _combat([make_defend()])
    combat.player.orb_slots = 2
    m = combat.monsters[0]
    hp_before = m.hp
    combat.channel_orb(make_lightning_orb())
    combat.channel_orb(make_lightning_orb())
    assert len(combat.player.orbs) == 2
    combat.channel_orb(make_lightning_orb())  # 3rd overflows -> evicts+evokes the 1st
    assert len(combat.player.orbs) == 2
    # Lightning evoke deals 8 to a random enemy; only one enemy here.
    assert m.hp == hp_before - 8


def test_focus_scales_orb_effects():
    combat = _combat([make_focus_card(), make_defend()])
    focus_card = next(c for c in combat.hand if c.name == "Focus")
    combat.play_card(focus_card, None)
    assert combat.player.get_power_amount("Focus") == 2
    combat.channel_orb(make_frost_orb())
    block_before = combat.player.block
    combat.end_player_turn()
    # Frost passive base 2 + Focus 2 = 4
    assert combat.player.block == block_before + 4


def test_dualcast_retriggers_most_recent_orb_passive():
    combat = _combat([make_zap(), make_dualcast()])
    combat.player.energy = 10
    zap = next(c for c in combat.hand if c.name == "Zap")
    combat.play_card(zap, None)
    m = combat.monsters[0]
    hp_before = m.hp
    dualcast = next(c for c in combat.hand if c.name == "Dualcast")
    combat.play_card(dualcast, None)
    # Dualcast re-fires Lightning's passive once (3 dmg, no Focus).
    assert hp_before - m.hp == 3


# --- Watcher: Stances ---

def test_wrath_doubles_damage_dealt_and_taken():
    combat = _combat([make_eruption(), make_strike()])
    eruption = next(c for c in combat.hand if c.name == "Eruption")
    m = combat.monsters[0]
    hp_before = m.hp
    combat.play_card(eruption, m)
    assert combat.player.stance == "Wrath"
    # Eruption deals its damage BEFORE entering Wrath (matches the real
    # game: stance changes don't retroactively apply to the card that
    # triggered them), so this hit is undoubled: base 9.
    assert hp_before - m.hp == 9

    # A later Strike, played while already in Wrath, IS doubled: 6*2=12.
    hp_before2 = m.hp
    strike = next(c for c in combat.hand if c.name == "Strike")
    combat.play_card(strike, m)
    assert hp_before2 - m.hp == 12

    hp_before_player = combat.player.hp
    m.take_turn(combat)  # Dummy deals base 5, doubled by the player's own Wrath
    assert hp_before_player - combat.player.hp == 10


def test_exit_calm_grants_energy():
    combat = _combat([make_vigilance(), make_empty_body()])
    vigilance = next(c for c in combat.hand if c.name == "Vigilance")
    combat.play_card(vigilance, None)
    assert combat.player.stance == "Calm"
    energy_before = combat.player.energy
    empty_body = next(c for c in combat.hand if c.name == "Empty Body")
    combat.play_card(empty_body, None)
    assert combat.player.stance is None
    # Empty Body costs 1 energy, then exiting Calm grants 2 back: net +1.
    assert combat.player.energy == energy_before - 1 + 2


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
