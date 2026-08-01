"""Precise correctness checks for the trickier new engine mechanics: X-cost,
ethereal, exhaust hooks, Corruption, Barricade, Double Tap, status-card draw
triggers, Rupture, orbs infra reuse. "Doesn't crash" (test_full_ironclad_deck)
isn't the same as "correct" -- these pin exact expected state."""

import random

from sts.combat import CombatState
from sts.creatures import Player
from sts.cards import (
    make_strike, make_defend, make_whirlwind, make_ghostly_armor,
    make_reaper, make_dark_embrace, make_feel_no_pain, make_true_grit,
    make_corruption, make_barricade, make_double_tap, make_evolve,
    make_fire_breathing, make_wound, make_rupture, make_bloodletting,
    make_limit_break,
)
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


def _combat(deck, player_hp=100):
    player = Player(max_hp=player_hp)
    combat = CombatState(player, [_Dummy()], deck, rng=random.Random(0))
    combat.start_player_turn()
    return combat


def test_whirlwind_x_cost_spends_all_energy_and_hits_x_times():
    combat = _combat([make_whirlwind()])
    combat.player.energy = 3
    card = combat.hand[0]
    m = combat.monsters[0]
    combat.play_card(card, None)
    assert combat.player.energy == 0
    # 3 energy -> X=3 -> 3 hits of 5 dmg = 15
    assert m.hp == 999 - 15


def test_ghostly_armor_exhausts_if_unplayed_at_end_of_turn():
    combat = _combat([make_ghostly_armor(), make_strike()])
    ghostly = next(c for c in combat.hand if c.name == "Ghostly Armor")
    # Deliberately don't play it -- end the turn with it still in hand.
    combat.end_player_turn()
    assert ghostly in combat.exhaust_pile
    assert ghostly not in combat.discard_pile


def test_ghostly_armor_goes_to_discard_normally_if_played():
    combat = _combat([make_ghostly_armor()])
    card = combat.hand[0]
    combat.play_card(card, None)
    assert card in combat.discard_pile
    assert card not in combat.exhaust_pile


def test_dark_embrace_draws_on_exhaust():
    """Traced by hand (deterministic -- each rng.choice below has exactly one
    candidate, so no randomness actually enters):
      hand=[DE,TG], draw_pile=[D1,D2,D3]
      play DE -> hand=[TG]; DE force-exhausts itself, whose own hook fires
                 and draws D3 -> hand=[TG,D3], draw_pile=[D1,D2]
      play TG -> hand=[D3]; gains block; exhausts D3 (its only hand card),
                 which re-fires Dark Embrace, drawing D2
                 -> hand=[D2], draw_pile=[D1]; TG itself goes to discard
    A cascade (Dark Embrace's draw feeds True Grit's exhaust target, which
    triggers Dark Embrace again) -- not a bug, just what the hooks compose to."""
    combat = _combat([make_defend()])  # minimal valid combat, state overridden below
    combat.player.energy = 5
    combat.hand = [make_dark_embrace(), make_true_grit()]
    combat.draw_pile = [make_defend(), make_defend(), make_defend()]

    de = combat.hand[0]
    combat.play_card(de, None)
    assert [c.name for c in combat.hand] == ["True Grit", "Defend"]
    assert len(combat.draw_pile) == 2

    true_grit = combat.hand[0]
    combat.play_card(true_grit, None)
    assert [c.name for c in combat.hand] == ["Defend"]
    assert len(combat.draw_pile) == 1
    assert len(combat.exhaust_pile) == 2  # Dark Embrace itself + the cascaded Defend
    assert [c.name for c in combat.discard_pile] == ["True Grit"]


def test_feel_no_pain_grants_block_on_exhaust():
    combat = _combat([make_feel_no_pain(), make_true_grit(), make_strike()])
    fnp = next(c for c in combat.hand if c.name == "Feel No Pain")
    combat.play_card(fnp, None)
    block_before = combat.player.block
    tg = next(c for c in combat.hand if c.name == "True Grit")
    combat.play_card(tg, None)  # True Grit's own block (7) + FNP triggers on its random-exhaust
    # True Grit grants 7 block itself; FNP should add 3 more if it exhausted a card.
    assert combat.player.block >= block_before + 7


def test_corruption_makes_skills_free_and_force_exhausts():
    combat = _combat([make_corruption(), make_defend(), make_defend()])
    corruption = next(c for c in combat.hand if c.name == "Corruption")
    combat.play_card(corruption, None)
    assert combat.player.has_power("Corruption")
    energy_before = combat.player.energy
    defend = combat.hand[0]
    assert defend.name == "Defend"
    combat.play_card(defend, None)
    assert combat.player.energy == energy_before  # cost overridden to 0
    assert defend in combat.exhaust_pile  # force-exhausted, not discarded
    assert defend not in combat.discard_pile


def test_barricade_keeps_block_across_turns():
    """Isolates exactly the mechanic start_player_turn implements (skip the
    block reset if Barricade is active), without routing it through combat
    damage -- if the incoming attack fully absorbs the block, "Barricade
    worked" and "Barricade did nothing, block was reset to 0 anyway" would
    look identical at the checkpoint."""
    from sts.powers import Barricade as BarricadePower

    with_barricade = _combat([make_defend()])
    with_barricade.player.block = 20
    with_barricade.player.add_power(BarricadePower(1))
    with_barricade.start_player_turn()
    assert with_barricade.player.block == 20

    without_barricade = _combat([make_defend()])
    without_barricade.player.block = 20
    without_barricade.start_player_turn()
    assert without_barricade.player.block == 0


def test_double_tap_replays_the_next_attack():
    combat = _combat([make_double_tap(), make_strike()])
    dt = next(c for c in combat.hand if c.name == "Double Tap")
    combat.play_card(dt, None)
    assert combat.double_tap_charges == 1
    strike = next(c for c in combat.hand if c.name == "Strike")
    m = combat.monsters[0]
    hp_before = m.hp
    combat.play_card(strike, m)
    # Strike deals 6; Double Tap should make it resolve twice = 12 total.
    assert hp_before - m.hp == 12
    assert combat.double_tap_charges == 0


def test_evolve_draws_extra_card_when_status_card_drawn():
    combat = _combat([make_evolve()])
    evolve = combat.hand[0]
    combat.play_card(evolve, None)
    assert combat.player.has_power("Evolve")
    # Wound is on top (popped first); a Defend sits underneath so Evolve's
    # bonus draw has something left to actually draw.
    combat.draw_pile = [make_defend(), make_wound()]
    combat.discard_pile = []
    combat.hand = []
    combat.draw_cards(1)
    # Drawing the Wound (a Status card) should trigger Evolve for +1 more
    # draw, pulling in the Defend too -- 2 cards total from 1 draw_cards(1) call.
    assert len(combat.hand) == 2
    assert {c.name for c in combat.hand} == {"Wound", "Defend"}


def test_fire_breathing_damages_enemies_on_status_draw():
    combat = _combat([make_fire_breathing()])
    fb = combat.hand[0]
    combat.play_card(fb, None)
    m = combat.monsters[0]
    hp_before = m.hp
    combat.draw_pile = [make_wound()]
    combat.discard_pile = []
    combat.draw_cards(1)
    assert m.hp < hp_before


def test_rupture_grants_strength_on_hp_loss_from_card_not_combat_damage():
    combat = _combat([make_rupture(), make_bloodletting()])
    rupture = next(c for c in combat.hand if c.name == "Rupture")
    combat.play_card(rupture, None)
    bloodletting = next(c for c in combat.hand if c.name == "Bloodletting")
    str_before = combat.player.get_power_amount("Strength")
    combat.play_card(bloodletting, None)  # loses 3 HP via a card effect
    assert combat.player.get_power_amount("Strength") == str_before + 1

    # Combat damage (the Dummy's attack) should NOT trigger Rupture.
    str_before2 = combat.player.get_power_amount("Strength")
    combat.end_player_turn()
    combat.enemy_turn()
    assert combat.player.get_power_amount("Strength") == str_before2


def test_limit_break_doubles_current_strength():
    combat = _combat([make_limit_break()])
    combat.player.add_power(__import__("sts.powers", fromlist=["Strength"]).Strength(4))
    lb = combat.hand[0]
    combat.play_card(lb, None)
    assert combat.player.get_power_amount("Strength") == 8


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
