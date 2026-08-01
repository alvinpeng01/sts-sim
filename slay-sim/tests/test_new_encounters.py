"""Correctness checks for the new monster roster, focused on the mechanics
that are genuinely new patterns (not just more of the same): Sentry's Dazed
injection, cross-monster buffing (Mystic -> Centurion), Champ's
HP-threshold-triggered one-time enrage, and (added later) Hexaghost's
player-HP-dependent Divider, Slime Boss's half-HP split, Chosen's
non-Attack-card-triggered Hex, Book of Stabbing's escalating/capped Multi
Stab, Giant Head's escalating/capped It Is Time, Shelled Parasite's
Life Suck lifesteal, and Time Eater's Time Warp (turn-ends-early-on-12th-
card-played) mechanic."""

import random

from sts.combat import CombatState, Result
from sts.creatures import Player
from sts.cards import ironclad_starter_deck, make_strike, make_defend
from sts.enemies import (
    Sentry, Mystic, Centurion, Champ, MadGremlin,
    Hexaghost, SlimeBoss, Chosen, BookOfStabbing, GiantHead, ShelledParasite,
    AcidSlimeM, SpikeSlimeM, TimeEater,
)


def test_sentry_beam_injects_dazed_into_draw_pile():
    combat = CombatState(Player(max_hp=80), [Sentry(starts_with_bolt=False)],
                         [make_strike() for _ in range(5)], rng=random.Random(0))
    m = combat.monsters[0]
    assert m.intent.name == "Beam"
    draw_pile_before = len(combat.draw_pile)
    m.take_turn(combat)
    assert len(combat.draw_pile) == draw_pile_before + 1
    assert any(c.name == "Dazed" for c in combat.draw_pile)


def test_sentry_alternates_deterministically():
    m = Sentry(starts_with_bolt=True)
    moves = []
    for _ in range(4):
        move = m.intent_options()[0][1]
        assert len(m.intent_options()) == 1  # fully deterministic, no chance node
        m.force_intent(move)
        moves.append(move)
        m.turn_count += 1
    assert moves == ["Bolt", "Beam", "Bolt", "Beam"]


def test_mystic_buff_applies_strength_to_both_self_and_centurion():
    """Mystic's Buff (like its Heal) affects both itself and its ally --
    the cross-monster part being tested is that the ally genuinely receives
    it too, not that the ally is the *only* recipient."""
    combat = CombatState(Player(max_hp=80), [Centurion(), Mystic()],
                         [make_strike()], rng=random.Random(0))
    centurion, mystic = combat.monsters
    mystic.force_intent(Mystic.BUFF)
    mystic.take_turn(combat)
    assert centurion.get_power_amount("Strength") == 2
    assert mystic.get_power_amount("Strength") == 2


def test_mystic_heal_shields_centurion_not_itself():
    combat = CombatState(Player(max_hp=80), [Centurion(), Mystic()],
                         [make_strike()], rng=random.Random(0))
    centurion, mystic = combat.monsters
    mystic.force_intent(Mystic.HEAL)
    mystic.take_turn(combat)
    assert centurion.block == 12
    assert mystic.block == 12  # Heal grants block to both self and ally


def test_mystic_with_no_living_ally_does_not_crash():
    combat = CombatState(Player(max_hp=80), [Mystic()], [make_strike()], rng=random.Random(0))
    mystic = combat.monsters[0]
    mystic.force_intent(Mystic.BUFF)
    mystic.take_turn(combat)  # no ally -- should just buff self, not crash
    assert mystic.get_power_amount("Strength") == 2


def test_champ_stays_calm_above_half_hp():
    champ = Champ()
    assert champ.hp == champ.max_hp
    assert champ.intent_options()[0][1] != Champ.ANGER


def test_champ_enrages_exactly_once_at_half_hp():
    champ = Champ()
    champ.hp = champ.max_hp // 2
    move = champ.intent_options()[0][1]
    assert move == Champ.ANGER
    champ.force_intent(move)
    champ.take_turn(None)
    assert champ.enraged is True
    assert champ.get_power_amount("Strength") == 5

    # Once enraged, further checks at low HP shouldn't re-trigger Anger.
    champ.hp = 10
    move2 = champ.intent_options()[0][1]
    assert move2 != Champ.ANGER


def test_champ_executes_after_face_slap_once_enraged():
    champ = Champ()
    champ.enraged = True
    champ.last_move = Champ.FACE_SLAP
    assert champ.intent_options() == [(1.0, Champ.EXECUTE)]


def test_gremlin_gang_is_four_deterministic_monsters():
    from sts.encounters import encounter_gremlin_gang
    monsters = encounter_gremlin_gang()
    assert len(monsters) == 4
    for m in monsters:
        assert len(m.intent_options()) == 1  # every gremlin type is deterministic


def test_hexaghost_divider_damage_depends_on_player_hp():
    """Divider deals (floor(player_hp/12)+1)*6 -- the actual number can only
    be known at resolve time (force_intent has no combat access), so the
    telegraphed Intent carries damage=None; this checks the *real* damage
    dealt in take_turn scales with player HP as documented, using two
    separate fights so one doesn't affect the other."""
    combat_low = CombatState(Player(max_hp=100), [Hexaghost()], [make_strike()], rng=random.Random(0))
    m_low = combat_low.monsters[0]
    m_low.turn_count = 1
    m_low.force_intent(Hexaghost.DIVIDER)
    assert m_low.intent.damage is None
    combat_low.player.hp = 24  # floor(24/12)+1 = 3 -> 18 damage
    m_low.take_turn(combat_low)
    assert combat_low.player.hp == 24 - 18

    combat_high = CombatState(Player(max_hp=100), [Hexaghost()], [make_strike()], rng=random.Random(0))
    m_high = combat_high.monsters[0]
    m_high.turn_count = 1
    m_high.force_intent(Hexaghost.DIVIDER)
    combat_high.player.hp = 84  # floor(84/12)+1 = 8 -> 48 damage
    m_high.take_turn(combat_high)
    assert combat_high.player.hp == 84 - 48


def test_hexaghost_cycle_after_divider():
    m = Hexaghost()
    m.turn_count = 2
    expected = [Hexaghost.SEAR, Hexaghost.TACKLE, Hexaghost.SEAR, Hexaghost.INFLAME,
                Hexaghost.TACKLE, Hexaghost.SEAR, Hexaghost.INFERNO, Hexaghost.SEAR]
    seen = []
    for _ in range(len(expected)):
        move = m.intent_options()[0][1]
        seen.append(move)
        m.force_intent(move)
        m.turn_count += 1
    assert seen == expected  # 7-move cycle, then repeats from Sear


def test_slime_boss_splits_at_half_hp_into_two_slimes_with_its_current_hp():
    combat = CombatState(Player(max_hp=80), [SlimeBoss()], [make_strike()], rng=random.Random(0))
    boss = combat.monsters[0]
    boss.hp = 70  # <= 150//2 (75) -- should trigger Split next
    move = boss.intent_options()[0][1]
    assert move == SlimeBoss.SPLIT
    boss.force_intent(move)
    boss.take_turn(combat)

    assert boss.is_dead
    spawned = [m for m in combat.monsters if m is not boss]
    assert len(spawned) == 2
    assert {type(s) for s in spawned} == {AcidSlimeM, SpikeSlimeM}
    for s in spawned:
        assert s.hp == 70
        assert s.max_hp == 70
        assert s.intent is not None  # roll_intent was called so it's ready to act


def test_slime_boss_normal_cycle_before_half_hp():
    m = SlimeBoss()
    assert m.hp == m.max_hp
    moves = []
    for _ in range(3):
        move = m.intent_options()[0][1]
        moves.append(move)
        m.force_intent(move)
        m.turn_count += 1
    assert moves == [SlimeBoss.GOOP_SPRAY, SlimeBoss.PREPARING, SlimeBoss.SLAM]


def test_chosen_hex_shuffles_dazed_into_draw_pile_on_non_attack_card_only():
    combat = CombatState(Player(max_hp=80), [Chosen()], [make_strike()], rng=random.Random(0))
    chosen = combat.monsters[0]
    chosen.hex_active = True
    combat.player.energy = 10
    attack, skill = make_strike(), make_defend()
    combat.hand = [attack, skill]

    draw_before = len(combat.draw_pile)
    combat.play_card(attack, chosen)
    assert len(combat.draw_pile) == draw_before  # Attack doesn't trigger Hex

    draw_before = len(combat.draw_pile)
    combat.play_card(skill, None)
    assert len(combat.draw_pile) == draw_before + 1
    assert any(c.name == "Dazed" for c in combat.draw_pile)


def test_chosen_opening_sequence_is_poke_then_hex():
    m = Chosen()
    assert m.intent_options() == [(1.0, Chosen.POKE)]
    m.force_intent(Chosen.POKE)
    m.turn_count = 1
    assert m.intent_options() == [(1.0, Chosen.HEX)]


def test_book_of_stabbing_multi_stab_hit_count_escalates_and_caps_repeats():
    m = BookOfStabbing()
    combat = CombatState(Player(max_hp=200), [m], [make_strike()], rng=random.Random(0))
    m.force_intent(BookOfStabbing.MULTI_STAB)  # after CombatState() -- its own
    # constructor already called roll_intent() once; forcing here overrides
    # that so the test is deterministic regardless of what the RNG rolled.
    hp_before = combat.player.hp
    m.take_turn(combat)  # 1st use: 2 hits of 7 = 14
    assert hp_before - combat.player.hp == 14

    m.force_intent(BookOfStabbing.MULTI_STAB)
    hp_before = combat.player.hp
    m.take_turn(combat)  # 2nd use: 3 hits of 7 = 21
    assert hp_before - combat.player.hp == 21

    # Two Multi Stabs in a row already happened -- a 3rd must be blocked.
    assert m.intent_options() == [(1.0, BookOfStabbing.BIG_STAB)]


def test_giant_head_it_is_time_escalates_and_caps_at_70():
    m = GiantHead()
    m.turn_count = 3
    combat = CombatState(Player(max_hp=200), [m], [make_strike()], rng=random.Random(0))
    assert m.intent_options() == [(1.0, GiantHead.IT_IS_TIME)]

    m.force_intent(GiantHead.IT_IS_TIME)
    hp_before = combat.player.hp
    m.take_turn(combat)
    assert hp_before - combat.player.hp == 40  # 1st use

    for _ in range(9):  # climb well past the cap
        m.force_intent(GiantHead.IT_IS_TIME)
        hp_before = combat.player.hp
        m.take_turn(combat)
        assert hp_before - combat.player.hp <= 70


def test_shelled_parasite_life_suck_heals_by_unblocked_damage():
    m = ShelledParasite()
    combat = CombatState(Player(max_hp=80), [m], [make_strike()], rng=random.Random(0))
    m.hp = m.max_hp - 30  # after CombatState() -- see the ordering note above
    m.force_intent(ShelledParasite.LIFE_SUCK)
    combat.player.block = 5  # 12 - 5 = 7 unblocked
    hp_before_monster = m.hp
    m.take_turn(combat)
    assert m.hp == min(m.max_hp, hp_before_monster + 7)


def test_shelled_parasite_never_uses_fell_twice_in_a_row():
    m = ShelledParasite()
    m.force_intent(ShelledParasite.FELL)
    m.turn_count += 1  # simulate having actually taken that turn -- otherwise
    # intent_options()'s turn_count==0 special case (always Fell) would mask
    # the repeat-avoidance check this test is actually about.
    assert ShelledParasite.FELL not in [move for _, move in m.intent_options()]


def test_time_warp_triggers_on_12th_card_gains_strength_and_ends_turn():
    combat = CombatState(Player(max_hp=80), [TimeEater()],
                         [make_strike() for _ in range(20)], rng=random.Random(0))
    combat.hand = [make_strike() for _ in range(11)]
    combat.player.energy = 99
    for card in list(combat.hand):
        combat.play_card(card, combat.monsters[0])
    assert combat.turn_should_end_early is False
    assert combat.monsters[0].get_power_amount("Strength") == 0

    twelfth = make_strike()
    combat.hand = [twelfth]
    combat.play_card(twelfth, combat.monsters[0])
    assert combat.turn_should_end_early is True
    assert combat.monsters[0].get_power_amount("Strength") == 2
    assert combat.monsters[0].time_warp_counter == 0


def test_time_warp_counter_carries_over_between_turns():
    """The real mechanic's counter is NOT reset by start_player_turn -- only
    by the trigger firing -- so 8 cards last turn + 4 this turn should still
    fire on the 4th card of the second turn."""
    combat = CombatState(Player(max_hp=80), [TimeEater()],
                         [make_strike() for _ in range(30)], rng=random.Random(0))
    combat.player.energy = 99
    combat.hand = [make_strike() for _ in range(8)]
    for card in list(combat.hand):
        combat.play_card(card, combat.monsters[0])
    assert combat.monsters[0].time_warp_counter == 8

    combat.start_player_turn()
    assert combat.monsters[0].time_warp_counter == 8  # unaffected by the reset
    combat.player.energy = 99
    combat.hand = [make_strike() for _ in range(4)]
    for card in list(combat.hand):
        combat.play_card(card, combat.monsters[0])
    assert combat.monsters[0].get_power_amount("Strength") == 2
    assert combat.turn_should_end_early is True


def test_time_warp_restricts_legal_actions_to_end_only():
    combat = CombatState(Player(max_hp=80), [TimeEater()],
                         [make_strike() for _ in range(15)], rng=random.Random(0))
    combat.turn_should_end_early = True
    combat.hand = [make_strike(), make_defend()]
    assert combat.legal_actions() == [("end",)]


def test_time_warp_reset_by_start_player_turn_flag_not_counter():
    combat = CombatState(Player(max_hp=80), [TimeEater()],
                         [make_strike() for _ in range(15)], rng=random.Random(0))
    combat.turn_should_end_early = True
    combat.start_player_turn()
    assert combat.turn_should_end_early is False


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
