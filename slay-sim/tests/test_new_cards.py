"""Correctness checks for this session's new Ironclad/colorless/curse cards
-- focused on the genuinely new mechanics they needed (not every card gets
a test; straightforward damage/block numbers aren't worth pinning
individually), matching test_new_encounters.py's own scope/style."""

import random

from sts.combat import CombatState
from sts.creatures import Player
from sts.enemies import JawWorm
from sts.powers import Artifact, Weak
from sts.cards import (
    make_strike, make_defend,
    make_juggernaut, make_sentinel, make_blood_for_blood,
    make_panache, make_the_bomb, make_panic_button,
    make_pain, make_normality, make_doubt, make_shame, make_regret, make_decay,
)


def _combat(hand, energy=10):
    p = Player(max_hp=70)
    combat = CombatState(p, [JawWorm()], [make_strike() for _ in range(5)], rng=random.Random(1))
    combat.player.energy = energy
    combat.hand = hand
    return combat


def test_juggernaut_procs_once_per_card_that_gains_block():
    jugg, defend = make_juggernaut(), make_defend()
    combat = _combat([jugg, defend])
    combat.play_card(jugg, None)
    hp_before = combat.monsters[0].hp
    combat.play_card(defend, None)
    assert combat.monsters[0].hp == hp_before - 5


def test_juggernaut_does_not_proc_when_no_block_gained():
    jugg, strike = make_juggernaut(), make_strike()
    combat = _combat([jugg, strike])
    combat.play_card(jugg, None)
    hp_before = combat.monsters[0].hp
    combat.play_card(strike, combat.monsters[0])
    # Strike deals 6 damage and grants no block -- only that 6 should apply,
    # no extra 5 from Juggernaut.
    assert combat.monsters[0].hp == hp_before - 6


def test_sentinel_grants_energy_only_when_exhausted_not_when_discarded():
    sentinel = make_sentinel()
    combat = _combat([sentinel])
    energy_before = combat.player.energy
    combat.play_card(sentinel, None)  # Sentinel isn't self-exhausting
    # Just its own 1-energy cost paid, no +2 exhaust bonus from a normal discard.
    assert combat.player.energy == energy_before - 1

    sentinel2 = make_sentinel()
    combat2 = _combat([sentinel2])
    energy_before2 = combat2.player.energy
    combat2._exhaust(sentinel2)
    assert combat2.player.energy == energy_before2 + 2


def test_blood_for_blood_cost_drops_with_hp_loss_and_floors_at_zero():
    bfb = make_blood_for_blood()
    combat = _combat([bfb])
    assert combat._effective_cost(bfb) == 2
    combat.player.lose_hp(1)
    assert combat._effective_cost(bfb) == 1
    combat.player.lose_hp(1)
    assert combat._effective_cost(bfb) == 0
    combat.player.lose_hp(1)
    assert combat._effective_cost(bfb) == 0  # floors, doesn't go negative


def test_panache_triggers_every_5th_card_played_this_turn():
    panache = make_panache()
    strikes = [make_strike() for _ in range(5)]
    combat = _combat([panache] + strikes)
    combat.play_card(panache, None)  # card #1 this turn
    hp_before = combat.monsters[0].hp
    for s in strikes[:3]:  # cards #2, #3, #4
        combat.play_card(s, combat.monsters[0])
    hp_before_5th = combat.monsters[0].hp
    combat.play_card(strikes[3], combat.monsters[0])  # card #5 -- should proc
    # 6 damage from the strike itself, plus 10 from Panache's proc.
    assert combat.monsters[0].hp == hp_before_5th - 6 - 10
    hp_before_6th = combat.monsters[0].hp
    combat.play_card(strikes[4], combat.monsters[0])  # card #6 -- no proc
    assert combat.monsters[0].hp == hp_before_6th - 6


def test_the_bomb_detonates_after_exactly_3_end_of_turns():
    bomb = make_the_bomb()
    combat = _combat([bomb])
    combat.play_card(bomb, None)
    hp_start = combat.monsters[0].hp
    combat.end_player_turn()
    assert combat.monsters[0].hp == hp_start  # 1st end-of-turn: no detonation yet
    combat.end_player_turn()
    assert combat.monsters[0].hp == hp_start  # 2nd: still not yet
    combat.end_player_turn()
    assert combat.monsters[0].hp == hp_start - 40  # 3rd: detonates
    assert "The Bomb" not in combat.player.powers  # removes itself after


def test_panic_button_blocks_further_block_gain_for_2_turns_then_recovers():
    pb, defend = make_panic_button(), make_defend()
    combat = _combat([pb, defend], energy=10)
    combat.play_card(pb, None)
    block_after_pb = combat.player.block
    assert block_after_pb == 30

    combat.play_card(defend, None)
    assert combat.player.block == block_after_pb  # suppressed, no extra block

    combat.start_player_turn()  # turn 1 of suppression ticks down
    combat.hand = [make_defend()]
    combat.play_card(combat.hand[0], None)
    assert combat.player.block == 0  # still suppressed (block reset at turn start)

    combat.start_player_turn()  # turn 2 -- suppression should be over now
    combat.hand = [make_defend()]
    combat.play_card(combat.hand[0], None)
    assert combat.player.block == 5  # normal Defend block now applies


def test_artifact_negates_the_next_debuff_and_is_consumed():
    p = Player(max_hp=70)
    p.add_power(Artifact(1))
    p.add_power(Weak(2))
    assert not p.has_power("Weak")
    assert not p.has_power("Artifact")


def test_artifact_stacks_negate_multiple_debuffs_before_running_out():
    p = Player(max_hp=70)
    p.add_power(Artifact(2))
    p.add_power(Weak(1))
    assert not p.has_power("Weak")
    assert p.get_power_amount("Artifact") == 1
    p.add_power(Weak(1))
    assert not p.has_power("Weak")
    assert not p.has_power("Artifact")
    p.add_power(Weak(1))
    assert p.has_power("Weak")  # Artifact ran out -- this one gets through


def test_pain_deals_1_damage_per_other_card_played_while_in_hand():
    pain, strike = make_pain(), make_strike()
    combat = _combat([pain, strike])
    hp_before = combat.player.hp
    combat.play_card(strike, combat.monsters[0])
    assert combat.player.hp == hp_before - 1


def test_normality_blocks_a_4th_card_play_but_not_the_first_3():
    normality = make_normality()
    strikes = [make_strike() for _ in range(4)]
    combat = _combat([normality] + strikes)
    for s in strikes[:3]:
        combat.play_card(s, combat.monsters[0])  # should not raise
    try:
        combat.play_card(strikes[3], combat.monsters[0])
        assert False, "4th card play should have been blocked by Normality"
    except ValueError:
        pass


def test_doubt_shame_regret_decay_apply_while_in_hand_at_end_of_turn():
    for factory, check in [
        (make_doubt, lambda c: c.player.has_power("Weak")),
        (make_shame, lambda c: c.player.has_power("Frail")),
        (make_regret, lambda c: c.player.hp < 70),
        (make_decay, lambda c: c.player.hp == 68),
    ]:
        combat = _combat([factory()])
        combat.end_player_turn()
        assert check(combat), f"{factory.__name__} effect did not apply"


def test_doubt_does_not_apply_if_played_away_before_end_of_turn():
    """Sanity check that these are genuinely "while in hand" effects, not
    unconditional -- a Doubt that's already been discarded shouldn't still
    apply at end of turn."""
    doubt = make_doubt()
    combat = _combat([doubt])
    combat.hand.remove(doubt)
    combat.discard_pile.append(doubt)
    combat.end_player_turn()
    assert not combat.player.has_power("Weak")


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
