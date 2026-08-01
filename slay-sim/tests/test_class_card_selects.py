"""Non-Ironclad card handling, pinned after three defects found on 2026-07-31.

All three produced wrong play rather than an error, and none is reachable from
the Ironclad-only training path, which is why they survived so long:

  * Action::enumerateCardSelectActions had no case for DISCARD, HOLOGRAM,
    NIGHTMARE, SETUP or RETAIN, so those selects returned an EMPTY action
    vector -- the same shape as the InputState::SCRY crash: nativeHeuristicPick
    indexes legal[0] and nativeExpandLeaf gives the searched node zero edges.
    A native playout of a Silent deck segfaulted on it.
  * SEEK emitted every pair unconditionally and MEDITATE emitted the empty
    selection, both of which isValidMultiCardSelectAction rejects. An invalid
    action is worse than an empty list: it still reaches
    executeMultiCardSelectActionHelper and acts.
  * Blizzard, Scrape, Self Repair and Pressure Points had their `case` in a
    switch that BattleContext::useCard never routes them to (it dispatches on
    cardTypes[]), so each fell to an "attempted to use unimplemented card"
    default and did nothing at all.

Every check here watches fd 2 rather than a return value. `sts_common.h:8`
defines `sts_asserts` unconditionally but CMakeLists.txt:17 passes -DNDEBUG, so
the assert() beside each diagnostic is a no-op in every release build while the
std::cerr write still happens -- stderr is the only runtime signal these
failures produce.
"""
from __future__ import annotations

import pytest
import slaythespire as sts

from lightspeed._class_card_audit import captured_cxx_stderr

CARD_SELECT = sts.INPUT_STATE_CARD_SELECT

FILLER = [("Strike_R", 0), ("Defend_R", 0), ("Bash", 0), ("Anger", 0)]
DRAW = [("Strike_R", 0), ("Defend_R", 0), ("Iron Wave", 0), ("Clothesline", 0),
        ("Shrug It Off", 0)]
DISCARD_PILE = [("Strike_R", 0), ("Defend_R", 0)]
EXHAUST_PILE = [("Anger", 0), ("Defend_R", 0)]


def ctx(hand, monster_hp=60):
    m = sts.NativeMonsterSpec()
    m.monster_id_name = "JAW_WORM"
    m.cur_hp = m.max_hp = monster_hp
    return sts.build_battle_context(
        player_hp=70, player_max_hp=80, player_block=0, player_energy=99,
        player_statuses=[], monsters=[m], hand_cards=hand,
        draw_pile_cards=DRAW, discard_pile_cards=DISCARD_PILE,
        exhaust_pile_cards=EXHAUST_PILE, potion_slots=[], relics=[], turn=1,
        ascension=20, rng_seed=4242)


def play_first(bc):
    """Play the card in hand slot 0, which every case here puts under test."""
    for a in bc.get_legal_actions():
        if a.action_type == sts.ActionType.CARD and a.source_idx == 0:
            a.execute(bc)
            return True
    return False


# (card_string_id, task it opens) -- the five tasks that returned an empty
# vector, one card per distinct trigger.
EMPTY_BEFORE = [
    ("Acrobatics", "DISCARD"),
    ("Prepared", "DISCARD"),
    ("Concentrate", "DISCARD"),
    ("Dagger Throw", "DISCARD"),
    ("Survivor", "DISCARD"),
    ("Hologram", "HOLOGRAM"),
    ("Night Terror", "NIGHTMARE"),
    ("Setup", "SETUP"),
]


@pytest.mark.parametrize("card,task", EMPTY_BEFORE)
def test_class_card_select_enumerates(card, task):
    bc = ctx([(card, 0)] + FILLER)
    assert play_first(bc), f"{card} was not playable"
    if sts.get_input_state_raw(bc) != CARD_SELECT:
        pytest.skip(f"{card} did not open a select in this state")
    assert bc.get_legal_actions(), (
        f"{card} opened {task} with an empty legal action list -- "
        "nativeHeuristicPick would index legal[0] on an empty vector"
    )


def test_well_laid_plans_retain_enumerates():
    """RETAIN fires from a power at end of turn, not on play."""
    bc = ctx([("Well Laid Plans", 0)] + FILLER)
    assert play_first(bc)
    for a in bc.get_legal_actions():
        if a.action_type == sts.ActionType.END_TURN:
            a.execute(bc)
            break
    if sts.get_input_state_raw(bc) != CARD_SELECT:
        pytest.skip("Well-Laid Plans did not open a retain select")
    assert bc.get_legal_actions(), "RETAIN enumerated no actions"


# Every enumerated action must pass isValidAction. Action::execute dumps the
# offending action and the whole BattleContext to stderr when it does not.
ENUMERATION_CARDS = ["Seek", "Meditate", "Acrobatics", "Concentrate",
                     "Night Terror", "Setup", "Hologram", "Armaments"]


@pytest.mark.parametrize("card", ENUMERATION_CARDS)
def test_enumerated_select_actions_are_all_valid(card):
    bc = ctx([(card, 0)] + FILLER)
    assert play_first(bc)
    if sts.get_input_state_raw(bc) != CARD_SELECT:
        pytest.skip(f"{card} did not open a select in this state")
    actions = list(bc.get_legal_actions())
    assert actions
    with captured_cxx_stderr() as cap:
        for a in actions:
            a.execute(bc.copy_self())
    assert not cap.text.strip(), (
        f"{card} enumerated an action isValidAction rejects:\n"
        f"{cap.text[:600]}"
    )


# Cards whose case sat in a switch their CardType does not route to. Each did
# nothing at all before the fix.
@pytest.mark.parametrize("card", ["Blizzard", "Scrape", "Self Repair",
                                  "PathToVictory"])
def test_card_is_not_unimplemented(card):
    bc = ctx([(card, 0)] + FILLER)
    with captured_cxx_stderr() as cap:
        played = play_first(bc)
    assert played, f"{card} was not playable"
    assert "unimplemented card" not in cap.text, (
        f"{card} fell to a type switch default: {cap.text.strip()[:200]}"
    )


def test_scrape_deals_its_damage():
    """Scrape's case was in useSkillCard while cardTypes calls it an ATTACK."""
    bc = ctx([("Scrape", 0)] + FILLER, monster_hp=60)
    assert play_first(bc)
    assert bc.monsters[0].cur_hp < 60, "Scrape dealt no damage"


def test_pressure_points_applies_mark():
    """Pressure Points' case was in useAttackCard; cardTypes calls it a SKILL."""
    bc = ctx([("PathToVictory", 0)] + FILLER, monster_hp=200)
    assert play_first(bc)
    assert bc.monsters[0].cur_hp < 200, "Pressure Points did nothing"


def test_blizzard_is_an_attack():
    """Pins the table against being "fixed" the wrong way round.

    Blizzard's case sat in useSkillCard, which makes cardTypes saying ATTACK
    look like the error. It is not: silverbot-reference's independent copy of
    the same upstream table also says ATTACK, and the misplaced case was the
    bug -- the same as Scrape, Self Repair and Pressure Points in this pass.
    """
    assert sts.get_card_type(sts.CardId.BLIZZARD) == sts.CardType.ATTACK
