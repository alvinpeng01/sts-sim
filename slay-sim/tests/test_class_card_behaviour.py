"""Behavioural tests for Silent, Defect and Watcher cards, on the C++ engine.

This closes the largest coverage gap in the project. The other card tests --
`test_new_cards.py`, `test_ironclad_mechanics.py`, `test_classes.py`,
`test_combat.py` -- all import `sts.combat`, the PURE-PYTHON engine, which
`AGENTS.md` states is "tests and live bridge only, NOT training" and which hazard
8 records as "not in parity" with the C++ one. Nothing executed the C++ engine's
cards and checked the result, and the ~227 Silent/Defect/Watcher card cases that
are this fork's main addition over upstream had no behavioural coverage at all.

**What these assert, and why they are properties rather than per-card
expectations.** Hand-written expected outcomes for 225 cards would mostly encode
one author's reading of the game's rules, and would miss the bugs this engine
actually produces. Every card defect found here so far was an INTERACTION or
STATE bug that static auditing passed:

  - five CardSelectTasks returned empty action vectors, segfaulting a Silent
    playout
  - four cards had their `case` in a switch `useCard` never routes them to, so
    they silently did nothing

None of those is a wrong constant, and `_game_jar_audit.py` / `_card_effect_audit.py`
verified the constants exhaustively while all three survived. So these tests
execute each card and assert properties that must hold whatever the card does.

States come from a real `GameContext` -- strip the starter deck, fill it with the
card under test, start a battle -- rather than from `build_battle_context`. The
bridge constructor is the one with a bug history (unset `uniqueId` let every card
be replayed forever; Entropic Brew still hangs it), and `_engine_invariants.py`
records that a `new_battle` path is what the training runtime actually uses.
"""
from __future__ import annotations

import pytest

import slaythespire as sts

from lightspeed._class_card_audit import captured_cxx_stderr

CARD_NAMES = {int(v): k for k, v in sts.CardId.__members__.items()}

CLASS_BY_COLOR = {
    "GREEN": sts.CharacterClass.SILENT,
    "BLUE": sts.CharacterClass.DEFECT,
    "PURPLE": sts.CharacterClass.WATCHER,
}
# A single weak monster: the point is to exercise the card, not to survive. Jaw
# Worm is the encounter the existing card audits use.
ENCOUNTER = sts.MonsterEncounter.JAW_WORM
DECK_COPIES = 8
SEED = 20260801


def _cards_of(color: str):
    out = []
    for name, value in sts.CardId.__members__.items():
        card_id = int(value)
        if card_id == 0:
            continue
        try:
            if str(sts.get_card_color(sts.CardId(card_id))) == f"CardColor.{color}":
                out.append((name, card_id))
        except (RuntimeError, ValueError):
            continue
    return sorted(out)


def _battle_with(card_id: int, character):
    """A battle whose deck is nothing but the card under test.

    `gc.deck` hands back a copy, so it cannot be mutated in place -- the real
    mutators are `obtain_card`/`remove_card`, and a version of this that used
    `clear()`/`append()` silently played 40 fights with the wrong deck once
    (docs/07-known-issues.md).
    """
    gc = sts.GameContext(character, SEED, 20)
    for _ in range(len(gc.deck)):
        gc.remove_card(0)
    for _ in range(DECK_COPIES):
        gc.obtain_card(sts.Card(sts.CardId(card_id)))
    assert len(gc.deck) == DECK_COPIES, "deck rebuild did not take"
    return sts.new_battle(gc, ENCOUNTER)


def _card_actions(battle, card_id=None):
    """Legal plays, optionally only those for one specific card.

    The filter is not optional in practice. Watcher's starter relic Pure Water
    puts a Miracle in the opening hand, so "the first legal card action" is often
    the Miracle rather than the card under test -- which is what made the first
    version of this file report 75 Watcher failures that were all the Miracle's
    +1 energy.
    """
    out = []
    for action in battle.get_legal_actions():
        if action.action_type != sts.ActionType.CARD:
            continue
        if card_id is None or int(battle.hand[action.source_idx].id) == card_id:
            out.append(action)
    return out


def _total_copies(battle, card_id):
    """How many copies of a card exist anywhere in the battle.

    Counted across all four piles rather than by hand size, because playing a
    DRAW card legitimately grows the hand -- Acrobatics, Backflip, Skim, Escape
    Plan and Prepared all do -- and with a deck made of one card those draws are
    more copies of the card under test.
    """
    piles = (battle.hand, battle.draw_pile, battle.discard_pile,
             battle.exhaust_pile)
    return sum(1 for pile in piles for card in pile if int(card.id) == card_id)


ALL_CARDS = [(color, name, card_id)
             for color in ("GREEN", "BLUE", "PURPLE")
             for name, card_id in _cards_of(color)]


@pytest.mark.parametrize("color,name,card_id", ALL_CARDS,
                         ids=[f"{c[0][0]}-{c[1]}" for c in ALL_CARDS])
def test_card_plays_without_corrupting_state(color, name, card_id):
    """Play the card and assert the properties that must hold regardless of it.

    A card with no legal action is not a failure -- plenty need a stance, an orb,
    a discard pile or more energy than turn one provides. What is asserted is
    that IF the engine offers the card, playing it behaves sanely.
    """
    battle = _battle_with(card_id, CLASS_BY_COLOR[color])
    actions = _card_actions(battle, card_id)
    if not actions:
        pytest.skip(f"{name} has no legal play in the opening state")

    copies_before = _total_copies(battle, card_id)

    with captured_cxx_stderr() as captured:
        actions[0].execute(battle)
    noise = captured.text

    # The engine writes "attempted to use unimplemented card: X" to std::cerr and
    # then asserts -- but -DNDEBUG makes assert() a no-op, so stderr is the ONLY
    # runtime signal that a card fell through its switch. This is how the four
    # misplaced-case bugs were eventually caught.
    assert "unimplemented" not in noise.lower(), (
        f"{name}: engine reported an unimplemented card: {noise.strip()[:200]}")
    assert "attempted" not in noise.lower(), (
        f"{name}: engine wrote a rejection dump: {noise.strip()[:200]}")

    # Not "the hand shrank" -- a draw card grows it. What must never happen is a
    # card multiplying itself, which is the shape the unset-uniqueId bug took:
    # the played card stayed in hand AND a copy went to the discard pile.
    assert _total_copies(battle, card_id) <= copies_before, (
        f"{name}: copies grew {copies_before} -> "
        f"{_total_copies(battle, card_id)} by playing it")

    assert 0 <= battle.player_hp <= battle.player_max_hp, (
        f"{name}: hp {battle.player_hp}/{battle.player_max_hp} out of range")
    assert battle.player_energy >= 0, f"{name}: negative energy"

    if battle.outcome == sts.BattleOutcome.UNDECIDED:
        assert battle.get_legal_actions(), (
            f"{name}: battle is undecided but offers no legal action -- the "
            f"empty-enumeration class of bug that segfaulted a Silent playout")


@pytest.mark.parametrize("color,name,card_id", ALL_CARDS,
                         ids=[f"{c[0][0]}-{c[1]}" for c in ALL_CARDS])
def test_card_select_states_resolve(color, name, card_id):
    """If a card opens a card-select screen, every option it offers must work.

    Enumeration and validation drifting apart is a defect class this engine has
    had twice: five tasks enumerated nothing at all, and SEEK/MEDITATE enumerated
    actions `isValidMultiCardSelectAction` rejects -- worse than nothing, since an
    invalid action still reaches the executor after writing a dump to stderr.
    """
    battle = _battle_with(card_id, CLASS_BY_COLOR[color])
    actions = _card_actions(battle, card_id)
    if not actions:
        pytest.skip(f"{name} has no legal play in the opening state")
    actions[0].execute(battle)

    if sts.get_input_state_raw(battle) != sts.INPUT_STATE_CARD_SELECT:
        pytest.skip(f"{name} does not open a card select")

    options = list(battle.get_legal_actions())
    assert options, f"{name}: opened a card select with no legal option"
    with captured_cxx_stderr() as captured:
        for option in options:
            option.execute(battle.copy_self())
    assert "attempted" not in captured.text.lower(), (
        f"{name}: an enumerated select option was rejected by the validator: "
        f"{captured.text.strip()[:200]}")


def test_status_and_curse_playability_follows_the_relics():
    """Status and curse cards are playable exactly when the relic says so.

    This replaces a test that asserted unplayable cards are never legal, which is
    false and was written on a misreading. Void turning up as a legal play in a
    benchmark fight looked like the -2 cost sentinel defeating an
    `energy >= cost` gate. It was not: the player held **Medical Kit**, and
    `CardInstance::canUse` was already correct --

        case CardType::STATUS:
            if (!bc.player.hasRelic<RelicId::MEDICAL_KIT>() && id != CardId::SLIMED)

    -- including the subtlety that Slimed is exempt, because Slimed is playable
    with no relic at all. Blue Candle does the same for curses. The lesson worth
    keeping: a conditional that fires in one fight out of 1,730 is a CONDITION,
    not a rare bug.
    """
    status_card, curse_card = sts.CardId.WOUND, sts.CardId.CLUMSY

    def legal_with(card_id, relic):
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, SEED, 20)
        for _ in range(len(gc.deck)):
            gc.remove_card(0)
        for _ in range(DECK_COPIES):
            gc.obtain_card(sts.Card(sts.CardId(card_id)))
        if relic is not None:
            gc.obtain_relic(relic)
        battle = sts.new_battle(gc, ENCOUNTER)
        return bool(_card_actions(battle, int(card_id)))

    assert not legal_with(status_card, None), "Wound is playable with no relic"
    assert legal_with(status_card, sts.RelicId.MEDICAL_KIT), (
        "Medical Kit does not make Status cards playable")
    assert not legal_with(curse_card, None), "Clumsy is playable with no relic"
    assert legal_with(curse_card, sts.RelicId.BLUE_CANDLE), (
        "Blue Candle does not make Curse cards playable")
    # Slimed is a Status card that needs no relic -- the exemption in canUse.
    assert legal_with(sts.CardId.SLIMED, None), "Slimed should be playable unaided"
