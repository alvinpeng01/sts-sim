"""Honest draw order must not invalidate actions that index the draw pile.

`honest_draw_order` permutes the draw pile so the search cannot read the future.
The permutation happens on a COPY of the parent state, just before the chosen
action executes -- but that action was enumerated against the PARENT's order, and
several action types are positions into `drawPile` rather than card identities:

    SECRET_WEAPON      legal only if drawPile[idx].getType() == ATTACK
    SECRET_TECHNIQUE   legal only if drawPile[idx].getType() == SKILL
    SEEK, OMNISCIENCE  any idx < drawPile.size()
    SCRY               the top N

(`isValidCardSelectAction`, sts_lightspeed/src/sim/search/Action.cpp.) Shuffling
between enumeration and execution re-points the index at a different card, so the
engine is handed an action it considers illegal, dumps the whole BattleContext to
stderr, and applies it anyway. Found by attributing a stray 6 KB stderr dump
during a 900-play sweep: honest=1, one fight in 900, CULTIST_AND_CHOSEN.

The engine fix is to skip the pre-execute permutation for exactly these action
types. That loses no honesty -- a card-select screen SHOWS the player the pile, so
the choice is over identities and the order carries no hidden information to
protect -- and the post-execute permutation that makes the next draw independent
still runs.

Rarity is why this is a test and not a scan: it needs the card in the deck, the
search to pick a select action, and the permutation to move that index onto a
card of the wrong type. Stacking the deck makes it fire every run.
"""
from __future__ import annotations

import pytest

import slaythespire as sts

from lightspeed._class_card_audit import captured_cxx_stderr

# Every card whose selection is a position into the draw pile. Omniscience and
# Seek are Watcher; the other three are colorless, so one Watcher deck holds all.
PILE_INDEXED = ["SECRET_WEAPON", "SECRET_TECHNIQUE", "SEEK", "OMNISCIENCE",
                "FORESIGHT"]
SEED = 20260801


def _deck_of(names):
    """A Watcher deck stacked with the pile-indexed cards plus fodder.

    Fodder matters: SECRET_WEAPON needs an ATTACK somewhere in the pile to have
    any legal selection at all, and SECRET_TECHNIQUE needs a SKILL. With only the
    cards under test the select screens would be empty and nothing is exercised.
    """
    gc = sts.GameContext(sts.CharacterClass.WATCHER, SEED, 20)
    for _ in range(len(gc.deck)):
        gc.remove_card(0)
    for name in names:
        card_id = sts.CardId.__members__.get(name)
        if card_id is not None:
            for _ in range(2):
                gc.obtain_card(sts.Card(card_id))
    for name in ("STRIKE_PURPLE", "DEFEND_PURPLE", "ERUPTION", "VIGILANCE"):
        for _ in range(3):
            gc.obtain_card(sts.Card(sts.CardId.__members__[name]))
    return gc


@pytest.fixture(autouse=True)
def _restore_shipped_regime():
    """Leave the regime as found. honest_draw_order is process-global state and
    a leaked 1.0 would silently change every later test's search."""
    yield
    sts.set_search_params({"honest_draw_order": 0.0})


@pytest.mark.parametrize("honest", [1.0, 2.0])
def test_pile_indexed_selection_is_silent_under_honest_draws(honest):
    """Search a stacked deck in each honest regime; the engine must not complain.

    Asserts on stderr rather than on a return value because an invalid card
    select does not raise or change the outcome -- it prints and proceeds. Only
    fd 2 sees it, which is why `captured_cxx_stderr` redirects the descriptor
    instead of swapping `sys.stderr`.
    """
    sts.set_search_params({"honest_draw_order": honest})
    with captured_cxx_stderr() as cap:
        for trial in range(12):
            battle = sts.new_battle(_deck_of(PILE_INDEXED),
                                    sts.MonsterEncounter.JAW_WORM)
            for step in range(60):
                if battle.outcome != sts.BattleOutcome.UNDECIDED:
                    break
                if not battle.get_legal_actions():
                    break
                action, _ = sts.run_mcts_search(battle, 60, None,
                                                trial * 7919 + step)
                action.execute(battle)
    assert cap.text == "", (
        f"honest_draw_order={honest} made the engine dump state:\n"
        f"{cap.text[:600]}")
