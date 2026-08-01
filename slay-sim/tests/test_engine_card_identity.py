"""Card identity invariants that three separate bugs violated silently.

Each test here corresponds to a defect found on 2026-07-30 that produced wrong
play rather than an error, which is why they are pinned rather than left to
manual checking:

  * build_battle_context left every card at CardInstance's -1 uniqueId default,
    so removeFromHandById never matched and a played card stayed in hand -- one
    Bash could kill a 44 HP Jaw Worm on its own;
  * cardColors[] had 8 transposed entries, which made getCardColor(Brutality)
    GREEN for an Ironclad and let a transform return the card it was given;
  * the CardColor pybind enum was missing BLUE entirely.
"""
from __future__ import annotations

import slaythespire as sts


HAND = [("Strike_R", 0), ("Defend_R", 0), ("Armaments", 0), ("Bash", 0),
        ("Warcry", 0)]
DRAW = [("Strike_R", 0)] * 4 + [("Defend_R", 0)] * 4


def _bridge_context(hand=None, energy=3, monster_hp=44):
    monster = sts.NativeMonsterSpec()
    monster.monster_id_name = "JAW_WORM"
    monster.cur_hp = monster_hp
    monster.max_hp = monster_hp
    return sts.build_battle_context(
        player_hp=60, player_max_hp=80, player_block=0, player_energy=energy,
        player_statuses=[], monsters=[monster],
        hand_cards=HAND if hand is None else hand,
        draw_pile_cards=DRAW, discard_pile_cards=[], exhaust_pile_cards=[],
        potion_slots=[], relics=[], turn=1, ascension=20, rng_seed=12345)


def _card_total(bc) -> int:
    return (len(bc.hand) + len(bc.draw_pile) + len(bc.discard_pile)
            + len(bc.exhaust_pile))


def _play_action(bc, card_id):
    hand = [card.id for card in bc.hand]
    for action in bc.get_legal_actions():
        if (action.action_type == sts.ActionType.CARD
                and 0 <= action.source_idx < len(hand)
                and hand[action.source_idx] == card_id):
            return action
    return None


def _resolve_pending_select(bc) -> None:
    actions = list(bc.get_legal_actions())
    if actions and actions[0].action_type in (
            sts.ActionType.SINGLE_CARD_SELECT, sts.ActionType.MULTI_CARD_SELECT):
        actions[0].execute(bc)


def test_bridge_played_card_leaves_hand():
    bc = _bridge_context(hand=[("Bash", 0)], energy=99)
    assert _play_action(bc, sts.CardId.BASH) is not None
    _play_action(bc, sts.CardId.BASH).execute(bc)
    assert _play_action(bc, sts.CardId.BASH) is None, (
        "Bash is still playable after being played -- the card was not removed "
        "from hand, so the search sees an infinite hand")
    assert bc.outcome == sts.BattleOutcome.UNDECIDED


def test_bridge_card_count_is_conserved_across_a_play():
    for card_id in (sts.CardId.STRIKE_RED, sts.CardId.BASH,
                    sts.CardId.ARMAMENTS, sts.CardId.WARCRY):
        bc = _bridge_context()
        before = _card_total(bc)
        action = _play_action(bc, card_id)
        assert action is not None, f"{card_id} not playable in the fixture hand"
        action.execute(bc)
        _resolve_pending_select(bc)
        assert _card_total(bc) == before, (
            f"playing {card_id} changed the card count {before} -> "
            f"{_card_total(bc)}")


def test_native_battle_card_count_is_conserved():
    """The training/eval path, which never uses build_battle_context."""
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 1, 20)
    bc = sts.new_battle(gc, sts.MonsterEncounter.JAW_WORM)
    before = _card_total(bc)
    action = next(a for a in bc.get_legal_actions()
                  if a.action_type == sts.ActionType.CARD)
    action.execute(bc)
    _resolve_pending_select(bc)
    assert _card_total(bc) == before


def test_card_colors_are_correct():
    expected = {
        # The eight entries that were transposed.
        "BRILLIANCE": "PURPLE", "COLLECT": "PURPLE",
        "BRUTALITY": "RED", "COMBUST": "RED",
        "BUFFER": "BLUE", "COMPILE_DRIVER": "BLUE",
        "BULLET_TIME": "GREEN", "CONCENTRATE": "GREEN",
        # Controls, correct throughout.
        "BASH": "RED", "NEUTRALIZE": "GREEN", "ZAP": "BLUE",
        "ERUPTION": "PURPLE", "BURN": "COLORLESS",
    }
    actual = {
        name: sts.get_card_color(getattr(sts.CardId, name)).name
        for name in expected}
    assert actual == expected


def test_card_color_enum_exposes_every_colour():
    """BLUE was missing from the pybind enum, so blue cards rendered as '???'."""
    for name in ("RED", "GREEN", "BLUE", "PURPLE", "COLORLESS", "CURSE"):
        assert hasattr(sts.CardColor, name), f"CardColor.{name} is not bound"


def test_ironclad_transform_pool_is_uniformly_red():
    """returnTrulyRandomCardFromAvailable's exclusion test compares the card's
    colour against the character's. Any pool card not reporting RED skips the
    exclusion branch and can transform into itself."""
    non_red = [
        card_id.name
        for card_id in (sts.CardId.BRUTALITY, sts.CardId.COMBUST,
                        sts.CardId.DEMON_FORM, sts.CardId.IMMOLATE,
                        sts.CardId.ARMAMENTS, sts.CardId.WHIRLWIND)
        if sts.get_card_color(card_id).name != "RED"]
    assert non_red == []
