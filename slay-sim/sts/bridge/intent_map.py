"""(engine monster name, CommunicationMod move_id) -> engine MonsterMoveId name.

CommunicationMod's ``move_id`` is a per-monster-class numeric with no name in
the protocol, which is why the bridge historically rolled a move from the
engine's own AI instead — measured **12.5% correct** against the telegraph the
game was actually reporting (`lightspeed/_bridge_intent_audit.py`), with worst
cases predicting zero incoming damage and thereby suppressing every Defend in
hand via ``defensiveCardSuppressionPenalty``.

This table was DERIVED, not transcribed: every (monster, move_id) pair observed
in the 52 MB live capture was matched by forcing each of the monster's
prefix-matching MonsterMoveId candidates through ``build_battle_context`` and
comparing ``get_monster_move_damage`` against the telegraphed base damage x
hits (attacks), or the move category against the intent family (non-attacks).
15 of 17 pairs matched uniquely; the two hand-resolved cases are marked. See
scratch derivation ``derive_intent_map.py`` and ``intent_map.json`` evidence
(2026-08-03).

Coverage is exactly the pairs the capture contains. Unmapped pairs return None
and the bridge falls back to the engine's AI roll, same as before — this table
can only improve on 12.5%, never regress it.
"""
from __future__ import annotations

INTENT_MOVE_MAP: dict[tuple[str, int], str] = {
    ("BLUE_SLAVER", 1): "BLUE_SLAVER_STAB",
    ("BLUE_SLAVER", 4): "BLUE_SLAVER_RAKE",
    # Damage matched (7 base); hits scale with the stab counter, which a bare
    # reconstruction cannot know, so the hits check was waived for this pair --
    # MULTI_STAB is the monster's only multi-hit move.
    ("BOOK_OF_STABBING", 1): "BOOK_OF_STABBING_MULTI_STAB",
    ("BOOK_OF_STABBING", 2): "BOOK_OF_STABBING_SINGLE_STAB",
    ("CENTURION", 1): "CENTURION_SLASH",
    ("CENTURION", 2): "CENTURION_DEFEND",
    ("GREMLIN_LEADER", 4): "GREMLIN_LEADER_STAB",
    ("MYSTIC", 1): "MYSTIC_ATTACK_DEBUFF",
    # Ambiguous between HEAL and BUFF by category (both non-attack); the game's
    # own Mystic.takeTurn uses byte 2 for the heal. Either choice telegraphs
    # zero incoming damage, so the search consequence of the tie is minimal.
    ("MYSTIC", 2): "MYSTIC_HEAL",
    ("RED_SLAVER", 1): "RED_SLAVER_STAB",
    ("RED_SLAVER", 3): "RED_SLAVER_SCRAPE",
    ("SNAKE_PLANT", 1): "SNAKE_PLANT_CHOMP",
    ("SNAKE_PLANT", 2): "SNAKE_PLANT_ENFEEBLING_SPORES",
    ("SNECKO", 1): "SNECKO_PERPLEXING_GLARE",
    ("SNECKO", 2): "SNECKO_BITE",
    ("SNECKO", 3): "SNECKO_TAIL_WHIP",
    ("TASKMASTER", 2): "TASKMASTER_SCOURING_WHIP",
}


def lookup_move_name(engine_monster_name: str, move_id) -> str | None:
    """Engine MonsterMoveId name for a telegraphed move, or None if unmapped."""
    if move_id is None:
        return None
    try:
        return INTENT_MOVE_MAP.get((engine_monster_name, int(move_id)))
    except (TypeError, ValueError):
        return None
