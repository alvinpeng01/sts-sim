"""Correctness checks for sts/bridge/native_recommend.py -- the native
sts_lightspeed-engine-backed recommendation layer wired into
communication_mod.py's _try_recommend. Complements test_autobattle.py
(which already exercises this end-to-end through handle_state) with
targeted checks on native_recommend's own mapping tables and the
_ShadowCombat monster-filter/index-alignment invariant its own docstring
calls out. Unlike an earlier version of this test file, none of this goes
through sts.bridge.state_mapper -- native_recommend is fully self-sufficient
(see native_recommend.py's own module docstring for why that changed)."""
from sts.bridge.native_recommend import native_recommend, UnmappedMonsterError


def _combat_json(monsters, hand=None, turn=1):
    return {
        "player": {"current_hp": 70, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
        "hand": hand if hand is not None else [{"id": "Strike_R", "upgrades": 0, "has_target": True} for _ in range(5)],
        "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
        "monsters": monsters,
        "turn": turn,
    }


def test_simple_fight_returns_valid_action():
    combat_state_json = _combat_json([
        {"id": "Cultist", "name": "Cultist", "current_hp": 50, "max_hp": 56, "block": 0,
         "intent": "ATTACK", "move_adjusted_damage": 6, "move_hits": 1, "is_gone": False, "powers": []},
    ])
    description, action, combat = native_recommend(combat_state_json)
    assert action[0] in ("play", "end")
    assert len(combat.hand) == 5
    assert len(combat.monsters) == 1
    print(f"test_simple_fight_returns_valid_action: OK ({description})")


def test_unmapped_monster_raises():
    combat_state_json = _combat_json([
        {"id": "TotallyUnknownMonsterId", "name": "???", "current_hp": 50, "max_hp": 56,
         "block": 0, "intent": "ATTACK", "move_adjusted_damage": 6, "move_hits": 1,
         "is_gone": False, "powers": []},
    ])
    try:
        native_recommend(combat_state_json)
        assert False, "expected UnmappedMonsterError"
    except UnmappedMonsterError:
        pass
    print("test_unmapped_monster_raises: OK")


def test_verified_real_capture_monster_ids():
    """These specific (raw id -> canonical name) pairs were confirmed against
    this project's own real CommunicationMod capture (sts_raw_states.log),
    not guessed -- see native_recommend.py's _MONSTER_ID_MAP comment. A
    regression here means a live run's Slaver/Taskmaster/Mugger fight would
    silently fall back to v1 (no recommendation) instead of using the
    native engine."""
    from sts.bridge.native_recommend import _map_monster_id
    assert _map_monster_id("SlaverBlue") == "BLUE_SLAVER"
    assert _map_monster_id("SlaverRed") == "RED_SLAVER"
    assert _map_monster_id("SlaverBoss") == "TASKMASTER"
    assert _map_monster_id("Mugger") == "MUGGER"
    assert _map_monster_id("Donu") == "DONU"
    assert _map_monster_id("Deca") == "DECA"
    print("test_verified_real_capture_monster_ids: OK")


def test_verified_real_capture_status_ids():
    from sts.bridge.native_recommend import _match_status, _PLAYER_STATUS_OVERRIDES, _PLAYER_STATUS_NAMES
    logged = []
    assert _match_status("Weakened", _PLAYER_STATUS_OVERRIDES, _PLAYER_STATUS_NAMES, logged.append) == "WEAK"
    assert _match_status("IntangiblePlayer", _PLAYER_STATUS_OVERRIDES, _PLAYER_STATUS_NAMES, logged.append) == "INTANGIBLE"
    assert _match_status("Feel No Pain", _PLAYER_STATUS_OVERRIDES, _PLAYER_STATUS_NAMES, logged.append) == "FEEL_NO_PAIN"
    assert _match_status("Demon Form", _PLAYER_STATUS_OVERRIDES, _PLAYER_STATUS_NAMES, logged.append) == "DEMON_FORM"
    assert _match_status("Strength", _PLAYER_STATUS_OVERRIDES, _PLAYER_STATUS_NAMES, logged.append) == "STRENGTH"
    assert not logged  # none of the above should have logged an unmapped warning
    print("test_verified_real_capture_status_ids: OK")


def test_taskmaster_fight_no_longer_blocked():
    """The real gap this module's self-sufficiency fix was built for: a
    Taskmaster (raw id "SlaverBoss") fight has NO class in sts/enemies.py at
    all, and used to be entirely blocked from native recommendations because
    an earlier version of this module required state_mapper.build_combat_state
    to succeed first. Confirmed via replaying 300 real captured states
    (sts_raw_states.log): 257/300 were exactly this fight, all previously
    stuck on v1 fallback. This must now succeed."""
    combat_state_json = _combat_json([
        {"id": "SlaverBoss", "name": "Taskmaster", "current_hp": 47, "max_hp": 61, "block": 0,
         "intent": "ATTACK_DEBUFF", "move_adjusted_damage": 7, "move_hits": 1,
         "is_gone": False, "powers": [{"amount": 2, "name": "Strength", "id": "Strength"}]},
    ])
    description, action, combat = native_recommend(combat_state_json)
    assert action[0] in ("play", "end")
    print(f"test_taskmaster_fight_no_longer_blocked: OK ({description})")


def test_dead_monster_index_alignment():
    """A dead monster ahead of a live one in the raw JSON: _ShadowCombat's
    monster list must skip it while still recording the LIVE monster's
    original (unfiltered) json_index -- CommunicationMod's own `play <card>
    <target>` command needs that original index, not a position in the
    filtered list. See native_recommend.py's own CRITICAL INVARIANT note."""
    combat_state_json = _combat_json([
        {"id": "JawWorm", "name": "Jaw Worm", "current_hp": 0, "max_hp": 44, "block": 0,
         "intent": "NONE", "is_gone": True, "powers": []},
        {"id": "Cultist", "name": "Cultist", "current_hp": 50, "max_hp": 56, "block": 0,
         "intent": "ATTACK", "move_adjusted_damage": 6, "move_hits": 1, "is_gone": False, "powers": []},
    ])
    description, action, combat = native_recommend(combat_state_json)
    assert len(combat.monsters) == 1  # dead Jaw Worm filtered out
    assert combat.monsters[0].json_index == 1  # but its ORIGINAL position (1) is preserved
    assert combat.monsters[0].name == "Cultist"
    if action[0] == "play" and action[2] is not None:
        assert action[2] is combat.monsters[0]
    print(f"test_dead_monster_index_alignment: OK ({description})")


if __name__ == "__main__":
    import sys
    tests = [
        test_simple_fight_returns_valid_action,
        test_unmapped_monster_raises,
        test_verified_real_capture_monster_ids,
        test_verified_real_capture_status_ids,
        test_taskmaster_fight_no_longer_blocked,
        test_dead_monster_index_alignment,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"FAILED: {t.__name__}: {e}", file=sys.stderr)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
