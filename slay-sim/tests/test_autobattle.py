"""Correctness checks for the bridge's autobattle mode (sts/bridge/
communication_mod.py) -- the toggle, and _build_command's translation from
a chosen (action, combat) into the literal string CommunicationMod expects.
Doesn't touch the real ~/sts_autobattle_enabled.txt / ~/sts_latest_
recommendation.txt paths -- each test points the module's path constants
at a scratch tempdir first, restoring them afterward, so this never reads
or writes the actual user's files."""

import tempfile
from pathlib import Path

import pytest

from sts.bridge import communication_mod as cm
from sts.bridge.state_mapper import build_combat_state


@pytest.fixture
def scratch_paths():
    scratch = Path(tempfile.mkdtemp())
    orig = (cm.LATEST_PATH, cm.AUTOBATTLE_PATH, cm.LOG_PATH)
    cm.LATEST_PATH = scratch / "latest.txt"
    cm.AUTOBATTLE_PATH = scratch / "autobattle.txt"
    cm.LOG_PATH = scratch / "predictions.log"
    yield scratch
    cm.LATEST_PATH, cm.AUTOBATTLE_PATH, cm.LOG_PATH = orig


def test_autobattle_defaults_off_when_file_missing(scratch_paths):
    assert cm._autobattle_enabled() is False


def test_autobattle_reads_true_and_false(scratch_paths):
    cm.AUTOBATTLE_PATH.write_text("true")
    assert cm._autobattle_enabled() is True
    cm.AUTOBATTLE_PATH.write_text("false")
    assert cm._autobattle_enabled() is False
    cm.AUTOBATTLE_PATH.write_text("garbage")
    assert cm._autobattle_enabled() is False


def _sample_combat():
    """One dead monster (json_index 0) ahead of the one living monster
    (json_index 1) -- the whole point of json_index is that combat.monsters
    compacts past the dead one, so this is the case that actually exercises
    it (a fight with no deaths yet wouldn't catch a bug here)."""
    combat_state_json = {
        "player": {"current_hp": 70, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
        "hand": [{"id": "Defend_R", "upgrades": 0}, {"id": "Bash", "upgrades": 0}],
        "draw_pile": [{"id": "Strike_R", "upgrades": 0} for _ in range(5)],
        "discard_pile": [], "exhaust_pile": [],
        "monsters": [
            {"id": "JawWorm", "name": "Jaw Worm", "current_hp": 0, "max_hp": 44,
             "block": 0, "intent": "NONE", "is_gone": True, "powers": []},
            {"id": "Cultist", "name": "Cultist", "current_hp": 50, "max_hp": 56,
             "block": 0, "intent": "ATTACK", "move_adjusted_damage": 6, "move_hits": 1,
             "is_gone": False, "powers": []},
        ],
    }
    return build_combat_state(combat_state_json)


def test_build_command_play_uses_1_based_hand_index_and_json_monster_index():
    combat = _sample_combat()
    bash_card = combat.hand[1]  # Bash, 0-based hand index 1
    cultist = combat.monsters[0]  # only living monster, but json_index is 1 (Jaw Worm died first)
    command = cm._build_command(("play", bash_card, cultist), combat, ["state", "play", "end"])
    assert command == "play 2 1"


def test_build_command_play_untargeted_omits_target_index():
    combat = _sample_combat()
    defend_card = combat.hand[0]
    command = cm._build_command(("play", defend_card, None), combat, ["state", "play", "end"])
    assert command == "play 1"


def test_build_command_end():
    combat = _sample_combat()
    assert cm._build_command(("end",), combat, ["state", "play", "end"]) == "end"


def test_build_command_refuses_when_command_type_not_available():
    combat = _sample_combat()
    bash_card = combat.hand[1]
    cultist = combat.monsters[0]
    assert cm._build_command(("play", bash_card, cultist), combat, ["state", "end"]) is None
    assert cm._build_command(("end",), combat, ["state", "play"]) is None


def test_build_command_refuses_card_no_longer_in_hand():
    combat = _sample_combat()
    from sts.cards import make_bash
    phantom_card = make_bash()  # a different instance, never actually in this hand
    cultist = combat.monsters[0]
    assert cm._build_command(("play", phantom_card, cultist), combat, ["state", "play", "end"]) is None


def test_handle_state_returns_none_when_autobattle_off(scratch_paths):
    payload = {
        "game_state": {"combat_state": {
            "player": {"current_hp": 70, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "hand": [{"id": "Defend_R", "upgrades": 0}, {"id": "Bash", "upgrades": 0}],
            "draw_pile": [{"id": "Strike_R", "upgrades": 0} for _ in range(5)],
            "discard_pile": [], "exhaust_pile": [],
            "monsters": [{"id": "Cultist", "name": "Cultist", "current_hp": 50, "max_hp": 56,
                          "block": 0, "intent": "ATTACK", "move_adjusted_damage": 6, "move_hits": 1,
                          "is_gone": False, "powers": []}],
            "turn": 1,
        }},
        "ready_for_command": True,
        "available_commands": ["state", "play", "end"],
    }
    assert cm.handle_state(payload) is None


def test_handle_state_returns_a_command_when_autobattle_on(scratch_paths):
    cm.AUTOBATTLE_PATH.write_text("true")
    payload = {
        "game_state": {"combat_state": {
            "player": {"current_hp": 70, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "hand": [{"id": "Defend_R", "upgrades": 0}, {"id": "Bash", "upgrades": 0}],
            "draw_pile": [{"id": "Strike_R", "upgrades": 0} for _ in range(5)],
            "discard_pile": [], "exhaust_pile": [],
            "monsters": [{"id": "Cultist", "name": "Cultist", "current_hp": 50, "max_hp": 56,
                          "block": 0, "intent": "ATTACK", "move_adjusted_damage": 6, "move_hits": 1,
                          "is_gone": False, "powers": []}],
            "turn": 1,
        }},
        "ready_for_command": True,
        "available_commands": ["state", "play", "end"],
    }
    command = cm.handle_state(payload)
    assert command is not None
    assert command == "end" or command.startswith("play ")
    # Overlay should visibly reflect that autobattle acted, not just recommended.
    overlay = cm.LATEST_PATH.read_text()
    assert "[autobattle ->" in overlay


def test_handle_state_never_acts_on_v1_fallback(scratch_paths):
    """An unmapped monster id forces the v1 damage-only fallback (no
    CombatState, no action) -- autobattle must not invent a command from
    that, even with the toggle on."""
    cm.AUTOBATTLE_PATH.write_text("true")
    payload = {
        "game_state": {"combat_state": {
            "player": {"current_hp": 70, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "hand": [{"id": "Defend_R", "upgrades": 0}],
            "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
            "monsters": [{"id": "TotallyUnknownMonsterId", "name": "???", "current_hp": 50,
                          "max_hp": 56, "block": 0, "intent": "ATTACK",
                          "move_adjusted_damage": 6, "move_hits": 1, "is_gone": False, "powers": []}],
            "turn": 1,
        }},
        "ready_for_command": True,
        "available_commands": ["state", "play", "end"],
    }
    assert cm.handle_state(payload) is None
