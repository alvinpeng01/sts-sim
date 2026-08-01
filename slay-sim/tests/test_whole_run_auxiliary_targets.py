from lightspeed.generate_whole_run_rollouts import (
    attach_episode_auxiliary_targets,
)


def test_attach_episode_auxiliary_targets_uses_future_events():
    rows = [
        {"act": 1, "_episode_decision": 2},
        {"act": 2, "_episode_decision": 9},
    ]
    attach_episode_auxiliary_targets(
        rows,
        combat_events=[
            {"decision": 4, "survived": True, "hp_fraction": 0.75},
            {"decision": 12, "survived": False, "hp_fraction": 0.0},
        ],
        rest_steps=[7],
        act_entry_hp={2: 0.60},
        terminal_floor=33,
        terminal_act=2,
        victory=False,
    )

    first = rows[0]["auxiliary_targets"]
    assert first["next_combat_survival"] == 1.0
    assert first["next_combat_hp"] == 0.75
    assert first["next_rest_reach"] == 1.0
    assert first["act_boss_survival"] == 1.0
    assert first["next_act_entry_hp"] == 0.60
    assert first["terminal_floor"] == 33 / 56

    second = rows[1]["auxiliary_targets"]
    assert second["next_combat_survival"] == 0.0
    assert second["next_combat_hp"] == 0.0
    assert second["next_rest_reach"] == 0.0
    assert second["act_boss_survival"] == 0.0
    assert second["next_act_entry_hp"] == 0.0
    assert "_episode_decision" not in rows[0]
    assert "_episode_decision" not in rows[1]
