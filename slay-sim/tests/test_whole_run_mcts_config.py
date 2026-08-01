import slaythespire as sts

from lightspeed.search_config import (
    active_search_config_mismatches,
    ensure_search_config,
    load_search_config,
)
from lightspeed.whole_run_env import RunConfig, WholeRunEnv


def test_whole_run_env_repairs_stale_native_search_state():
    config = load_search_config("lightspeed/tuned_search_params.json")
    # A parameter the artifact does NOT override, so the assertion below tests the
    # half of the repair that active_search_config_mismatches cannot see: reset to
    # the COMPILED default, not merely agreement with the config. Chosen from the
    # live parameter set rather than hardcoded -- this test asserted `block_weight
    # == 0.0` until the 42-parameter tuning run of 2026-07-31 put block_weight in
    # the artifact at 4.139, at which point the assertion was testing nothing but
    # its own staleness and failed.
    sts.reset_search_config()
    defaults = sts.get_search_params()
    untuned = next(k for k in sorted(defaults) if k not in config["params"])

    sts.set_search_params({"c_ucb": 0.01, untuned: 999.0})
    sts.set_seq_halving(False)
    sts.set_rave(True)
    sts.set_state_merging(True)
    sts.set_leaf_eval_mode("value")
    sts.set_early_act_card_biases({1: 123.0})

    WholeRunEnv(RunConfig(combat_sims=17))

    assert active_search_config_mismatches(config) == []
    assert sts.get_seq_halving() is True
    assert sts.get_rave() is False
    assert sts.get_state_merging() is False
    assert sts.get_leaf_eval_mode()[0] == "rollout"
    assert sts.get_search_params()[untuned] == defaults[untuned]
    assert dict(sts.get_early_act_card_biases()) == {}


def test_native_mcts_uses_exact_budget_and_seed_reproducibly():
    ensure_search_config()
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 991_177, 20)
    battle = sts.new_battle(gc, sts.MonsterEncounter.JAW_WORM)

    action_a, visits_a = sts.run_mcts_search(battle, 73, None, 123_456)
    action_b, visits_b = sts.run_mcts_search(battle, 73, None, 123_456)

    assert sum(visits_a) == 73
    assert action_a.bits == action_b.bits
    assert visits_a == visits_b


def test_whole_run_battle_result_reports_actual_search_mode():
    env = WholeRunEnv(RunConfig(
        ascension=20, combat_sims=19, deterministic_combat=True))
    observation = env.reset(884_422)
    for _ in range(20):
        if env.last_battle_result is not None:
            break
        observation, _, done, _ = env.step(0)
        assert not done

    result = env.last_battle_result
    assert result is not None
    assert result["simulations_per_decision"] == 19
    assert result["search_decisions"] > 0
    assert (result["searched_decisions"] + result["forced_decisions"]
            + result["stall_fallback_decisions"] == result["search_decisions"])
    assert result["search_simulations_total"] == result["searched_decisions"] * 19
    assert result["soft_tempo_override_decisions"] >= 0
    assert result["stall_recovery_search_decisions"] >= 0
    assert result["max_consecutive_stall_fallbacks"] <= 3
    assert result["first_stall_turn"] == -1 or result["first_stall_turn"] >= 20
    assert (result["first_tempo_override_turn"] == -1
            or result["first_tempo_override_turn"] >= 12)
    assert result["sequential_halving"] is True
    assert result["deterministic_search"] is True
    assert result["turn_limit_reached"] is False
    assert env.combat_audit["battles"] == env.battles
    assert env.combat_audit["search_decisions"] >= result["search_decisions"]
    assert env.combat_audit["search_simulations_total"] >= result["search_simulations_total"]
    assert isinstance(env.combat_audit["fallback_battles"], list)
    assert env.combat_audit["turn_limit_battles"] == 0
