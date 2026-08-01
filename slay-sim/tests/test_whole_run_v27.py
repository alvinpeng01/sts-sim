import torch

from lightspeed.whole_run_env import RunConfig, WholeRunEnv
from lightspeed.whole_run_transformer import WholeRunTransformerPolicy
from lightspeed.whole_run_transformer_v27 import WholeRunTransformerPolicyV27


def test_v27_is_exact_noop_when_loading_a_base_policy():
    torch.manual_seed(27)
    base = WholeRunTransformerPolicy().eval()
    v27 = WholeRunTransformerPolicyV27().eval()
    missing, unexpected = v27.load_state_dict(base.state_dict(), strict=False)
    assert missing
    assert unexpected == []
    env = WholeRunEnv(RunConfig(combat_sims=7, deterministic_combat=True))
    observation = env.reset(727_027)
    with torch.no_grad():
        expected, _ = base(observation)
        actual, _, auxiliary, uncertainty, ensemble = v27.forward_detailed(
            observation)
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(uncertainty) == 0
    assert ensemble.shape == (3, len(expected))
    assert set(auxiliary) == {
        "next_combat_survival", "next_combat_hp", "next_rest_reach",
        "act_boss_survival", "next_act_entry_hp", "terminal_floor",
    }


def test_v27_neow_has_an_isolated_decision_expert():
    assert WholeRunTransformerPolicyV27._expert_id({
        "screen": 1,
        "action_neow_bonuses": [1, 2],
        "action_neow_drawbacks": [0, 0],
    }) == 9
    assert WholeRunTransformerPolicyV27._expert_id({
        "screen": 1,
        "action_neow_bonuses": [0, 0],
        "action_neow_drawbacks": [0, 0],
    }) == 1
    assert WholeRunTransformerPolicyV27._expert_id({"screen": 5}) == 5


def test_v27_fast_forward_preserves_ensemble_mean_and_action():
    torch.manual_seed(270)
    policy = WholeRunTransformerPolicyV27().eval()
    env = WholeRunEnv(RunConfig(combat_sims=7, deterministic_combat=True))
    observation = env.reset(727_270)
    with torch.no_grad():
        detailed, detailed_value, *_ = policy.forward_detailed(observation)
        fast, fast_value = policy(observation)
        action, log_prob, _, logits = policy.act(observation, sample=False)
    assert torch.allclose(fast, detailed, rtol=1e-6, atol=1e-7)
    assert torch.equal(fast_value, detailed_value)
    assert torch.equal(logits, fast)
    assert action == int(torch.argmax(detailed))
    assert torch.allclose(
        log_prob, torch.log_softmax(detailed, dim=-1)[action],
        rtol=1e-6, atol=1e-7)
