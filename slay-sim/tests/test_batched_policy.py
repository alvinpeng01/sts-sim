"""The batched forward must agree with the single-observation one it replaces.

`forward_batch` exists to make on-policy RL affordable, but every number this
project has on record came from the per-observation path. So that path is the
oracle: if these two disagree, the loop is right and the batch is wrong.

Agreement is checked on REAL observations -- states an actual policy reaches
walking an actual seed -- rather than synthetic ones, because the batching is
conditional on `screen`, `act` and `floor // 6`, and synthetic observations would
not exercise the grouped heads those select.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from lightspeed.batched_policy import forward_batch
from lightspeed.whole_run_env import RunConfig, WholeRunEnv
from lightspeed.whole_run_transformer_v27 import WholeRunTransformerPolicyV27


def _collect(limit: int = 40):
    """Real observations from a real run, taken with the untrained policy."""
    torch.manual_seed(0)
    policy = WholeRunTransformerPolicyV27(dim=96, layers=2, heads=4)
    policy.eval()
    env = WholeRunEnv(RunConfig(combat_sims=8, ascension=20))
    observations = []
    for episode in range(4):
        obs = env.reset(77_000_000 + episode)
        done = False
        while not done and len(observations) < limit:
            observations.append(obs)
            with torch.no_grad():
                logits, _ = policy(obs)
            obs, _, done, _ = env.step(int(torch.argmax(logits)))
        if len(observations) >= limit:
            break
    return policy, observations


@pytest.fixture(scope="module")
def collected():
    return _collect()


def test_batched_logits_match_single_observation(collected):
    policy, observations = collected
    assert len(observations) >= 8, "need a few real decisions to compare"

    with torch.no_grad():
        reference = [policy(obs) for obs in observations]
        logits, mask, values = forward_batch(policy, observations)

    for i, (ref_logits, ref_value) in enumerate(reference):
        count = ref_logits.shape[0]
        assert int(mask[i].sum()) == count, f"action count differs at row {i}"
        assert torch.allclose(logits[i, :count], ref_logits, atol=1e-4), (
            f"logits differ at row {i}: "
            f"max |delta| = {(logits[i, :count] - ref_logits).abs().max():.2e}")
        assert torch.allclose(values[i], ref_value, atol=1e-4), (
            f"value differs at row {i}")


def test_padding_is_negative_infinity(collected):
    """Padded slots must not survive a softmax, so PPO needs no extra masking."""
    policy, observations = collected
    with torch.no_grad():
        logits, mask, _ = forward_batch(policy, observations)
    if bool((~mask).any()):
        assert torch.isinf(logits[~mask]).all()
        probabilities = torch.softmax(logits, dim=-1)
        assert float(probabilities[~mask].abs().max()) == 0.0


def test_batch_of_one_matches(collected):
    """The degenerate case, since a PPO minibatch can end up with one step."""
    policy, observations = collected
    with torch.no_grad():
        ref_logits, ref_value = policy(observations[0])
        logits, mask, values = forward_batch(policy, observations[:1])
    assert torch.allclose(logits[0, :ref_logits.shape[0]], ref_logits, atol=1e-4)
    assert torch.allclose(values[0], ref_value, atol=1e-4)


def test_gradients_flow_through_the_batch(collected):
    """PPO backprops through this; a detached path would train nothing."""
    policy, observations = collected
    logits, mask, values = forward_batch(policy, observations[:8])
    loss = logits[mask].sum() + values.sum()
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert any(bool(torch.any(g != 0)) for g in grads)
    assert all(bool(torch.isfinite(g).all()) for g in grads), (
        "a -inf padding slot leaked into the gradient")
