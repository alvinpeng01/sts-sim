"""Batched forward for the whole-run transformer.

Nothing in this project has ever run this model on a batch. `forward` takes one
observation dict; `train_whole_run_v27.py`'s `--batch` is gradient ACCUMULATION
over a per-row Python loop (`:182`), not batching; every evaluator steps one
decision at a time. That was affordable for supervised training on 4,008 labels
and is not affordable for on-policy RL, which re-forwards every collected step
once per epoch: measured 22.3 ms per step for forward+backward, i.e. ~35 minutes
for one 512-game PPO iteration whose collection takes 0.4 minutes on 11 workers.

Threads do not help -- the model is dispatch-bound, not FLOP-bound, and goes
22.3 -> 38.1 ms/step from 1 to 12 threads. Batching is the only lever, and the
headroom is uneven, which is why this batches what it batches:

    encoder (60 tokens, dim 96, 2 layers)   1.56 -> 0.67 ms/obs    2.3x
    a dim*3 -> dim -> 1 scoring head       0.090 -> 0.001 ms/item   86x

The encoder does real arithmetic and has a hard floor; the per-action heads are
almost pure dispatch overhead and nearly vanish. So the win comes from flattening
every action in the batch into ONE (total_actions, 3*dim) matrix and calling each
head once.

Three things in `forward_detailed` select a module with a Python int and are what
made this non-trivial -- `act_score[act_id]`, `phase_score[act*4 + floor//6]` and
`decision_experts[expert_id]`, plus two `screen in (...)` conditionals. Each is
handled by grouping rows and applying the module to its subset: at most 5, 20 and
10 groups respectively, so a batch of any real size amortises them.

The single-observation path is left exactly as it is. It is what every evaluator
and checkpoint comparison on record used, and this file is verified AGAINST it --
see `tests/test_batched_policy.py`, which asserts agreement to 1e-4 on real
observations. If the two ever disagree, the loop is right and this is wrong.

    from .batched_policy import forward_batch
    logits, mask, values = forward_batch(policy, [obs_a, obs_b, ...])
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .v27_features import augment_v27_observation


def _group_apply(modules, index: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Apply `modules[i]` to the rows whose index is `i`, in one pass per group.

    The alternative -- one module call per ROW -- is what the single-observation
    path effectively does across a batch, and it is the cost this file exists to
    remove. `index` is per-row, so this is a scatter of at most len(modules)
    batched calls regardless of how many rows there are.
    """
    out = rows.new_zeros(rows.shape[0])
    for module_id in torch.unique(index).tolist():
        selected = index == module_id
        out[selected] = modules[module_id](rows[selected]).squeeze(-1)
    return out


def _masked_apply(module, mask: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Apply `module` only to rows where `mask` is true; zero elsewhere.

    Mirrors the `if int(obs["screen"]) in (...)` conditionals, which are
    per-observation and therefore per-row once actions are flattened.
    """
    out = rows.new_zeros(rows.shape[0])
    if bool(mask.any()):
        out[mask] = module(rows[mask]).squeeze(-1)
    return out


def _stack_actions(obs_list, key, width, dtype):
    """Concatenate one per-action field across the batch into (total_actions, width).

    Falls back to the same defaults `forward_detailed` uses for an observation
    that predates a field, so an old cached observation batches identically to
    the way it forwards today.
    """
    parts = []
    for obs in obs_list:
        count = len(obs["action_content_ids"])
        value = obs.get(key)
        if value is None:
            value = np.zeros((count, width), dtype=dtype) if width > 1 else np.zeros(count, dtype=dtype)
        array = np.asarray(value, dtype=dtype)
        parts.append(array.reshape(count, width) if width > 1 else array.reshape(count))
    return np.concatenate(parts, axis=0)


def forward_batch(policy, raw_obs_list, diagnostics: bool = False):
    """Logits for every observation in one pass.

    Returns `(logits, mask, values)` where logits is (B, max_actions) padded with
    -inf, mask is (B, max_actions) marking real actions, and values is (B,).
    -inf padding means a softmax over the row is already correct without a
    separate masking step, which is what every PPO consumer wants.
    """
    if not raw_obs_list:
        raise ValueError("forward_batch needs at least one observation")
    obs_list = [augment_v27_observation(obs) for obs in raw_obs_list]
    device = next(policy.parameters()).device
    batch = len(obs_list)

    # --- state tokens -------------------------------------------------------
    # Token construction stays per-observation: the sequences are ragged in a
    # way that differs per token TYPE (deck size, relic count, map size all vary
    # independently), so batching it means padding four separate groups. The
    # encoder is the expensive part and it is batched below; this is the
    # deliberate stopping point for a first verified version.
    token_seqs = [policy._state_tokens(obs, device)[0] for obs in obs_list]
    lengths = [seq.shape[0] for seq in token_seqs]
    max_len = max(lengths)
    padded = torch.zeros(batch, max_len, policy.dim, device=device)
    # True marks a padded slot. Attention gives those -inf weight, so the CLS
    # output at index 0 is unaffected by padding -- the property that makes this
    # equal to the unpadded per-observation encode.
    key_padding = torch.ones(batch, max_len, dtype=torch.bool, device=device)
    for i, (seq, length) in enumerate(zip(token_seqs, lengths)):
        padded[i, :length] = seq
        key_padding[i, :length] = False
    state = policy.encoder(padded, src_key_padding_mask=key_padding)[:, 0]

    deck_summary = torch.as_tensor(
        np.stack([obs["v27_deck_summary"] for obs in obs_list]),
        dtype=torch.float32, device=device)
    strategic = torch.as_tensor(
        np.stack([obs["v27_strategic_context"] for obs in obs_list]),
        dtype=torch.float32, device=device)
    state = state + policy.deck_summary_adapter(deck_summary)
    state = state + policy.strategic_context_adapter(strategic)

    # --- per-action features, flattened across the batch ---------------------
    counts = torch.tensor([len(obs["action_content_ids"]) for obs in obs_list],
                          dtype=torch.long, device=device)

    def flat(key, width, dtype=np.float32):
        array = _stack_actions(obs_list, key, width, dtype)
        return torch.as_tensor(
            array, dtype=torch.float32 if dtype == np.float32 else torch.long,
            device=device)

    features = flat("action_features", 6)
    content = flat("action_content_ids", 1, np.int64).clamp(0, 699)
    action = policy.action_features(features)
    action_content = policy.action_content(content)
    action = (
        action
        + policy.target_room(flat("action_target_rooms", 1, np.int64).clamp(0, 15))
        + policy.target_coord(flat("action_target_coords", 2))
        + policy.route_cone(flat("action_route_cones", 4))
        + policy.route_resources(flat("action_route_resources", 4))
        + policy.shop_price(flat("action_prices", 2))
        + policy.event(flat("action_event_ids", 1, np.int64).clamp(0, 63))
        + policy.neow_bonus(flat("action_neow_bonuses", 1, np.int64).clamp(0, 31))
        + policy.neow_drawback(flat("action_neow_drawbacks", 1, np.int64).clamp(0, 15))
        + policy.action_card_structure_adapter(flat("v27_action_card_structure", 14))
    )

    # Per-observation integers, expanded to one entry per action row so every
    # grouped head below can index rows directly.
    screens = torch.tensor([min(8, max(0, int(obs.get("screen", 0)))) for obs in obs_list],
                           dtype=torch.long, device=device)
    acts = torch.tensor([min(4, max(0, int(obs.get("act", 1)))) for obs in obs_list],
                        dtype=torch.long, device=device)
    phases = acts * 4 + torch.tensor(
        [min(3, max(0, int(obs.get("floor", 0))) // 6) for obs in obs_list],
        dtype=torch.long, device=device)
    experts = torch.tensor([policy._expert_id(obs) for obs in obs_list],
                           dtype=torch.long, device=device)
    row_screen = torch.repeat_interleave(screens, counts)
    row_act = torch.repeat_interleave(acts, counts)
    row_phase = torch.repeat_interleave(phases, counts)
    row_expert = torch.repeat_interleave(experts, counts)

    consequence_rows = (row_screen == 2) | (row_screen == 3) | (row_screen == 8)
    consequences = flat("action_consequences", 10)
    if bool(consequence_rows.any()):
        contribution = torch.zeros_like(action)
        contribution[consequence_rows] = policy.action_consequence(
            consequences[consequence_rows])
        action = action + contribution

    action = (action * (1.0 + policy.act_action_scale(row_act))
              + policy.act_action(row_act))
    score_input = torch.cat(
        (torch.repeat_interleave(state, counts, dim=0), action, action_content), dim=1)

    logits_flat = policy.score(score_input).squeeze(-1)
    logits_flat = logits_flat + _group_apply(policy.act_score, row_act, score_input)
    logits_flat = logits_flat + _group_apply(policy.phase_score, row_phase, score_input)
    logits_flat = logits_flat + _masked_apply(
        policy.human_score, (row_screen == 2) | (row_screen == 3), score_input)
    logits_flat = logits_flat + _group_apply(
        policy.decision_experts, row_expert, score_input)

    if diagnostics:
        ensemble = torch.stack([head(score_input).squeeze(-1)
                                for head in policy.uncertainty_heads])
        logits_flat = logits_flat + ensemble.mean(dim=0)
    else:
        # The same fused three-head mean the single-observation path uses. Its
        # cache is keyed on device/dtype there; reuse that logic rather than
        # duplicating the fusion, so the two cannot drift.
        cached = policy._fast_ensemble_weights
        if (cached is None or cached[0].device != score_input.device
                or cached[0].dtype != score_input.dtype):
            policy._refresh_fast_ensemble()
            cached = policy._fast_ensemble_weights
        first_weight, first_bias, second_weight, second_bias = cached
        logits_flat = logits_flat + F.linear(
            F.gelu(F.linear(score_input, first_weight, first_bias)),
            second_weight, second_bias).squeeze(-1)

    # --- scatter back into a padded (B, max_actions) matrix ------------------
    max_actions = int(counts.max())
    logits = torch.full((batch, max_actions), float("-inf"), device=device)
    mask = torch.zeros(batch, max_actions, dtype=torch.bool, device=device)
    offset = 0
    for i, count in enumerate(counts.tolist()):
        logits[i, :count] = logits_flat[offset:offset + count]
        mask[i, :count] = True
        offset += count
    return logits, mask, policy.value(state).squeeze(-1)
