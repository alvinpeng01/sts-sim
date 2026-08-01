"""DRAFT -- combines policy_relic_draft.py's relic-aware architecture with
potion_features.py's potion-identity embedding into one policy, so promoting
both happens as ONE architecture change / ONE retrain once train_relics_v1
finishes, instead of two separate invalidating changes back to back. See
policy_relic_draft.py's own docstring for why this stays a standalone class
(state_encoder's input width changes) and potion_features.py's docstring for
why potions need BOTH a state-level pooled embedding (what's held right now)
and an action-level one (which specific potion this action would use) where
relics only ever needed the state-level kind (a relic is never itself an
action).

Not yet wired into the live env.py/az_search.py, same reason as both drafts
above: train_relics_v1's per-chunk worker respawn re-imports env.py/policy.py
from disk, so live edits there would reach and break it mid-run.

Promotion checklist once train_relics_v1 finishes (extends
policy_relic_draft.py's own 4-step list with the potion-specific pieces):
  1. Merge this class into policy.py (or replace ActionScoringPolicy outright).
  2. env.py's reset() calls relic_features.capture_active_relic_idxs(gc) once
     (relics: static for the episode); _observation() calls
     potion_features.capture_active_potion_idxs(bc) EVERY call (potions:
     change turn to turn) and potion_features.encode_action_with_potion(...)
     in place of _encode_action_and_card_idx for each legal action.
  3. ACTION_FEATURES becomes 12 + POTION_ACTION_FEATURES (14); env.py's
     IroncladFightEnv gains potion_generator/potion_count constructor params
     mirroring relic_generator/relic_count, granting via gc.obtain_potion(...)
     in reset() (after relics, before the hp override, same ordering
     rationale as relics -- see env.py's own comment on why).
  4. az_search.py's _expand()/choose_action/run_episode_with_search/
     evaluate_with_search all gain relic_idxs/relic_mask (threaded once from
     the root, unchanged all search) and potion_idxs/potion_mask (recomputed
     at every _expand() call, since potions can be consumed mid-search).
  5. Rebuild slaythespire.pyd (obtain_potion/Potion enum/potions property/
     POTION-action getLegalActions support -- already written this session,
     just blocked on the same DLL lock train_relics_v1 holds).
  6. Retrain from scratch -- state encoder input width and scorer input width
     both change, invalidating every existing checkpoint's first layer,
     same rule as every prior architecture change this project has made.
  7. Broad crash-fuzz the potion pool the same way relics were (see
     potions.py's docstring on Attack/Colorless/Power/Skill/Liquid Memories/
     Entropic Brew/Gambler's Brew being un-verified-but-plausible) --
     exclude whatever actually crashes, same methodology, before committing
     to a long real run.

Tested standalone below against synthetic tensors, no env.py/live-training
dependency.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .cards import EMBEDDING_VOCAB_SIZE, OTHER_CARD_INDEX
from .env import STATE_FEATURES, ACTION_FEATURES
from .monsters import MONSTER_EMBEDDING_VOCAB_SIZE, OTHER_MONSTER_INDEX
from .relic_features import RELIC_EMBEDDING_VOCAB_SIZE, MAX_ACTIVE_RELICS
from .potion_features import POTION_EMBEDDING_VOCAB_SIZE, MAX_POTION_SLOTS, POTION_ACTION_FEATURES, EMPTY_POTION_INDEX

CARD_EMBED_DIM = 16
MONSTER_EMBED_DIM = 8
RELIC_EMBED_DIM = 8
POTION_EMBED_DIM = 8  # same order of magnitude as relics: identity + implicit-presence signal, not a rich per-card-like feature
STATE_HIDDEN = 64
ACTION_HIDDEN = 32
NEW_ACTION_FEATURES = ACTION_FEATURES + POTION_ACTION_FEATURES  # 12 + 2 = 14


class RelicAndPotionAwareActionScoringPolicy(nn.Module):
    def __init__(self, card_embed_dim: int = CARD_EMBED_DIM, monster_embed_dim: int = MONSTER_EMBED_DIM,
                 relic_embed_dim: int = RELIC_EMBED_DIM, potion_embed_dim: int = POTION_EMBED_DIM):
        super().__init__()
        self.card_embedding = nn.Embedding(EMBEDDING_VOCAB_SIZE, card_embed_dim)
        self.monster_embedding = nn.Embedding(MONSTER_EMBEDDING_VOCAB_SIZE, monster_embed_dim)
        self.relic_embedding = nn.Embedding(RELIC_EMBEDDING_VOCAB_SIZE, relic_embed_dim)
        # One shared potion embedding table serves BOTH roles (state-level
        # pooled "what's held" and action-level "which potion is this
        # action") -- see potion_features.py's docstring for why potions
        # need both while relics only needed the state-level kind.
        self.potion_embedding = nn.Embedding(POTION_EMBEDDING_VOCAB_SIZE, potion_embed_dim)

        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_FEATURES + relic_embed_dim + potion_embed_dim, STATE_HIDDEN),
            nn.ReLU(),
            nn.Linear(STATE_HIDDEN, STATE_HIDDEN),
            nn.ReLU(),
        )

        self.scorer = nn.Sequential(
            nn.Linear(STATE_HIDDEN + NEW_ACTION_FEATURES + card_embed_dim + monster_embed_dim + potion_embed_dim,
                       ACTION_HIDDEN),
            nn.ReLU(),
            nn.Linear(ACTION_HIDDEN, 1),
        )

        self.value_head = nn.Linear(STATE_HIDDEN, 1)

    @staticmethod
    def _masked_pool(embedding: nn.Embedding, idxs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Shared by relic and potion state-pooling -- sum embeddings over
        real (mask=True) slots, zero vector when nothing's active. idxs/mask:
        (K,) or (batch, K) for either K=MAX_ACTIVE_RELICS or K=MAX_POTION_SLOTS."""
        emb = embedding(idxs)
        return (emb * mask.unsqueeze(-1).float()).sum(dim=-2)

    def encode_state(self, state: torch.Tensor, relic_idxs: torch.Tensor, relic_mask: torch.Tensor,
                      potion_idxs: torch.Tensor, potion_mask: torch.Tensor) -> torch.Tensor:
        relic_emb = self._masked_pool(self.relic_embedding, relic_idxs, relic_mask)
        potion_emb = self._masked_pool(self.potion_embedding, potion_idxs, potion_mask)
        return self.state_encoder(torch.cat([state, relic_emb, potion_emb], dim=-1))

    def value(self, state: torch.Tensor, relic_idxs: torch.Tensor, relic_mask: torch.Tensor,
              potion_idxs: torch.Tensor, potion_mask: torch.Tensor) -> torch.Tensor:
        return self.value_head(
            self.encode_state(state, relic_idxs, relic_mask, potion_idxs, potion_mask)
        ).squeeze(-1)

    def score_actions(self, state: torch.Tensor, action_features: torch.Tensor,
                       card_idxs: torch.Tensor, monster_idxs: torch.Tensor, action_potion_idxs: torch.Tensor,
                       relic_idxs: torch.Tensor, relic_mask: torch.Tensor,
                       potion_idxs: torch.Tensor, potion_mask: torch.Tensor,
                       state_emb: torch.Tensor = None) -> torch.Tensor:
        """action_potion_idxs: per-action potion identity (EMPTY_POTION_INDEX
        for any non-potion action, e.g. card plays/END_TURN -- distinct role
        from potion_idxs/potion_mask above, which is the STATE-level pooled
        "what's currently held" signal shared across every action this call
        scores)."""
        n_actions = action_features.shape[0]
        if state_emb is None:
            state_emb = self.encode_state(state, relic_idxs, relic_mask, potion_idxs, potion_mask)
        state_emb = state_emb.unsqueeze(0).expand(n_actions, -1)
        card_emb = self.card_embedding(card_idxs)
        monster_emb = self.monster_embedding(monster_idxs)
        action_potion_emb = self.potion_embedding(action_potion_idxs)
        combined = torch.cat([state_emb, action_features, card_emb, monster_emb, action_potion_emb], dim=-1)
        return self.scorer(combined).squeeze(-1)


if __name__ == "__main__":
    policy = RelicAndPotionAwareActionScoringPolicy()

    relic_idxs = torch.full((MAX_ACTIVE_RELICS,), 0, dtype=torch.long)
    relic_mask = torch.zeros(MAX_ACTIVE_RELICS, dtype=torch.bool)
    potion_idxs = torch.full((MAX_POTION_SLOTS,), EMPTY_POTION_INDEX, dtype=torch.long)
    potion_mask = torch.zeros(MAX_POTION_SLOTS, dtype=torch.bool)

    state = torch.randn(STATE_FEATURES)
    state_emb = policy.encode_state(state, relic_idxs, relic_mask, potion_idxs, potion_mask)
    assert state_emb.shape == (STATE_HIDDEN,)

    value = policy.value(state, relic_idxs, relic_mask, potion_idxs, potion_mask)
    assert value.shape == ()

    n_actions = 6
    action_features = torch.randn(n_actions, NEW_ACTION_FEATURES)
    card_idxs = torch.full((n_actions,), OTHER_CARD_INDEX, dtype=torch.long)
    monster_idxs = torch.full((n_actions,), OTHER_MONSTER_INDEX, dtype=torch.long)
    action_potion_idxs = torch.full((n_actions,), EMPTY_POTION_INDEX, dtype=torch.long)
    scores = policy.score_actions(state, action_features, card_idxs, monster_idxs, action_potion_idxs,
                                   relic_idxs, relic_mask, potion_idxs, potion_mask)
    assert scores.shape == (n_actions,)

    # A real held potion should change the state embedding vs. all-empty.
    potion_idxs2 = potion_idxs.clone()
    potion_idxs2[0] = 3
    potion_mask2 = potion_mask.clone()
    potion_mask2[0] = True
    state_emb2 = policy.encode_state(state, relic_idxs, relic_mask, potion_idxs2, potion_mask2)
    assert not torch.allclose(state_emb, state_emb2), "a held potion should change the pooled state embedding"

    print("RelicAndPotionAwareActionScoringPolicy: all shape/sanity checks passed")
    print(f"  NEW_ACTION_FEATURES={NEW_ACTION_FEATURES}, POTION_EMBED_DIM={POTION_EMBED_DIM}, "
          f"RELIC_EMBED_DIM={RELIC_EMBED_DIM}")
