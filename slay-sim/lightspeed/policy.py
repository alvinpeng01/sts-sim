"""Action-scoring policy network: the architecture discussed for "learn all
cards" -- not a fixed one-hot action vocabulary (poor generalization to
rarely-seen cards), but a shared network that scores each currently-legal
action, conditioned on a learned per-card embedding. Softmax is over however
many actions are legal this turn, not a fixed width.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cards import EMBEDDING_VOCAB_SIZE, OTHER_CARD_INDEX
from .env import STATE_FEATURES, ACTION_FEATURES
from .monsters import MONSTER_EMBEDDING_VOCAB_SIZE, OTHER_MONSTER_INDEX
from .relic_features import RELIC_EMBEDDING_VOCAB_SIZE
from .potion_features import POTION_EMBEDDING_VOCAB_SIZE, EMPTY_POTION_INDEX

CARD_EMBED_DIM = 16
MONSTER_EMBED_DIM = 8
RELIC_EMBED_DIM = 8  # smaller than card/monster: relic identity is a coarser
                     # signal (presence + implicit effect), no per-relic
                     # action features to complement it the way cards/
                     # monsters have their own ACTION_FEATURES entries
POTION_EMBED_DIM = 8  # same coarse-identity reasoning as relics, but potions
                      # ALSO get a per-action lookup (drinking is an active
                      # choice, unlike a relic's passive presence) -- see
                      # potion_features.py's docstring for why potions need
                      # both roles where relics only needed the state one
STATE_HIDDEN = 64
ACTION_HIDDEN = 32


def prep_obs(obs: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """obs (dict from IroncladFightEnv) -> (state, action_features, card_idxs,
    monster_idxs, relic_idxs, relic_mask, action_potion_idxs, potion_idxs,
    potion_mask) tensors. Shared by ActionScoringPolicy.act() and the PPO
    collection loop so both build tensors the same way."""
    state = torch.as_tensor(obs["state"], dtype=torch.float32)
    action_features = torch.as_tensor(np.stack(obs["action_features"]), dtype=torch.float32)
    # END_TURN's card_idx is None -> reuse OTHER_CARD_INDEX as a "no card"
    # row; the explicit is_end_turn feature in ACTION_FEATURES is what
    # actually disambiguates it for the net, not this embedding row alone.
    card_idxs = torch.as_tensor(
        [OTHER_CARD_INDEX if c is None else c for c in obs["action_card_idx"]],
        dtype=torch.long,
    )
    # Same "no target" convention as card_idxs -- END_TURN and untargeted
    # cards get OTHER_MONSTER_INDEX's row, disambiguated for the net by
    # target_is_attacking/is_end_turn already being 0 for those actions.
    monster_idxs = torch.as_tensor(
        [OTHER_MONSTER_INDEX if m is None else m for m in obs["action_monster_idx"]],
        dtype=torch.long,
    )
    relic_idxs = torch.as_tensor(obs["relic_idxs"], dtype=torch.long)
    relic_mask = torch.as_tensor(obs["relic_mask"], dtype=torch.bool)
    # Same None-means-"no lookup for this action" convention as card/monster
    # idxs above -- non-potion actions (card plays, END_TURN) get
    # EMPTY_POTION_INDEX's row, disambiguated by the is_potion action
    # feature (see potion_features.encode_action_with_potion), not by this
    # embedding row alone.
    action_potion_idxs = torch.as_tensor(
        [EMPTY_POTION_INDEX if p is None else p for p in obs["action_potion_idx"]],
        dtype=torch.long,
    )
    potion_idxs = torch.as_tensor(obs["potion_idxs"], dtype=torch.long)
    potion_mask = torch.as_tensor(obs["potion_mask"], dtype=torch.bool)
    return (state, action_features, card_idxs, monster_idxs, relic_idxs, relic_mask,
            action_potion_idxs, potion_idxs, potion_mask)


class ActionScoringPolicy(nn.Module):
    def __init__(self, card_embed_dim: int = CARD_EMBED_DIM, monster_embed_dim: int = MONSTER_EMBED_DIM,
                 relic_embed_dim: int = RELIC_EMBED_DIM, potion_embed_dim: int = POTION_EMBED_DIM):
        super().__init__()
        self.card_embedding = nn.Embedding(EMBEDDING_VOCAB_SIZE, card_embed_dim)
        # Added so the network can condition its strategy on WHICH monster
        # it's targeting (Donu & Deca's focus-fire logic, Reptomancer's
        # dagger-priority, Time Eater's card-count discipline, Awakened
        # One's revive-expectation are all boss-specific, not derivable
        # from anonymous HP/strength/block numbers alone) -- see monsters.py.
        self.monster_embedding = nn.Embedding(MONSTER_EMBEDDING_VOCAB_SIZE, monster_embed_dim)
        # Several relics have effects invisible without knowing they're
        # present -- Orichalcum ("gain 6 Block if you end turn with none"),
        # Nunchaku/Centennial Puzzle-style internal counters -- since
        # STATE_FEATURES doesn't track them (only Time Warp's counter got an
        # explicit fix; see env.py's own comment on that). Sum-pooled over
        # whatever's active rather than concatenated per-slot, same
        # reasoning as card multisets in a deck: relic count varies 0-12
        # across tiers and order has no game meaning. padding_idx not set:
        # a masked sum already excludes padding slots entirely (see
        # encode_state), so the pad row is never looked up regardless.
        self.relic_embedding = nn.Embedding(RELIC_EMBEDDING_VOCAB_SIZE, relic_embed_dim)
        # One shared embedding table serves TWO roles, unlike relics (state-
        # pool only): a state-level sum-pooled "what's currently held"
        # signal (same masked-sum mechanism as relics, via potion_idxs/
        # potion_mask) AND a per-action lookup for "which potion would THIS
        # action use" (via action_potion_idxs, since drinking a potion is an
        # active per-decision choice, unlike a relic's passive presence --
        # see potion_features.py's docstring). padding_idx not set, same
        # reasoning as relic_embedding: EMPTY_POTION_INDEX's row IS looked
        # up (unlike relic padding, an empty potion SLOT is real
        # information -- see potion_features.py), so it needs a real,
        # trainable embedding, not a zeroed-out one.
        self.potion_embedding = nn.Embedding(POTION_EMBEDDING_VOCAB_SIZE, potion_embed_dim)

        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_FEATURES + relic_embed_dim + potion_embed_dim, STATE_HIDDEN),
            nn.ReLU(),
            nn.Linear(STATE_HIDDEN, STATE_HIDDEN),
            nn.ReLU(),
        )

        # Scores one action: [state_embedding ; action_features ; card_embedding ;
        # monster_embedding ; action_potion_embedding] -> scalar. END_TURN/
        # untargeted cards get a zero vector in the card/monster-embedding
        # slots (disambiguated for the net via ACTION_FEATURES' own explicit
        # is_end_turn/target_is_attacking flags, not by the zero vectors
        # alone); non-potion actions get EMPTY_POTION_INDEX's row in the
        # potion slot, disambiguated by the is_potion action feature.
        self.scorer = nn.Sequential(
            nn.Linear(STATE_HIDDEN + ACTION_FEATURES + card_embed_dim + monster_embed_dim + potion_embed_dim,
                       ACTION_HIDDEN),
            nn.ReLU(),
            nn.Linear(ACTION_HIDDEN, 1),
        )

        # State-value baseline for PPO's advantage estimate (return - value)
        # -- reduces gradient variance versus REINFORCE's raw-return
        # advantage. Not used by plain REINFORCE (train.py ignores it).
        self.value_head = nn.Linear(STATE_HIDDEN, 1)

    @staticmethod
    def _masked_pool(embedding: nn.Embedding, idxs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Shared by relic and potion state-pooling: sum embeddings over
        real (mask=True) slots, zero vector if nothing's active. idxs/mask:
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
        """state: (state_dim,) or (batch, state_dim). Returns scalar V(s) (or (batch,))."""
        return self.value_head(
            self.encode_state(state, relic_idxs, relic_mask, potion_idxs, potion_mask)
        ).squeeze(-1)

    def score_actions(self, state: torch.Tensor, action_features: torch.Tensor,
                      card_idxs: torch.Tensor, monster_idxs: torch.Tensor, action_potion_idxs: torch.Tensor,
                      relic_idxs: torch.Tensor, relic_mask: torch.Tensor,
                      potion_idxs: torch.Tensor, potion_mask: torch.Tensor,
                      state_emb: torch.Tensor = None) -> torch.Tensor:
        """state: (state_dim,). action_features: (n_actions, action_dim).
        card_idxs/monster_idxs/action_potion_idxs: (n_actions,) int64 --
        action_potion_idxs is WHICH potion this specific action would use
        (EMPTY_POTION_INDEX for non-potion actions), distinct from potion_idxs
        below. relic_idxs/relic_mask/potion_idxs/potion_mask: fixed-width,
        constant across all actions this decision (state-level, not
        per-action), unlike card/monster/action_potion idxs. Returns
        (n_actions,) raw scores (not yet softmaxed, so callers can mask/
        combine across batches if needed). Pass a precomputed state_emb to
        avoid re-running the state encoder when the caller already has it
        (e.g. alongside a value())."""
        n_actions = action_features.shape[0]
        if state_emb is None:
            state_emb = self.encode_state(state, relic_idxs, relic_mask, potion_idxs, potion_mask)
        state_emb = state_emb.unsqueeze(0).expand(n_actions, -1)
        card_emb = self.card_embedding(card_idxs)
        monster_emb = self.monster_embedding(monster_idxs)
        action_potion_emb = self.potion_embedding(action_potion_idxs)
        combined = torch.cat([state_emb, action_features, card_emb, monster_emb, action_potion_emb], dim=-1)
        return self.scorer(combined).squeeze(-1)

    def score_actions_batched(self, state_batch: torch.Tensor, action_features_padded: torch.Tensor,
                              card_idxs_padded: torch.Tensor, monster_idxs_padded: torch.Tensor,
                              action_potion_idxs_padded: torch.Tensor,
                              relic_idxs_batch: torch.Tensor, relic_mask_batch: torch.Tensor,
                              potion_idxs_batch: torch.Tensor, potion_mask_batch: torch.Tensor,
                              mask: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized form of score_actions() across N steps at once, for
        PPO's batched update (see ppo.py) -- the per-step Python-loop version
        measured ~4.8x slower in practice (21.7 vs 103.9 eps/sec), purely an
        implementation artifact of scoring one state's variable-length legal-
        action set at a time rather than batching, not anything inherent to
        PPO's algorithm.

        state_batch: (N, STATE_FEATURES). action_features_padded: (N, A,
        ACTION_FEATURES) where A = max legal-action count in this batch,
        shorter steps zero-padded. card_idxs_padded/monster_idxs_padded/
        action_potion_idxs_padded: (N, A) int64, same padding.
        relic_idxs_batch/relic_mask_batch: (N, MAX_ACTIVE_RELICS).
        potion_idxs_batch/potion_mask_batch: (N, MAX_POTION_SLOTS) -- one
        set per STEP (not per action, unlike card/monster/action_potion
        idxs), since these are the state-level "what's held" signal.
        mask: (N, A) bool, True at real (non-padding) positions.

        Returns (scores, state_emb): scores is (N, A) with padded positions
        set to a large FINITE negative number (-1e9, not literal -inf) so
        downstream softmax/entropy stays numerically finite -- exp(-1e9) at
        float32 correctly underflows to exact 0.0 without ever multiplying
        0 by inf, which is what actually produces NaN. state_emb (N,
        STATE_HIDDEN) is returned too so the value head can reuse it
        without a second forward pass through the state encoder."""
        N, A, _ = action_features_padded.shape
        state_emb = self.encode_state(state_batch, relic_idxs_batch, relic_mask_batch,
                                       potion_idxs_batch, potion_mask_batch)
        state_emb_exp = state_emb.unsqueeze(1).expand(N, A, -1)
        card_emb = self.card_embedding(card_idxs_padded)
        monster_emb = self.monster_embedding(monster_idxs_padded)
        action_potion_emb = self.potion_embedding(action_potion_idxs_padded)
        combined = torch.cat([state_emb_exp, action_features_padded, card_emb, monster_emb, action_potion_emb], dim=-1)
        scores = self.scorer(combined).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        return scores, state_emb

    def act(self, obs: dict, sample: bool = True) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """obs: the dict IroncladFightEnv.reset()/step() returns. Returns
        (chosen_index_into_obs['actions'], log_prob, scores) for training."""
        (state, action_features, card_idxs, monster_idxs, relic_idxs, relic_mask,
         action_potion_idxs, potion_idxs, potion_mask) = prep_obs(obs)
        scores = self.score_actions(state, action_features, card_idxs, monster_idxs, action_potion_idxs,
                                     relic_idxs, relic_mask, potion_idxs, potion_mask)
        probs = F.softmax(scores, dim=-1)
        # validate_args=False: see ppo.py's collect_batch/ppo_update for the
        # same fix with the profiling that motivated it -- safe here too,
        # `probs` is always fresh softmax output.
        dist = torch.distributions.Categorical(probs=probs, validate_args=False)
        idx = dist.sample() if sample else torch.argmax(probs)
        return int(idx.item()), dist.log_prob(idx), scores
