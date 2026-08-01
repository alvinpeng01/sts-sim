"""Relic-identity features for the policy -- wired into env.py (capture at
reset(), exposed via _observation()'s relic_idxs/relic_mask), policy.py
(relic_embedding, sum-pooled into encode_state), ppo.py/az_search.py/
distillation.py (threaded through alongside `policy` wherever those already
call encode_state/score_actions/choose_action). Started as an isolated
draft while train_relics_v1 (built against the pre-relic-feature 46-input
architecture) was still running; that run was stopped and this was merged
in for real once its sunk cost was low (0.46h into a 4h budget) -- a fresh
training run is required regardless of timing, since STATE_FEATURES itself
doesn't change but the state ENCODER's input width does (see policy.py),
same checkpoint-invalidating category as any other architecture change this
project has always treated as a fresh-training-run event.

Why this exists (see the train_relics_v1 session discussion): several
relics have effects invisible to the network without knowing the relic is
present -- Orichalcum ("gain 6 Block if you end turn with none"), Nunchaku/
Centennial Puzzle-style internal counters -- because those effects are
either delayed, conditional, or tracked by state this project's 46 features
don't expose at all. Only Time Warp's counter got an explicit fix earlier
this session; every other relic-specific counter is currently invisible.

Design, mirroring the existing card_embedding/monster_embedding pattern
(policy.py) rather than a giant multi-hot vector: a learned embedding per
relic, summed across whatever's active this episode, concatenated into the
state encoder's input. Sum-pooling (not concatenation of a fixed per-relic
slot) because relic count varies 0-12 across tiers and relic ORDER has no
game meaning -- same reasoning that already applies to card multisets in a
deck, just for relics instead.

Why capture happens at reset() time from GameContext.relics rather than
querying live during search: BattleContext only exposes has_relic() (bound
to Player::hasRelicRuntime), which this session already found unreliable
for one-time battle-start-effect relics (Vajra's strength grant fires
correctly but the runtime bit doesn't stay set afterward -- confirmed not a
bug, just means has_relic() only reflects relics the ENGINE's own logic
re-checks repeatedly, like Ginger/Turnip, not "does the player own this").
Since relics never change mid-combat, a snapshot taken once at reset() (via
GameContext.relics, before combat starts) is both correct and sidesteps
that unreliability entirely -- nothing during search needs to re-query it,
it just needs to be threaded down alongside `policy` as an argument
wherever encode_state/score_actions currently gets called (env.py's
_observation(), az_search.py's _expand()/choose_action/run_episode_with_
search), same plumbing already used for those.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import slaythespire as sts

from .relics import RELIC_IDS, BURNING_BLOOD

# Every relic this project can actually grant (relics.py's own trainable
# pool, plus Burning Blood which is granted unconditionally rather than
# sampled -- see env.py's reset()). +1 for OTHER_RELIC_INDEX (a defensive
# catch-all, mirroring OTHER_CARD_INDEX/OTHER_MONSTER_INDEX's precedent for
# "should never happen in practice, but don't crash if it does") and +1 for
# PAD_RELIC_INDEX (an empty-slot marker for padding below MAX_ACTIVE_RELICS,
# distinct from OTHER so the network can tell "no relic here" apart from
# "a relic here I don't have a real embedding for").
_TRAINABLE_RELICS = [BURNING_BLOOD] + RELIC_IDS
RELIC_TO_EMBED_IDX = {int(r): i for i, r in enumerate(_TRAINABLE_RELICS)}
OTHER_RELIC_INDEX = len(_TRAINABLE_RELICS)
PAD_RELIC_INDEX = len(_TRAINABLE_RELICS) + 1
RELIC_EMBEDDING_VOCAB_SIZE = len(_TRAINABLE_RELICS) + 2

# Act3 boss tier grants up to 11 extra relics + Burning Blood = 12 (see
# env.py's ACT_TIER_RESOURCES) -- generous headroom, not a tight fit, since
# going over just means excess relics get silently dropped (see
# capture_active_relic_idxs's truncation) rather than crashing.
MAX_ACTIVE_RELICS = 12


def capture_active_relic_idxs(gc) -> List[int]:
    """Snapshot which relics are present, as embedding indices -- call ONCE
    at reset() time, right after granting relics and before hp/deck setup
    (same point env.py already does its own relic-related work), store the
    result on the env wrapper, and thread it through unchanged for the rest
    of that episode (see module docstring for why bc.has_relic() isn't used
    instead)."""
    idxs = [RELIC_TO_EMBED_IDX.get(int(ri.id), OTHER_RELIC_INDEX) for ri in gc.relics]
    return idxs[:MAX_ACTIVE_RELICS]


def pad_relic_idxs(idxs: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed-width (MAX_ACTIVE_RELICS,) idx array + bool mask (True = real
    relic, False = padding) -- same masked-pooling shape convention
    score_actions_batched already uses for variable-length legal-action
    sets, applied here to a variable-length relic SET instead."""
    n = min(len(idxs), MAX_ACTIVE_RELICS)
    padded = np.full(MAX_ACTIVE_RELICS, PAD_RELIC_INDEX, dtype=np.int64)
    mask = np.zeros(MAX_ACTIVE_RELICS, dtype=bool)
    padded[:n] = idxs[:n]
    mask[:n] = True
    return padded, mask
