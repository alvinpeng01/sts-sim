"""DRAFT -- potion-identity features for the policy, mirroring
relic_features.py's structure but NOT identical to it: relics are captured
ONCE at reset() (they never change mid-combat), potions are actively drunk/
discarded turn to turn, so held-potion state has to be recomputed from
bc.potions on every decision, same cadence _encode_state already runs at.
Potions also need an ACTION-level embedding on top of the state-level one
(relics never appear as an action -- their effect is passive -- but "drink
potion X" is a real per-decision choice the policy has to condition on),
so this covers both roles with one shared embedding table.

Not yet wired into the live env.py/policy.py, for the same reason
relic_features.py isn't: train_relics_v1 is running against the current
architecture, and each PPO training chunk spawns a FRESH multiprocessing
worker pool that re-imports env.py/policy.py from disk (confirmed by
reading ppo.py's train_ppo -- mp.Pool is created inside train_ppo itself,
called once per chunk, not once for the whole run) -- so editing those
files' shared STATE_FEATURES/ACTION_FEATURES/_encode_state/
_encode_action_and_card_idx WOULD reach and break the next chunk's workers
mid-run, unlike a plain new module which no running code imports yet.

This module is self-contained and safe to import/test right now. See this
file's __main__ block and policy_relic_draft.py's combined promotion notes
for what wiring both relics and potions in together (one architecture
change, one retrain, not two) looks like once train_relics_v1 finishes.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import slaythespire as sts

from .potions import POTION_IDS

# Every potion this project can actually grant (potions.py's own trainable
# pool). +1 for OTHER_POTION_INDEX (defensive catch-all, same precedent as
# OTHER_CARD_INDEX/OTHER_RELIC_INDEX) and +1 for EMPTY_POTION_INDEX -- unlike
# relics' PAD_RELIC_INDEX (purely structural, always masked out), an empty
# potion SLOT is real game state worth its own embedding row: "I have an
# open slot" is different information from "I have no information about
# this slot at all" (which never happens here, since bc.potions is always
# read up to potionCapacity, not padded past it).
_TRAINABLE_POTIONS = POTION_IDS
POTION_TO_EMBED_IDX = {int(p): i for i, p in enumerate(_TRAINABLE_POTIONS)}
OTHER_POTION_INDEX = len(_TRAINABLE_POTIONS)
EMPTY_POTION_INDEX = len(_TRAINABLE_POTIONS) + 1
POTION_EMBEDDING_VOCAB_SIZE = len(_TRAINABLE_POTIONS) + 2

# BattleContext.h's potions array is a fixed std::array<Potion,5> --
# potionCapacity (default 3, only grows via Potion Belt/similar relics this
# project doesn't grant) is the REAL slot count, but padding to the array's
# max size is simplest and gives headroom if capacity-increasing relics are
# added to the trainable pool later without needing to revisit this file.
MAX_POTION_SLOTS = 5


def _potion_embed_idx(p) -> int:
    if p == sts.Potion.EMPTY_POTION_SLOT or p == sts.Potion.INVALID:
        return EMPTY_POTION_INDEX
    return POTION_TO_EMBED_IDX.get(int(p), OTHER_POTION_INDEX)


def capture_active_potion_idxs(bc) -> List[int]:
    """Snapshot of held-potion embedding indices -- call EVERY decision
    (unlike relics' capture-once-at-reset), since potions change turn to
    turn. bc.potions is already potionCapacity-long (see the binding's own
    comment), so no separate capacity lookup is needed here."""
    return [_potion_embed_idx(p) for p in bc.potions]


def pad_potion_idxs(idxs: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed-width (MAX_POTION_SLOTS,) idx array + bool mask. mask=True for
    every slot actually returned by capture_active_potion_idxs (EMPTY slots
    included -- an empty slot is real information, see module docstring),
    mask=False only for padding past potionCapacity up to MAX_POTION_SLOTS."""
    n = min(len(idxs), MAX_POTION_SLOTS)
    padded = np.full(MAX_POTION_SLOTS, EMPTY_POTION_INDEX, dtype=np.int64)
    mask = np.zeros(MAX_POTION_SLOTS, dtype=bool)
    padded[:n] = idxs[:n]
    mask[:n] = True
    return padded, mask


# --- action-level: 2 new scalar features + a potion_idx, layered on top of
# env.py's existing 12 ACTION_FEATURES + card_idx/monster_idx. Kept as a
# wrapper around the live _encode_action_and_card_idx (calling it is safe --
# reading/invoking a live function doesn't touch anything the running job
# depends on, only EDITING env.py in place would) rather than a duplicated
# reimplementation, so this can't silently drift out of sync with it.

POTION_ACTION_FEATURES = 2  # is_potion, is_potion_discard -- appended, not replacing the existing 12


def encode_action_with_potion(bc, action, total_living_hp: int, hand: list):
    """Extends env.py's _encode_action_and_card_idx with is_potion/
    is_potion_discard flags and a potion_idx (None for non-potion actions,
    same None-means-"no embedding lookup for this action" convention
    card_idx/monster_idx already use for END_TURN). Returns
    (action_features [14-wide], card_idx, monster_idx, potion_idx)."""
    from .env import _encode_action_and_card_idx  # local import: draft-only dependency, not a module-level coupling

    base_features, card_idx, monster_idx = _encode_action_and_card_idx(bc, action, total_living_hp, hand)
    is_potion = action.action_type == sts.ActionType.POTION
    potion_idx = None
    is_discard = 0.0
    if is_potion:
        potion_idx = _potion_embed_idx(bc.potions[action.source_idx])
        is_discard = 1.0 if action.target_idx > 5 else 0.0
    extra = np.array([1.0 if is_potion else 0.0, is_discard], dtype=np.float32)
    return np.concatenate([base_features, extra]), card_idx, monster_idx, potion_idx


if __name__ == "__main__":
    # Standalone sanity checks -- synthetic idx lists only, no BattleContext
    # dependency (capture_active_potion_idxs itself needs a live bc, tested
    # separately once this gets a real end-to-end smoke test post-rebuild).
    idxs_empty, mask_empty = pad_potion_idxs([])
    assert not mask_empty.any(), "no slots captured -> nothing should be masked in"

    idxs_a, mask_a = pad_potion_idxs([EMPTY_POTION_INDEX, 3, OTHER_POTION_INDEX])
    assert mask_a.tolist() == [True, True, True, False, False]
    assert idxs_a[3] == EMPTY_POTION_INDEX and idxs_a[4] == EMPTY_POTION_INDEX, "padding beyond captured slots uses EMPTY_POTION_INDEX"

    print("potion_features: all shape/sanity checks passed")
    print(f"  POTION_EMBEDDING_VOCAB_SIZE={POTION_EMBEDDING_VOCAB_SIZE}, MAX_POTION_SLOTS={MAX_POTION_SLOTS}, "
          f"POTION_ACTION_FEATURES={POTION_ACTION_FEATURES}")
