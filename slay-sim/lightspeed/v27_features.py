"""Structured features used only by the isolated v27 policy."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import slaythespire as sts


CARD_STRUCTURE_SIZE = 14
DECK_SUMMARY_SIZE = 16
STRATEGIC_CONTEXT_SIZE = 12


@lru_cache(maxsize=400)
def _card_structure(card_id: int) -> np.ndarray:
    values = np.zeros(CARD_STRUCTURE_SIZE, dtype=np.float32)
    try:
        card = sts.Card(sts.CardId(int(card_id)))
        card_type = min(4, max(0, int(card.type)))
        rarity = min(4, max(0, int(card.rarity)))
        values[card_type] = 1.0
        values[5 + rarity] = 1.0
        values[10] = float(card.innate)
        values[11] = float(card.is_strikeCard)
        values[12] = float(card.is_starter_strike_or_defend)
        values[13] = float(card.upgraded)
    except (RuntimeError, TypeError, ValueError):
        pass
    return values


def augment_v27_observation(obs):
    """Add deterministic structured features without mutating stored rows."""
    if "v27_strategic_context" in obs:
        return obs
    result = dict(obs)
    deck_ids = np.asarray(obs.get("deck_ids", []), dtype=np.int64)
    upgrades = np.asarray(obs.get("deck_upgrades", []), dtype=np.float32)
    deck_structure = np.asarray(
        [_card_structure(int(card_id)).copy() for card_id in deck_ids],
        dtype=np.float32).reshape((-1, CARD_STRUCTURE_SIZE))
    if len(deck_structure):
        deck_structure[:, 13] = np.maximum(
            deck_structure[:, 13], upgrades[:len(deck_structure)] > 0)
        means = deck_structure.mean(axis=0)
    else:
        means = np.zeros(CARD_STRUCTURE_SIZE, dtype=np.float32)
    result["v27_deck_structure"] = deck_structure
    result["v27_deck_summary"] = np.concatenate((
        means,
        np.asarray((
            min(1.0, len(deck_ids) / 40.0),
            min(1.0, float(np.sum(upgrades > 0)) / max(1, len(deck_ids))),
        ), dtype=np.float32),
    ))

    content = np.asarray(obs.get("action_content_ids", []), dtype=np.int64)
    action_structure = np.zeros(
        (len(content), CARD_STRUCTURE_SIZE), dtype=np.float32)
    for index, content_id in enumerate(content):
        # Card candidate vocabulary is 1 + CardId; relics and potions occupy
        # disjoint ranges and intentionally retain all-zero card metadata.
        if 1 <= int(content_id) < 400:
            action_structure[index] = _card_structure(int(content_id) - 1)
    result["v27_action_card_structure"] = action_structure

    fixed = np.asarray(obs.get("fixed", np.zeros(10)), dtype=np.float32)
    hp, max_hp = float(fixed[0]), max(1.0, float(fixed[1]))
    floor = int(obs.get("floor", fixed[3] if len(fixed) > 3 else 0))
    act = int(obs.get("act", 1))
    result["v27_strategic_context"] = np.asarray((
        hp / max_hp,
        max_hp / 100.0,
        float(fixed[2]) / 500.0,
        floor / 56.0,
        float(fixed[4]) / 10.0,
        float(fixed[5]) / 40.0,
        float(fixed[6]) / 20.0,
        float(fixed[7]),
        float(fixed[8]),
        float(fixed[9]),
        act / 4.0,
        min(3, max(0, floor // 6)) / 3.0,
    ), dtype=np.float32)
    return result
