"""Convert curated Baalorlord floor histories into audited imitation rows.

The archive is not simulator-complete.  This importer therefore emits only
decisions whose chosen action and visible alternatives are known:

* ordinary post-combat card rewards (including explicit skips), and
* boss relic rewards.

State is reconstructed from the preceding floor history.  These rows are
useful as a human prior, but should be simulator-relabelled before promotion.
Every output row retains provenance and an explicit state-quality marker.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
import re
from typing import Any, Iterable

import numpy as np
import slaythespire as sts
import torch


STARTING_CARDS = [
    *(["Strike"] * 5),
    *(["Defend"] * 4),
    "Bash",
    "Ascender's Bane",
]
CARD_REWARD_NODES = {
    "fight_normal", "fight_elite", "event_fight", "boss_node",
}


def _canonical(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _split_upgrade(name: str) -> tuple[str, int]:
    match = re.match(r"^(.*?)(?:\+(\d*)| \+(\d*))?$", name.strip())
    if not match:
        return name.strip(), 0
    base = match.group(1).strip()
    suffix = match.group(2) or match.group(3)
    upgraded = 1 if name.strip().endswith("+") else int(suffix or 0)
    return base, upgraded


def _card_name_map() -> dict[str, int]:
    result: dict[str, int] = {}
    for enum_value in sts.CardId.__members__.values():
        if enum_value == sts.CardId.INVALID:
            continue
        rendered = repr(sts.Card(enum_value))
        match = re.match(r"<slaythespire\.Card (.*?)(?:\+\d*)?>$", rendered)
        if match:
            result[_canonical(match.group(1))] = int(enum_value)
        result[_canonical(enum_value.name)] = int(enum_value)
    return result


def _enum_name_map(enum_type) -> dict[str, int]:
    return {
        _canonical(name): int(value)
        for name, value in enum_type.__members__.items()
        if name != "INVALID"
    }


CARD_IDS = _card_name_map()
RELIC_IDS = _enum_name_map(sts.RelicId)
POTION_IDS = _enum_name_map(sts.Potion)
# The archive drops the "Potion" suffix on a few names and spells two others
# differently than the enum does.
POTION_ALIASES = {
    "elixir": "elixir_potion",
    "gamblers_brew": "gamblers_potion",
    "fairy_in_a_bottle": "fairy_potion",
}

# Potion slots are 3 at base and 2 from Ascension 11 up; this dataset is all A20.
POTION_SLOTS_A20 = 2


def resolve_card(name: str) -> tuple[int, int] | None:
    base, upgraded = _split_upgrade(name)
    card_id = CARD_IDS.get(_canonical(base))
    return None if card_id is None else (card_id, upgraded)


def resolve_relic(name: str) -> int | None:
    return RELIC_IDS.get(_canonical(name))


def resolve_potion(name: str) -> int | None:
    key = _canonical(name)
    return POTION_IDS.get(POTION_ALIASES.get(key, key))


def _max_multiset(*groups: Iterable[str]) -> list[str]:
    merged: Counter[str] = Counter()
    for group in groups:
        counts = Counter(group or ())
        for item, count in counts.items():
            merged[item] = max(merged[item], count)
    return list(merged.elements())


class ReconstructedRun:
    def __init__(self):
        self.deck: list[tuple[int, int]] = []
        self.relics: list[int] = []
        # Potion inventory held ENTERING a floor. The archive records what was
        # obtained, used and discarded per floor, so the inventory is a replay of
        # those three. It matters for anything comparing our combat to his: he
        # drinks something in 23.7% of fights and 38.6% of elites, so a
        # potionless reconstruction hands him a resource we never get.
        self.potions: list[int] = []
        self.red_key = self.green_key = self.blue_key = 0
        self.unresolved_state_items: Counter[str] = Counter()
        for name in STARTING_CARDS:
            resolved = resolve_card(name)
            if resolved:
                self.deck.append(resolved)
        burning_blood = resolve_relic("Burning Blood")
        if burning_blood is not None:
            self.relics.append(burning_blood)

    def _remove_card(self, name: str) -> None:
        resolved = resolve_card(name)
        if resolved is None:
            self.unresolved_state_items[f"card:{name}"] += 1
            return
        card_id, upgraded = resolved
        candidates = [
            i for i, (existing_id, existing_upgrade) in enumerate(self.deck)
            if existing_id == card_id
            and (existing_upgrade == upgraded or upgraded == 0)
        ]
        if candidates:
            self.deck.pop(candidates[0])

    def _upgrade_card(self, name: str) -> None:
        resolved = resolve_card(name)
        if resolved is None:
            self.unresolved_state_items[f"card:{name}"] += 1
            return
        card_id, requested_upgrade = resolved
        for i, (existing_id, existing_upgrade) in enumerate(self.deck):
            if existing_id == card_id and existing_upgrade == 0:
                self.deck[i] = (card_id, max(1, requested_upgrade))
                return

    def apply_floor(self, row: dict[str, Any]) -> None:
        raw = row.get("raw_detail", "")
        self.red_key |= int("Ruby key" in raw)
        self.green_key |= int("Emerald Key" in raw)
        self.blue_key |= int("Sapphire key" in raw)

        shop = row.get("shop_purchases") or {}
        removals = _max_multiset(
            row.get("cards_removed", ()), shop.get("removals", ()))
        for name in removals:
            self._remove_card(name)
        for name in row.get("cards_upgraded", ()):
            self._upgrade_card(name)

        additions = _max_multiset(
            row.get("card_picked", ()),
            row.get("cards_obtained", ()),
            shop.get("cards", ()),
        )
        for name in additions:
            # The archive renders Singing Bowl's +2 max-HP choice under
            # "Picked". It is the reward-screen skip action, not a card.
            if name == "Singing Bowl":
                continue
            resolved = resolve_card(name)
            if resolved is None:
                self.unresolved_state_items[f"card:{name}"] += 1
            else:
                self.deck.append(resolved)

        relic_additions = _max_multiset(
            row.get("relics_obtained", ()), shop.get("relics", ()))
        for name in relic_additions:
            relic_id = resolve_relic(name)
            if relic_id is None:
                self.unresolved_state_items[f"relic:{name}"] += 1
            elif relic_id not in self.relics:
                self.relics.append(relic_id)

        # Consume before acquiring: a potion bought and drunk on the same floor
        # must not survive into the next one, and the archive gives no ordering
        # within a floor to tell us otherwise.
        for name in list(row.get("potions_used", ())) + list(row.get("potions_discarded", ())):
            potion_id = resolve_potion(name)
            if potion_id is None:
                self.unresolved_state_items[f"potion:{name}"] += 1
            elif potion_id in self.potions:
                self.potions.remove(potion_id)
        for name in row.get("potions_obtained", ()):
            potion_id = resolve_potion(name)
            if potion_id is None:
                self.unresolved_state_items[f"potion:{name}"] += 1
            else:
                self.potions.append(potion_id)
        # Over-cap means the archive shows him holding more than the slots allow,
        # which happens when a floor's obtained/used ordering is ambiguous. Keep
        # the most recent, since those are the ones he had not yet had a chance
        # to drink.
        if len(self.potions) > POTION_SLOTS_A20:
            self.potions = self.potions[-POTION_SLOTS_A20:]


def _base_observation(
    state: ReconstructedRun,
    row: dict[str, Any],
    screen: int,
    candidate_count: int,
) -> dict[str, Any]:
    hp = int(row.get("hp_current") or 0)
    max_hp = int(row.get("hp_max") or max(hp, 1))
    gold = int(row.get("gold") or 0)
    floor = int(row["floor"])
    fixed = np.asarray([
        hp, max_hp, gold, floor, 0, 0, int(row.get("ascension") or 20),
        state.red_key, state.green_key, state.blue_key,
    ], dtype=np.int16)
    return {
        "fixed": fixed,
        "deck_ids": np.asarray([card_id for card_id, _ in state.deck], dtype=np.int16),
        "deck_upgrades": np.asarray([upgrade for _, upgrade in state.deck], dtype=np.int8),
        "relic_ids": np.asarray(state.relics, dtype=np.int16),
        "relic_counters": np.zeros(len(state.relics), dtype=np.int16),
        "potions": np.empty(0, dtype=np.int16),
        "map_xs": np.empty(0, dtype=np.int8),
        "map_ys": np.empty(0, dtype=np.int8),
        "map_rooms": np.empty(0, dtype=np.int8),
        "map_paths": np.empty((0, 3), dtype=np.int8),
        "map_x": -1,
        "map_y": -1,
        "act": int(row["act"]),
        "floor": floor,
        "screen": screen,
        "action_bits": np.arange(candidate_count, dtype=np.int64),
        "action_target_rooms": np.zeros(candidate_count, dtype=np.int64),
        "action_target_coords": np.full((candidate_count, 2), -1.0, dtype=np.float32),
        "action_prices": np.zeros((candidate_count, 2), dtype=np.float32),
        "action_event_ids": np.zeros(candidate_count, dtype=np.int64),
        "action_neow_bonuses": np.zeros(candidate_count, dtype=np.int64),
        "action_neow_drawbacks": np.zeros(candidate_count, dtype=np.int64),
    }


def card_reward_demo(
    state: ReconstructedRun,
    row: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    if row.get("map_node") not in CARD_REWARD_NODES:
        return None, "not_card_reward_node"
    picked = list(row.get("card_picked", ()))
    skipped = list(row.get("skipped", ()))
    if len(picked) > 1:
        return None, "multiple_picks_ambiguous"
    if not skipped:
        return None, "no_visible_alternatives"
    bowl_choice = picked == ["Singing Bowl"]
    actual_picks = [] if bowl_choice else picked
    resolved_offers = [resolve_card(name) for name in actual_picks + skipped]
    if any(item is None for item in resolved_offers):
        return None, "unresolved_card"

    # The archive lists picked cards first. Add the always-legal skip action.
    candidates = actual_picks + skipped
    chosen_index = 0 if actual_picks else len(candidates)
    candidate_ids = [item[0] for item in resolved_offers]
    candidate_ids.append(None)
    candidate_names = candidates + [
        "Singing Bowl (+2 Max HP)" if bowl_choice else "Skip"
    ]
    reward_types = [
        int(sts.RewardsActionType.CARD) for _ in candidates
    ] + [int(sts.RewardsActionType.SKIP)]
    obs = _base_observation(
        state, row, int(sts.ScreenState.REWARDS), len(candidate_ids))
    obs["action_features"] = np.asarray([
        [i / 96.0, 0.0, 0.0, reward_type / 8.0,
         int(sts.ScreenState.REWARDS) / 10.0, 1.0]
        for i, reward_type in enumerate(reward_types)
    ], dtype=np.float32)
    obs["action_content_ids"] = np.asarray([
        0 if card_id is None else 1 + card_id for card_id in candidate_ids
    ], dtype=np.int64)
    obs["action_text"] = candidate_names
    target = np.zeros(len(candidate_ids), dtype=np.float32)
    target[chosen_index] = 1.0
    return {
        "observation": obs,
        "target_probabilities": target,
        "decision_type": "human_card_reward",
        "confidence": "high_action_set_approximate_state",
        "provenance": {
            "run_id": row["run_id"], "floor": int(row["floor"]),
            "seed": row["seed"], "source_url": row["source_url"],
        },
    }, None


def boss_relic_demo(
    state: ReconstructedRun,
    row: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    if row.get("map_node") != "boss_chest":
        return None, "not_boss_chest"
    picked = list(row.get("relics_obtained", ()))
    skipped = list(row.get("relics_skipped", ()))
    if len(picked) != 1 or len(skipped) < 1:
        return None, "incomplete_boss_relic_set"
    names = picked + skipped
    ids = [resolve_relic(name) for name in names]
    if any(relic_id is None for relic_id in ids):
        return None, "unresolved_relic"
    obs = _base_observation(
        state, row, int(sts.ScreenState.BOSS_RELIC_REWARDS), len(ids))
    obs["action_features"] = np.asarray([
        [i / 96.0, 0.0, 0.0, int(sts.RewardsActionType.RELIC) / 8.0,
         int(sts.ScreenState.BOSS_RELIC_REWARDS) / 10.0, 1.0]
        for i in range(len(ids))
    ], dtype=np.float32)
    obs["action_content_ids"] = np.asarray(
        [400 + relic_id for relic_id in ids], dtype=np.int64)
    obs["action_text"] = names
    target = np.zeros(len(ids), dtype=np.float32)
    target[0] = 1.0
    return {
        "observation": obs,
        "target_probabilities": target,
        "decision_type": "human_boss_relic",
        "confidence": "high_action_set_approximate_state",
        "provenance": {
            "run_id": row["run_id"], "floor": int(row["floor"]),
            "seed": row["seed"], "source_url": row["source_url"],
        },
    }, None


def load_runs(path: str) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("character") != "Ironclad" or int(row.get("ascension") or -1) != 20:
                raise ValueError(f"line {line_number}: expected Ironclad A20")
            runs[str(row["run_id"])].append(row)
    for rows in runs.values():
        rows.sort(key=lambda row: int(row["floor"]))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--validation-out", default=None)
    parser.add_argument(
        "--validation-every", type=int, default=5,
        help="with --validation-out, hold out every Nth sorted run")
    args = parser.parse_args()

    runs = load_runs(args.input)
    output_rows = []
    rejected = Counter()
    unresolved = Counter()
    per_run = {}
    for run_id, floor_rows in runs.items():
        state = ReconstructedRun()
        before = len(output_rows)
        for row in floor_rows:
            for extractor in (card_reward_demo, boss_relic_demo):
                demo, reason = extractor(state, row)
                if demo is not None:
                    output_rows.append(demo)
                elif reason not in ("not_card_reward_node", "not_boss_chest"):
                    rejected[f"{extractor.__name__}:{reason}"] += 1
            state.apply_floor(row)
        unresolved.update(state.unresolved_state_items)
        per_run[run_id] = len(output_rows) - before

    decision_counts = Counter(row["decision_type"] for row in output_rows)
    audit = {
        "source": os.path.abspath(args.input),
        "runs": len(runs),
        "floors": sum(len(rows) for rows in runs.values()),
        "demonstrations": len(output_rows),
        "decision_counts": dict(decision_counts),
        "rejected": dict(rejected),
        "unresolved_state_items": dict(unresolved),
        "per_run_demonstrations": per_run,
        "state_quality": "reconstructed from prior floor summaries; no map, potion, "
                         "boss identity, Neow, relic counters, or exact pre-decision RNG",
    }
    train_rows = output_rows
    validation_rows = []
    validation_run_ids: list[str] = []
    if args.validation_out:
        sorted_run_ids = sorted(runs)
        validation_run_ids = [
            run_id for index, run_id in enumerate(sorted_run_ids)
            if index % max(2, args.validation_every) == 0
        ]
        validation_ids = set(validation_run_ids)
        train_rows = [
            row for row in output_rows
            if row["provenance"]["run_id"] not in validation_ids
        ]
        validation_rows = [
            row for row in output_rows
            if row["provenance"]["run_id"] in validation_ids
        ]
        audit["training_demonstrations"] = len(train_rows)
        audit["validation_demonstrations"] = len(validation_rows)
        audit["validation_run_ids"] = validation_run_ids

    metadata = {
        "kind": "baalorlord_human_demonstrations",
        "audit": audit,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "rows": train_rows,
        "metadata": {
            **metadata, "split": "train" if args.validation_out else "all",
        },
    }, args.out)
    if args.validation_out:
        os.makedirs(os.path.dirname(args.validation_out) or ".", exist_ok=True)
        torch.save({
            "rows": validation_rows,
            "metadata": {**metadata, "split": "validation"},
        }, args.validation_out)
    audit_path = args.audit or args.out + ".audit.json"
    with open(audit_path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
