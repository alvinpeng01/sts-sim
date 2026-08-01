"""Small native whole-run RL environment.

This is deliberately independent of Silverbot: the simulator owns the state,
legal actions, and combat.  A policy only chooses between currently legal
overworld actions; battles are resolved by our native expectimax MCTS.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import slaythespire as sts

from .search_config import DEFAULT_SEARCH_CONFIG_PATH, ensure_search_config


MAP_LOOKAHEAD_CAP = 8


def action_consequence_features(gc, action) -> tuple[np.ndarray, bool]:
    """Probe deterministic immediate effects without advancing the real run."""
    branch = gc.copy()
    potion_count_before = sum(
        int(int(p) != 1) for p in sts.getNNRepresentation(gc).potions)
    before = (
        int(gc.cur_hp), int(gc.max_hp), int(gc.gold), len(gc.deck),
        len(gc.relics), potion_count_before,
        int(gc.floor_num), int(gc.screen_state),
    )
    action.execute(branch)
    potion_count_after = sum(
        int(int(p) != 1) for p in sts.getNNRepresentation(branch).potions)
    after = (
        int(branch.cur_hp), int(branch.max_hp), int(branch.gold), len(branch.deck),
        len(branch.relics), potion_count_after,
        int(branch.floor_num), int(branch.screen_state),
    )
    loses_now = branch.outcome == sts.GameOutcome.PLAYER_LOSS
    features = np.asarray((
        (after[0] - before[0]) / max(1.0, float(before[1])),
        (after[1] - before[1]) / max(1.0, float(before[1])),
        (after[2] - before[2]) / 300.0,
        (after[3] - before[3]) / 10.0,
        (after[4] - before[4]) / 5.0,
        (after[5] - before[5]) / 3.0,
        (after[6] - before[6]) / 16.0,
        float(branch.screen_state == sts.ScreenState.BATTLE),
        float(loses_now),
        float(after[7] != before[7]),
    ), dtype=np.float32)
    return features, loses_now


def partition_legal_actions(gc):
    """Return safe candidates and number of provably immediate losses removed."""
    if gc.outcome != sts.GameOutcome.UNDECIDED:
        return [], 0
    actions = list(sts.GameAction.getAllActionsInState(gc))
    if len(actions) <= 1:
        return actions, 0
    safe, immediately_losing = [], []
    for action in actions:
        try:
            _, loses_now = action_consequence_features(gc, action)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            loses_now = False
        (immediately_losing if loses_now else safe).append(action)
    if safe and immediately_losing:
        return safe, len(immediately_losing)
    return actions, 0


def map_route_features(
        map_rep, target_xs: list[int], map_y: int) -> tuple[np.ndarray, np.ndarray]:
    """Return choice-local route lookahead and resource/risk exposure.

    The raw map remains in the observation.  These aggregates ground each path
    candidate in the part of the DAG it selects, avoiding the need for the
    policy to learn multi-hop graph traversal merely to compare two routes.
    """
    xs = [int(v) for v in map_rep.xs]
    ys = [int(v) for v in map_rep.ys]
    rooms = [int(v) for v in map_rep.room_types]
    paths = np.asarray(map_rep.path_xs, dtype=np.int16).reshape((-1, 3))
    index = {(x, y): i for i, (x, y) in enumerate(zip(xs, ys))}
    successors: list[list[int]] = [[] for _ in xs]
    for i, (y, edges) in enumerate(zip(ys, paths)):
        for edge_x in edges:
            child = index.get((int(edge_x), y + 1))
            if int(edge_x) >= 0 and child is not None:
                successors[i].append(child)

    elite, rest = int(sts.Room.ELITE), int(sts.Room.REST)
    monster, shop = int(sts.Room.MONSTER), int(sts.Room.SHOP)
    min_elites = [0] * len(xs)
    max_elites = [0] * len(xs)
    min_monsters = [0] * len(xs)
    max_monsters = [0] * len(xs)
    rest_distance = [MAP_LOOKAHEAD_CAP] * len(xs)
    elite_distance = [MAP_LOOKAHEAD_CAP] * len(xs)
    shop_distance = [MAP_LOOKAHEAD_CAP] * len(xs)
    for i in sorted(range(len(xs)), key=lambda j: -ys[j]):
        is_elite = int(rooms[i] == elite)
        is_monster = int(rooms[i] == monster)
        children = successors[i]
        min_elites[i] = is_elite + (min(min_elites[j] for j in children) if children else 0)
        max_elites[i] = is_elite + (max(max_elites[j] for j in children) if children else 0)
        min_monsters[i] = is_monster + (
            min(min_monsters[j] for j in children) if children else 0)
        max_monsters[i] = is_monster + (
            max(max_monsters[j] for j in children) if children else 0)
        if rooms[i] == rest:
            rest_distance[i] = 0
        elif children:
            rest_distance[i] = min(
                MAP_LOOKAHEAD_CAP, 1 + min(rest_distance[j] for j in children))
        if rooms[i] == elite:
            elite_distance[i] = 0
        elif children:
            elite_distance[i] = min(
                MAP_LOOKAHEAD_CAP, 1 + min(elite_distance[j] for j in children))
        if rooms[i] == shop:
            shop_distance[i] = 0
        elif children:
            shop_distance[i] = min(
                MAP_LOOKAHEAD_CAP, 1 + min(shop_distance[j] for j in children))

    burning_x = int(getattr(map_rep, "burning_elite_x", -1))
    burning_y = int(getattr(map_rep, "burning_elite_y", -1))
    burning_index = index.get((burning_x, burning_y))
    reaches_burning = [False] * len(xs)
    if burning_index is not None:
        for i in sorted(range(len(xs)), key=lambda j: -ys[j]):
            reaches_burning[i] = (
                i == burning_index
                or any(reaches_burning[j] for j in successors[i])
            )

    destination_y = int(map_y) + 1
    cones = []
    resources = []
    for target_x in target_xs:
        i = index.get((int(target_x), destination_y))
        if i is None:
            cones.append((0.0, 0.0, 1.0, 0.0))
            resources.append((0.0, 0.0, 1.0, 1.0))
        else:
            cones.append((
                min_elites[i] / MAP_LOOKAHEAD_CAP,
                max_elites[i] / MAP_LOOKAHEAD_CAP,
                rest_distance[i] / MAP_LOOKAHEAD_CAP,
                float(reaches_burning[i]),
            ))
            resources.append((
                min_monsters[i] / MAP_LOOKAHEAD_CAP,
                max_monsters[i] / MAP_LOOKAHEAD_CAP,
                elite_distance[i] / MAP_LOOKAHEAD_CAP,
                shop_distance[i] / MAP_LOOKAHEAD_CAP,
            ))
    return (
        np.asarray(cones, dtype=np.float32).reshape((-1, 4)),
        np.asarray(resources, dtype=np.float32).reshape((-1, 4)),
    )


@dataclass
class RunConfig:
    ascension: int = 20
    # Strong enough to be a meaningful combat oracle; callers can lower this
    # for fast curriculum rollouts or raise it for final evaluation.
    combat_sims: int = 300
    max_decisions: int = 256
    deterministic_combat: bool = False
    # Whole-run combat must not silently inherit native defaults or stale
    # process-global experiment settings.
    search_config_path: str | Path | None = DEFAULT_SEARCH_CONFIG_PATH


class WholeRunEnv:
    """Gym-like interface with candidate actions instead of a fixed action ID space."""

    def __init__(self, config: RunConfig | None = None):
        self.config = config or RunConfig()
        if self.config.search_config_path is not None:
            ensure_search_config(self.config.search_config_path)
        self.gc = None
        self.steps = 0
        self.battles = 0
        self.search_seed_base = 0
        self.last_battle_result = None
        self.combat_audit = {}

    def _reset_combat_audit(self) -> None:
        self.combat_audit = {
            "battles": 0,
            "search_decisions": 0,
            "searched_decisions": 0,
            "forced_decisions": 0,
            "stall_fallback_decisions": 0,
            "stall_progress_override_decisions": 0,
            "soft_tempo_override_decisions": 0,
            "stall_recovery_search_decisions": 0,
            "search_simulations_total": 0,
            "turn_limit_battles": 0,
            "safety_filter_events": 0,
            "immediate_loss_actions_filtered": 0,
            "fallback_battles": [],
        }

    def _record_battle_audit(self, result: dict[str, Any]) -> None:
        self.combat_audit["battles"] += 1
        for key in (
                "search_decisions", "searched_decisions", "forced_decisions",
                "stall_fallback_decisions", "stall_progress_override_decisions",
                "soft_tempo_override_decisions",
                "stall_recovery_search_decisions", "search_simulations_total"):
            self.combat_audit[key] += int(result.get(key, 0))
        self.combat_audit["turn_limit_battles"] += int(
            bool(result.get("turn_limit_reached", False)))
        if (int(result.get("stall_fallback_decisions", 0)) > 0
                or int(result.get("soft_tempo_override_decisions", 0)) > 0):
            self.combat_audit["fallback_battles"].append({
                "floor": int(self.gc.floor_num),
                "encounter": int(result.get("encounter", -1)),
                "is_boss": bool(result.get("is_boss", False)),
                "outcome": int(result.get("outcome", -1)),
                "player_hp": int(result.get("player_hp", -1)),
                "monster_hp": int(result.get("monster_hp", -1)),
                "turn": int(result.get("turn", -1)),
                "stall_fallback_decisions": int(
                    result.get("stall_fallback_decisions", 0)),
                "stall_progress_override_decisions": int(
                    result.get("stall_progress_override_decisions", 0)),
                "soft_tempo_override_decisions": int(
                    result.get("soft_tempo_override_decisions", 0)),
                "stall_recovery_search_decisions": int(
                    result.get("stall_recovery_search_decisions", 0)),
                "max_consecutive_stall_fallbacks": int(
                    result.get("max_consecutive_stall_fallbacks", 0)),
                "first_stall_turn": int(result.get("first_stall_turn", -1)),
                "last_stall_turn": int(result.get("last_stall_turn", -1)),
                "first_stall_player_hp": int(
                    result.get("first_stall_player_hp", -1)),
                "first_stall_monster_hp": int(
                    result.get("first_stall_monster_hp", -1)),
                "first_tempo_override_turn": int(
                    result.get("first_tempo_override_turn", -1)),
                "first_tempo_override_player_hp": int(
                    result.get("first_tempo_override_player_hp", -1)),
                "first_tempo_override_monster_hp": int(
                    result.get("first_tempo_override_monster_hp", -1)),
            })

    def reset(self, seed: int) -> dict[str, Any]:
        self.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, int(seed), self.config.ascension)
        self.steps = 0
        self.battles = 0
        self.search_seed_base = int(seed)
        self.last_battle_result = None
        self._reset_combat_audit()
        self._resolve_battles()
        return self.observation()

    def _resolve_battles(self) -> None:
        while (self.gc.outcome == sts.GameOutcome.UNDECIDED
               and self.gc.screen_state == sts.ScreenState.BATTLE):
            if self.config.deterministic_combat:
                search_seed = (
                    self.search_seed_base
                    ^ (int(self.gc.floor_num) * 0x9E3779B97F4A7C15)
                    ^ self.battles
                ) & ((1 << 64) - 1)
                self.last_battle_result = dict(sts.native_playout_current_battle_result(
                    self.gc, self.config.combat_sims, search_seed)
                )
            else:
                self.last_battle_result = dict(sts.native_playout_current_battle_result(
                    self.gc, self.config.combat_sims)
                )
            self.battles += 1
            self._record_battle_audit(self.last_battle_result)

    def _partition_legal_actions(self):
        return partition_legal_actions(self.gc)

    def legal_actions(self):
        return self._partition_legal_actions()[0]

    def observation(self) -> dict[str, Any]:
        rep = sts.getNNRepresentation(self.gc)
        actions = self.legal_actions()
        action_features = []
        action_content_ids = []
        # Map choices are semantically actions *towards a particular node*,
        # not merely small integers.  Preserve that relationship explicitly
        # for the policy: idx1 alone cannot tell it whether a path leads to a
        # fight, elite, shop, or rest site.
        target_rooms = []
        target_coords = []
        action_prices = []
        action_consequences = []
        event_ids = []
        neow_bonuses = []
        neow_drawbacks = []
        map_target_xs = []
        node_rooms = {
            (int(x), int(y)): int(room)
            for x, y, room in zip(rep.map.xs, rep.map.ys, rep.map.room_types)
        }
        is_map = self.gc.screen_state == sts.ScreenState.MAP_SCREEN
        for action in actions:
            try:
                consequence, _ = action_consequence_features(self.gc, action)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                consequence = np.zeros(10, dtype=np.float32)
            action_consequences.append(consequence)
            try:
                reward_type = float(int(action.rewards_action_type)) / 8.0
            except (AttributeError, TypeError, ValueError):
                reward_type = 0.0
            action_features.append([
                float(action.idx1) / 96.0, float(action.idx2) / 96.0,
                float(action.idx3) / 96.0, reward_type,
                float(self.gc.screen_state) / 10.0, 1.0,
            ])
            # A single vocabulary shared across candidate cards/relics/
            # potions.  Position alone cannot distinguish a Bash reward from
            # a Clash reward, so this is essential for generalization.
            content = 0
            try:
                rt = action.rewards_action_type
                if self.gc.screen_state == sts.ScreenState.REWARDS:
                    if rt == sts.RewardsActionType.CARD:
                        content = 1 + int(self.gc.screen_state_info.rewards_container.cards[action.idx1][action.idx2].id)
                    elif rt == sts.RewardsActionType.RELIC:
                        content = 400 + int(self.gc.screen_state_info.rewards_container.relics[action.idx1])
                    elif rt == sts.RewardsActionType.POTION:
                        content = 600 + int(self.gc.screen_state_info.rewards_container.potions[action.idx1])
                elif self.gc.screen_state == sts.ScreenState.SHOP_ROOM:
                    if rt == sts.RewardsActionType.CARD:
                        content = 1 + int(self.gc.screen_state_info.shop.cards[action.idx1].id)
                    elif rt == sts.RewardsActionType.RELIC:
                        content = 400 + int(self.gc.screen_state_info.shop.relics[action.idx1])
                    elif rt == sts.RewardsActionType.POTION:
                        content = 600 + int(self.gc.screen_state_info.shop.potions[action.idx1])
                elif self.gc.screen_state == sts.ScreenState.BOSS_RELIC_REWARDS and rt == sts.RewardsActionType.RELIC:
                    content = 400 + int(self.gc.screen_state_info.boss_relics[action.idx1])
                elif self.gc.screen_state == sts.ScreenState.CARD_SELECT:
                    content = 1 + int(self.gc.screen_state_info.to_select_cards[action.idx1].id)
            except (AttributeError, IndexError, TypeError, ValueError):
                content = 0
            action_content_ids.append(content)
            # Shop candidates are already legal, but their opportunity cost
            # is essential: buying a 190g relic and a 50g card cannot be
            # treated as equally attractive simply because both are valid.
            price = 0
            if self.gc.screen_state == sts.ScreenState.SHOP_ROOM:
                try:
                    shop = self.gc.screen_state_info.shop
                    rt = action.rewards_action_type
                    if rt == sts.RewardsActionType.CARD:
                        price = shop.prices[action.idx1]
                    elif rt == sts.RewardsActionType.RELIC:
                        price = shop.prices[7 + action.idx1]
                    elif rt == sts.RewardsActionType.POTION:
                        price = shop.prices[10 + action.idx1]
                    elif rt == sts.RewardsActionType.CARD_REMOVE:
                        price = shop.remove_cost
                except (AttributeError, IndexError, TypeError, ValueError):
                    price = 0
            action_prices.append((float(max(0, price)) / 300.0,
                                  float(max(0, price)) / max(1.0, float(self.gc.gold))))
            try:
                event_id = int(self.gc.cur_event) if self.gc.screen_state == sts.ScreenState.EVENT_SCREEN else 0
            except (TypeError, ValueError):
                event_id = 0
            event_ids.append(event_id)
            bonus = drawback = 0
            if event_id == int(sts.Event.NEOW):
                try:
                    option = self.gc.screen_state_info.neowRewards[action.idx1]
                    bonus, drawback = int(option.r), int(option.d)
                except (AttributeError, IndexError, TypeError, ValueError):
                    pass
            neow_bonuses.append(bonus)
            neow_drawbacks.append(drawback)
            if is_map:
                # Before the first selection mapY is -1, so the next row is
                # row zero.  Thereafter every map action advances one row.
                target_x = int(action.idx1)
                target_y = int(rep.mapY) + 1
                target_rooms.append(node_rooms.get((target_x, target_y), 0))
                target_coords.append((target_x / 7.0, target_y / 16.0))
                map_target_xs.append(target_x)
            else:
                target_rooms.append(0)
                target_coords.append((-1.0, -1.0))
                map_target_xs.append(-1)
        route_cones = np.zeros((len(actions), 4), dtype=np.float32)
        route_resources = np.zeros((len(actions), 4), dtype=np.float32)
        if is_map:
            route_cones, route_resources = map_route_features(
                rep.map, map_target_xs, int(rep.mapY))
        return {
            "fixed": np.asarray(rep.fixed_observation, dtype=np.int16),
            "deck_ids": np.asarray(rep.deck.cards, dtype=np.int16),
            "deck_upgrades": np.asarray(rep.deck.upgrades, dtype=np.int8),
            "relic_ids": np.asarray(rep.relics.relics, dtype=np.int16),
            "relic_counters": np.asarray(rep.relics.relic_counters, dtype=np.int16),
            "potions": np.asarray(rep.potions, dtype=np.int16),
            "map_xs": np.asarray(rep.map.xs, dtype=np.int8),
            "map_ys": np.asarray(rep.map.ys, dtype=np.int8),
            "map_rooms": np.asarray(rep.map.room_types, dtype=np.int8),
            "map_paths": np.asarray(rep.map.path_xs, dtype=np.int8),
            "map_x": int(rep.mapX), "map_y": int(rep.mapY),
            "act": int(self.gc.act),
            "floor": int(self.gc.floor_num),
            "screen": int(self.gc.screen_state),
            "action_bits": np.asarray([a.bits for a in actions], dtype=np.int64),
            "action_features": np.asarray(action_features, dtype=np.float32).reshape((-1, 6)),
            "action_content_ids": np.asarray(action_content_ids, dtype=np.int64),
            "action_target_rooms": np.asarray(target_rooms, dtype=np.int64),
            "action_target_coords": np.asarray(target_coords, dtype=np.float32).reshape((-1, 2)),
            "action_route_cones": route_cones,
            "action_route_resources": route_resources,
            "action_prices": np.asarray(action_prices, dtype=np.float32).reshape((-1, 2)),
            "action_consequences": np.asarray(
                action_consequences, dtype=np.float32).reshape((-1, 10)),
            "action_event_ids": np.asarray(event_ids, dtype=np.int64),
            "action_neow_bonuses": np.asarray(neow_bonuses, dtype=np.int64),
            "action_neow_drawbacks": np.asarray(neow_drawbacks, dtype=np.int64),
            "action_text": [a.getDesc(self.gc) for a in actions],
        }

    def step(self, action_index: int):
        actions, filtered_actions = self._partition_legal_actions()
        if not actions:
            return self.observation(), 0.0, True, {"outcome": str(self.gc.outcome)}
        if not 0 <= int(action_index) < len(actions):
            raise IndexError(f"action index {action_index} outside [0,{len(actions)})")
        hp_before, floor_before = self.gc.cur_hp, self.gc.floor_num
        action = actions[int(action_index)]
        if not action.isValidAction(self.gc):
            raise RuntimeError(f"invalid action: {action.getDesc(self.gc)}")
        if filtered_actions:
            self.combat_audit["safety_filter_events"] += 1
            self.combat_audit["immediate_loss_actions_filtered"] += filtered_actions
        action.execute(self.gc)
        self.steps += 1
        self._resolve_battles()
        done = self.gc.outcome != sts.GameOutcome.UNDECIDED or self.steps >= self.config.max_decisions
        # Small dense progress signal; terminal outcome remains dominant.
        reward = 0.01 * (self.gc.floor_num - floor_before)
        reward += 0.002 * (self.gc.cur_hp - hp_before)
        if self.gc.outcome == sts.GameOutcome.PLAYER_VICTORY:
            reward += 1.0
        elif self.gc.outcome == sts.GameOutcome.PLAYER_LOSS:
            reward -= 1.0
        info = {"outcome": str(self.gc.outcome), "floor": self.gc.floor_num,
                "hp": self.gc.cur_hp, "action": action.getDesc(self.gc),
                "combat_audit": dict(self.combat_audit)}
        return (self.observation() if not done else None), reward, done, info
