"""Native self-play labels for the whole-run policy.

No external model is used.  A candidate action is scored by branching our
GameContext and completing that branch with our own out-of-combat heuristic
and native expectimax combat MCTS.
"""
from __future__ import annotations

import slaythespire as sts

from .whole_run_env import RunConfig, WholeRunEnv
from .search_config import ensure_search_config


def terminal_score(gc) -> float:
    victory = gc.outcome == sts.GameOutcome.PLAYER_VICTORY
    # Lexicographic objective expressed as a scalar: Heart win dominates,
    # then progress/keys, then survivability as a branch tie-breaker.
    cleared_act3 = gc.act >= 4 or victory
    keys = (int(gc.red_key) + int(gc.green_key) + int(gc.blue_key)) if cleared_act3 else 0
    return (10_000.0 if victory else 0.0) + 100.0 * gc.floor_num + 50.0 * keys + gc.cur_hp / max(gc.max_hp, 1)


def rank_legal_actions(gc, combat_sims: int = 100):
    """Return [(action_index, score)] ranked best-first for the current state."""
    ensure_search_config()
    actions = list(sts.GameAction.getAllActionsInState(gc))
    scored = []
    for idx, action in enumerate(actions):
        branch = gc.copy()
        action.execute(branch)
        agent = sts.Agent()
        agent.pause_on_card_reward = False
        agent.playout_hybrid(branch, combat_sims)
        scored.append((idx, terminal_score(branch)))
    return sorted(scored, key=lambda item: item[1], reverse=True)


def rank_actions_with_value(gc, model, combat_sims: int = 100):
    """Cheap policy-improvement target using an immediate transition + V(s')."""
    ensure_search_config()
    actions = list(sts.GameAction.getAllActionsInState(gc))
    scored = []
    for idx, action in enumerate(actions):
        branch = gc.copy()
        action.execute(branch)
        while branch.outcome == sts.GameOutcome.UNDECIDED and branch.screen_state == sts.ScreenState.BATTLE:
            sts.native_playout_current_battle(branch, combat_sims)
        if branch.outcome != sts.GameOutcome.UNDECIDED:
            score = terminal_score(branch) / 10_000.0
        else:
            probe = WholeRunEnv(RunConfig(combat_sims=combat_sims))
            probe.gc = branch
            with __import__("torch").no_grad():
                score = float(model(probe.observation())[1])
        scored.append((idx, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)
