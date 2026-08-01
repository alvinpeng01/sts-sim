"""Expectimax search driven by the native slaythespire (C++) BattleContext,
instead of sts/combat.py's pure-Python engine -- same bounded-turns
algorithm as sts/search.py (see its docstring for the max/chance-node
rationale), ported onto compiled state so search cost stops being
dominated by Python object cloning/hashing. This is what makes deeper
turns_left (3+) affordable: sts/search.py's own clone() chain was ~50% of
total search time even after this session's Python-side optimizations.

Structural differences from sts/search.py, and why:

  - Ending a turn (Action(END_TURN).execute(bc)) resolves the ENTIRE enemy
    turn AND draws the next hand in one atomic native call. There's no
    separate intent-branch / draw-sample split like
    _after_end_turn/_expected_next_turn -- one sampling loop covers both,
    since a single execute() already bakes in both kinds of randomness.

  - BattleContext's copy constructor is a bit-for-bit struct copy, RNG
    state included: cloning twice and executing the same action on both
    replays an IDENTICAL outcome (verified directly -- three END_TURN
    executions on three fresh unmodified clones came back bit-identical).
    `.decorrelate_rng()` (added to bindings/slaythespire.cpp for this) must
    be called on the PARENT before every clone meant to sample a different
    hypothetical future -- exactly the same fix already made to
    sts/combat.py's CombatState.clone() this session, same reasoning (the
    exact draw sequence isn't correctness-relevant, only decorrelation is).

  - No transposition-table caching (yet). The Python bindings expose only
    strength/vulnerable/weak on Monster -- no frail/poison/artifact/etc. --
    so a cache key built from what's exposed could silently merge two
    states that actually differ in an unexposed status effect. That's a
    worse failure mode (silently wrong recommendation) than the speed a
    cache would buy, so it's left out until more status bindings exist.

  - No value-net cutoff (yet). sts/value_net.py's encoder reads
    CombatState-specific fields; a BattleContext-reading equivalent would
    need its own encoder plus retraining on native-engine self-play data
    (HP/damage numbers differ between the two engines). Uses the same
    hand-written fallback formula as sts/search.py's evaluate() for now.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import slaythespire as sts

WIN_SCORE = 100.0
LOSS_SCORE = -1000.0


def evaluate(bc) -> float:
    """Mirrors sts/search.py's evaluate() fallback term exactly (same
    formula, same terminal-state handling), just reading BattleContext's
    exposed fields instead of CombatState's."""
    if bc.outcome == sts.BattleOutcome.PLAYER_VICTORY:
        return WIN_SCORE + bc.player_hp
    if bc.outcome == sts.BattleOutcome.PLAYER_LOSS:
        return LOSS_SCORE
    enemy_hp = sum(m.cur_hp for m in bc.monsters if m.cur_hp > 0)
    return bc.player_hp * 2.0 - enemy_hp * 0.5 + bc.player_block * 0.5


def _distinct_actions(bc) -> List:
    """get_legal_actions() emits one action per (hand-index, target) pair
    with no dedup -- multiple identical Strikes targeting the same monster
    would otherwise be explored as separate branches for no benefit, same
    motivation as sts/search.py's _distinct_actions."""
    seen = set()
    actions = []
    for a in bc.get_legal_actions():
        if a.action_type == sts.ActionType.END_TURN:
            key = ("end",)
        else:
            card = bc.hand[a.source_idx]
            key = ("card", int(card.id), card.upgraded, a.target_idx)
        if key in seen:
            continue
        seen.add(key)
        actions.append(a)
    return actions


def _search(bc, turns_left: int, draw_samples: int) -> float:
    if bc.outcome != sts.BattleOutcome.UNDECIDED or turns_left <= 0:
        return evaluate(bc)
    return max(
        _action_value(bc, a, turns_left, draw_samples)
        for a in _distinct_actions(bc)
    )


def _action_value(parent_bc, action, turns_left: int, draw_samples: int) -> float:
    if action.action_type == sts.ActionType.END_TURN:
        # Monte Carlo determinization over both the enemy's move roll and
        # the player's next draw at once (see module docstring) -- decorrelate
        # the shared PARENT before each clone, not the clone itself, so each
        # sample actually diverges (mirrors _expected_next_turn's pattern).
        total = 0.0
        for _ in range(draw_samples):
            parent_bc.decorrelate_rng()
            sample = sts.BattleContext(parent_bc)
            action.execute(sample)
            total += _search(sample, turns_left - 1, draw_samples)
        return total / draw_samples
    child = sts.BattleContext(parent_bc)
    action.execute(child)
    return _search(child, turns_left, draw_samples)


def choose_action(
    bc, turns_left: int = 1, draw_samples: int = 1,
) -> Tuple[object, float]:
    """Best action for the player right now. Same turns_left/draw_samples
    semantics as sts/search.py's choose_action (see its docstring) --
    turns_left=1 fully solves this turn and peeks at next turn's expected
    value; draw_samples only matters once you're averaging over a sampled
    future (i.e. always, here, since every leaf is a real END_TURN sample
    rather than an exact enumeration)."""
    best_action: Optional[object] = None
    best_val = float("-inf")
    for action in _distinct_actions(bc):
        val = _action_value(bc, action, turns_left, draw_samples)
        if val > best_val:
            best_val, best_action = val, action
    assert best_action is not None
    return best_action, best_val
