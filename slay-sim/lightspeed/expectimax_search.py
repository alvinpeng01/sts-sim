"""Pure expectimax MCTS combat search -- no neural network anywhere, unlike
az_search.py's PUCT search (which scores priors/leaf-values with the
trained ActionScoringPolicy). Built after a real, measured comparison
against Silver Automaton's own combat search (daniel-ziegler/sts_lightspeed,
see silverbot-reference/comparison_tests/): its no-NN, heuristic-rollout
UCT search beat our NN-guided PUCT search 100% vs 0-20% on the exact same
deck/HP/encounters (Time Eater, Donu & Deca) -- a real result, not a guess,
and unsurprising given our network is trained on a small fraction of the
compute AlphaZero-style methods assume, so its priors/value estimates
aren't yet better than a cheap heuristic on the fights that matter most.

This is NOT meant to replace az_search.py for good. The plan this session
settled on: use THIS (cheap, strong, network-free) search as the
distillation TEACHER for a fast blind policy (see distillation.py's
existing Stage-2 pipeline), rather than trying to make an undertrained
network guide search directly. A fast deployed bot still needs a cheap
blind policy at inference time -- this search is expensive per-decision
(many heuristic rollouts) exactly like Silverbot's own combat search is,
which is fine for a training-time teacher, not for deployment.

Reuses az_search.py's ALREADY-FIXED correctness machinery directly (DPW,
the transposition table, and -- most importantly -- the RNG-counter-probe
action classification) rather than reimplementing it: those are about
search CORRECTNESS regardless of what evaluates a leaf, not about whether
a network is involved. See az_search.py's own module/_simulate docstrings
for the full history (two real crashes -- an assertion, then a segfault --
both traced to assuming action_type alone tells you whether an action is
deterministic, when some CARD actions secretly consume RNG via shuffle-
into-draw-pile effects).

What's actually different from az_search.py:
  - Selection: plain UCB1 (Q + C*sqrt(log(parent_N)/(N+1)), unvisited
    edges explored first) instead of PUCT's prior-weighted term -- there's
    no learned prior here, so PUCT's P term would just be uniform, which
    is exactly what UCB1 already assumes. Mirrors Silver Automaton's own
    BattleSearcher::evaluateEdge formula (same shape, including the same
    Childs et al. 2008 edge-visit-count-not-node-count reasoning for
    transposition-shared nodes).
  - Leaf evaluation: _heuristic_playout (a real rollout to a terminal
    state, no network call) at EVERY newly-reached node, not just ones
    past a depth cutoff (az_search.py's ROLLOUT_DEPTH_TURNS existed
    specifically to bound network-forward-pass cost, which doesn't apply
    here at all -- there is no network cost to bound).
  - Backup: kept as env.py's dense-reward-shaped, GAMMA-discounted
    accumulation (same as az_search.py), NOT switched to Silver
    Automaton's plain terminal-score average -- that shaping (BETA-
    weighted incoming damage, turn-efficiency terms, ...) already encodes
    real domain knowledge validated over this whole session's training
    runs; there's no reason to throw it away just because the leaf
    estimator changed.
  - No relic_idxs/relic_mask/policy threading anywhere: nothing here reads
    the network, so there's nothing that needs the embedding inputs that
    were only ever for it. Relics/potions still affect gameplay normally
    (through bc itself, which potential()/dense_reward already read).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

import slaythespire as sts

from .az_search import (
    _heuristic_pick, _heuristic_playout, _terminal_reward, _predicted_incoming_damage,
    _state_key, _crn_seed, WC_CHANCE, WA_CHANCE, MAX_CALL_DEPTH,
)
from .env import GAMMA, W_SHAPE, W_HP, BETA
import slaythespire as _sts_bo

C_UCB = 1.5  # UCB1 exploration constant -- Silver Automaton's own explorationParameter is also O(1). Tuned via a real sweep (0.7/1.0/1.5 x the-then-existing IN_DANGER_HP_FRACTION 0.15/0.25/0.35, 10 eps/cell, Spheric Guardian/Time Eater/Donu & Deca): 1.5 tied for best (16/30) with no regression elsewhere. NOTE: IN_DANGER_HP_FRACTION itself no longer exists -- diagnosing Time Eater's still-0/15 result after that sweep found _heuristic_pick's SKILL score was structurally always below ATTACK's even when "in danger" (9.0 < 10.0), so the whole binary in_danger gate was replaced with a continuous danger_fraction scaling (see az_search.py's _heuristic_pick). C_UCB=1.5 itself doesn't depend on that mechanism and wasn't re-swept after the fix, but should be if defense's new behavior shifts what exploration constant is optimal.

# Flat HP-equivalent penalty (BETA-weighted, same units as the incoming-
# damage term) for a living, non-half-dead monster whose CURRENT move deals
# 0 damage -- a cheap stand-in for "this monster is building toward
# something instead of attacking right now" (a buff, a block grant, a
# multi-turn setup), added after direct evidence Donu & Deca specifically
# punishes leaving either monster alive through its non-attack turn (Circle
# of Power/Square of Protection both buff BOTH monsters, confirmed reading
# MonsterSpecific.cpp -- see the earlier analysis this session). Not gated
# on classify_monster_move's actual self_buffs/buffs_ally flags -- that
# binding runs the move on a throwaway BattleContext copy internally (a
# real simulate-and-execute, not a cheap query), which would be a genuine
# new cost on this function's hot path (called every _dense_reward, i.e.
# every simulate() step). A flat penalty for ANY 0-damage intent is blunter
# (also catches purely-defensive 0-damage moves that aren't actually
# escalating anything) but stays cheap, and "don't leave idle monsters
# alone" is a reasonable general instinct even where it's not precisely
# targeted. Kept OUT of env.py's own potential() deliberately -- that
# function is also PPO's training reward, and changing it invalidates
# every existing checkpoint's value head (see env.py's own note on this);
# this search has no trained value head to invalidate, so it's free to
# extend the shaping without that cost.
BREWING_THREAT_ESTIMATE = 8.0


def _potential(bc) -> float:
    """A standalone reimplementation of env.py's potential() shape (W_HP
    * player_hp, minus each living monster's HP and BETA-weighted incoming
    damage), NOT a call into env.py's actual potential() -- deliberately,
    same reasoning as BREWING_THREAT_ESTIMATE above (this search has no
    trained value head to invalidate, so it's free to diverge from the
    training-reward-shared function). The divergence that matters: the
    incoming-damage term uses _predicted_incoming_damage (az_search.py),
    NOT bc.get_monster_move_damage's raw value directly -- see that
    function's own docstring for the full story, but in short,
    get_monster_move_damage is only a static base-damage table lookup that
    ignores the monster's current Strength, the monster's own Weak status,
    and the player's Vulnerable status, all of which the engine's actual
    damage resolution applies. env.py's potential() has this same gap, but
    fixing it THERE would be a training-reward change (checkpoint-
    invalidating, out of scope for a search-only session); fixing it HERE
    costs nothing since nothing depends on this function's exact shape."""
    if bc.outcome != _sts_bo.BattleOutcome.UNDECIDED:
        return 0.0
    phi = W_HP * bc.player_hp
    vuln_mult = 1.5 if bc.get_player_status_value("VULNERABLE") > 0 else 1.0
    for i, m in enumerate(bc.monsters):
        if m.half_dead:
            phi -= m.max_hp
            continue
        if m.cur_hp <= 0:
            continue
        dmg = _predicted_incoming_damage(bc, i, m, vuln_mult)
        phi -= m.cur_hp + BETA * dmg
        if dmg == 0:
            phi -= BETA * BREWING_THREAT_ESTIMATE
    return phi


def _dense_reward(bc_before, bc_after) -> float:
    """Same shape as az_search.py's _dense_reward, using THIS module's own
    _potential (see its docstring for why) instead of env.py's directly."""
    return W_SHAPE * (GAMMA * _potential(bc_after) - _potential(bc_before))


class _Node:
    __slots__ = ("bc", "actions", "N", "W", "children", "chance_children",
                 "chance_samples_drawn", "key", "is_terminal", "terminal_value", "visit_count")

    def __init__(self, bc):
        self.bc = bc
        self.actions: Optional[List] = None
        self.N: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None
        self.children: Dict[int, "_Node"] = {}
        self.chance_children: Dict[int, List["_Node"]] = {}
        self.chance_samples_drawn: Dict[int, int] = {}
        self.key: Optional[tuple] = None
        self.is_terminal = False
        self.terminal_value = 0.0
        self.visit_count = 0


def _select_idx(node: _Node) -> int:
    """UCB1 selection -- see module docstring for how this differs from
    az_search.py's PUCT. Unvisited edges (N==0) are explored before any
    UCB comparison, matching Silver Automaton's own
    `if edge.visitCount == 0: return infinity` rather than folding it into
    the same formula (avoids a div-by-zero and makes "try everything once"
    an explicit first phase, not an artifact of a large-but-finite bonus).

    Plain Python loops over .tolist()'d copies, not vectorized numpy ops on
    node.N/node.W directly -- profiling (cProfile, 4 full episodes) found
    THIS function alone responsible for 36% of total search time, almost
    entirely numpy call-dispatch overhead (flatnonzero/argmax/sqrt/sum),
    not real computation: node.N/W are tiny (one entry per legal action,
    typically <10), and numpy's per-call overhead for vectorized ops on
    arrays that small dominates over just looping in Python. Semantics are
    unchanged: still returns the FIRST index with N==0 if any exists (not
    just "an" unvisited index), and still breaks ties on the max UCB score
    by first-occurrence (matching np.argmax's own tie-breaking) via a
    strict `>` comparison rather than `>=`. Validated against the old
    numpy implementation across many random (N, W) arrays including ties
    and all-visited/some-unvisited cases before this replaced it."""
    N = node.N.tolist()
    n = len(N)
    for i in range(n):
        if N[i] == 0:
            return i
    log_parent = math.log(float(sum(N)) + 1.0)
    W = node.W.tolist()
    best_idx = 0
    best_score = float("-inf")
    for i in range(n):
        ni = N[i]
        score = W[i] / ni + C_UCB * math.sqrt(log_parent / (ni + 1.0))
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _dpw_chance_child(node: _Node, idx: int, action, bc, crn_base: Optional[int]) -> _Node:
    """Same DPW widen-or-revisit step as az_search.py's own (see its
    docstring for the full rationale) -- duplicated rather than imported
    because it closes over this module's OWN _Node type."""
    siblings = node.chance_children.setdefault(idx, [])
    n = int(node.N[idx])
    widening_cap = math.ceil(WC_CHANCE * (n + 1) ** WA_CHANCE)
    if len(siblings) < widening_cap:
        local_sample_index = node.chance_samples_drawn.get(idx, 0)
        node.chance_samples_drawn[idx] = local_sample_index + 1
        if crn_base is not None:
            bc.seed_rng(_crn_seed(crn_base, _state_key(bc), local_sample_index))
        else:
            bc.decorrelate_rng()
        sample = sts.BattleContext(bc)
        action.execute(sample)
        sample_key = _state_key(sample)
        child = None
        for existing in siblings:
            if existing.key == sample_key:
                child = existing
                break
        if child is None:
            child = _Node(sample)
            child.key = sample_key
            if sample.outcome != sts.BattleOutcome.UNDECIDED:
                child.is_terminal = True
                child.terminal_value = _terminal_reward(sample, sample.turn)
            siblings.append(child)
        return child
    weights = np.array([c.visit_count + 1 for c in siblings], dtype=np.float64)
    return siblings[int(np.random.choice(len(siblings), p=weights / weights.sum()))]


def _expand_leaf(node: _Node) -> float:
    """First visit to a node: populate its legal actions (so future visits
    can do real UCB1 selection among them) and return a value estimate for
    THIS call's backup via a heuristic rollout -- no network call anywhere,
    see module docstring. Unlike az_search.py's _expand (which also
    computes priors from the network), there's no separate prior here:
    _select_idx's "explore every action once" phase serves the role priors
    would have, at the cost of needing len(legal) visits before UCB1
    proper kicks in -- acceptable since a heuristic-rollout visit is much
    cheaper than a network-scored one was."""
    bc = node.bc
    legal = bc.get_legal_actions()
    node.actions = legal
    node.N = np.zeros(len(legal), dtype=np.int64)
    node.W = np.zeros(len(legal), dtype=np.float64)
    return _heuristic_playout(bc)


def _simulate(node: _Node, trans_table: Dict[tuple, "_Node"],
              call_depth: int = 0, crn_base: Optional[int] = None) -> float:
    """Same overall shape and action-classification fix as az_search.py's
    _simulate (see its docstring for the full RNG-consumption-probe
    rationale) -- UCB1 selection and heuristic-rollout leaf evaluation are
    the only real differences (see module docstring)."""
    node.visit_count += 1
    if node.is_terminal:
        return node.terminal_value
    if call_depth >= MAX_CALL_DEPTH:
        return _heuristic_playout(node.bc)
    if node.actions is None:
        return _expand_leaf(node)

    idx = _select_idx(node)
    action = node.actions[idx]
    bc = node.bc

    if action.action_type == sts.ActionType.END_TURN or idx in node.chance_children:
        child = _dpw_chance_child(node, idx, action, bc, crn_base)
        r = _dense_reward(bc, child.bc)
        if child.is_terminal:
            value = r + child.terminal_value
        else:
            value = r + GAMMA * _simulate(child, trans_table, call_depth + 1, crn_base)
    elif idx in node.children:
        child = node.children[idx]
        r = _dense_reward(bc, child.bc)
        value = r + GAMMA * _simulate(child, trans_table, call_depth + 1, crn_base)
    else:
        # First visit to this (node, idx) pair, action isn't END_TURN --
        # probe for RNG consumption (see az_search.py's _simulate docstring
        # for why action_type alone can't classify this).
        child_bc = sts.BattleContext(bc)
        counter_before = bc.rng_counter_sum()
        action.execute(child_bc)
        consumed_rng = child_bc.rng_counter_sum() != counter_before
        r = _dense_reward(bc, child_bc)
        child = _Node(child_bc)
        if child_bc.outcome != sts.BattleOutcome.UNDECIDED:
            child.is_terminal = True
            child.terminal_value = _terminal_reward(child_bc, child_bc.turn)
        if consumed_rng:
            child.key = _state_key(child_bc)
            node.chance_children[idx] = [child]
            node.chance_samples_drawn[idx] = 1
        elif not child.is_terminal:
            key = _state_key(child_bc)
            existing = trans_table.get(key)
            if existing is not None:
                child = existing
            else:
                trans_table[key] = child
            node.children[idx] = child
        else:
            node.children[idx] = child
        if child.is_terminal:
            value = r + child.terminal_value
        else:
            value = r + GAMMA * _simulate(child, trans_table, call_depth + 1, crn_base)

    node.N[idx] += 1
    node.W[idx] += value
    return value


def choose_action_python(bc, n_simulations: int = 200, crn_base: Optional[int] = None):
    """The original, pure-Python expectimax MCTS loop above (UCB1 selection,
    DPW, RNG-probe classification, transposition sharing). NO LONGER KEPT IN
    SYNC with the native side (bindings/slaythespire.cpp's run_mcts_search)
    as a hard rule -- native is now the maintained, authoritative
    implementation (validated three independent ways: exact state-key match,
    exact rollout-policy match, and statistical MCTS parity, see
    run_mcts_search's own docstring) and the only path anything in
    production actually runs. This function is kept purely as a debugging
    aid -- a slower, simpler implementation to diff against if a native
    result looks wrong -- the same role az_search_debug.py plays for
    az_search.py's own search. Expect it to drift from native's behavior
    over time as native-only changes land; do not treat a mismatch between
    the two as a bug on its own without checking which one actually matches
    the intended fix. Returns (action, visit_counts), visit_counts as a
    real numpy array."""
    root = _Node(sts.BattleContext(bc))
    trans_table: Dict[tuple, _Node] = {}
    for _ in range(n_simulations):
        _simulate(root, trans_table, crn_base=crn_base)
    best_idx = int(np.argmax(root.N))
    return root.actions[best_idx], root.N


def _tree_seed(search_seed: int, tree_index: int = 0) -> int:
    """Stable SplitMix64-derived seed for one independent native tree."""
    x = (int(search_seed) + 0x9E3779B97F4A7C15 * (tree_index + 1)) & ((1 << 64) - 1)
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return (x ^ (x >> 31)) & ((1 << 64) - 1)


def choose_action_native(bc, n_simulations: int = 200, crn_base: Optional[int] = None,
                         search_seed: Optional[int] = None):
    """All-native replacement for choose_action_python -- runs the ENTIRE
    MCTS loop in C++ (sts.run_mcts_search, bindings/slaythespire.cpp) in
    ONE call instead of one Python call per simulation step. NOT bit-
    identical to choose_action_python (DPW's revisit-weighted sampling uses
    its own std::mt19937_64 instead of numpy's global RNG, and CRN seeding
    uses its own hash combiner instead of CPython's tuple hash -- see
    run_mcts_search's own docstring/DRIFT WARNING for why neither is
    reproducible in C++ without porting numpy/CPython itself, and why
    neither needs to be for correctness). Validated via exact statistical
    parity as of its introduction, and again after the Strength/Weak/
    Vulnerable damage-prediction fix (scratchpad/compare_native_vs_python_mcts.py:
    90/160 Python vs 87/160 native across 8 encounters, well within noise
    at n=20/cell), and roughly 9x faster wall-clock on the same workload --
    this is the DEFAULT path (see choose_action below) and the one actually
    maintained going forward; choose_action_python is a debugging aid only,
    not required to be kept in sync (see its own docstring)."""
    action, visits = sts.run_mcts_search(bc, n_simulations, crn_base, search_seed)
    visits = np.array(visits, dtype=np.int64)
    if sts.get_seq_halving():
        # Sequential-halving visit totals are a budget-allocation artifact,
        # while its selected action is the highest surviving mean-Q action.
        # Returning a hard target is honest for downstream distillation;
        # normalizing the allocation counts would teach a different policy.
        legal = bc.get_legal_actions()
        target = np.zeros(len(legal), dtype=np.int64)
        target[next(i for i, candidate in enumerate(legal) if candidate.bits == action.bits)] = 1
        return action, target
    return action, visits


def root_parallel_search(bc, n_simulations: int = 200, n_trees: int = 4,
                         crn_base: Optional[int] = None, search_seed: Optional[int] = None):
    """Root parallelization: run n_trees INDEPENDENT native searches (each
    with n_simulations // n_trees of the total budget) concurrently on real
    OS threads, then combine by SUMMING each tree's root visit counts and
    taking the argmax over the total -- a well-established MCTS technique
    for improving robustness under a fixed total sim budget compared to one
    bigger sequential tree (independent trees sample different random
    rollout/DPW paths -- run_mcts_search reseeds its own RNG every call --
    so combining several reduces variance the same way averaging any set of
    independent estimates does).

    True thread parallelism here (not just concurrent scheduling under the
    GIL) relies on sts.run_mcts_search releasing the GIL during the actual
    search (see its own docstring/DRIFT WARNING in bindings/slaythespire.cpp)
    -- confirmed safe to do because nativeSimulate and everything it calls
    touch no global MUTABLE state (the arena/transposition table are locals
    of each call; the only "global" data read, cardTypes, is a compile-time
    `static constexpr` array). Structurally much lower-risk than tree reuse
    across decisions (see choose_action_native's sibling attempt, reverted
    twice after real crashes) -- there is no shared mutable state between
    the parallel trees at all, nothing to reroot, nothing that can go stale.

    Requires every tree to see the IDENTICAL starting bc so their internal
    legal-action enumerations (and therefore visit-count array indices)
    line up -- guaranteed here since bc is only ever read, never mutated,
    by any of this function's own code or by run_mcts_search itself."""
    if sts.get_seq_halving():
        # Visit counts cannot be aggregated across sequential-halving trees:
        # they encode phase allocation, not action preference. One full-budget
        # tree preserves the selector's own mean-Q decision and yields the
        # hard target supplied by choose_action_native.
        return choose_action_native(bc, n_simulations=n_simulations, crn_base=crn_base,
                                    search_seed=search_seed)

    import concurrent.futures

    sims_per_tree = max(1, n_simulations // n_trees)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_trees) as pool:
        futures = [pool.submit(sts.run_mcts_search, bc, sims_per_tree, crn_base,
                               None if search_seed is None else _tree_seed(search_seed, i))
                   for i in range(n_trees)]
        results = [f.result() for f in futures]

    combined = None
    for _, visits in results:
        v = np.array(visits, dtype=np.int64)
        combined = v.copy() if combined is None else combined + v

    best_idx = int(np.argmax(combined))
    legal = bc.get_legal_actions()  # same deterministic enumeration every tree used internally, since bc is unmutated
    return legal[best_idx], combined


def choose_action(bc, n_simulations: int = 200, crn_base: Optional[int] = None,
                  use_native: bool = True, search_seed: Optional[int] = None):
    """Expectimax-MCTS-searched action for the current decision. Returns
    (action, visit_counts) -- same return shape as az_search.py's
    choose_action, for drop-in use anywhere that consumes it (distillation
    targets, evaluation harnesses). use_native=True (default) dispatches to
    choose_action_native; pass False to force the pure-Python debugging
    path (choose_action_python) -- e.g. to sanity-check a native result
    that looks wrong. Not required after routine native-side changes;
    native is the maintained implementation now (see choose_action_native's
    own docstring)."""
    if use_native:
        return choose_action_native(bc, n_simulations=n_simulations, crn_base=crn_base,
                                    search_seed=search_seed)
    return choose_action_python(bc, n_simulations=n_simulations, crn_base=crn_base)


def run_episode_with_search(env, n_simulations: int = 200, seed=None, use_native: bool = True):
    """Same shape as az_search.py's run_episode_with_search, no policy
    argument (nothing here reads one)."""
    obs = env.reset(seed=seed)
    done = False
    info = None
    while not done:
        action, _visits = choose_action(env.bc, n_simulations=n_simulations, use_native=use_native)
        obs, reward, done, info = env.step(action)
    return info


def evaluate_with_search(env, n: int, n_simulations: int = 200,
                          seed_offset: int = 0, per_encounter: bool = False):
    """Same shape as az_search.py's evaluate_with_search."""
    wins = hp_sum = 0
    breakdown = {}
    for i in range(n):
        info = run_episode_with_search(env, n_simulations=n_simulations, seed=seed_offset + i)
        won = info["outcome"] == sts.BattleOutcome.PLAYER_VICTORY
        if won:
            wins += 1
            hp_sum += info["player_hp"]
        if per_encounter:
            key = env.last_encounter
            w, total = breakdown.get(key, [0, 0])
            breakdown[key] = [w + (1 if won else 0), total + 1]

    result = (wins / n, hp_sum / max(wins, 1))
    if per_encounter:
        return result, {enc: w / total for enc, (w, total) in breakdown.items()}
    return result


# --- multiprocessing-parallel evaluation -----------------------------------
#
# Episode-level parallelism, ported from az_search.py's own
# evaluate_with_search_parallel (see its comment for the full rationale --
# each episode's search is fully independent of every other episode's, so
# this is embarrassingly parallel, no root-parallelization/visit-merging
# needed). This module had no such helper until now even though every
# evaluation/comparison run this session (encounter sweeps, tuning sweeps,
# the Silverbot comparison) drove it through the serial evaluate_with_search
# loop above -- on a 12-core machine that's an easy, safe, near-linear
# wall-clock win with zero risk to the search's own correctness-critical
# logic (nothing about DPW/transposition/RNG-classification changes here,
# just which process runs which episodes). No torch/policy involved here
# (unlike az_search.py's worker, which loads a state_dict) since this search
# has no network at all -- the worker init only needs to build its own env.

import multiprocessing as mp

_ex_worker_env = None


def _ex_worker_init(env_kwargs: dict) -> None:
    global _ex_worker_env
    from .env import IroncladFightEnv
    _ex_worker_env = IroncladFightEnv(**env_kwargs)


def _ex_worker_run_episodes(args):
    n_episodes, n_simulations, seed_offset = args
    wins = hp_sum = 0
    breakdown = {}
    for i in range(n_episodes):
        info = run_episode_with_search(_ex_worker_env, n_simulations=n_simulations, seed=seed_offset + i)
        won = info["outcome"] == sts.BattleOutcome.PLAYER_VICTORY
        if won:
            wins += 1
            hp_sum += info["player_hp"]
        key = _ex_worker_env.last_encounter
        w, total = breakdown.get(key, [0, 0])
        breakdown[key] = [w + (1 if won else 0), total + 1]
    return wins, hp_sum, breakdown


def evaluate_with_search_parallel(env, n: int, n_simulations: int = 200,
                                   seed_offset: int = 0, per_encounter: bool = False, n_workers: int = 6):
    """Same result shape as evaluate_with_search, split across n_workers
    processes by episode -- each worker gets its own contiguous seed range,
    so results are exactly the union of what evaluate_with_search would have
    produced for those same seeds serially, not a different sample, just
    faster."""
    from .ppo import _env_kwargs_from
    env_kwargs = _env_kwargs_from(env)

    counts = [n // n_workers + (1 if i < n % n_workers else 0) for i in range(n_workers)]
    args_list = []
    offset = seed_offset
    for c in counts:
        if c > 0:
            args_list.append((c, n_simulations, offset))
        offset += c

    with mp.Pool(len(args_list), initializer=_ex_worker_init, initargs=(env_kwargs,)) as pool:
        results = pool.map(_ex_worker_run_episodes, args_list)

    wins = hp_sum = 0
    breakdown = {}
    for w_wins, w_hp_sum, w_breakdown in results:
        wins += w_wins
        hp_sum += w_hp_sum
        for enc, (w, total) in w_breakdown.items():
            bw, btotal = breakdown.get(enc, [0, 0])
            breakdown[enc] = [bw + w, btotal + total]

    result = (wins / n, hp_sum / max(wins, 1))
    if per_encounter:
        return result, {enc: w / total for enc, (w, total) in breakdown.items()}
    return result
