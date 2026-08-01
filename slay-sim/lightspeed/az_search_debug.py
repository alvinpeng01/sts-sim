"""DEBUG/INVESTIGATION ARTIFACT -- NOT used by any production code path (no
other module imports this one; az_search.py is the real, live version).
Forked from az_search.py to isolate two crashes (an Action.cpp:459
assertion, then a segfault) hit while trying to add DPW-dedup and tree-
reuse-across-decisions to the real module -- both were reverted there.

Root cause, confirmed here via DEBUG_CHECK_CONSISTENCY (compares a
rerooted node's cached state against the real env.bc immediately after
rerooting): some CARD actions -- not just END_TURN -- consume RNG
internally (shuffle-into-draw-pile effects: Wild Strike, Reckless Charge,
similar), so they are NOT the pure, RNG-independent functions of parent
state that both the DPW-dedup merge and the tree-reuse reroot assumed.
Direct evidence: two states identical on HP/block/energy/turn/monsters/
hand differed ONLY in draw-pile order after a "deterministic" reroot.

ENABLE_DEDUP / ENABLE_TREE_REUSE / INCLUDE_MOVE_STATE_IN_KEY /
PERSIST_TRANS_TABLE / DEBUG_CHECK_CONSISTENCY (module-level flags, all
default False/off) let each variable be isolated independently from a
test script (`az_search_debug.ENABLE_DEDUP = True`, etc.) -- kept here
rather than deleted since re-attempting the real fix (detect RNG
consumption after the fact via Random::counter, mirroring Silver
Automaton's actual technique, instead of trusting action_type) will want
this same isolation harness again.

--- original az_search.py docstring below, describes the LIVE module this
was forked from, not this file's own (nonexistent) production role ---

AlphaZero-style search-augmented inference: wraps the already-trained
ActionScoringPolicy in a PUCT search over BattleContext, instead of using it
as a blind single forward pass (policy.act()). Stage 1 only -- this changes
INFERENCE, not training. The network itself (policy.py) is unchanged; this
is a new search procedure built on top of its existing policy/value heads,
reusing the same BattleContext primitives as native_search.py (the copy
constructor and decorrelate_rng added to bindings/slaythespire.cpp this
session, plus get_legal_actions/execute).

Structural choice, mirroring the same deterministic/stochastic split
search.py and native_search.py both make: the search tree with real PUCT
statistics (N/W/P per action) is only built for the DETERMINISTIC intra-turn
part of a decision (card-play sequences before ending the turn). Ending the
turn is stochastic -- the enemy's move roll and the next draw are both baked
into one execute() call -- so trying to cache and revisit a shared subtree
there would mean tracking statistics for states that essentially never
recur (a different draw/roll every visit). Instead, each simulation that
selects END_TURN takes ONE decorrelated sample and scores the result
directly with the value head: a leaf, not a further-expanded node. PUCT
accumulation only makes sense across REPEATED visits to the SAME state,
which is exactly the deterministic part.

A 4-sample-averaged version of this leaf was tried and reverted. It was
added chasing a real, measured regression -- on Awakened One (a long
two-phase fight with far more END_TURN transitions than most encounters),
search underperformed the blind policy (15.0% vs 28.0% win rate, n=100, a
statistically real gap), and doubling the simulation budget (40->100) left
it completely unchanged, ruling out "just needs more search width" and
pointing at leaf-estimate variance instead. But averaging 4 samples left
Awakened One at exactly the same 15.0% -- no improvement at all -- while
measurably costing 2.66-3.15x more search time (real A/B timing, not
estimated). Paying a real, confirmed cost for zero confirmed benefit on
the one case it was built to fix isn't worth it; the actual explanation
was more likely structural: this search used to look ahead only within the
current turn, then trust the value function for everything past the next
enemy turn -- a fight where the critical decision is genuinely multi-turn,
like surviving a boss's revival two turns out, may be outside what a
single-value-estimate horizon can reason about, regardless of leaf
variance.

A first attempt at targeting that structural gap added a hand-rolled bounded
lookahead: at each END_TURN, take one decorrelated sample, then spend a fixed
sub-simulation budget exploring FROM that one sampled state, recursing a
fixed number of turns before falling back to a single _expand() estimate.
Measured improvement on Awakened One (15.0% -> 24.0%, n=100), confirming the
structural theory. But it had two real limitations inherent to being
hand-bounded rather than principled: (1) the recursion depth (how many
turns get real search) and per-turn sub-budget were both hardcoded constants,
not something the search itself could allocate adaptively based on where the
value estimate actually looked uncertain or promising; (2) each END_TURN
visit took exactly ONE fresh sample and committed the entire sub-budget to
exploring only from it, so a single unlucky decorrelated draw (a bad card
draw, a harsh enemy roll) could dominate an entire branch's value estimate
even though the search had budget to also check a few OTHER plausible
draws before committing depth to any of them.

DPW (Double Progressive Widening -- Couetoux et al., "Continuous Upper
Confidence Trees", 2011) replaces that hand-rolled version with the standard
technique for exactly this problem: stochastic transitions with a large or
continuous outcome space, where you can't enumerate every possible outcome
(here: every possible draw x every possible enemy-move roll) but also
shouldn't collapse to a single sample per visit. At a chance node (an
END_TURN selection), each visit either draws a NEW decorrelated outcome and
caches it as a proper child node, or REVISITS an existing cached outcome
--picked with probability proportional to that child's own visit count,
mirroring how PUCT itself concentrates visits on promising options. Which
of the two happens is governed by a widening cap k(n) = ceil(WC * n^WA):
as long as the number of cached outcomes is below the cap for this chance
node's total visit count n, a fresh sample gets added (building width);
once the cap catches up, revisits are forced (building depth into the
outcomes that have already earned it). This is where "double" comes from
-- both the outcome branching AND the depth into each outcome widen
progressively with total search budget, rather than either being fixed in
advance. Depth beyond the current turn falls out naturally rather than
needing its own constant: a revisited chance-child is a real cached node,
so a later simulation that reaches it and then reaches its OWN END_TURN
recurses through the exact same DPW logic one level deeper, for as many
levels as accumulated visits justify. No MAX_LOOKAHEAD_TURNS-style cap is
needed; the only limit is the total simulation budget passed to
choose_action, same as it already limits everything else the search does.

Backup uses the same reward accounting PPO trained the value head against
(see ppo.py's _collect_one_episode: G = reward + gamma * G) -- each edge's
backed-up value is env.py's own per-step reward plus gamma times the
child/leaf value, not just the leaf value alone, so the search stays
consistent with what the network actually learned to predict.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch

import slaythespire as sts

from .env import potential, terminal_reward as _env_terminal_reward, W_SHAPE
from .env import _encode_state
from .cards import OTHER_CARD_INDEX
from .monsters import OTHER_MONSTER_INDEX
from .policy import ActionScoringPolicy
from .potion_features import capture_active_potion_idxs, pad_potion_idxs, encode_action_with_potion, EMPTY_POTION_INDEX

C_PUCT = 1.5
GAMMA = 0.99
# Double Progressive Widening, for END_TURN chance nodes only (see module
# docstring). k(n) = ceil(WC_CHANCE * n^WA_CHANCE) caps the number of cached
# outcome-samples at a chance node once it has been visited n times; WA=0.5
# (sqrt growth) is the standard default from Couetoux et al. for continuous/
# large stochastic outcome spaces -- narrow at first (build a few different
# next-turn samples before committing), widening slowly so repeated visits
# increasingly favor deepening an already-cached outcome over sampling a new
# one.
WC_CHANCE = 1.0
WA_CHANCE = 0.5

# DEBUG-ONLY toggles for isolating the two crashes found in az_search.py
# (an Action.cpp:459 assertion, then a segfault) -- flip from a test
# script (az_search_debug.ENABLE_DEDUP = ...) to test each independently
# before ever considering re-merging anything into the real az_search.py.
ENABLE_DEDUP = False
ENABLE_TREE_REUSE = False
PERSIST_TRANS_TABLE = True  # DEBUG: isolate whether cross-decision trans_table persistence (as opposed to node reroot itself) is the actual bug
DEBUG_CHECK_CONSISTENCY = False  # DEBUG: verify cached rerooted node's state actually matches real env.bc
INCLUDE_MOVE_STATE_IN_KEY = False

# --- transposition table (deterministic intra-turn nodes only) ---
#
# sts_lightspeed's own native_search.py explored the same idea and stopped
# short: "No transposition-table caching (yet). The Python bindings expose
# only strength/vulnerable/weak on Monster -- no frail/poison/artifact/etc.
# -- so a cache key built from what's exposed could silently merge two
# states that actually differ in an unexposed status effect." Since then,
# this session added get_player_status_value/get_monster_status_value
# (generic per-name getters, not one binding per power), which is enough
# status coverage to make a real attempt -- built from the SAME canonical
# status lists env.py's own _encode_state already reads (not a separately
# hand-picked subset, so any gap is a gap _encode_state already has too).
# The same residual caveat still applies, though: a state dimension with no
# getter at all can't be included, so this isn't proven airtight, just
# built from everything currently exposed.
#
# Deliberately excludes RNG state, matching native_search.py's own
# precedent -- forward dynamics from an identical deterministic state are
# equivalent in expectation regardless of which specific RNG state produced
# it, since every chance-node sample already calls decorrelate_rng() first
# (see the END_TURN branch below). Scoped to the DETERMINISTIC intra-turn
# part only, same split as everywhere else in this file: END_TURN's sampled
# outcomes stay in chance_children, keyed by DPW's own visit-weighted
# revisit logic, not folded into this table.
_PLAYER_STATUS_NAMES = [
    "ARTIFACT", "BARRICADE", "METALLICIZE", "RITUAL", "RAGE", "RUPTURE",
    "COMBUST", "DEMON_FORM", "DARK_EMBRACE", "EVOLVE", "FEEL_NO_PAIN",
    "FIRE_BREATHING", "JUGGERNAUT", "PANACHE", "ENVENOM", "FLAME_BARRIER",
    "BRUTALITY", "REGEN", "CORRUPTION",
]
_MONSTER_STATUS_NAMES = ["POISON", "PLATED_ARMOR", "ARTIFACT", "METALLICIZE", "MODE_SHIFT", "TIME_WARP"]


def _crn_seed(crn_base: int, state_key: tuple, local_sample_index: int) -> int:
    """Common Random Numbers seed for the local_sample_index'th DPW sample
    drawn from the chance node whose PARENT deterministic state is
    state_key. Keyed off the node's own state + a LOCAL (per-node) sample
    count -- not a monotonic counter shared across the whole tree -- so
    that two SEPARATE search runs (comparing two configs/policies) which
    each happen to reach an equivalent decision point get the SAME
    underlying randomness for its k-th chance-outcome sample, regardless of
    what order each run's own traversal visited things in. This is what
    makes it a real port of Silver Automaton's semantic (shared
    randomnessBase for a matched comparison) rather than the earlier
    monotonic-counter version, which was measured (via scratch_crn_test.py/
    scratch_crn_test2.py) to give no variance reduction at all -- once two
    trees' accumulated values diverge even slightly, a shared counter no
    longer points at the same logical decision in both, so there's nothing
    left to cancel. Keying off state_key instead means the correlation only
    requires the two runs to revisit an EQUIVALENT state, not to have taken
    the same number of total simulation steps to get there.

    hash() on state_key is safe to use as a stable seed here: _state_key's
    tuple is built entirely from ints/floats/bools (hp, statuses, card ids,
    ...), never strings, so it isn't subject to Python's per-process string-
    hash randomization -- unlike hashing a str/bytes object, this reproduces
    identically across separate process runs, which paired A/B comparisons
    launched as separate processes depend on."""
    h = hash((crn_base, state_key, local_sample_index))
    return h & 0xFFFFFFFFFFFFFFFF  # mask to unsigned 64-bit for bc.seed_rng's std::uint64_t parameter


def _state_key(bc):
    """Hashable key for the deterministic intra-turn part of search -- see
    the block comment above for what's included/excluded and why."""
    p_statuses = tuple(bc.get_player_status_value(n) for n in _PLAYER_STATUS_NAMES)
    if INCLUDE_MOVE_STATE_IN_KEY:
        monsters = tuple(
            (
                m.cur_hp, m.block, m.strength, m.vulnerable, m.weak, m.half_dead,
                m.move_history, m.misc_info,
                tuple(bc.get_monster_status_value(i, n) for n in _MONSTER_STATUS_NAMES),
            )
            for i, m in enumerate(bc.monsters)
        )
    else:
        monsters = tuple(
            (
                m.cur_hp, m.block, m.strength, m.vulnerable, m.weak, m.half_dead,
                tuple(bc.get_monster_status_value(i, n) for n in _MONSTER_STATUS_NAMES),
            )
            for i, m in enumerate(bc.monsters)
        )
    hand = tuple(sorted(int(c.id) for c in bc.hand))  # order-independent: any card in hand is playable regardless of storage order
    draw = tuple(int(c.id) for c in bc.draw_pile)  # order-DEPENDENT: determines future draws
    discard = tuple(sorted(int(c.id) for c in bc.discard_pile))  # reshuffled via RNG on empty draw, order doesn't matter
    return (bc.player_hp, bc.player_block, bc.player_energy, bc.turn,
            p_statuses, monsters, hand, draw, discard)


# --- heuristic-rollout fallback (deep leaf evaluation without a network call) ---
#
# Silver Automaton (daniel-ziegler/sts_lightspeed's combat search) evaluates
# every brand-new node with "a fast randomized heuristic-agent playout to
# the end of combat" rather than a trained network -- no NN call anywhere
# in their combat search at all, which is a big part of how they afford
# ~10,000 iterations/decision. This project's search is network-guided
# (AlphaZero-style PUCT: real priors + a real learned value estimate), which
# is worth keeping for the near-term decision that actually matters most --
# but DPW's whole point is letting depth grow past that near-term horizon
# as budget allows, and the network was never trained to be accurate many
# hops past the current real decision anyway. Past ROLLOUT_DEPTH_TURNS real
# END_TURN transitions deep in a single simulate() call, a freshly-created
# node is evaluated with a cheap heuristic playout instead of a full
# _expand() (network) call -- lets DPW deepen arbitrarily without every
# extra hop costing a real forward pass, same spirit as Silver Automaton's
# approach, just applied only past the horizon the network was ever
# calibrated for instead of everywhere.
ROLLOUT_DEPTH_TURNS = 2
ROLLOUT_MAX_ACTIONS = 200  # safety cap so a pathological rollout can't loop forever

# Transposition sharing (see _state_key below) turns the search tree into a
# graph: the SAME cached node can be reachable as a child of many different
# parents at once. That's the whole point for the common case (a handful of
# card-play orders landing on an equivalent state), but it also means PUCT's
# own under-explored-node bonus (which favors low-N actions) can, in a
# degenerate case, drive one simulate() call through a long chain of shared
# nodes each still individually low-visit from EVERY OTHER parent's
# perspective, recursing far deeper within a single turn than any real
# card-play sequence could -- observed in practice as a Python
# RecursionError, not a game-logic issue. call_depth is a blunt safety
# valve against exactly that: incremented on every recursive call
# (deterministic AND chance), unrelated to `depth`'s turn-counting role.
# Past MAX_CALL_DEPTH, bail out via the same cheap heuristic-playout escape
# hatch ROLLOUT_DEPTH_TURNS already uses, rather than inventing a second
# fallback mechanism.
MAX_CALL_DEPTH = 150


def _heuristic_pick(sim, legal: List):
    """1-ply-greedy action choice with no network involved: try every legal
    action against a throwaway copy and keep whichever leaves potential()
    highest. Reuses potential() (env.py's own shaping function, already
    imported) rather than inventing a separate StS-specific heuristic --
    cheap (pure C++ copy+execute per candidate, no Python/torch tensor
    overhead), not sophisticated, which matches Silver Automaton's own
    "randomized heuristic-agent" framing: good enough for a deep rollout
    tail, not meant to replace the real network-guided search above it."""
    non_end = [a for a in legal if a.action_type != sts.ActionType.END_TURN]
    if not non_end:
        return legal[0]
    best, best_score = non_end[0], float("-inf")
    for a in non_end:
        trial = sts.BattleContext(sim)
        a.execute(trial)
        score = potential(trial)
        if score > best_score:
            best_score = score
            best = a
    return best


def _heuristic_playout(bc) -> float:
    """Plays a COPY of bc to completion using _heuristic_pick, no network
    calls at all, and returns env.py's terminal_reward once decided (or a
    potential()-based fallback if ROLLOUT_MAX_ACTIONS is hit first, which
    should be rare). Operates on a copy specifically so the caller's own
    node.bc -- still needed by future _dense_reward calls against this same
    node on subsequent revisits -- is never mutated by the rollout."""
    sim = sts.BattleContext(bc)
    for _ in range(ROLLOUT_MAX_ACTIONS):
        if sim.outcome != sts.BattleOutcome.UNDECIDED:
            return _terminal_reward(sim, sim.turn)
        action = _heuristic_pick(sim, sim.get_legal_actions())
        action.execute(sim)
    return W_SHAPE * potential(sim)


class _Node:
    __slots__ = ("bc", "actions", "P", "N", "W", "children", "chance_children",
                 "chance_samples_drawn", "key",
                 "is_terminal", "terminal_value", "visit_count", "rollout_value")

    def __init__(self, bc):
        self.bc = bc
        self.actions: Optional[List] = None
        self.P: Optional[np.ndarray] = None
        self.N: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None
        self.children: Dict[int, "_Node"] = {}
        # Only used for END_TURN action indices: a chance node's cached
        # outcome-samples, each a real _Node (see module docstring on DPW).
        # Kept separate from `children` (which assumes one deterministic
        # outcome per action) rather than overloading it with a list.
        self.chance_children: Dict[int, List["_Node"]] = {}
        self.chance_samples_drawn: Dict[int, int] = {}  # DEBUG: ENABLE_DEDUP
        self.key: Optional[tuple] = None  # DEBUG: ENABLE_DEDUP
        self.is_terminal = False
        self.terminal_value = 0.0
        # Total times this node has been reached by _simulate, incremented
        # unconditionally on entry (including terminal nodes) -- DPW's
        # revisit selection weights existing chance-children by this.
        self.visit_count = 0
        # Cached _heuristic_playout() result, once computed -- see
        # ROLLOUT_DEPTH_TURNS. None until a rollout has actually run; a
        # node that gets a rollout value never gets a real _expand() (its
        # actions/P/N/W stay None forever), same "dead end leaf" treatment
        # a real terminal node gets.
        self.rollout_value: Optional[float] = None


def _terminal_reward(bc, turn_at_terminal: int) -> float:
    """env.py's own terminal_reward, W_SHAPE-scaled -- env.py's step() now
    scales the terminal term too (previously it didn't), see its own
    comments on W_WIN/W_DEATH being sized for that."""
    return W_SHAPE * _env_terminal_reward(bc, turn_at_terminal)


def _dense_reward(bc_before, bc_after) -> float:
    """env.py's own Phi-based shaping term. Previously this took four plain
    hp numbers and re-derived a one-line formula locally ("kept in sync
    deliberately rather than imported"); now that Phi weighs telegraphed
    incoming damage per-monster (BETA) as well as raw HP, re-deriving it
    here risked silently drifting out of sync with env.py's actual reward,
    so this now calls env.py's potential() directly instead -- the same
    reasoning that already made this search's backup reuse env.py's reward
    accounting rather than inventing its own (see module docstring).
    potential() already returns 0.0 for a terminal bc_after, so no separate
    done-check is needed here."""
    return W_SHAPE * (GAMMA * potential(bc_after) - potential(bc_before))


@torch.no_grad()
def _expand(node: _Node, policy: ActionScoringPolicy, relic_idxs: torch.Tensor, relic_mask: torch.Tensor) -> float:
    """Score node.bc's legal actions for priors + this state's value
    estimate. Returns the value (used as the backed-up leaf value by the
    caller). relic_idxs/relic_mask are constant for the whole search tree
    (relics never change mid-combat) -- threaded down from choose_action's
    caller, same as `policy` itself, rather than re-derived per node.
    Potion state, unlike relics, is captured FRESH here from node.bc every
    call instead of threaded down -- potions can be drunk/discarded within
    the search tree itself (a simulated future turn might use one), so a
    value captured once at the root would go stale exactly the same way
    env.py's own _observation() (not reset()) recomputes it every call."""
    bc = node.bc
    legal = bc.get_legal_actions()
    hand = bc.hand
    state, total_living_hp = _encode_state(bc, hand)
    action_features = []
    card_idxs = []
    monster_idxs = []
    action_potion_idxs = []
    for a in legal:
        feats, card_idx, monster_idx, potion_idx = encode_action_with_potion(bc, a, total_living_hp, hand)
        action_features.append(feats)
        card_idxs.append(card_idx)
        monster_idxs.append(monster_idx)
        action_potion_idxs.append(potion_idx)

    state_t = torch.as_tensor(state, dtype=torch.float32)
    action_features_t = torch.as_tensor(np.stack(action_features), dtype=torch.float32)
    card_idxs_t = torch.as_tensor(
        [OTHER_CARD_INDEX if c is None else c for c in card_idxs], dtype=torch.long,
    )
    monster_idxs_t = torch.as_tensor(
        [OTHER_MONSTER_INDEX if m is None else m for m in monster_idxs], dtype=torch.long,
    )
    action_potion_idxs_t = torch.as_tensor(
        [EMPTY_POTION_INDEX if p is None else p for p in action_potion_idxs], dtype=torch.long,
    )
    potion_idxs_np, potion_mask_np = pad_potion_idxs(capture_active_potion_idxs(bc))
    potion_idxs = torch.as_tensor(potion_idxs_np, dtype=torch.long)
    potion_mask = torch.as_tensor(potion_mask_np, dtype=torch.bool)

    state_emb = policy.encode_state(state_t, relic_idxs, relic_mask, potion_idxs, potion_mask)
    scores = policy.score_actions(state_t, action_features_t, card_idxs_t, monster_idxs_t, action_potion_idxs_t,
                                   relic_idxs, relic_mask, potion_idxs, potion_mask, state_emb=state_emb)
    priors = torch.softmax(scores, dim=-1).numpy()
    value = policy.value_head(state_emb).item()  # reuses state_emb -- no second encode_state pass needed

    node.actions = legal
    node.P = priors
    node.N = np.zeros(len(legal), dtype=np.int64)
    node.W = np.zeros(len(legal), dtype=np.float64)
    return value


def _simulate(node: _Node, policy: ActionScoringPolicy, trans_table: Dict[tuple, "_Node"],
              relic_idxs: torch.Tensor, relic_mask: torch.Tensor,
              depth: int = 0, call_depth: int = 0,
              crn_base: Optional[int] = None) -> float:
    """depth: how many real END_TURN transitions deep this call is, starting
    from choose_action's root at 0 -- only ever incremented crossing an
    END_TURN (see the chance-node branch below), used solely to decide when
    a freshly-reached node gets _heuristic_playout instead of _expand (see
    ROLLOUT_DEPTH_TURNS). Unlike the old turns_left, this never bounds DPW's
    own recursion -- it only changes HOW a new leaf gets evaluated.

    call_depth: incremented on EVERY recursive call regardless of branch --
    an unrelated safety valve against transposition-driven degenerate
    recursion (see MAX_CALL_DEPTH).

    relic_idxs/relic_mask: constant for the whole search tree (relics never
    change mid-combat) -- threaded down from choose_action's caller, same as
    `policy` itself. _heuristic_playout doesn't need them (it uses env.py's
    potential(), not the network).

    crn_base: Common Random Numbers (see module docstring, bindings/
    slaythespire.cpp's seed_rng comment, and _crn_seed's own docstring) --
    when not None, every NEW DPW-widened chance-node sample seeds via
    _crn_seed(crn_base, state_key(parent), local_sample_index) instead of
    the default bc.decorrelate_rng(). A first version keyed this off a
    single counter shared across the whole tree instead of per-node state;
    measured (scratch_crn_test.py/scratch_crn_test2.py) to give NO variance
    reduction, because two compared trees diverge in accumulated values
    almost immediately, so a shared counter stops pointing at the same
    logical decision in both almost right away. Keying off the node's own
    state instead means two runs only need to revisit an EQUIVALENT state
    to get matched randomness there, not to have taken identical paths to
    reach it. None (the default) preserves the original independently-
    decorrelated behavior exactly."""
    node.visit_count += 1
    if node.is_terminal:
        return node.terminal_value
    if call_depth >= MAX_CALL_DEPTH:
        if node.rollout_value is None:
            node.rollout_value = _heuristic_playout(node.bc)
        return node.rollout_value
    if node.actions is None:
        if node.rollout_value is not None:
            return node.rollout_value
        if depth >= ROLLOUT_DEPTH_TURNS:
            value = _heuristic_playout(node.bc)
            node.rollout_value = value
            return value
        return _expand(node, policy, relic_idxs, relic_mask)

    q = np.divide(node.W, node.N, out=np.zeros_like(node.W), where=node.N > 0)
    u = C_PUCT * node.P * math.sqrt(node.N.sum() + 1) / (1 + node.N)
    idx = int(np.argmax(q + u))
    action = node.actions[idx]
    bc = node.bc

    if action.action_type == sts.ActionType.END_TURN:
        # Stochastic chance node: DPW over cached outcome-samples (see
        # module docstring). node.N[idx] is this chance node's own total
        # visit count n (incremented at the bottom of this function, same
        # as every other action index) -- used for the widening cap, not
        # node.visit_count, which counts visits to `node` itself rather
        # than to this specific chance-producing action.
        siblings = node.chance_children.setdefault(idx, [])
        n = int(node.N[idx])
        widening_cap = math.ceil(WC_CHANCE * (n + 1) ** WA_CHANCE)
        if len(siblings) < widening_cap:
            local_sample_index = node.chance_samples_drawn.get(idx, 0) if ENABLE_DEDUP else len(siblings)
            node.chance_samples_drawn[idx] = local_sample_index + 1
            if crn_base is not None:
                bc.seed_rng(_crn_seed(crn_base, _state_key(bc), local_sample_index))
            else:
                bc.decorrelate_rng()
            sample = sts.BattleContext(bc)
            action.execute(sample)
            child = None
            if ENABLE_DEDUP:
                sample_key = _state_key(sample)
                for existing in siblings:
                    if existing.key == sample_key:
                        child = existing
                        break
            if child is None:
                child = _Node(sample)
                if ENABLE_DEDUP:
                    child.key = sample_key
                if sample.outcome != sts.BattleOutcome.UNDECIDED:
                    child.is_terminal = True
                    child.terminal_value = _terminal_reward(sample, sample.turn)
                siblings.append(child)
        else:
            # Revisit an existing sampled outcome, weighted by its own
            # visit count -- the chance-node analogue of PUCT concentrating
            # visits on promising options, per Couetoux et al.
            weights = np.array([c.visit_count + 1 for c in siblings], dtype=np.float64)
            child = siblings[int(np.random.choice(len(siblings), p=weights / weights.sum()))]
        r = _dense_reward(bc, child.bc)
        if child.is_terminal:
            value = r + child.terminal_value
        else:
            value = r + GAMMA * _simulate(child, policy, trans_table, relic_idxs, relic_mask,
                                            depth + 1, call_depth + 1, crn_base)
    else:
        child = node.children.get(idx)
        if child is None:
            child_bc = sts.BattleContext(bc)
            action.execute(child_bc)
            r = _dense_reward(bc, child_bc)
            if child_bc.outcome != sts.BattleOutcome.UNDECIDED:
                child = _Node(child_bc)
                child.is_terminal = True
                child.terminal_value = _terminal_reward(child_bc, child_bc.turn)
                node.children[idx] = child
                value = r + child.terminal_value
            else:
                # Transposition lookup -- a different card-play ORDER within
                # this same turn can land on an identical deterministic
                # state (same hp/block/energy/statuses/hand/piles); reusing
                # the cached node means its accumulated N/W/P get shared
                # instead of re-explored from scratch (see the table's own
                # block comment above for what the key does/doesn't cover).
                key = _state_key(child_bc)
                child = trans_table.get(key)
                if child is None:
                    child = _Node(child_bc)
                    trans_table[key] = child
                node.children[idx] = child
                value = r + GAMMA * _simulate(child, policy, trans_table, relic_idxs, relic_mask,
                                                depth, call_depth + 1, crn_base)
        else:
            r = _dense_reward(bc, child.bc)
            value = r + GAMMA * _simulate(child, policy, trans_table, relic_idxs, relic_mask,
                                            depth, call_depth + 1, crn_base)

    node.N[idx] += 1
    node.W[idx] += value
    return value


def choose_action(bc, policy: ActionScoringPolicy, relic_idxs: torch.Tensor, relic_mask: torch.Tensor,
                   n_simulations: int = 100, crn_base: Optional[int] = None):
    """PUCT-searched action for the current decision, using `policy`'s own
    priors/value instead of a blind forward pass. Returns (action,
    visit_counts) -- visit_counts (not raw Q) is the AlphaZero-standard
    signal for "how good the search thought each option was", and is what
    Stage 2 (search-refined training targets) would consume if built.

    relic_idxs/relic_mask: this episode's active relics (see env.py's
    capture_active_relic_idxs) -- constant for the whole search tree, so
    threaded through once here rather than re-derived per node.

    trans_table is scoped to this one call (a fresh dict each time), same
    as the tree itself -- no cross-decision persistence, matching every
    other piece of this search's state.

    A tree/rerootAt-style reuse across decisions (mirroring Silver
    Automaton's BattleSearcher) was attempted twice and reverted both
    times. First attempt: matched a real post-END_TURN state against a
    cached DPW sample via _state_key -- hit a hard C++ assertion crash
    (Action.cpp:459) on Awakened One, traced to _state_key omitting
    monster move_history/misc_info (Monster.h documents misc_info as
    move-selection-relevant for several bosses: Time Eater's has-used-
    Haste flag, Awakened One's isPhase2, ...). Second attempt: added
    move_history/misc_info bindings and to _state_key, then re-tried BOTH
    a same-tree DPW-dedup merge (using the fuller key to merge same-
    action-same-parent samples) and the deterministic-only reroot case --
    this time hit a SEGFAULT (not even a clean assertion) on the exact
    same fight, before the specific cause could be isolated further.
    Given a memory-safety crash is a more serious failure mode than a
    logic assertion, and two attempts at building confidence in a fuller
    key both ended in a hard crash, this was reverted a second time rather
    than continuing to iterate live -- worth a dedicated, careful
    investigation before retrying rather than another quick patch.

    crn_base: Common Random Numbers seed (see _simulate's own docstring and
    bindings/slaythespire.cpp's seed_rng) -- None (default) preserves the
    original independently-decorrelated behavior; a caller doing a paired
    comparison across two configurations should pass the SAME crn_base to
    both choose_action calls for that same decision point."""
    root = _Node(sts.BattleContext(bc))
    trans_table: Dict[tuple, _Node] = {}
    best_idx = _run_search(root, trans_table, policy, relic_idxs, relic_mask, n_simulations, crn_base)
    return root.actions[best_idx], root.N


def _run_search(root: _Node, trans_table: Dict[tuple, _Node], policy: ActionScoringPolicy,
                 relic_idxs: torch.Tensor, relic_mask: torch.Tensor,
                 n_simulations: int, crn_base: Optional[int] = None) -> int:
    for _ in range(n_simulations):
        _simulate(root, policy, trans_table, relic_idxs, relic_mask, crn_base=crn_base)
    return int(np.argmax(root.N))


def _find_child_for_real_state(node: _Node, idx: int) -> Optional[_Node]:
    """DEBUG: ENABLE_TREE_REUSE. Deterministic-only reroot lookup."""
    child = node.children.get(idx)
    if child is not None and (child.actions is not None or child.is_terminal):
        return child
    return None


def run_episode_with_search(env, policy: ActionScoringPolicy, n_simulations: int = 50, seed=None):
    """DEBUG version: reuse gated by ENABLE_TREE_REUSE for isolation testing."""
    obs = env.reset(seed=seed)
    done = False
    info = None
    root: Optional[_Node] = None
    trans_table: Optional[Dict[tuple, _Node]] = None
    while not done:
        relic_idxs = torch.as_tensor(obs["relic_idxs"], dtype=torch.long)
        relic_mask = torch.as_tensor(obs["relic_mask"], dtype=torch.bool)
        if not ENABLE_TREE_REUSE or root is None:
            root = _Node(sts.BattleContext(env.bc))
            if trans_table is None or not PERSIST_TRANS_TABLE:
                trans_table = {}
        best_idx = _run_search(root, trans_table, policy, relic_idxs, relic_mask, n_simulations)
        action = root.actions[best_idx]
        obs, reward, done, info = env.step(action)
        if not done and ENABLE_TREE_REUSE:
            root = _find_child_for_real_state(root, best_idx)
            if root is not None and DEBUG_CHECK_CONSISTENCY:
                cached_key = _state_key(root.bc)
                real_key = _state_key(env.bc)
                if cached_key != real_key:
                    print("MISMATCH DETECTED between cached rerooted node and real env.bc!", flush=True)
                    print("cached:", cached_key, flush=True)
                    print("real:  ", real_key, flush=True)
                    raise RuntimeError("cached/real state mismatch after reroot")
    return info


def evaluate_with_search(env, policy: ActionScoringPolicy, n: int, n_simulations: int = 50,
                          seed_offset: int = 0, per_encounter: bool = False):
    """Same signature/shape as train.py's evaluate(), for a direct,
    apples-to-apples comparison of inference-time search vs a blind forward
    pass -- against the SAME already-trained policy, no retraining involved.
    Substitutes lookahead for whatever strategy the network hasn't learned,
    so it's the cheapest thing to check before investing in more training."""
    wins = hp_sum = 0
    breakdown = {}
    with torch.no_grad():
        for i in range(n):
            info = run_episode_with_search(env, policy, n_simulations=n_simulations, seed=seed_offset + i)
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
# Episode-level parallelism, not tree-level: each episode's search is fully
# independent of every other episode's (unlike a single decision's tree,
# where simulations share state and would need real merging), so this is
# the embarrassingly-parallel case -- no root-parallelization/visit-count
# merging complexity needed, just N workers each running a slice of the
# episodes and combining the win/hp/breakdown tallies at the end.
#
# Mirrors ppo.py's _worker_init/_worker_collect_chunk pattern exactly,
# including the same torch.set_num_threads(1) fix -- without it, each of
# N_WORKERS processes defaults to multi-threaded intra-op parallelism,
# oversubscribing the actual core count and measuring SLOWER than serial
# (ppo.py's own comment on this; same cause applies here since this is the
# same tiny-forward-pass-per-node workload, just under search instead of a
# blind policy.act() call).

import multiprocessing as mp

_search_worker_env = None
_search_worker_policy: Optional[ActionScoringPolicy] = None


def _search_worker_init(env_kwargs: dict) -> None:
    global _search_worker_env, _search_worker_policy
    torch.set_num_threads(1)
    from .env import IroncladFightEnv
    _search_worker_env = IroncladFightEnv(**env_kwargs)
    _search_worker_policy = ActionScoringPolicy()


def _search_worker_run_episodes(args):
    state_dict, n_episodes, n_simulations, seed_offset = args
    _search_worker_policy.load_state_dict(state_dict)
    wins = hp_sum = 0
    breakdown = {}
    with torch.no_grad():
        for i in range(n_episodes):
            info = run_episode_with_search(_search_worker_env, _search_worker_policy,
                                            n_simulations=n_simulations, seed=seed_offset + i)
            won = info["outcome"] == sts.BattleOutcome.PLAYER_VICTORY
            if won:
                wins += 1
                hp_sum += info["player_hp"]
            key = _search_worker_env.last_encounter
            w, total = breakdown.get(key, [0, 0])
            breakdown[key] = [w + (1 if won else 0), total + 1]
    return wins, hp_sum, breakdown


def evaluate_with_search_parallel(env, policy: ActionScoringPolicy, n: int, n_simulations: int = 50,
                                   seed_offset: int = 0, per_encounter: bool = False, n_workers: int = 6):
    """Same result shape as evaluate_with_search, split across n_workers
    processes by episode. Each worker gets its own contiguous seed range
    (seed_offset + worker's slice) rather than interleaving, so results are
    exactly the union of what evaluate_with_search would have produced for
    those same seeds serially -- not a different sample, just faster."""
    from .ppo import _env_kwargs_from
    env_kwargs = _env_kwargs_from(env)
    state_dict = policy.state_dict()

    counts = [n // n_workers + (1 if i < n % n_workers else 0) for i in range(n_workers)]
    args_list = []
    offset = seed_offset
    for c in counts:
        if c > 0:
            args_list.append((state_dict, c, n_simulations, offset))
        offset += c

    with mp.Pool(len(args_list), initializer=_search_worker_init, initargs=(env_kwargs,)) as pool:
        results = pool.map(_search_worker_run_episodes, args_list)

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
