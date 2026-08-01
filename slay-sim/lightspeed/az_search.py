"""AlphaZero-style search-augmented inference: wraps the already-trained
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
    the block comment above for what's included/excluded and why.

    monsters' move_history/misc_info included after root-causing a real
    crash (two, actually -- an assertion then a segfault) to two combined
    gaps: (1) this key omitted them, and Monster.h documents misc_info as
    move-selection-relevant for several bosses (Time Eater's has-used-
    Haste flag, Awakened One's isPhase2, ...) and move_history as what
    several bosses check to avoid/force move repeats; (2) separately and
    more fundamentally, _simulate used to classify any non-END_TURN action
    as deterministic by action_type alone, which is false -- some CARD
    actions (shuffle-into-draw-pile effects: Wild Strike, Reckless Charge)
    consume RNG internally too. (2) is the primary fix (see _simulate's
    RNG-counter probe), confirmed via direct evidence: two states that
    matched on every field THIS key already tracked, reached via a
    misclassified "deterministic" action, differed only in draw-pile order
    -- not a move-history/misc_info gap at all. Both gaps are fixed now;
    documenting both since either alone would still have been unsafe.

    Delegates to bc.state_key_bundle() (bindings/slaythespire.cpp), a single
    C++ call that computes this exact tuple directly (ordinal/templated
    status dispatch, no string comparisons) instead of the ~19 + 6*n_monsters
    separate get_player_status_value/get_monster_status_value pybind11
    crossings the pure-Python version used to make -- profiling found this
    function the dominant remaining cost of search rollouts once
    get_card_type caching was already fixed (see _cached_card_type's own
    comment). Validated field-for-field identical to the old pure-Python
    implementation across 760 random states spanning 7 encounters
    (scratchpad/validate_state_key_bundle.py) before this replaced it --
    _state_key's own crash history makes an exact match, not just "close
    enough", the bar here. _PLAYER_STATUS_NAMES/_MONSTER_STATUS_NAMES above
    are now purely documentation of what the C++ side computes (also used
    by az_search_debug.py's own separate, non-production copy of this
    function)."""
    return bc.state_key_bundle()


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


_CARD_TYPE_CACHE: Dict[int, "sts.CardType"] = {}


def _cached_card_type(card_id) -> "sts.CardType":
    """sts.get_card_type(id) is a pure function of a static, unchanging
    property (a card's type never depends on game state) -- caching it in
    a plain Python dict avoids a real, measured cost: profiling
    expectimax_search.py found 66,105 get_card_type calls (a pybind11
    cross-language call each) across just 5 choose_action calls, entirely
    avoidable since the same handful of card ids repeat constantly across
    a rollout's many _heuristic_pick calls."""
    key = int(card_id)
    cached = _CARD_TYPE_CACHE.get(key)
    if cached is None:
        cached = sts.get_card_type(card_id)
        _CARD_TYPE_CACHE[key] = cached
    return cached


def _predicted_incoming_damage(sim, monster_idx: int, m, vuln_mult: float) -> int:
    """Real predicted damage from monster `monster_idx`'s CURRENTLY QUEUED
    move -- NOT bc.get_monster_move_damage's raw value, which is only a
    static per-move-id base-damage TABLE LOOKUP (see sts_lightspeed's
    MonsterMoveDamage.cpp: Monster::getMoveBaseDamage is a pure switch on
    moveHistory[0]) with none of the adjustments the engine's OWN damage
    resolution actually applies (Monster::calculateDamageToPlayer, in
    Monster.cpp): the monster's current Strength is ADDED, the monster's
    own Weak status multiplies by 0.75, and the player's Vulnerable status
    multiplies by 1.5, all before flooring and clamping at 0. Every
    heuristic/reward signal in this file that used the raw table lookup as
    if it were the real predicted damage was silently wrong by exactly
    this gap -- confirmed empirically on Time Eater (a fight that stacks
    +2 Strength via repeated Time Warp procs and applies Weak via Ripple):
    directly comparing get_monster_move_damage's prediction against the
    ACTUAL HP loss one END_TURN later showed mismatches from -12 to +16,
    both signs, exactly matching the missing Weak/Strength/Vulnerable
    terms. Diagnosed while investigating why the player was still losing
    Time Eater fights after the _heuristic_pick defense-scoring fix (see
    that function's own comment) despite defense now being ABLE to win
    the scoring competition -- the DANGER SIGNAL feeding that scoring was
    itself understating real incoming damage, sometimes by 50%+."""
    dmg, hits = sim.get_monster_move_damage(monster_idx)
    if dmg <= 0:
        return 0
    weak_mult = 0.75 if m.weak > 0 else 1.0
    per_hit = max(0, int((dmg + m.strength) * weak_mult * vuln_mult))
    return per_hit * hits


def _heuristic_pick(sim, legal: List):
    """Rule-based action choice with no network AND no per-candidate
    simulation involved: scores each legal action from direct field reads
    (card type, target HP fraction, whether the player is in danger) rather
    than executing it against a throwaway copy first. A first version did
    exactly that (copy+execute+potential() per candidate) -- correct, but
    measured as the dominant cost of expectimax_search.py's rollouts (a
    single call to this function, needed at every step of every rollout,
    was paying for len(legal) full BattleContext copies + card-effect
    executions + potential() evaluations each time). Silver Automaton's own
    "randomized heuristic-agent" framing implies something cheap and
    approximate is exactly the right tool for a rollout policy -- it only
    needs to be decent on average across many samples, not precise on any
    one -- so trading exactness for a large constant-factor speedup here is
    the right tradeoff, unlike in the real search tree above, where
    decisions get backed up individually and precision matters more.

    incoming/unblocked computed ONCE per call (not per candidate) -- an
    O(monsters) cost shared by every candidate's scoring, rather than
    O(legal_actions) copies of the same information.

    Two boss-specific adjustments, both cheap (direct field reads only, no
    classify_monster_move/simulate-based calls on this hot path):
      - Time Eater's Haste: triggers on ITS next turn once already at/below
        50% HP (confirmed reading MonsterSpecific.cpp), healing back to
        50% and wiping every player-applied debuff. Once that's true and
        it hasn't used Haste yet (misc_info == 0, see Monster.h's own
        comment on what that field means for this specific boss), any
        debuff SKILL played this turn is guaranteed wasted -- deprioritized
        in favor of attacks (whose damage this turn is real, unlike a
        debuff that gets wiped before it matters) or other options.
        Deliberately NOT trying to model the extra nuance that damage
        beyond what's needed to reach 50% also gets partially undone by
        the heal-back-to-50% -- that needs simulating exact damage, which
        this heuristic doesn't do; "stop wasting debuffs here" is the
        clear, high-confidence part.
      - Time Warp: triggers once 12 cards have been played across the
        whole fight (tracked via the TIME_WARP status value), granting the
        monster +2 Strength and ending the turn immediately regardless of
        remaining energy/cards. Once one more card play would hit that
        threshold, END_TURN gets folded into the SAME scoring competition
        as card plays (normally skipped entirely) so the heuristic can
        choose to stop on its own instead of always maximizing cards
        played this turn."""
    monsters = sim.monsters
    hand = sim.hand
    vuln_mult = 1.5 if sim.get_player_status_value("VULNERABLE") > 0 else 1.0
    incoming = 0
    for i, m in enumerate(monsters):
        if m.cur_hp > 0:
            incoming += _predicted_incoming_damage(sim, i, m, vuln_mult)
    unblocked = max(0, incoming - sim.player_block)

    time_warp_risk = any(
        m.cur_hp > 0 and sim.get_monster_status_value(i, "TIME_WARP") >= 11
        for i, m in enumerate(monsters)
    )
    haste_wasted_debuffs = any(
        m.cur_hp > 0 and m.name == "TIME_EATER" and m.misc_info == 0
        and m.cur_hp <= m.max_hp * 0.5
        for m in monsters
    )

    best, best_score = None, float("-inf")
    if time_warp_risk:
        end_turn_action = next((a for a in legal if a.action_type == sts.ActionType.END_TURN), None)
        if end_turn_action is not None:
            # Competitive with a plain attack (10.0), not an automatic
            # override -- a genuinely strong play this turn can still win
            # out, this just stops END_TURN from being excluded outright.
            best, best_score = end_turn_action, 11.0

    for a in legal:
        if a.action_type == sts.ActionType.END_TURN:
            continue
        if a.action_type == sts.ActionType.CARD:
            card = hand[a.source_idx]
            ctype = _cached_card_type(card.id)
            if ctype == sts.CardType.ATTACK:
                score = 10.0
                if 0 <= a.target_idx < len(monsters) and monsters[a.target_idx].cur_hp > 0:
                    target = monsters[a.target_idx]
                    score += (1.0 - target.cur_hp / max(1, target.max_hp)) * 5.0  # finish off low-HP targets
                    # Prefer targets whose block won't eat the whole hit --
                    # added after confirming Spheric Guardian's actual move
                    # cycle (MonsterSpecific.cpp): 25-35 block turn one, then
                    # +15 more every other turn on TOP of whatever remains,
                    # while still attacking every turn -- stalling doesn't
                    # help (the block only grows, nothing decays it faster
                    # than attacking does), so a target with high current
                    # block is specifically where chip damage is wasted, not
                    # where patience pays off. Capped rather than unbounded:
                    # sometimes attacking into heavy block IS still the
                    # right call (single target, no alternative, or already
                    # partway through breaking it), this should discourage
                    # that choice when a genuinely better target exists, not
                    # rule it out.
                    score -= min(target.block, 20) * 0.15
            elif ctype == sts.CardType.SKILL:
                # Continuous in danger_fraction, NOT the old binary in_danger
                # gate -- that gave SKILL a flat 9.0 even when "in danger",
                # which is STILL below ATTACK's base 10.0, so attacking
                # structurally always won the comparison regardless of how
                # much danger the player was actually in. Confirmed the hard
                # way: diagnosing Time Eater's 0/15 win rate found the
                # player had ZERO block at the start of EVERY turn across
                # every traced seed -- the rollout heuristic (which drives
                # leaf value estimates throughout the tree) was never
                # choosing to block at all, in a 456 HP grind fight where
                # sustained chip damage over many turns is exactly what
                # defense is for. Scaling continuously with how much of the
                # player's current HP is unblocked-and-incoming means a
                # genuinely dangerous turn (danger_fraction near/above 1.0)
                # scores well above any attack, while a safe turn still
                # deprioritizes defense the same as before (score near 4.0).
                danger_fraction = unblocked / max(1, sim.player_hp)
                score = 4.0 + danger_fraction * 30.0
                if haste_wasted_debuffs:
                    score -= 5.0  # any debuff this skill applies is about to be wiped for free
            elif ctype == sts.CardType.POWER:
                score = 6.0  # generally worth playing early, no per-power distinction attempted
            else:
                score = 1.0
        else:
            score = 5.0  # potion or other non-card action -- no finer heuristic yet, a flat mid priority
        if score > best_score:
            best_score = score
            best = a
    if best is None:
        best = legal[0]  # only reachable if legal contained nothing but END_TURN and time_warp_risk was False
    return best


def _heuristic_playout(bc) -> float:
    """Plays a COPY of bc to completion using _heuristic_pick, no network
    calls at all, and returns env.py's terminal_reward once decided (or a
    potential()-based fallback if ROLLOUT_MAX_ACTIONS is hit first, which
    should be rare). Operates on a copy specifically so the caller's own
    node.bc -- still needed by future _dense_reward calls against this same
    node on subsequent revisits -- is never mutated by the rollout.

    Delegates to bc.heuristic_playout() (bindings/slaythespire.cpp), a
    single native C++ call that runs this ENTIRE rollout -- heuristic
    scoring, action execution, and the terminal/potential reward -- with no
    Python<->C++ crossings per action, instead of this function's old
    Python for-loop calling _heuristic_pick + action.execute once per step
    (each of those a real language-boundary crossing). Profiling found this
    the second-largest remaining search cost after _state_key (already
    fixed via state_key_bundle): potential() alone was ~26% of total search
    time, largely from re-copying bc.monsters across the boundary on every
    call. Validated field-for-field (exact reward across 1359 paired
    rollouts with matched RNG, spanning 10 encounters -- see scratchpad/
    validate_heuristic_playout.py) against this exact Python implementation
    before it was replaced. The Python implementation above is preserved,
    not deleted, as the reference _heuristic_pick/ROLLOUT_MAX_ACTIONS still
    documents and as the thing any future re-validation compares against --
    see the C++ side's own DRIFT WARNING comment (bindings/slaythespire.cpp)
    for why touching _heuristic_pick's scoring or these constants requires
    re-syncing both."""
    return bc.heuristic_playout()


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
        # Chance-node action indices only -- END_TURN always, plus any
        # other action idx that _simulate's RNG-counter probe discovers
        # actually consumes randomness (see _simulate's own docstring on
        # why action_type alone isn't a reliable classifier). Each entry
        # is a list of cached outcome-samples for that idx, DPW-managed
        # (see module docstring). Kept separate from `children` (which
        # assumes one deterministic outcome per action) rather than
        # overloading it with a list.
        self.chance_children: Dict[int, List["_Node"]] = {}
        # Per-idx count of widen ATTEMPTS at a chance node, distinct from
        # len(chance_children[idx]) (distinct outcomes) -- a widen attempt
        # that turns out to duplicate an existing outcome must still not
        # reuse the same CRN seed offset next time, or it would keep
        # resampling the identical duplicate forever. Also seeded to 1 the
        # moment an action idx is FIRST discovered to be stochastic (see
        # _simulate), since that discovery already consumed one sample.
        self.chance_samples_drawn: Dict[int, int] = {}
        # This node's own _state_key, cached at creation for any node that
        # might ever be compared as a chance-node dedup candidate (see
        # _dpw_chance_child) -- avoids recomputing every sibling's key on
        # every widen attempt. None for nodes never used that way
        # (deterministic children, root).
        self.key: Optional[tuple] = None
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


def _dpw_chance_child(node: _Node, idx: int, action, bc, crn_base: Optional[int]) -> _Node:
    """Shared DPW widen-or-revisit step for a chance-node action index,
    used for BOTH END_TURN and any other action idx _simulate discovers
    consumes RNG (see _simulate's own docstring). node.N[idx] is this
    chance node's own total visit count (incremented at the bottom of
    _simulate, same as every other index) -- used for the widening cap,
    not node.visit_count, which counts visits to `node` itself rather than
    to this specific chance-producing action."""
    siblings = node.chance_children.setdefault(idx, [])
    n = int(node.N[idx])
    widening_cap = math.ceil(WC_CHANCE * (n + 1) ** WA_CHANCE)
    if len(siblings) < widening_cap:
        # local_sample_index: a widen-ATTEMPT counter, distinct from
        # len(siblings) (distinct outcomes) -- a widen attempt that turns
        # out to duplicate an existing outcome must still not reuse the
        # same CRN seed offset next time, or it would keep resampling the
        # identical duplicate forever instead of ever advancing to a new
        # draw.
        local_sample_index = node.chance_samples_drawn.get(idx, 0)
        node.chance_samples_drawn[idx] = local_sample_index + 1
        if crn_base is not None:
            bc.seed_rng(_crn_seed(crn_base, _state_key(bc), local_sample_index))
        else:
            bc.decorrelate_rng()
        sample = sts.BattleContext(bc)
        action.execute(sample)
        sample_key = _state_key(sample)
        # Dedup against EXISTING siblings before creating a new node -- a
        # widen attempt can land on a state we already have (a low-entropy
        # transition, e.g. a two-move enemy with few real branches), and
        # without this check that outcome's visits would split across
        # duplicate nodes instead of pooling into one (mirrors Silver
        # Automaton's BattleSearcher.cpp chanceSiblingReuse handling).
        # Safe: this only ever compares outcomes of the SAME action on the
        # SAME parent bc (just different RNG), so a key collision can only
        # mean the two rolls genuinely agreed on every tracked field.
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
    # Capped: revisit an existing sampled outcome, weighted by its own
    # visit count -- the chance-node analogue of PUCT concentrating visits
    # on promising options, per Couetoux et al.
    weights = np.array([c.visit_count + 1 for c in siblings], dtype=np.float64)
    return siblings[int(np.random.choice(len(siblings), p=weights / weights.sum()))]


def _simulate(node: _Node, policy: ActionScoringPolicy, trans_table: Dict[tuple, "_Node"],
              relic_idxs: torch.Tensor, relic_mask: torch.Tensor,
              depth: int = 0, call_depth: int = 0,
              crn_base: Optional[int] = None) -> float:
    """depth: how many real END_TURN transitions deep this call is, starting
    from choose_action's root at 0 -- only ever incremented crossing an
    END_TURN, used solely to decide when a freshly-reached node gets
    _heuristic_playout instead of _expand (see ROLLOUT_DEPTH_TURNS).
    Deliberately NOT incremented for a mid-turn action that turns out to be
    stochastic (see below) -- depth tracks real TURN boundaries, matching
    ROLLOUT_DEPTH_TURNS' existing calibration, not "any chance event".
    Unlike the old turns_left, this never bounds DPW's own recursion -- it
    only changes HOW a new leaf gets evaluated.

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
    the default bc.decorrelate_rng(). Keying off the node's own state means
    two separate search runs only need to revisit an EQUIVALENT state to
    get matched randomness there, not to have taken identical paths to
    reach it. None (the default) preserves the original independently-
    decorrelated behavior exactly.

    Action classification (chance vs deterministic): END_TURN is always a
    chance node (unchanged). For any OTHER action, action_type alone does
    NOT reliably say whether it's deterministic -- some CARD actions
    (shuffle-into-draw-pile effects: Wild Strike, Reckless Charge, similar)
    consume RNG internally too. Confirmed the hard way: two search-
    improvement attempts (a DPW-dedup merge, a tree-reuse reroot) that both
    assumed action_type-based classification each hit a real crash (an
    Action.cpp assertion, then a segfault), traced by direct evidence to
    exactly this -- two states matching on every OTHER tracked field
    differed only in draw-pile order after executing a "deterministic"
    action. The fix: on the FIRST visit to any non-END_TURN action index,
    execute it once and compare bc.rng_counter_sum() before/after: if it
    moved, this index gets treated as a chance node (chance_children, DPW,
    from here on) exactly like END_TURN; if not, as a genuinely
    deterministic one (children, transposition-eligible). Once an index is
    classified (present in either children or chance_children), later
    visits skip the probe and go straight to the appropriate path."""
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

    if action.action_type == sts.ActionType.END_TURN or idx in node.chance_children:
        child = _dpw_chance_child(node, idx, action, bc, crn_base)
        r = _dense_reward(bc, child.bc)
        if child.is_terminal:
            value = r + child.terminal_value
        else:
            value = r + GAMMA * _simulate(child, policy, trans_table, relic_idxs, relic_mask,
                                            depth + 1, call_depth + 1, crn_base)
    elif idx in node.children:
        child = node.children[idx]
        r = _dense_reward(bc, child.bc)
        value = r + GAMMA * _simulate(child, policy, trans_table, relic_idxs, relic_mask,
                                        depth, call_depth + 1, crn_base)
    else:
        # First visit to this (node, idx) pair, action isn't END_TURN --
        # probe by executing once and checking bc.rng_counter_sum() to
        # find out whether THIS action, from THIS state, actually consumes
        # randomness (see this function's own docstring on why action_type
        # alone can't answer that).
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
            # Transposition lookup -- a different card-play ORDER within
            # this same turn can land on an identical deterministic state
            # (same hp/block/energy/statuses/hand/piles); reusing the
            # cached node means its accumulated N/W/P get shared instead
            # of re-explored from scratch.
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
    for _ in range(n_simulations):
        _simulate(root, policy, trans_table, relic_idxs, relic_mask, crn_base=crn_base)
    best_idx = int(np.argmax(root.N))
    return root.actions[best_idx], root.N


def run_episode_with_search(env, policy: ActionScoringPolicy, n_simulations: int = 50, seed=None):
    """Same shape as train.py's run_episode, but each decision comes from
    choose_action's PUCT search instead of a blind policy.act() forward
    pass. Drives env.step() (not a bare bc.execute loop) so done/outcome/
    reward bookkeeping stays identical to every other eval path -- the
    action returned by choose_action(env.bc, ...) is valid against env.bc
    itself even though the search built its tree on an internal COPY
    (sts.BattleContext(bc) inside choose_action), since search::Action is a
    position-indexed instruction (source_idx/target_idx into hand/monsters),
    not a reference tied to one specific BattleContext instance -- the same
    assumption _simulate's own child_bc copies already rely on."""
    obs = env.reset(seed=seed)
    done = False
    info = None
    while not done:
        relic_idxs = torch.as_tensor(obs["relic_idxs"], dtype=torch.long)
        relic_mask = torch.as_tensor(obs["relic_mask"], dtype=torch.bool)
        action, _visits = choose_action(env.bc, policy, relic_idxs, relic_mask, n_simulations=n_simulations)
        obs, reward, done, info = env.step(action)
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
