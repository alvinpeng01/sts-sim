"""Expectimax fight solver.

StS enemies aren't an adversary to minimize against -- they follow a scripted
probability table (see ``Monster.intent_options`` in enemies.py), so the
correct target is the *expectation* over that table, not a worst case. That's
why this is expectimax, not minimax: max nodes for the player's choices,
chance nodes for the enemy's move roll and the player's next draw.

Within a single turn there's no randomness to branch over at all -- the
enemy's intent for *this* round is already fixed, and card effects are
deterministic -- so that portion is an exact search over play sequences and
targets. Two chance points sit between turns:

  1. the enemy's *next* intent roll -- small discrete distribution, enumerated
     exactly via ``intent_options()``.
  2. the player's next draw -- the draw-pile order is hidden information, so
     instead of searching it exactly we use Monte Carlo determinization:
     sample a handful of shuffles and average.

Because ``legal_actions()``/``play_card()`` are generic over whatever cards
and monsters are in play, this solver works unmodified against any deck or
encounter -- no coupling to the card/monster roster.
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, List, Optional, Tuple

from .combat import CombatState, Result
from .enemies import _sample_weighted
from .powers import Power
from .orbs import Orb
from .enemies import Intent
from .relics import Relic
from . import value_net

WIN_SCORE = 100.0
LOSS_SCORE = -1000.0
# Joint enemy-intent combinations beyond this are sampled instead of
# enumerated exactly (see _intent_branches) -- multi-enemy fights otherwise
# blow up combinatorially: 2 enemies with 3 options each is already 9, and a
# 13-card deck plus 2 stochastic enemies measured at ~120s for a SINGLE
# decision before this fix (vs ~0.1-0.3s for 1 enemy).
MAX_EXACT_INTENT_BRANCHES = 8
INTENT_SAMPLE_COUNT = 6


def evaluate(combat: CombatState) -> float:
    """Value of a state: used at terminal nodes and at search-depth cutoffs.

    Terminal states are always scored exactly (win => HP carried forward,
    loss => the loss floor) -- those are ground truth, never guessed. For
    NON-terminal cutoffs, if a learned value net is installed (see
    value_net.set_value_net) it's used instead of the hand-written term
    below; that's the whole point of the net -- the hand term has no power
    features, so banked buffs whose payoff lands past turns_left=1 are
    invisible to it. Falls back to the hand term when no net is loaded, so
    nothing that doesn't opt in changes behavior."""
    result = combat.result()
    if result == Result.WIN:
        return WIN_SCORE + combat.player.hp
    if result == Result.LOSS:
        return LOSS_SCORE
    if value_net.get_value_net() is not None:
        return value_net.learned_value(combat)
    p = combat.player
    enemy_hp = sum(m.hp for m in combat.living_monsters)
    return p.hp * 2.0 - enemy_hp * 0.5 + p.block * 0.5


class _Cache:
    """Transposition table: state_key -> computed value, with hit/miss
    counters as real attributes rather than sentinel entries stashed inside
    the same dict used for state keys. (An earlier version did exactly that
    with "_hits"/"_misses" string keys -- fragile, and broke the first time
    a second caller, mcts.py, passed in a plain {} without those sentinels
    pre-seeded.) Any caller can now safely construct one of these instead of
    having to know about the bookkeeping convention."""

    __slots__ = ("_table", "hits", "misses")

    def __init__(self):
        self._table: Dict = {}
        self.hits = 0
        self.misses = 0

    def get(self, key):
        val = self._table.get(key)
        if val is not None:
            self.hits += 1
        else:
            self.misses += 1
        return val

    def set(self, key, value) -> None:
        self._table[key] = value


_PLAIN_HASHABLE_TYPES = (int, str, bool, type(None))


def _hashable(value):
    """Recursively turn engine objects into hashable, structurally-equal
    keys for the transposition table below."""
    # Fast path: most Creature fields (hp, block, name, stance, Enums, ...)
    # are already plain/hashable, so check that first rather than falling
    # through dict/list/Power/Orb/Intent/Relic isinstance checks every time
    # -- measured via cProfile as 2.68M calls with 16M+ isinstance checks
    # total on a representative search workload, most of them failed checks
    # on plain values before reaching the `return value` fallback below.
    if type(value) in _PLAIN_HASHABLE_TYPES:
        return value
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, Power):
        return (type(value).__name__, value.amount)
    if isinstance(value, Orb):
        return value.name
    if isinstance(value, Intent):
        # Intent is a non-frozen dataclass -> unhashable by default.
        return (value.type, value.damage, value.name)
    if isinstance(value, Relic):
        # Without this case, relics would fall through to identity-based
        # hashing -- every clone has its own Relic instances (see
        # Player.clone), so no two states would ever compare equal once any
        # relic is present, silently collapsing the cache's hit rate to zero
        # instead of erroring. Relics with no mutable state (Vajra, Anchor,
        # ...) hash as a constant and cost nothing; counter/once-per-combat
        # relics (Nunchaku, Akabeko, ...) correctly fragment the cache by
        # their actual state, same tradeoff as Powers.
        return (type(value).__name__, value.counter, value.used)
    return value  # int, str, bool, None, Enum


def _creature_key(creature):
    """Generic over vars() rather than hand-picking fields, deliberately:
    every Monster subclass stashes its own AI state in different attributes
    (last_move, turn_count, asleep, mode, dmg_taken_this_turn, ...), and
    hand-picking risks silently missing one -- which would corrupt the cache
    by merging two states that actually have different futures. Reading
    everything in __dict__ costs a little more per hash but is correct by
    construction for any current or future monster/power."""
    return tuple(sorted((k, _hashable(v)) for k, v in vars(creature).items()))


def _pile_key(cards, ordered: bool):
    """Card objects are shared by reference and stateless, so two cards with
    the same name are interchangeable -- hash by name, not identity. Hand/
    discard/exhaust order never affects anything downstream, so those are
    hashed as an order-independent multiset; draw_pile order does matter
    (it's literally what gets drawn next), so it's kept as a sequence."""
    names = [c.name for c in cards]
    return tuple(names) if ordered else tuple(sorted(names))


def _state_key(combat: CombatState, turns_left: int):
    """Cache key = everything that can affect the value of `combat` looking
    `turns_left` further turns ahead. Deliberately excludes: combat.turn
    (unread by any value-relevant logic), combat._x_value (a transient
    scratch var only live during a card's own play() call, never read at a
    decision-point boundary), and combat.rng (the search doesn't depend on
    *which* random samples produced a cached estimate, only on the estimate
    itself -- reusing it is standard Monte Carlo practice, same as reusing
    any other previously-computed expectation).
    """
    return (
        turns_left,
        _creature_key(combat.player),
        tuple(_creature_key(m) for m in combat.monsters),
        _pile_key(combat.hand, ordered=False),
        _pile_key(combat.draw_pile, ordered=True),
        _pile_key(combat.discard_pile, ordered=False),
        _pile_key(combat.exhaust_pile, ordered=False),
        combat.double_tap_charges,
    )


def _distinct_actions(combat: CombatState):
    """De-duplicate functionally-identical actions -- e.g. two Strikes in
    hand targeting the same enemy resolve identically, so treating them as
    separate branches only wastes search budget."""
    seen = set()
    for action in combat.legal_actions():
        if action[0] == "end":
            key = ("end",)
        else:
            _, card, target = action
            key = ("play", card.name, id(target) if target is not None else None)
        if key in seen:
            continue
        seen.add(key)
        yield action


def _map_target(parent: CombatState, child: CombatState, target):
    """Monsters are deep-copied on clone, so a target from the parent's
    action list must be resolved to its counterpart in the child by index."""
    if target is None:
        return None
    idx = parent.monsters.index(target)
    return child.monsters[idx]


def _resolve_enemy_actions(combat: CombatState) -> None:
    """Deterministically execute each living monster's already-telegraphed
    intent for this round -- no branching, the intent was fixed before the
    player's turn started."""
    for m in combat.living_monsters:
        m.block = 0
        m.start_turn(combat)
        m.take_turn(combat)
        m.end_turn(combat)
        if combat.player.is_dead:
            return


def _intent_branches(monsters, rng: random.Random):
    """Joint distribution over every living monster's next intent. Exact
    (itertools.product) when the combination count is small; otherwise
    importance-sampled from the same per-monster distributions, with each
    sampled branch weighted 1/N (sampling already draws branches
    proportional to their true probability, so uniform reweighting is
    correct in expectation)."""
    per_monster = [m.intent_options() for m in monsters]
    exact_count = 1
    for opts in per_monster:
        exact_count *= len(opts)

    if exact_count <= MAX_EXACT_INTENT_BRANCHES:
        for combo in itertools.product(*per_monster):
            prob = 1.0
            moves = []
            for p, move in combo:
                prob *= p
                moves.append(move)
            if prob > 0:
                yield prob, moves
        return

    for _ in range(INTENT_SAMPLE_COUNT):
        moves = [_sample_weighted(opts, rng) for opts in per_monster]
        yield 1.0 / INTENT_SAMPLE_COUNT, moves


def _expected_next_turn(combat: CombatState, turns_left: int, draw_samples: int,
                         cache: "_Cache") -> float:
    """Monte Carlo determinization over the hidden draw-pile order."""
    if combat.result() != Result.ONGOING:
        return evaluate(combat)
    total = 0.0
    for _ in range(draw_samples):
        # No manual reseed needed here: CombatState.clone() itself now seeds
        # the child's rng from a fresh draw of the parent's stream, so each
        # of these draw_samples clones -- all taken from the same `combat`
        # -- already gets a decorrelated rng (parent.rng advances between
        # clone() calls). Re-reseeding on top of that was only necessary
        # back when clone() did an exact getstate/setstate copy (which would
        # otherwise give every sample here the identical stream).
        sample = combat.clone()
        sample.start_player_turn()
        total += _search(sample, turns_left - 1, draw_samples, cache)
    return total / draw_samples


def _after_end_turn(combat: CombatState, turns_left: int, draw_samples: int,
                     cache: "_Cache") -> float:
    combat.end_player_turn()
    _resolve_enemy_actions(combat)
    if combat.result() != Result.ONGOING:
        return evaluate(combat)

    living = combat.living_monsters
    total = 0.0
    for prob, moves in _intent_branches(living, combat.rng):
        branch = combat.clone()
        for m, move in zip(branch.living_monsters, moves):
            m.force_intent(move)
        total += prob * _expected_next_turn(branch, turns_left, draw_samples, cache)
    return total


def _action_value(parent: CombatState, action: tuple, turns_left: int,
                   draw_samples: int, cache: "_Cache") -> float:
    child = parent.clone()
    if action[0] == "end":
        return _after_end_turn(child, turns_left, draw_samples, cache)
    _, card, target = action
    child.play_card(card, _map_target(parent, child, target))
    return _search(child, turns_left, draw_samples, cache)


def _search(combat: CombatState, turns_left: int, draw_samples: int,
            cache: "_Cache") -> float:
    if combat.result() != Result.ONGOING or turns_left <= 0:
        return evaluate(combat)

    key = _state_key(combat, turns_left)
    cached = cache.get(key)
    if cached is not None:
        return cached

    val = max(
        _action_value(combat, a, turns_left, draw_samples, cache)
        for a in _distinct_actions(combat)
    )
    cache.set(key, val)
    return val


def choose_action(
    combat: CombatState, turns_left: int = 1, draw_samples: int = 6,
    stats: Optional[Dict] = None,
) -> Tuple[tuple, float]:
    """Best action for the player right now: ('play', card, target) or ('end',).

    ``turns_left`` counts player turns of lookahead. turns_left=1 fully
    solves the rest of *this* turn (arbitrarily deep -- unbounded by
    turns_left, since only ending a turn consumes budget), resolves the
    incoming enemy turn exactly, and peeks at the enemy's next intent and the
    player's next opening hand -- but doesn't optimize how that next turn is
    played. turns_left=2+ recurses into optimizing future turns too, at
    roughly turns_left-times the cost.

    ``draw_samples`` only matters once turns_left >= 2 (it's how many
    shuffles get averaged when planning *into* a future turn); at
    turns_left=1 the sampled hand never gets played, so draw_samples=1 is
    enough and cheaper.

    A transposition table (keyed on everything that affects a state's future
    value -- see _state_key) is shared across this whole call: different
    orderings of the same cards routinely land on identical resulting
    states, and re-deriving that subtree's value each time is exactly what
    made deep/multi-enemy searches blow up (a single decision measured at
    ~120s before this). The table is fresh per call, not persisted across
    turns -- see the project notes for why that's deliberate. Pass a dict to
    ``stats`` to get back {"hits": int, "misses": int} for introspection.
    """
    cache = _Cache()
    best_action: Optional[tuple] = None
    best_val = float("-inf")
    for action in _distinct_actions(combat):
        val = _action_value(combat, action, turns_left, draw_samples, cache)
        if val > best_val:
            best_val, best_action = val, action
    assert best_action is not None
    if stats is not None:
        stats["hits"] = cache.hits
        stats["misses"] = cache.misses
    return best_action, best_val
