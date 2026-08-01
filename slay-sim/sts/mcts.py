"""Monte Carlo Tree Search over within-turn card sequencing.

search.py's expectimax is exhaustive: it expands every action at every node.
That's exact, but the benchmarks in the project notes show it compounds
multiplicatively with multi-target/multi-card branching -- a 4-enemy fight's
within-turn action tree is wide (every single-target card branches once per
living enemy), and turns_left=2 means exploring that whole wide tree twice,
nested.

MCTS fixes exactly that axis: instead of expanding every branch, it spends a
fixed simulation budget selectively via UCB1, favoring lines that look
promising or under-explored rather than exhausting all of them. It's
anytime -- more simulations converge toward the exhaustive answer, fewer
simulations still return *something* reasonable, instead of a hard cliff
between "cheap and shallow" and "exact but slow."

Deliberately NOT reimplemented here: the chance-node handling (enemy intent
rolls, hidden draw-pile order) that search.py already does exactly/well.
Every MCTS leaf -- whichever node "end turn" is chosen from, or a terminal
WIN/LOSS reached mid-turn -- is scored by calling straight into search.py's
existing _after_end_turn / evaluate, which already average over those chance
events (exactly, or via Monte Carlo determinization) and already benefit
from the transposition table. So MCTS only ever explores the deterministic
part of the tree: which sequence of card plays to make this turn. This is a
deliberate scope split, not a shortcut -- the exhaustive expectimax was never
the bottleneck on the chance-node axis (that was fixed separately), only on
the action-branching axis, which is what UCB1 selection targets here.

A non-leaf node's value, before it's been expanded deep enough to reach a
real leaf, is estimated with the same evaluate() heuristic search.py uses at
its depth cutoffs -- standard "replace the random rollout with a value
function" MCTS practice, and reuses a function that's already tuned and
tested rather than adding a second one.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .combat import CombatState, Result
from .search import evaluate, _distinct_actions, _map_target, _after_end_turn, _Cache

# Tuned to evaluate()'s natural score scale (roughly -1000..230, not [0,1]),
# not the [0,1]-normalized constant textbook UCB1 examples use.
EXPLORATION_CONSTANT = 40.0


class _Node:
    __slots__ = ("combat", "is_leaf", "leaf_value", "untried", "children", "visits", "total_value")

    def __init__(self, combat: CombatState, is_leaf: bool, leaf_value: Optional[float] = None):
        self.combat = combat
        self.is_leaf = is_leaf
        self.leaf_value = leaf_value
        self.untried: List[tuple] = [] if is_leaf else list(_distinct_actions(combat))
        self.children: List[Tuple[tuple, "_Node"]] = []
        self.visits = 0
        self.total_value = 0.0

    @property
    def mean(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0


def _ucb_select_child(node: _Node, exploration: float) -> Tuple[tuple, _Node]:
    log_n = math.log(node.visits)
    best_action, best_child, best_score = None, None, float("-inf")
    for action, child in node.children:
        score = child.mean + exploration * math.sqrt(log_n / child.visits)
        if score > best_score:
            best_score, best_action, best_child = score, action, child
    return best_action, best_child


def _expand(parent: _Node, action: tuple, leaf_turns_left: int, draw_samples: int) -> _Node:
    child_combat = parent.combat.clone()
    if action[0] == "end":
        # The entire chance-node computation (enemy intent, hidden draw
        # order) happens inside this one call, exactly/sampled as
        # search.py already does, with its own transposition table.
        value = _after_end_turn(child_combat, leaf_turns_left, draw_samples, _Cache())
        return _Node(child_combat, is_leaf=True, leaf_value=value)

    _, card, target = action
    child_combat.play_card(card, _map_target(parent.combat, child_combat, target))
    if child_combat.result() != Result.ONGOING:
        return _Node(child_combat, is_leaf=True, leaf_value=evaluate(child_combat))
    return _Node(child_combat, is_leaf=False)


def _simulate(root: _Node, leaf_turns_left: int, draw_samples: int, exploration: float) -> None:
    node = root
    path = [node]
    while not node.is_leaf and not node.untried:
        action, child = _ucb_select_child(node, exploration)
        node = child
        path.append(node)

    if not node.is_leaf:
        action = node.untried.pop()
        child = _expand(node, action, leaf_turns_left, draw_samples)
        node.children.append((action, child))
        node = child
        path.append(node)

    value = node.leaf_value if node.is_leaf else evaluate(node.combat)
    for n in path:
        n.visits += 1
        n.total_value += value


def mcts_choose_action(
    combat: CombatState,
    simulations: int = 400,
    leaf_turns_left: int = 1,
    draw_samples: int = 3,
    exploration: float = EXPLORATION_CONSTANT,
) -> Tuple[tuple, float]:
    """Best action for the player right now, found via a fixed simulation
    budget instead of exhaustive expansion.

    ``simulations`` is the anytime knob: more simulations converge toward
    what exhaustive expectimax (search.choose_action) would return; fewer
    are faster but noisier. ``leaf_turns_left``/``draw_samples`` mean exactly
    what they mean in search.choose_action, since they're passed straight
    into the same _after_end_turn call used there.

    Final action selection uses the most-visited child (standard MCTS
    practice -- visit count is less noisy than mean value early in search,
    since UCB1 already concentrates visits on the best-looking lines).
    """
    if combat.result() != Result.ONGOING:
        raise ValueError(f"combat is already decided: {combat.result()}")

    root = _Node(combat, is_leaf=False)
    for _ in range(simulations):
        _simulate(root, leaf_turns_left, draw_samples, exploration)

    best_action, best_child = max(root.children, key=lambda ac: ac[1].visits)
    return best_action, best_child.mean
