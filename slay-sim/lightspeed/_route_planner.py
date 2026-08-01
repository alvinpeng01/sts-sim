"""Survival-weighted route planning over the act map, as a drop-in for the policy's
map decision.

The overworld is the binding constraint -- the 2026-07-31 layer swap put +15.71
floors in the run policy with combat held byte-identical -- and routing is the
decision class where v31 is measurably inverted: ELITE logit **-2.55** against
Baalorlord's +1.93 and Silverbot's +0.22, and hp_frac x REST **+1.19** where both
references are around -1.8. Learning has not fixed either: 10x the labels moved
Act 2 survival and left Act 1 routing identical (42/189 elites captured, both
versions), and cloning the human's preference outright killed the policy at floor
7.08 because elite-taking is capability-dependent.

This takes the other route. An act map is a DAG of ~54 nodes with <=3 successors
each, which is a *small planning problem*, not a learning problem -- the shape the
orienteering literature calls stochastic orienteering with survival constraints
(maximize collected value subject to surviving the traversal). The one thing that
formulation needs, and that a capability-independent human demonstration could
never supply, is *our own* probability of surviving a fight. v31's
`next_combat_survival` auxiliary head supplies exactly that, measured on
2026-08-01 at **AUC 0.817** over 4,617 on-policy decisions, and it is conditioned
on our combat rather than a human's.

The value of a node is then its room reward discounted by the probability of
still being alive when reached, and the value of a choice is the best path from
it -- one backward pass over the DAG in y order:

    V(n) = room_value(n) + p_fight(n) * max over successors s of V(s)

with p_fight = P(survive a fight) at MONSTER/ELITE nodes and 1 elsewhere. That is
the orienteering objective computed exactly, not approximated, because the graph
is tiny.

**Two approximations, both deliberate and both worth knowing before reading any
result.** The survival head predicts P(surviving the NEXT combat) from the CURRENT
state; it cannot be queried at a hypothetical node three floors ahead without
simulating to get there. So one probability is applied homogeneously to every
fight on the route -- routes are discriminated by how many fights they carry and
where the value sits, not by which fights are harder. And the head is
under-dispersed (predicted sd 0.185 vs actual 0.313 on `next_combat_hp`; survival
is the better-spread one at sd 1.84 in logit space), which compresses differences
between routes.

Room weights are parameters rather than learned. They are the two coefficients the
audit says are wrong plus reference values for the ones it says are already
right -- SHOP and EVENT agree across v31, the human and Silverbot, so the planner
deliberately does not disturb them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import slaythespire as sts


@dataclass(frozen=True)
class RouteWeights:
    """Room values, with MONSTER as the unit of account.

    `rest` is multiplied by (1 - hp_frac), which is the whole point of it being
    here: v31's hp_frac x REST coefficient is +1.19 where the human's is -1.93 and
    Silverbot's -1.72, i.e. it rests LESS when hurt. Making rest value scale with
    missing HP encodes the correct sign structurally instead of hoping a net
    learns it.
    """
    elite: float = 3.0
    rest: float = 4.0
    shop: float = 0.8
    event: float = 0.6
    monster: float = 1.0
    treasure: float = 1.0


_ROOM_ELITE = int(sts.Room.ELITE)
_ROOM_REST = int(sts.Room.REST)
_ROOM_SHOP = int(sts.Room.SHOP)
_ROOM_EVENT = int(sts.Room.EVENT)
_ROOM_MONSTER = int(sts.Room.MONSTER)
_ROOM_TREASURE = int(sts.Room.TREASURE)


def build_graph(map_rep):
    """(xs, ys, rooms, successors) for the current act map.

    Successor construction mirrors `whole_run_env.map_route_features` exactly --
    an edge value of -1 means "no outgoing edge", and edges point from row y into
    row y+1 -- so the planner and the policy's own route features can never
    disagree about the shape of the graph they are reading.
    """
    xs = [int(v) for v in map_rep.xs]
    ys = [int(v) for v in map_rep.ys]
    rooms = [int(v) for v in map_rep.room_types]
    edges = np.asarray(map_rep.path_xs, dtype=np.int16).reshape((-1, 3))
    index = {(x, y): i for i, (x, y) in enumerate(zip(xs, ys))}
    successors: list[list[int]] = [[] for _ in xs]
    for i, (y, row) in enumerate(zip(ys, edges)):
        for edge_x in row:
            if int(edge_x) < 0:
                continue
            child = index.get((int(edge_x), y + 1))
            if child is not None:
                successors[i].append(child)
    return xs, ys, rooms, successors, index


def room_value(room: int, hp_frac: float, weights: RouteWeights) -> float:
    if room == _ROOM_ELITE:
        return weights.elite
    if room == _ROOM_REST:
        return weights.rest * max(0.0, 1.0 - hp_frac)
    if room == _ROOM_SHOP:
        return weights.shop
    if room == _ROOM_EVENT:
        return weights.event
    if room == _ROOM_MONSTER:
        return weights.monster
    if room == _ROOM_TREASURE:
        return weights.treasure
    return 0.0


def node_values(map_rep, hp_frac: float, survive_p: float,
                weights: RouteWeights) -> list[float]:
    """V(n) for every node, by one backward pass in descending y.

    Rows are processed deepest-first, so every successor is final before its
    parent is computed -- the map is a layered DAG with edges only from y to y+1,
    which makes a y-sort a valid topological order and needs no cycle handling.
    """
    xs, ys, rooms, successors, _ = build_graph(map_rep)
    values = [0.0] * len(xs)
    for i in sorted(range(len(xs)), key=lambda j: -ys[j]):
        best_child = max((values[s] for s in successors[i]), default=0.0)
        survives = survive_p if rooms[i] in (_ROOM_MONSTER, _ROOM_ELITE) else 1.0
        values[i] = room_value(rooms[i], hp_frac, weights) + survives * best_child
    return values


def survival_probability(auxiliary) -> float:
    """P(survive the next combat) from the auxiliary head, as a probability.

    The head is trained with binary_cross_entropy_with_logits, so its output is a
    LOGIT and must be squashed. Clamped away from the endpoints because it feeds a
    product over a whole route: a single saturated 0.0 would zero every downstream
    node and silently turn the planner into "avoid all fights".
    """
    logit = float(auxiliary["next_combat_survival"])
    probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
    return min(0.999, max(0.5, probability))


def score_actions(gc, map_rep, map_y: int, actions, auxiliary,
                  weights: RouteWeights = RouteWeights()) -> list[float]:
    """Planner value for each legal map action, in the order given.

    An action's `idx1` is the destination x on the next row (`map_y + 1`), the same
    decoding `whole_run_env.observation` uses for `action_target_rooms`. An action
    whose destination is not on the map scores -inf so it can never be chosen; that
    should not happen and a silent 0.0 would make it look merely unattractive.
    """
    hp_frac = float(gc.cur_hp) / max(1.0, float(gc.max_hp))
    values = node_values(map_rep, hp_frac, survival_probability(auxiliary), weights)
    _, _, _, _, index = build_graph(map_rep)
    target_y = int(map_y) + 1
    scores = []
    for action in actions:
        node = index.get((int(action.idx1), target_y))
        scores.append(values[node] if node is not None else -math.inf)
    return scores


def choose(gc, map_rep, map_y: int, actions, auxiliary,
           weights: RouteWeights = RouteWeights()) -> int:
    """Index of the highest-value legal map action.

    `map_y` is the CURRENT row (-1 before the first pick); it lives on the
    top-level NNRepresentation rather than on its `.map`, which is why it is
    threaded in rather than read off map_rep.
    """
    scores = score_actions(gc, map_rep, map_y, actions, auxiliary, weights)
    return max(range(len(scores)), key=lambda i: scores[i])
