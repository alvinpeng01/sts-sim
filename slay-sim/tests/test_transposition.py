"""Correctness checks for the transposition table in search.py: it must be
completely transparent (same answer with or without it -- it's a pure
memoization of an already-deterministic computation, not an approximation),
plus targeted checks on the state-key function itself and a measured
hit-rate/speedup demonstration."""

import random
import time
from typing import Dict

from sts.combat import CombatState, Result
from sts.creatures import Player
from sts.cards import ironclad_starter_deck, varied_ironclad_deck, make_strike, make_defend
from sts.enemies import JawWorm, Cultist, Louse
from sts.encounters import encounter_louse_pair
from sts.search import (
    choose_action, _action_value, _distinct_actions, _state_key, _Cache,
)


class _NoReuseCache(_Cache):
    """Same interface as the real transposition cache, but .get() always
    reports a miss, forcing every node to be recomputed from scratch. Lets
    tests compute a "no transposition table" reference value using the
    exact same code path as the real (caching) one, so the only variable
    being tested is whether reuse changes the answer."""

    def get(self, key):
        self.misses += 1
        return None


def _reference_value(combat, action, turns_left, draw_samples):
    cache = _NoReuseCache()
    return _action_value(combat, action, turns_left, draw_samples, cache)


def _cached_value(combat, action, turns_left, draw_samples):
    cache = _Cache()
    return _action_value(combat, action, turns_left, draw_samples, cache)


def test_cache_is_transparent_single_enemy():
    rng = random.Random(3)
    player = Player(max_hp=45)
    combat = CombatState(player, [JawWorm()], ironclad_starter_deck(), rng=rng)
    combat.start_player_turn()
    for action in _distinct_actions(combat):
        ref = _reference_value(combat, action, 2, 3)
        cached = _cached_value(combat, action, 2, 3)
        assert abs(ref - cached) < 1e-9, f"cache changed the value for {action}"


def test_cache_is_transparent_multi_enemy():
    rng = random.Random(4)
    player = Player(max_hp=40)
    combat = CombatState(player, encounter_louse_pair(rng), varied_ironclad_deck(), rng=rng)
    combat.start_player_turn()
    for action in _distinct_actions(combat):
        ref = _reference_value(combat, action, 2, 2)
        cached = _cached_value(combat, action, 2, 2)
        assert abs(ref - cached) < 1e-9, f"cache changed the value for {action}"


def test_choose_action_matches_uncached_argmax():
    """End-to-end: the action/value choose_action returns (cache on) must
    match what an uncached search would pick."""
    rng = random.Random(5)
    player = Player(max_hp=40)
    combat = CombatState(player, [Cultist()], ironclad_starter_deck(), rng=rng)
    combat.start_player_turn()

    cached_action, cached_val = choose_action(combat, turns_left=2, draw_samples=3)

    best_ref_action, best_ref_val = None, float("-inf")
    for action in _distinct_actions(combat):
        val = _reference_value(combat, action, 2, 3)
        if val > best_ref_val:
            best_ref_val, best_ref_action = val, action
    assert abs(cached_val - best_ref_val) < 1e-9


def test_state_key_equal_for_states_reached_via_different_card_order():
    """The core premise the whole cache is built on: two different play
    orders of the same cards should converge to an identical key."""
    rng = random.Random(0)
    player = Player(max_hp=80)
    combat = CombatState(player, [JawWorm()], [make_strike(), make_defend()], rng=rng)
    combat.start_player_turn()
    combat.player.energy = 5

    # Each clone has its own monster instances (clone() deep-copies them);
    # a target captured before cloning is stale for either clone, so it must
    # be re-resolved against each clone's own monster list -- exactly what
    # search.py's _map_target does for real. Forgetting this is a classic
    # trap when hand-rolling test scenarios against a cloned CombatState.
    order_a = combat.clone()
    sa, da = (order_a.hand[0], order_a.hand[1])
    order_a.play_card(sa if sa.name == "Strike" else da, order_a.monsters[0])
    order_a.play_card(da if sa.name == "Strike" else sa, None)

    order_b = combat.clone()
    sb, db = (order_b.hand[0], order_b.hand[1])
    order_b.play_card(db if sb.name == "Strike" else sb, None)
    order_b.play_card(sb if sb.name == "Strike" else db, order_b.monsters[0])

    assert order_a.monsters[0].hp == order_b.monsters[0].hp == 42 - 6  # Jaw Worm HP is A20 (42)
    assert _state_key(order_a, 1) == _state_key(order_b, 1)


def test_state_key_differs_for_genuinely_different_states():
    rng = random.Random(0)
    player = Player(max_hp=80)
    combat = CombatState(player, [JawWorm()], [make_strike()], rng=rng)
    combat.start_player_turn()
    key_full_hp = _state_key(combat, 1)
    combat.player.hp -= 5
    key_damaged = _state_key(combat, 1)
    assert key_full_hp != key_damaged

    combat.player.hp += 5  # restore
    key_depth1 = _state_key(combat, 1)
    key_depth2 = _state_key(combat, 2)
    assert key_depth1 != key_depth2, "turns_left must be part of the key"


def test_cache_actually_gets_hits_and_speeds_up_deep_search():
    rng = random.Random(6)
    player = Player(max_hp=45)
    combat = CombatState(player, [JawWorm()], ironclad_starter_deck(), rng=rng)
    combat.start_player_turn()

    stats: Dict = {}
    t0 = time.perf_counter()
    action, val = choose_action(combat, turns_left=2, draw_samples=4, stats=stats)
    dt_cached = time.perf_counter() - t0

    assert stats["hits"] > 0, "expected at least some transposition hits at turns_left=2"

    t0 = time.perf_counter()
    best_ref_val = float("-inf")
    for a in _distinct_actions(combat):
        v = _reference_value(combat, a, 2, 4)
        best_ref_val = max(best_ref_val, v)
    dt_uncached = time.perf_counter() - t0

    assert abs(val - best_ref_val) < 1e-9
    assert dt_cached < dt_uncached, (
        f"expected caching to be faster: cached={dt_cached:.3f}s "
        f"uncached={dt_uncached:.3f}s"
    )


if __name__ == "__main__":
    import sys, traceback
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); passed += 1
            except Exception:
                failed += 1; print(f"FAIL {name}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
