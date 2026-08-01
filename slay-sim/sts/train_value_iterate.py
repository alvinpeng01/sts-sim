"""Policy/value iteration pass 1: fixes the Nemesis regression the v1 net
showed (72%->28% win vs the hand eval, confirmed at n=40 -- not noise).

Root cause: v1's training data came from GREEDY self-play (train_value.py),
and greedy plays Nemesis badly (a stochastic 45-dmg burst attacker), so the
net mostly saw losing Nemesis trajectories and learned an uninformative
"these states are doomed" signal there -- which then misled the search on
exactly that matchup, even though the net was a clear net win everywhere
else (Guardian 40%->100%, overall 85.7%->91.4% at n=10).

Fix: regenerate self-play data using EXPECTIMAX SEARCH WITH THE V1 NET
LOADED as its leaf eval, instead of greedy -- so the data now reflects
actually-competent Nemesis play, not greedy's. Retrain a v2 net on that.
This is standard approximate policy/value iteration: the policy used to
generate data is the (better) policy the previous value net enables, not a
fixed weak baseline.

Run:  PYTHONPATH=. .venv/bin/python -m sts.train_value_iterate
"""

from __future__ import annotations

import random
import sys
import time

import numpy as np

sys.path.insert(0, "/home/alvin/slay-sim")

from sts.combat import CombatState, Result
from sts.creatures import Player
from sts.cards import big_ironclad_deck
from sts.value_net import encode_state, ValueNet, set_value_net, STATE_DIM
from sts.search import choose_action
from sts.train_value import (
    ALL_ENCS, LOSS_FLOOR, train, export, BENCH_SAMPLE, BENCH_HP, BENCH_N,
    expectimax_policy, play_fight,
)

V1_WEIGHTS = "sts/value_net_weights.npz"
V2_WEIGHTS = "sts/value_net_weights_v2.npz"


def search_policy(combat):
    return choose_action(combat, turns_left=1, draw_samples=1)[0]


def play_and_record_search(enc_fn, player_hp, seed, states_out):
    """Same recording shape as train_value.play_and_record, but driven by
    expectimax search (with whatever net is currently active) instead of
    greedy -- the actual policy fix for this iteration."""
    rng = random.Random(seed)
    player = Player(max_hp=player_hp)
    combat = CombatState(player, enc_fn(), big_ironclad_deck(), rng=rng)
    my_states = []
    for _ in range(60):
        if combat.result() != Result.ONGOING:
            break
        combat.start_player_turn()
        for _ in range(20):
            if combat.result() != Result.ONGOING:
                break
            my_states.append(encode_state(combat))
            action = search_policy(combat)
            if action[0] == "end":
                break
            combat.play_card(action[1], action[2])
        if combat.result() != Result.ONGOING:
            break
        combat.end_player_turn()
        combat.enemy_turn()

    result = combat.result()
    target = combat.player.hp / max(player_hp, 1) if result == Result.WIN else LOSS_FLOOR
    for s in my_states:
        states_out.append((s, target))


def generate_data_search(n_fights, seed0=0):
    data = []
    rng = random.Random(seed0)
    t0 = time.time()
    for i in range(n_fights):
        enc_fn = rng.choice(ALL_ENCS)
        player_hp = rng.choice([60, 80, 100, 120, 150])
        play_and_record_search(enc_fn, player_hp, seed=seed0 + i, states_out=data)
        if (i + 1) % 250 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_fights} fights ({elapsed:.0f}s, {len(data)} states so far)")
    print(f"  generated {len(data)} states from {n_fights} fights in {time.time()-t0:.1f}s")
    return data


def bench_named(label, net_path=None):
    print(f"  --- {label} ---")
    set_value_net(ValueNet.load(net_path) if net_path else None)
    grand_w = grand_hp = grand_n = 0
    for enc_fn in BENCH_SAMPLE:
        wins = hp_total = 0
        for i in range(BENCH_N):
            c = play_fight(expectimax_policy, enc_fn, seed=1000 + i)
            if c.result() == Result.WIN:
                wins += 1
                hp_total += c.player.hp
        grand_w += wins
        grand_hp += hp_total
        grand_n += BENCH_N
        print(f"    {enc_fn.__name__:28s} win {wins/BENCH_N:5.1%}  avg HP {hp_total/max(wins,1):5.1f}")
    print(f"    {'OVERALL':28s} win {grand_w/grand_n:5.1%}  avg HP {grand_hp/max(grand_w,1):5.1f}")


if __name__ == "__main__":
    print("[1] loading v1 net as the data-generation policy's eval...")
    set_value_net(ValueNet.load(V1_WEIGHTS))

    print("[2] generating self-play data via SEARCH (not greedy)...")
    data = generate_data_search(n_fights=2500, seed0=10000)

    print("[3] training v2 value net on search-quality data...")
    net = train(data, epochs=40)
    export(net, V2_WEIGHTS)

    print("[4] benchmark: hand eval vs v1 (greedy-trained) vs v2 (search-trained)")
    bench_named("hand eval", None)
    bench_named("v1 net (greedy-trained)", V1_WEIGHTS)
    bench_named("v2 net (search-trained)", V2_WEIGHTS)
