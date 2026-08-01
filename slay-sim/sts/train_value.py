"""Train the learned leaf-evaluation value net (see value_net.py) and
benchmark it inside expectimax against the hand-written eval.

Method (deliberately the simplest thing that captures banked-power value,
not the fanciest):
  1. DATA -- self-play many fights across the WHOLE roster with the greedy
     policy, recording every player-decision state and labelling each with
     that fight's terminal outcome (final HP on a win, a floor on a loss).
     This is the plain Monte-Carlo target V(s) = E[final HP | s]: a state
     where a buff was banked and then paid off ends the fight with more HP,
     so that state gets a higher label -- which is exactly how "1 Strength is
     worth X HP" gets regressed instead of hand-picked.
  2. TRAIN -- regress a small MLP (torch) onto those normalized targets.
  3. EXPORT -- dump weights to .npz for value_net.ValueNet's numpy forward.
  4. BENCH -- expectimax with the net vs expectimax with the hand eval, same
     seeds, on a fixed encounter sample; report win% / avg HP.

Known v1 limitation, noted not hidden: data comes from GREEDY play, so the
net learns "value under greedy," while search plays better than greedy --
a train/eval policy mismatch. The standard fix is to iterate (regenerate
data with search+net, retrain: approximate policy/value iteration), which is
the natural next pass if the v1 numbers justify it.

Run:  PYTHONPATH=. .venv/bin/python -m sts.train_value
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
from sts.encounters import (
    ACT1_BASIC, ACT1_ELITE, ACT1_BOSS, ACT2_BASIC, ACT2_ELITE, ACT2_BOSS,
    ACT3_BASIC, ACT3_ELITE, ACT3_BOSS,
)
from demo import greedy_policy

ALL_ENCS = (ACT1_BASIC + ACT1_ELITE + ACT1_BOSS + ACT2_BASIC + ACT2_ELITE
            + ACT2_BOSS + ACT3_BASIC + ACT3_ELITE + ACT3_BOSS)

LOSS_FLOOR = -0.5  # normalized target for any state on a losing trajectory


def play_and_record(enc_fn, player_hp, seed, states_out):
    """Greedy playthrough; append (encoded_state,) markers and return the
    fight's normalized terminal value so the caller can label them. Records
    at every player decision point for state diversity (mid-turn energy/hand
    variety, not just turn boundaries)."""
    rng = random.Random(seed)
    player = Player(max_hp=player_hp)
    combat = CombatState(player, enc_fn(), big_ironclad_deck(), rng=rng)
    my_states = []
    for _ in range(60):
        if combat.result() != Result.ONGOING:
            break
        combat.start_player_turn()
        while combat.result() == Result.ONGOING:
            my_states.append(encode_state(combat))
            action = greedy_policy(combat)
            if action[0] == "end":
                break
            combat.play_card(action[1], action[2])
        if combat.result() != Result.ONGOING:
            break
        combat.end_player_turn()
        combat.enemy_turn()

    result = combat.result()
    if result == Result.WIN:
        target = combat.player.hp / max(player_hp, 1)
    else:
        target = LOSS_FLOOR
    for s in my_states:
        states_out.append((s, target))


def generate_data(n_fights, seed0=0):
    data = []
    rng = random.Random(seed0)
    t0 = time.time()
    for i in range(n_fights):
        enc_fn = rng.choice(ALL_ENCS)
        player_hp = rng.choice([60, 80, 100, 120, 150])
        play_and_record(enc_fn, player_hp, seed=seed0 + i, states_out=data)
    print(f"  generated {len(data)} states from {n_fights} fights in {time.time()-t0:.1f}s")
    return data


def train(data, epochs=40, lr=1e-3, hidden=64, seed=0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    X = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)
    y = torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32).unsqueeze(1)

    net = nn.Sequential(
        nn.Linear(STATE_DIM, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1), nn.Tanh(),
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    n = len(X)
    idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        total = 0.0
        for start in range(0, n, 256):
            b = idx[start:start + 256]
            xb, yb = X[b], y[b]
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1:3d}: train MSE {total/n:.4f}")
    return net


def export(net, path):
    import torch  # noqa: F401
    layers = [m for m in net if hasattr(m, "weight")]
    W1 = layers[0].weight.detach().numpy().T.astype(np.float32)
    b1 = layers[0].bias.detach().numpy().astype(np.float32)
    W2 = layers[1].weight.detach().numpy().T.astype(np.float32)
    b2 = layers[1].bias.detach().numpy().astype(np.float32)
    W3 = layers[2].weight.detach().numpy().T.astype(np.float32)
    b3 = layers[2].bias.detach().numpy().astype(np.float32)
    np.savez(path, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
    print(f"  exported weights to {path}")


# --- benchmark -------------------------------------------------------------
random.seed(42)
BENCH_SAMPLE = random.sample(ALL_ENCS, 6) + [f for f in ACT1_BOSS if f.__name__ == "encounter_guardian"]
BENCH_HP = 150
BENCH_N = 10


def expectimax_policy(combat):
    return choose_action(combat, turns_left=1, draw_samples=1)[0]


def play_fight(policy, enc_fn, seed):
    rng = random.Random(seed)
    combat = CombatState(Player(max_hp=BENCH_HP), enc_fn(), big_ironclad_deck(), rng=rng)
    for _ in range(60):
        if combat.result() != Result.ONGOING:
            break
        combat.start_player_turn()
        for _ in range(20):
            if combat.result() != Result.ONGOING:
                break
            action = policy(combat)
            if action[0] == "end":
                break
            combat.play_card(action[1], action[2])
        if combat.result() != Result.ONGOING:
            break
        combat.end_player_turn()
        combat.enemy_turn()
    return combat


def bench(label):
    print(f"  --- {label} ---")
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
    print("[1] generating self-play data...")
    data = generate_data(n_fights=4000, seed0=0)

    print("[2] training value net...")
    net = train(data, epochs=40)
    export(net, "sts/value_net_weights.npz")

    print("[3] benchmark: expectimax with HAND eval (baseline)")
    set_value_net(None)
    bench("hand eval")

    print("[4] benchmark: expectimax with LEARNED eval")
    set_value_net(ValueNet.load("sts/value_net_weights.npz"))
    bench("learned eval")
