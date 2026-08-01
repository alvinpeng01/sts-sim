"""Train a leaf value net on the widened features, and measure whether it pays.

The leaf estimator is the evaluation half of the search, and the search spends
~90% of its time there running a full rollout. Replacing that rollout with a
static estimate is the backgammon shape -- shallow simulation, strong evaluator --
and it was tried and refuted here: `leaf_eval_mode=value` measured
**-19.14 +/- 1.16 HP** against a rollout at matched simulations and **-14.96** at
4x the simulations.

That refutation was against a **deck-blind** ten-feature vector. `nativeLeafFeatures`
is now 30 and can see hand composition, pile composition, worst incoming hit and
the debuffs that rescale damage. On 1,494 states with 6-rollout-averaged targets
the linear held-out R^2 goes 0.562 -> 0.611, so the new features carry signal --
modestly. This trains the nonlinear estimator the socket already exists for
(`load_value_net`, `leaf_eval_mode=valuenet`) and puts it through the only test
that decides anything.

**Pre-registered gate.** Ship only if the net beats a rollout on the human
benchmark at MATCHED WALL CLOCK, measured at k=3 seeds per fight. Accuracy is not
the deliverable: a net is evaluated at every leaf, so it buys down the simulation
count directly, and 5 points of R^2 on a linear fit does not predict recovering a
19 HP deficit. Expect this to fail; run it because the one identified cause of the
previous failure has been addressed and nothing else has.

Targets are the mean of `--playouts` heuristic playouts per state. A single
playout is a hopeless label -- the per-fight sd of the search's own outcome is
10.82 HP (see docs/04-evaluation.md) -- and averaging is what makes the label
worth fitting.

    python -m lightspeed._leaf_value_net --collect runs/leaf_value_data.json
    python -m lightspeed._leaf_value_net --data runs/leaf_value_data.json --out runs/leaf_value_net.json
    python -m lightspeed._leaf_value_net --net runs/leaf_value_net.json --evaluate
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import numpy as np
import torch
import torch.nn as nn

import slaythespire as sts

from ._human_deck_combat import build_battle, play
from .paths import HUMAN_BENCHMARK
from .search_config import DEFAULT_SEARCH_CONFIG_PATH, ensure_search_config

DIM = 30


def load_fights(split: str, limit: int):
    with open(HUMAN_BENCHMARK, encoding="utf-8") as handle:
        records = json.load(handle)
    fights = [r for r in records if r["split"] == split]
    return fights[:limit] if limit else fights


def collect(split: str, limit: int, sims: int, playouts: int, every: int) -> dict:
    """States from real searches, labelled with averaged playout value."""
    ensure_search_config(DEFAULT_SEARCH_CONFIG_PATH)
    fights = load_fights(split, limit)
    features, targets = [], []
    start = time.time()
    for index, record in enumerate(fights):
        battle, _ = build_battle(
            record["deck"], record["relics"], record["cur_hp"], record["max_hp"],
            getattr(sts.MonsterEncounter, record["encounter"]),
            20, record["act"], record.get("potions", ()))
        for step in range(80):
            if battle.outcome != sts.BattleOutcome.UNDECIDED:
                break
            actions = battle.get_legal_actions()
            if not actions:
                break
            if step % every == 0:
                features.append(list(battle.leaf_features()))
                targets.append(statistics.mean(
                    battle.heuristic_playout() for _ in range(playouts)))
            action, _ = sts.run_mcts_search(battle, sims, None,
                                            index * 977 + step)
            action.execute(battle)
        if (index + 1) % 100 == 0:
            print(f"  {index+1}/{len(fights)} fights, {len(features)} states, "
                  f"{time.time()-start:.0f}s", flush=True)
    return {"x": features, "y": targets,
            "meta": {"split": split, "sims": sims, "playouts": playouts}}


class ValueNet(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        layers, prev = [], DIM
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.Tanh()]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(payload, hidden, epochs: int, out_path: str) -> float:
    x = torch.tensor(np.asarray(payload["x"], dtype=np.float64))
    y = torch.tensor(np.asarray(payload["y"], dtype=np.float64))
    split = int(len(x) * 0.75)
    xtr, xva, ytr, yva = x[:split], x[split:], y[:split], y[split:]
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
    # Target normalisation is folded into the OUTPUT layer below, because the
    # engine applies only (x - mu)/sd on the way in and expects the raw leaf
    # value on the way out -- no post-scaling hook exists.
    ymu, ysd = ytr.mean(), ytr.std().clamp_min(1e-6)

    model = ValueNet(hidden).double()
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    best, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        loss = nn.functional.mse_loss(model((xtr - mu) / sd), (ytr - ymu) / ysd)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        model.eval()
        with torch.no_grad():
            pred = model((xva - mu) / sd) * ysd + ymu
            mse = float(((pred - yva) ** 2).mean())
        if mse < best:
            best, best_state = mse, {k: v.clone()
                                     for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    with torch.no_grad():
        pred = model((xva - mu) / sd) * ysd + ymu
        r2 = 1 - float(((pred - yva) ** 2).sum()
                       / ((yva - yva.mean()) ** 2).sum())

    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    layers = []
    for i, linear in enumerate(linears):
        weight = linear.weight.detach().clone()
        bias = linear.bias.detach().clone()
        if i == len(linears) - 1:
            # Fold de-normalisation into the last layer: out = W*h*ysd + b*ysd + ymu.
            weight, bias = weight * ysd, bias * ysd + ymu
        layers.append({"W": weight.tolist(), "b": bias.tolist(),
                       "activation": "tanh" if i < len(linears) - 1 else "none"})
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"input_mu": mu.tolist(), "input_sd": sd.tolist(),
                   "layers": layers}, handle)
    print(f"held-out R^2 {r2:.3f}, hidden {hidden} -> {out_path}")
    return r2


def evaluate(net_path: str, split: str, limit: int, sims: int, seeds: int):
    """Matched wall clock, k seeds per fight. The only test that ships anything."""
    fights = load_fights(split, limit)
    with open(net_path, encoding="utf-8") as handle:
        net = json.load(handle)

    def run(n_sims: int) -> list[float]:
        out = []
        for index, record in enumerate(fights):
            per_seed = []
            for seed in range(seeds):
                battle, _ = build_battle(
                    record["deck"], record["relics"], record["cur_hp"],
                    record["max_hp"],
                    getattr(sts.MonsterEncounter, record["encounter"]),
                    20, record["act"], record.get("potions", ()))
                damage, outcome = play(battle, n_sims, index + seed * 1_000_003)
                died = outcome != sts.BattleOutcome.PLAYER_VICTORY
                per_seed.append(record["human_damage"]
                                - (record["cur_hp"] if died else damage))
            out.append(statistics.mean(per_seed))
        return out

    ensure_search_config(DEFAULT_SEARCH_CONFIG_PATH)
    sts.set_leaf_eval_mode("rollout")
    start = time.time()
    baseline = run(sims)
    rollout_seconds = time.time() - start
    print(f"  rollout   {statistics.mean(baseline):+8.3f}  {sims:5d} sims  "
          f"{rollout_seconds:6.1f}s", flush=True)

    sts.load_value_net(net)
    sts.set_leaf_eval_mode("valuenet")
    probe_n = max(10, len(fights) // 10)
    probe_start = time.time()
    for index, record in enumerate(fights[:probe_n]):
        battle, _ = build_battle(
            record["deck"], record["relics"], record["cur_hp"],
            record["max_hp"],
            getattr(sts.MonsterEncounter, record["encounter"]),
            20, record["act"], record.get("potions", ()))
        play(battle, sims, index)
    cost = ((time.time() - probe_start) / probe_n) / (
        rollout_seconds / (len(fights) * seeds))
    matched = max(1, int(round(sims / cost)))
    print(f"  net costs {cost:.2f}x a rollout -> matched budget {matched} sims",
          flush=True)

    values = run(matched)
    deltas = [b - a for a, b in zip(baseline, values)]
    mean = statistics.mean(deltas)
    stderr = statistics.stdev(deltas) / math.sqrt(len(deltas))
    print(f"  valuenet  {statistics.mean(values):+8.3f}  {matched:5d} sims  "
          f"vs rollout {mean:+.2f} +/- {stderr:.2f} (t={mean/stderr:+.2f})")
    print("\nShip only if this is positive and clears its standard error. The "
          "pre-registered expectation is that it does not.")
    sts.set_leaf_eval_mode("rollout")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", metavar="OUT")
    parser.add_argument("--data")
    parser.add_argument("--out", default="runs/leaf_value_net.json")
    parser.add_argument("--net")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--playouts", type=int, default=6)
    parser.add_argument("--every", type=int, default=4)
    parser.add_argument("--hidden", default="32,32")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    if args.collect:
        payload = collect(args.split, args.limit, args.sims, args.playouts,
                          args.every)
        with open(args.collect, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        print(f"{len(payload['x'])} states -> {args.collect}")
        return
    if args.data:
        with open(args.data, encoding="utf-8") as handle:
            payload = json.load(handle)
        hidden = [int(h) for h in args.hidden.split(",") if h.strip()]
        train(payload, hidden, args.epochs, args.out)
        return
    if args.evaluate and args.net:
        evaluate(args.net, "val", args.limit, args.sims, args.seeds)
        return
    parser.error("pass --collect, --data or --net with --evaluate")


if __name__ == "__main__":
    main()
