"""Train the rollout-scoring net to imitate the search, and export it for the engine.

Companion to `collect_rollout_policy_data.py`; see that file for why this exists
and what the ceiling on it is. In short: at 1 simulation the rollout policy scores
-31.83 where 100 simulations score -1.28, so search is worth +30.55 HP over the
policy it rolls out with, and that is the gap being distilled.

The objective is a softmax over each decision's legal actions with the search's
pick as the target -- a ranking loss, not a regression, because
`nativePolicyNetScore`'s output is only ever compared between actions of the same
state. Absolute scale is irrelevant; `g_params.policyNetWeight` sets how loudly
the net speaks relative to the hand-tuned heuristic it is added to.

Two things this must respect to be usable at inference:

- The engine applies `(x - input_mu) / input_sd` before the first layer, so
  normalisation has to be exported rather than baked into the training data.
- Layers are dense with `activation` either "tanh" or anything else (treated as
  linear). No other op is available, so the architecture is fixed to that shape.

**Size it against the clock, not just the loss.** A loaded net is evaluated for
every legal action at every rollout step, so it directly buys down the simulation
count: measured at ~5x for a 32-unit hidden layer, which turns 100 sims into ~20
effective, and 20 sims scores 4.08 HP worse than 100. A net therefore has to
recover ~4 HP before it breaks even at all. Prefer the smallest hidden layer that
trains well.

    python -m lightspeed.train_rollout_policy_net --data runs/rollout_policy_data.pt \\
        --hidden 16 --out runs/rollout_policy_net.json
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn


class RankNet(nn.Module):
    """Dense stack ending in a scalar score per action."""

    def __init__(self, dim: int, hidden: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = dim
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.Tanh()]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def grouped_cross_entropy(scores, groups, chosen):
    """Softmax cross-entropy within each decision's own action set.

    Vectorised via scatter rather than looping over decisions. The obvious
    implementation -- slice each group and call cross_entropy -- is one Python
    iteration and one autograd node per decision, which at 42k decisions x 25
    epochs is ~1M calls and takes longer than collecting the data did.
    """
    n_groups = len(groups)
    group_of = torch.repeat_interleave(torch.arange(n_groups, device=scores.device),
                                       groups)
    offsets = torch.zeros(n_groups + 1, dtype=torch.long, device=scores.device)
    offsets[1:] = groups.cumsum(0)

    # Per-group log-sum-exp, shifted by the group max for stability.
    group_max = scores.new_full((n_groups,), float("-inf"))
    group_max = group_max.scatter_reduce(0, group_of, scores, reduce="amax")
    shifted = (scores - group_max[group_of]).exp()
    group_sum = scores.new_zeros(n_groups).index_add(0, group_of, shifted)
    logsumexp = group_max + group_sum.log()

    chosen_flat = offsets[:-1] + chosen
    loss = (logsumexp - scores[chosen_flat]).mean()

    # Top-1: a group's argmax is the position of its max within that group.
    best = scores.new_full((n_groups,), float("-inf")).scatter_reduce(
        0, group_of, scores, reduce="amax")
    is_best = scores == best[group_of]
    first_best = torch.zeros(n_groups, dtype=torch.long, device=scores.device)
    first_best.scatter_reduce_(0, group_of[is_best],
                               (torch.arange(len(scores), device=scores.device)
                                - offsets[:-1][group_of])[is_best],
                               reduce="amin", include_self=False)
    accuracy = (first_best == chosen).float().mean().item()
    return loss, accuracy


def export(model: RankNet, mu, sd, path: str) -> None:
    """Write the dict `load_policy_net` expects."""
    layers = []
    modules = [m for m in model.net if isinstance(m, nn.Linear)]
    for index, linear in enumerate(modules):
        # Every Linear except the last is followed by Tanh in __init__.
        layers.append({
            "W": linear.weight.detach().tolist(),
            "b": linear.bias.detach().tolist(),
            "activation": "tanh" if index < len(modules) - 1 else "none",
        })
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"input_mu": mu.tolist(), "input_sd": sd.tolist(),
                   "layers": layers}, handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--hidden", default="16",
                        help="comma-separated widths, e.g. 16 or 32,32")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-decisions", type=int, default=2048)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = torch.load(args.data, weights_only=False)
    x, groups, chosen = payload["x"], payload["groups"], payload["chosen"]
    print(f"{len(groups)} decisions, {x.shape[0]} actions, dim {x.shape[1]}")
    print(f"meta: {payload['meta']}")

    # Split by DECISION, and contiguously, so a decision's actions never straddle
    # the boundary and the validation set is a distinct stretch of play.
    n_val = max(1, int(len(groups) * args.val_frac))
    n_train = len(groups) - n_val
    train_actions = int(groups[:n_train].sum())
    xtr, xva = x[:train_actions], x[train_actions:]
    gtr, gva = groups[:n_train], groups[n_train:]
    ctr, cva = chosen[:n_train], chosen[n_train:]

    mu = xtr.mean(0)
    sd = xtr.std(0).clamp_min(1e-6)
    xtr_n, xva_n = (xtr - mu) / sd, (xva - mu) / sd

    hidden = [int(h) for h in args.hidden.split(",") if h.strip()]
    model = RankNet(x.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    params = sum(p.numel() for p in model.parameters())
    print(f"hidden {hidden}, {params} parameters")

    # The baseline any net must beat: how often does picking uniformly at random
    # land on the search's choice? Accuracy above this is the only evidence the
    # 18 features carry usable signal at all.
    baseline = float((1.0 / gva.float()).mean())
    print(f"random-pick accuracy on val: {baseline:.3f}")

    # Full-batch. The whole training set is ~240k x 18 floats, so a forward pass
    # is milliseconds and minibatching would only add index bookkeeping -- the
    # grouped loss needs each decision's actions kept together, which makes
    # shuffled minibatches meaningfully more fiddly than they are worth here.
    best_acc, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        loss, tacc = grouped_cross_entropy(model(xtr_n), gtr, ctr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vloss, vacc = grouped_cross_entropy(model(xva_n), gva, cva)
        if vacc > best_acc:
            best_acc = vacc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % max(1, args.epochs // 8) == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:4d}: train {loss:.4f}/{tacc:.3f}  "
                  f"val {vloss:.4f}/{vacc:.3f}  (best {best_acc:.3f})")

    model.load_state_dict(best_state)
    export(model, mu, sd, args.out)
    print(f"\nbest val top-1 {best_acc:.3f} vs {baseline:.3f} random "
          f"({best_acc/baseline:.2f}x)")
    print(f"wrote {args.out}")
    print("\nNext: load it and measure on the benchmark. Accuracy is NOT the "
          "deliverable -- the net costs ~5x search speed, so it has to recover "
          "~4 HP before it breaks even.")


if __name__ == "__main__":
    main()
