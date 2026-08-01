"""Trains a small MLP to imitate expectimax search's own per-decision visit-count
distribution over legal actions, for use as a blended term in the native rollout
heuristic (nativeScoreAction's g_params.policy_net_weight -- see slaythespire.cpp's
nativePolicyNetScore).

Deliberately NOT the big embedding-based ActionScoringPolicy (policy.py) that
distillation.py trains -- this reuses the same small, hand-picked feature
philosophy already validated for the leaf value net (leaf_features()/
nativeLeafFeatures, 10 raw scalars), concatenated with a matching small
per-action feature vector (action_features()/nativeActionFeatures, 8 raw
scalars) -- 18 inputs total, not hundreds of embedding dims. Small enough to
run natively in C++ at rollout speed (thousands of calls per search), which
the big transformer-style ActionScoringPolicy never could be without embedding
LibTorch in the C++ engine itself.

Training target: search's OWN visit-count distribution at each decision (the
same AlphaZero-standard target distillation.py's policy head trains against),
not the final win/loss outcome -- this is a policy-imitation net, not a value
net. g_valueNet (a separate, already-tried experiment) replaces a full rollout
with a static state estimate and measurably lost to rollout as a LEAF
evaluator (~53% vs 83% win rate held-out). This net never replaces the
rollout -- it's blended into the same per-action heuristic score
(nativeScoreAction) that already drives every rollout step, closer to a PUCT
prior's role than a leaf estimator's.

Run:  PYTHONPATH=".;../sts_lightspeed/build" python -m lightspeed.train_policy_net
"""

from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .search_config import apply_search_config, load_search_config

# Same roster tune_search_cma.py trains the hand-tuned heuristic weights against --
# using a different set here would mean this net learns to imitate search on a
# DIFFERENT distribution of fights than the heuristic it's being blended with was
# tuned on, which would make the blend's net effect harder to attribute to either piece.
ENCOUNTERS = ["TIME_EATER", "DONU_AND_DECA", "SPHERIC_GUARDIAN", "GREMLIN_NOB", "HEXAGHOST", "THE_GUARDIAN",
              "COLLECTOR", "CHAMP", "AUTOMATON", "CENTURION_AND_HEALER"]
SIMS = 150
N_EPISODES_PER_ENCOUNTER = 40
N_WORKERS = 10
HIDDEN_DIM = 32
EPOCHS = 30
LR = 1e-3
OUT_PATH = "lightspeed/policy_net_weights.json"


@dataclass
class PolicyDecision:
    state_features: np.ndarray  # (10,)
    action_features: np.ndarray  # (A, 8)
    pi_target: np.ndarray  # (A,), sums to 1


def _collect_episode(env, n_simulations: int, seed=None) -> List[PolicyDecision]:
    from . import expectimax_search as ex

    env.reset(seed=seed)
    decisions: List[PolicyDecision] = []
    done = False
    while not done:
        bc = env.bc
        legal = bc.get_legal_actions()
        action, visits = ex.choose_action_native(bc, n_simulations=n_simulations)
        if len(legal) > 1:  # a single-legal-action decision has nothing to imitate (target is degenerate)
            state_features = np.array(bc.leaf_features(), dtype=np.float32)
            action_features = np.array([bc.action_features(a) for a in legal], dtype=np.float32)
            pi_target = visits.astype(np.float32)
            pi_target = pi_target / max(pi_target.sum(), 1e-8)
            decisions.append(PolicyDecision(state_features, action_features, pi_target))
        _, _, done, _ = env.step(action)
    return decisions


_worker_envs = None


def _worker_init(env_kwargs_by_encounter: dict, search_config: dict) -> None:
    # MODULE-LEVEL function taking plain picklable data via initargs, not a nested closure --
    # Windows multiprocessing uses spawn (not fork), so each worker is a fresh interpreter that
    # pickles the initializer + its args to send over; a closure capturing local variables from
    # collect_dataset() is NOT picklable and would fail at pool-creation time. Same pattern
    # distillation.py's _distill_worker_init already uses for exactly this reason.
    global _worker_envs
    torch.set_num_threads(1)
    from .env import IroncladFightEnv
    _worker_envs = {name: IroncladFightEnv(**kwargs) for name, kwargs in env_kwargs_by_encounter.items()}
    apply_search_config(search_config)


def _worker_collect(args) -> list:
    enc_name, n_episodes, n_simulations, seed_offset = args
    env = _worker_envs[enc_name]
    all_decisions = []
    for i in range(n_episodes):
        for d in _collect_episode(env, n_simulations, seed=seed_offset + i):
            all_decisions.append((d.state_features, d.action_features, d.pi_target))
    return all_decisions


def collect_dataset() -> List[PolicyDecision]:
    import sys
    sys.path.insert(0, r"C:\Users\Alvin\grok\sts-project\sts_lightspeed\build")
    import slaythespire as sts
    from lightspeed.env import build_full_encounter_resources
    from lightspeed.cards import weighted_ironclad_deck

    tuned_config = load_search_config("lightspeed/tuned_search_params.json")

    encounter_resources = build_full_encounter_resources()
    env_kwargs_by_encounter = {
        name: dict(encounter=getattr(sts.MonsterEncounter, name),
                   encounter_resources=encounter_resources,
                   deck_generator=weighted_ironclad_deck)
        for name in ENCOUNTERS
    }

    tasks = [(name, N_EPISODES_PER_ENCOUNTER, SIMS, 0) for name in ENCOUNTERS]
    with mp.Pool(N_WORKERS, initializer=_worker_init,
                 initargs=(env_kwargs_by_encounter, tuned_config)) as pool:
        results = pool.map(_worker_collect, tasks)

    decisions = []
    for worker_decisions in results:
        for sf, af, pt in worker_decisions:
            decisions.append(PolicyDecision(sf, af, pt))
    return decisions


class PolicyScoreNet(nn.Module):
    """Scores ONE (state_features, action_features) pair -> a scalar. Called once per
    action in a decision; the caller stacks per-action scores and softmaxes them to get
    a distribution to train against pi_target. Mirrors the value net's small-MLP shape
    (tanh hidden layers, linear output) so the same export/native-forward-pass pattern
    (ValueNetLayer in slaythespire.cpp) can load either."""

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train(decisions: List[PolicyDecision]) -> tuple[PolicyScoreNet, np.ndarray, np.ndarray]:
    """Normalizes inputs (mean/std across every (state, action) pair seen, pooled --
    NOT per-decision), trains via cross-entropy against each decision's own pi_target,
    looping decisions one at a time (action counts vary per decision, and this net's
    forward pass is cheap enough that padding for batched tensor ops isn't worth the
    added bookkeeping at this data scale -- unlike distillation.py's big embedding net,
    which pads because its own forward pass is the expensive part)."""
    all_inputs = []
    for d in decisions:
        state_rep = np.tile(d.state_features, (d.action_features.shape[0], 1))
        all_inputs.append(np.concatenate([state_rep, d.action_features], axis=1))
    stacked = np.concatenate(all_inputs, axis=0)
    mu = stacked.mean(axis=0)
    sd = stacked.std(axis=0)
    sd[sd < 1e-6] = 1.0

    input_dim = stacked.shape[1]
    net = PolicyScoreNet(input_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    mu_t = torch.as_tensor(mu, dtype=torch.float32)
    sd_t = torch.as_tensor(sd, dtype=torch.float32)

    for epoch in range(EPOCHS):
        perm = np.random.permutation(len(decisions))
        total_loss = 0.0
        for i in perm:
            d = decisions[i]
            state_rep = np.tile(d.state_features, (d.action_features.shape[0], 1))
            x = np.concatenate([state_rep, d.action_features], axis=1)
            x_t = (torch.as_tensor(x, dtype=torch.float32) - mu_t) / sd_t
            logits = net(x_t)
            log_probs = F.log_softmax(logits, dim=-1)
            target = torch.as_tensor(d.pi_target, dtype=torch.float32)
            loss = -(target * log_probs).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"epoch {epoch+1}/{EPOCHS}: mean_loss={total_loss/len(decisions):.4f}", flush=True)

    return net, mu, sd


def export(net: PolicyScoreNet, mu: np.ndarray, sd: np.ndarray, path: str) -> None:
    layers = []
    linear_layers = [m for m in net.net if isinstance(m, nn.Linear)]
    for idx, layer in enumerate(linear_layers):
        is_last = idx == len(linear_layers) - 1
        layers.append({
            "W": layer.weight.detach().numpy().tolist(),
            "b": layer.bias.detach().numpy().tolist(),
            "activation": "linear" if is_last else "tanh",
        })
    out = {"input_mu": mu.tolist(), "input_sd": sd.tolist(), "layers": layers}
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"exported to {path}", flush=True)


def main():
    print(f"collecting dataset: {len(ENCOUNTERS)} encounters x {N_EPISODES_PER_ENCOUNTER} episodes, "
          f"sims={SIMS}", flush=True)
    decisions = collect_dataset()
    print(f"collected {len(decisions)} decisions", flush=True)
    net, mu, sd = train(decisions)
    export(net, mu, sd, OUT_PATH)


if __name__ == "__main__":
    main()
