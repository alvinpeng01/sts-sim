"""Measure whether the routing policy does anything, and what it prefers.

Two diagnostics, both borrowed from silverbot-reference's map-representation
program (`EXPERIMENT_LOG.md`, 2026-06-04 and 2026-06-10):

1. `--randomize-paths` — replace every map path choice with a uniform random
   legal choice, keep the policy everywhere else. Silverbot measured its routing
   policy as causally worth +9.8pp win rate this way (0.794 -> 0.696). If our
   floors and act-3 reach do not drop, our routing contributes nothing.

2. Conditional logit over path decisions — per-room-type coefficients relative
   to MONSTER, plus an hp_frac x REST interaction. Silverbot's honest1 learned
   REST +1.51 (z=56), SHOP +1.34, EVENT +0.64, ELITE +0.22; its earlier
   "cheat-era" policy "never exceeded +/-0.1 nats (all n.s.)", i.e. no
   detectable routing preference at all. This says which of those we resemble.

The observation already carries `action_target_rooms`, so no engine change is
needed to know each option's destination room.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random

import numpy as np
import torch

import slaythespire as sts
from .eval_whole_run_policy import load_policy
from .generate_whole_run_rollouts import decision_type
from .whole_run_env import RunConfig, WholeRunEnv
from .search_config import DEFAULT_SEARCH_CONFIG_PATH


ROOM_NAMES = {int(getattr(sts.Room, n)): n for n in (
    "MONSTER", "ELITE", "REST", "SHOP", "EVENT", "TREASURE", "BOSS")}
REFERENCE = int(sts.Room.MONSTER)


def collect(policy, *, runs: int, seed_base: int, sims: int, ascension: int,
            randomize_paths: bool, rng: random.Random):
    """Play `runs` seeds, returning per-run outcomes and every path decision."""
    outcomes, decisions = [], []
    for offset in range(runs):
        env = WholeRunEnv(RunConfig(
            ascension=ascension, combat_sims=sims, deterministic_combat=True,
            search_config_path=DEFAULT_SEARCH_CONFIG_PATH))
        obs = env.reset(seed_base + offset)
        with torch.inference_mode():
            while (env.gc.outcome.name == "UNDECIDED"
                   and env.steps < env.config.max_decisions):
                is_map = decision_type(env.gc) == "map"
                action, _, _, _ = policy.act(obs, sample=False)
                if is_map:
                    rooms = [int(r) for r in obs["action_target_rooms"]]
                    legal = env.legal_actions()
                    if len(rooms) >= 2:
                        chosen = int(action) if int(action) < len(rooms) else 0
                        decisions.append({
                            "rooms": rooms,
                            "chosen": chosen,
                            "hp_frac": float(env.gc.cur_hp)
                            / max(1.0, float(env.gc.max_hp)),
                        })
                    if randomize_paths and legal:
                        action = rng.randrange(len(legal))
                obs, _, done, _ = env.step(action)
                if done:
                    break
        outcomes.append({
            "seed": seed_base + offset, "floor": int(env.gc.floor_num),
            "act": int(env.gc.act), "outcome": str(env.gc.outcome)})
    return outcomes, decisions


def conditional_logit(decisions, steps: int = 4000, ridge: float = 1e-3):
    """Fit per-room utilities with MONSTER pinned at zero.

    utility_k = beta[room_k] + gamma * hp_frac * 1[room_k == REST]
    P(choose k) = softmax_k(utility). Standard errors from the inverse Hessian
    via the observed information, so coefficients come with z-scores rather than
    bare point estimates.
    """
    rooms_seen = sorted({r for d in decisions for r in d["rooms"]})
    free = [r for r in rooms_seen if r != REFERENCE]
    index = {r: i for i, r in enumerate(free)}
    beta = torch.zeros(len(free) + 1, dtype=torch.float64, requires_grad=True)
    rest = int(sts.Room.REST)

    rows = []
    for d in decisions:
        onehot = torch.zeros(len(d["rooms"]), len(free) + 1, dtype=torch.float64)
        for k, room in enumerate(d["rooms"]):
            if room in index:
                onehot[k, index[room]] = 1.0
            if room == rest:
                onehot[k, -1] = d["hp_frac"]
        rows.append((onehot, d["chosen"]))

    optimizer = torch.optim.LBFGS([beta], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        total = torch.zeros((), dtype=torch.float64)
        for onehot, chosen in rows:
            logits = onehot @ beta
            total = total - (logits[chosen] - torch.logsumexp(logits, 0))
        total = total + ridge * (beta ** 2).sum()
        total.backward()
        return total

    optimizer.step(closure)

    # Observed information -> standard errors.
    def negative_log_likelihood(b):
        total = torch.zeros((), dtype=torch.float64)
        for onehot, chosen in rows:
            logits = onehot @ b
            total = total - (logits[chosen] - torch.logsumexp(logits, 0))
        return total + ridge * (b ** 2).sum()

    hessian = torch.autograd.functional.hessian(
        negative_log_likelihood, beta.detach())
    try:
        cov = torch.linalg.inv(hessian)
        errors = torch.sqrt(torch.diag(cov).clamp_min(0))
    except Exception:
        errors = torch.full_like(beta.detach(), float("nan"))

    out = []
    for room, i in index.items():
        out.append((ROOM_NAMES.get(room, str(room)),
                    float(beta[i]), float(errors[i])))
    out.append(("hp_frac x REST", float(beta[-1]), float(errors[-1])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--seed-base", type=int, default=18_900_000)
    parser.add_argument("--sims", type=int, default=300)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument(
        "--intervention", action="store_true",
        help="also run with randomized path choices and report the paired delta")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = load_policy(args.checkpoint, torch.device("cpu"))

    base, decisions = collect(
        policy, runs=args.runs, seed_base=args.seed_base, sims=args.sims,
        ascension=args.ascension, randomize_paths=False,
        rng=random.Random(0))
    print(f"\npolicy routing: {len(decisions)} path decisions over {args.runs} runs")
    counts = collections.Counter(
        ROOM_NAMES.get(d["rooms"][d["chosen"]], "?") for d in decisions)
    offered = collections.Counter(
        ROOM_NAMES.get(r, "?") for d in decisions for r in d["rooms"])
    print("  chosen room mix:  " + " ".join(
        f"{k}={v}" for k, v in counts.most_common()))
    print("  offered room mix: " + " ".join(
        f"{k}={v}" for k, v in offered.most_common()))

    print("\n  conditional logit (MONSTER = 0, positive = preferred):")
    for name, coef, err in conditional_logit(decisions):
        z = coef / err if err and not math.isnan(err) and err > 0 else float("nan")
        print(f"    {name:16s} {coef:+7.3f} +/- {err:5.3f}   z={z:+6.1f}")

    if args.intervention:
        rand, _ = collect(
            policy, runs=args.runs, seed_base=args.seed_base, sims=args.sims,
            ascension=args.ascension, randomize_paths=True,
            rng=random.Random(12345))
        bf = {r["seed"]: r["floor"] for r in base}
        rf = {r["seed"]: r["floor"] for r in rand}
        seeds = sorted(set(bf) & set(rf))
        delta = [bf[s] - rf[s] for s in seeds]
        mean = sum(delta) / len(delta)
        sem = (np.std(delta, ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else 0.0
        a3b = sum(r["act"] >= 3 for r in base)
        a3r = sum(r["act"] >= 3 for r in rand)
        print(f"\n  ROUTING INTERVENTION over {len(seeds)} paired seeds")
        print(f"    policy paths   mean floor {sum(bf.values())/len(bf):.2f}  act3+ {a3b}")
        print(f"    random paths   mean floor {sum(rf.values())/len(rf):.2f}  act3+ {a3r}")
        print(f"    causal value of routing: {mean:+.2f} floors (sem {sem:.2f}, "
              f"t={mean/sem if sem else float('nan'):+.2f})")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"outcomes": base, "decisions": decisions}, handle)


if __name__ == "__main__":
    main()
