"""Does Gumbel + sequential halving pick better labels than uniform allocation?

`label_state` spends its budget uniformly -- `rollouts` continuations for every
legal action -- then softmaxes the means.  Measured consequence: median paired
SNR 0.803 with two-thirds of labels below 1.0, i.e. on most decisions the gap
between the best and second-best action is smaller than the noise in measuring
it (06-experiment-log.md).

Gumbel AlphaZero (Danihelka et al.) attacks exactly this regime.  AlphaZero-style
targets carry no improvement guarantee at small simulation budgets; the Gumbel
construction -- sample Gumbel noise over the prior's logits, take the top-m
candidates, then run SEQUENTIAL HALVING over them -- minimizes simple regret and
keeps improving at budgets as small as 2-3 simulations.  This project already
runs sequential halving at the root of its *combat* search; it has never been
applied to overworld labels.

Method.  For each state, draw a reference pool of `--reference` continuation
scores per legal action.  The best action by reference mean is treated as
ground truth.  Every allocation strategy is then replayed against that pool by
sampling WITHOUT replacement, so all strategies see the same underlying score
distributions at the same budget and differ only in how they spend it.  This is
the standard way to compare bandit allocation rules, and it makes the comparison
affordable: the pool is drawn once and reused by every arm at every budget.

Reported metric is **simple regret** -- reference_mean[best] minus
reference_mean[chosen] -- which is what a label's argmax costs when it is wrong,
plus the plain rate of agreeing with the reference best.

Run from slay-sim/:
    python -m lightspeed._gumbel_label_probe --states 60 --workers 6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

DEFAULT_POLICY = "runs/whole_run_transformer_postfix-trunc_a20_v37.pt"

_STATE: dict = {}


def _worker_init(policy_path: str, sims: int, ascension: int,
                 rollout_decisions: int, temperature: float) -> None:
    from .eval_whole_run_policy import load_policy
    from .search_config import DEFAULT_SEARCH_CONFIG_PATH

    torch.set_num_threads(1)
    _STATE.update(
        policy=load_policy(policy_path, torch.device("cpu")), sims=sims,
        ascension=ascension, rollout_decisions=rollout_decisions,
        temperature=temperature, search_config=DEFAULT_SEARCH_CONFIG_PATH)


def _new_env():
    from .whole_run_env import RunConfig, WholeRunEnv

    env = WholeRunEnv(RunConfig(
        ascension=_STATE["ascension"], combat_sims=_STATE["sims"],
        deterministic_combat=True, search_config_path=_STATE["search_config"]))
    env._reset_combat_audit()
    return env


def _branch_score(gc, action_index: int, seed: int) -> float:
    """One continuation after forcing an action, scored like `rollout_score`."""
    from .generate_whole_run_rollouts import rollout_score

    env = _new_env()
    env.gc = gc.copy()
    env.search_seed_base = seed
    start_floor, start_act = int(env.gc.floor_num), int(env.gc.act)
    generator = torch.Generator().manual_seed(seed)
    policy = _STATE["policy"]
    actions = env.legal_actions()
    if action_index >= len(actions):
        return float("nan")
    _, _, done, _ = env.step(action_index)
    decisions = 0
    with torch.inference_mode():
        while (not done and env.gc.outcome.name == "UNDECIDED"
               and decisions < _STATE["rollout_decisions"]):
            obs = env.observation()
            logits, _ = policy(obs)
            probs = torch.softmax(
                logits / max(1e-6, _STATE["temperature"]), dim=-1)
            index = int(torch.multinomial(probs, 1, generator=generator))
            _, _, done, _ = env.step(index)
            decisions += 1
    return float(rollout_score(env, start_floor, start_act,
                               floor_weight=0.10, act_weight=1.50,
                               loss_penalty=0.40, boss_progress_weight=0.0))


def _collect_pool(job: dict) -> dict:
    """Play to a target decision, then draw the reference pool at that state."""
    policy = _STATE["policy"]
    env = _new_env()
    obs = env.reset(job["seed"])
    generator = torch.Generator().manual_seed(job["seed"] * 31 + 7)
    step = 0
    with torch.inference_mode():
        while (env.gc.outcome.name == "UNDECIDED"
               and env.steps < env.config.max_decisions and step < job["target_step"]):
            logits, _ = policy(obs)
            probs = torch.softmax(
                logits / max(1e-6, _STATE["temperature"]), dim=-1)
            index = int(torch.multinomial(probs, 1, generator=generator))
            obs, _, done, _ = env.step(index)
            step += 1
            if done:
                return {"seed": job["seed"], "ok": False}
        if env.gc.outcome.name != "UNDECIDED":
            return {"seed": job["seed"], "ok": False}
        actions = env.legal_actions()
        if len(actions) < 2:
            return {"seed": job["seed"], "ok": False}
        logits, _ = policy(env.observation())
        prior = (logits - logits.logsumexp(dim=-1)).numpy().astype(np.float64)

    pool = np.full((len(actions), job["reference"]), np.nan, dtype=np.float64)
    for action_index in range(len(actions)):
        for draw in range(job["reference"]):
            pool[action_index, draw] = _branch_score(
                env.gc, action_index, job["seed"] * 977 + draw * 13 + 1)
    return {"seed": job["seed"], "ok": True, "screen": int(env.gc.screen_state),
            "floor": int(env.gc.floor_num), "pool": pool.tolist(),
            "prior": prior.tolist()}


def uniform_choice(pool: np.ndarray, budget: int, rng) -> int:
    """Current estimator: equal draws per action, argmax of means."""
    actions, reference = pool.shape
    per_action = max(1, budget // actions)
    means = []
    for index in range(actions):
        draws = rng.choice(reference, size=min(per_action, reference),
                           replace=False)
        means.append(pool[index, draws].mean())
    return int(np.argmax(means))


def sequential_halving_choice(pool: np.ndarray, budget: int, rng,
                              candidates: list[int] | None = None) -> int:
    """Spend the budget in halving phases over the surviving candidates."""
    reference = pool.shape[1]
    alive = list(range(pool.shape[0])) if candidates is None else list(candidates)
    if len(alive) == 1:
        return alive[0]
    phases = max(1, math.ceil(math.log2(len(alive))))
    means = {index: (0.0, 0) for index in alive}
    while len(alive) > 1:
        per_action = max(1, budget // (len(alive) * phases))
        for index in alive:
            draws = rng.choice(reference, size=min(per_action, reference),
                               replace=False)
            total, count = means[index]
            means[index] = (total + pool[index, draws].sum(),
                            count + len(draws))
        alive.sort(key=lambda i: means[i][0] / max(1, means[i][1]), reverse=True)
        alive = alive[:max(1, len(alive) // 2)]
    return alive[0]


def gumbel_choice(pool: np.ndarray, prior: np.ndarray, budget: int, rng,
                  top_m: int) -> int:
    """Gumbel top-m over the prior, then sequential halving among them."""
    gumbel = rng.gumbel(size=len(prior))
    order = np.argsort(-(prior + gumbel))
    candidates = order[:min(top_m, len(prior))].tolist()
    return sequential_halving_choice(pool, budget, rng, candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--states", type=int, default=60)
    parser.add_argument("--seed-base", type=int, default=8_000_000)
    parser.add_argument("--reference", type=int, default=24,
                        help="continuations per action forming the reference pool")
    parser.add_argument("--budgets", default="8,16,32",
                        help="total continuations an estimator may spend")
    parser.add_argument("--top-m", type=int, default=2)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--rollout-decisions", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=1.05)
    parser.add_argument("--replicates", type=int, default=200,
                        help="resamples of each strategy per state")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default="runs/gumbel_label_probe.json")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    jobs = [{"seed": args.seed_base + index,
             "target_step": int(rng.integers(2, 30)),
             "reference": args.reference}
            for index in range(args.states)]
    print(f"{args.states} states x {args.reference} reference continuations "
          f"per action at {args.sims} sims, {args.workers} workers", flush=True)

    pools = []
    started = time.perf_counter()
    with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_worker_init,
            initargs=(args.policy, args.sims, args.ascension,
                      args.rollout_decisions, args.temperature)) as pool_executor:
        for done, row in enumerate(pool_executor.map(_collect_pool, jobs,
                                                     chunksize=1), start=1):
            if row["ok"]:
                pools.append(row)
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)} states "
                      f"({time.perf_counter() - started:.0f}s, "
                      f"{len(pools)} usable)", flush=True)

    print(f"\n{len(pools)} usable states "
          f"({time.perf_counter() - started:.0f}s)\n")

    # Gumbel top-m can only work if the prior's top-m usually CONTAINS the
    # best action -- that is the assumption the method rests on, and this
    # policy's priors are near-uniform (median top-1/top-2 gap 0.129 nats), so
    # it is worth checking rather than assuming.
    print(f"{'m':>3}  {'reference-best inside prior top-m':>34}")
    for m in (1, 2, 3):
        inside = []
        for row in pools:
            truth = np.asarray(row["pool"], dtype=np.float64).mean(axis=1)
            prior = np.asarray(row["prior"], dtype=np.float64)
            top = np.argsort(-prior)[:m]
            inside.append(float(int(np.argmax(truth)) in set(top.tolist())))
        print(f"{m:>3}  {np.mean(inside):>34.3f}")
    print(f"{'':>3}  {'(mean legal actions: ':>34}"
          f"{np.mean([len(r['prior']) for r in pools]):.2f})")

    budgets = [int(b) for b in args.budgets.split(",")]
    summary = {}
    replicate_rng = np.random.default_rng(1)
    print(f"{'budget':>7}  {'strategy':>22}  {'simple regret':>14}  "
          f"{'+/-':>7}  {'picks best':>11}")
    for budget in budgets:
        for name in ("uniform", "sequential-halving", "gumbel-top2+SH",
                     "gumbel-top3+SH"):
            regrets, hits = [], []
            for row in pools:
                pool_array = np.asarray(row["pool"], dtype=np.float64)
                prior = np.asarray(row["prior"], dtype=np.float64)
                truth = pool_array.mean(axis=1)
                best = int(np.argmax(truth))
                for _ in range(args.replicates):
                    if name == "uniform":
                        choice = uniform_choice(pool_array, budget, replicate_rng)
                    elif name == "sequential-halving":
                        choice = sequential_halving_choice(
                            pool_array, budget, replicate_rng)
                    else:
                        choice = gumbel_choice(
                            pool_array, prior, budget, replicate_rng,
                            2 if name.startswith("gumbel-top2") else 3)
                    regrets.append(truth[best] - truth[choice])
                    hits.append(float(choice == best))
            mean = float(np.mean(regrets))
            sem = float(np.std(regrets) / math.sqrt(len(pools)))
            summary[f"{budget}:{name}"] = {
                "regret": mean, "sem": sem, "hit_rate": float(np.mean(hits))}
            print(f"{budget:>7}  {name:>22}  {mean:>14.4f}  {sem:>7.4f}  "
                  f"{np.mean(hits):>11.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "states": len(pools),
                   "pools": pools, "config": vars(args)}, handle, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
