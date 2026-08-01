"""AIVAT-style variance reduction for stochastic-policy run evaluation.

AIVAT (Burch et al., AAAI 2018) subtracts provably zero-mean control variates
from an outcome: one term per chance event, and one per decision of a player
whose strategy is known.  In poker it cut the standard deviation of a
man-machine match by 85%, needing 44x fewer games for the same conclusion.

**Where it does not apply here, and why that is worth writing down.**  This
project's standard evaluation plays a GREEDY policy with `deterministic_combat`,
which makes a run a deterministic function of its seed.  Both AIVAT terms are
then identically zero: there is no chance left to correct, and a deterministic
policy has no randomness to average over.  All the variance is seed-to-seed
heterogeneity, and paired seeds already remove what can be removed from it --
measured at 39-43% on `runs/sharp_rebaseline_600seeds.jsonl` (rho 0.39-0.43).
No amount of control-variate machinery improves a protocol that is already
deterministic.  The engine also exposes no RNG reseed, so nature cannot be
resampled in a fork even if we wanted the chance term.

**Where it does apply.**  Every stochastic-return estimate in the stack:
`ppo_collect` samples at T=0.2, `label_state`'s continuations sample at T=1.05,
and the vine estimates in `_advantage_estimators`.  Those carry real policy
randomness, and their noise is the project's documented binding constraint --
paired label SNR 0.803 with two-thirds of labels below 1.0.

The correction implemented here is AIVAT's decision term:

    corrected = outcome - sum_t [ V(s'_chosen) - sum_a pi(a) V(s'_a) ]

Each bracket has mean zero under pi, so the estimator is unbiased for ANY V --
V only has to correlate with the outcome to reduce variance, not be accurate.
V is a floor-predicting head fitted on `runs/run_value_data.pt`, since mean
final floor is the metric this project promotes on.

A decision is corrected only when every candidate's post-action state can be
reached without triggering a battle, because resolving a battle per candidate
costs more than the variance is worth.  Skipping terms costs reduction, never
unbiasedness.

The unbiasedness check is the point of the harness, not an afterthought: the
corrected mean must match the plain mean within noise while its spread shrinks.

Run from slay-sim/:
    python -m lightspeed._aivat_eval --runs 300 --temperature 0.2 --workers 6
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
from torch import nn

DEFAULT_POLICY = "runs/whole_run_transformer_v37_critic.pt"
FLOOR_CRITIC = "runs/floor_critic_v37.pt"
VALUE_DATA = "runs/run_value_data.pt"

_STATE: dict = {}


def fit_floor_critic(data_path: str, out_path: str, epochs: int = 25) -> float:
    """V(s) -> the final floor of the run that visited s.

    The control variate must predict the METRIC being estimated. `run_critic`
    predicts the env's shaped return, which is a different quantity, so this
    fits a head on the same cached features against final floor.
    """
    from .collect_run_value_data import stack
    from .train_value_from_harvest import fit_head, r2

    payload = torch.load(data_path, weights_only=False, map_location="cpu")
    episodes = payload["episodes"]
    for episode in episodes:
        episode = episode
    cut = int(len(episodes) * 0.8)

    def design(subset):
        features = np.concatenate([e["features"] for e in subset])
        floors = np.concatenate([
            np.full(len(e["returns"]), e["final_floor"], dtype=np.float32)
            for e in subset])
        scalars = np.concatenate([
            np.stack([e["floors"] / 56.0, e["acts"] / 4.0,
                      e["screens"] / 10.0], axis=1).astype(np.float32)
            for e in subset])
        return np.concatenate([features, scalars], axis=1), floors

    train_x, train_y = design(episodes[:cut])
    val_x, val_y = design(episodes[cut:])
    print(f"fitting floor critic on {len(train_y)} states "
          f"(target sd {train_y.std():.2f} floors)", flush=True)
    head = nn.Sequential(nn.Linear(train_x.shape[1], 96), nn.GELU(),
                         nn.Linear(96, 1))
    head, best = fit_head(head, train_x, train_y, val_x, val_y, epochs, 256,
                          1e-3, "floor")
    torch.save({"state_dict": head.state_dict(), "in_dim": train_x.shape[1],
                "val_r2": best}, out_path)
    print(f"floor critic val R2 {best:+.4f} -> {out_path}")
    return best


def load_floor_critic(path: str):
    payload = torch.load(path, weights_only=False, map_location="cpu")
    head = nn.Sequential(nn.Linear(payload["in_dim"], 96), nn.GELU(),
                         nn.Linear(96, 1))
    head.load_state_dict(payload["state_dict"])
    return head.eval()


def _worker_init(policy_path: str, floor_critic_path: str, sims: int,
                 ascension: int, temperature: float, correct: bool) -> None:
    from .eval_whole_run_policy import load_policy
    from .search_config import DEFAULT_SEARCH_CONFIG_PATH

    torch.set_num_threads(1)
    _STATE.update(
        policy=load_policy(policy_path, torch.device("cpu")),
        floor_critic=load_floor_critic(floor_critic_path), sims=sims,
        ascension=ascension, temperature=temperature, correct=correct,
        search_config=DEFAULT_SEARCH_CONFIG_PATH)


def _new_env():
    from .whole_run_env import RunConfig, WholeRunEnv

    env = WholeRunEnv(RunConfig(
        ascension=_STATE["ascension"], combat_sims=_STATE["sims"],
        deterministic_combat=True, search_config_path=_STATE["search_config"]))
    env._reset_combat_audit()
    return env


def _floor_value(env) -> float:
    from .run_critic import scalars_from_obs
    from .train_value_from_harvest import _state_features

    obs = env.observation()
    features = _state_features(_STATE["policy"], obs)
    vector = np.concatenate([features, scalars_from_obs(obs)]).astype(np.float32)
    return float(_STATE["floor_critic"](torch.from_numpy(vector)))


def _candidate_values(env, actions) -> list[float] | None:
    """V at each candidate's post-action state, or None if any triggers combat."""
    import slaythespire as sts

    values = []
    for index in range(len(actions)):
        branch = _new_env()
        branch.gc = env.gc.copy()
        branch.steps = env.steps
        legal = branch.legal_actions()
        if index >= len(legal):
            return None
        legal[index].execute(branch.gc)
        if (branch.gc.outcome != sts.GameOutcome.UNDECIDED
                or branch.gc.screen_state == sts.ScreenState.BATTLE):
            # Pricing this candidate means resolving a fight; the correction is
            # not worth that, and dropping the whole decision keeps the
            # remaining terms' zero-mean property intact.
            return None
        values.append(_floor_value(branch))
    return values


def _play(job: tuple[int, int]) -> dict:
    seed, policy_seed = job
    policy = _STATE["policy"]
    generator = torch.Generator().manual_seed(policy_seed)
    env = _new_env()
    obs = env.reset(seed)
    correction = 0.0
    corrected_decisions = total_decisions = 0
    started = time.perf_counter()
    with torch.inference_mode():
        while (env.gc.outcome.name == "UNDECIDED"
               and env.steps < env.config.max_decisions):
            logits, _ = policy(obs)
            probabilities = torch.softmax(
                logits / max(1e-6, _STATE["temperature"]), dim=-1)
            index = int(torch.multinomial(probabilities, 1, generator=generator))
            total_decisions += 1

            if _STATE["correct"] and len(logits) > 1:
                actions = env.legal_actions()
                values = _candidate_values(env, actions)
                if values is not None:
                    expected = float(
                        (probabilities.numpy() * np.asarray(values)).sum())
                    correction += values[index] - expected
                    corrected_decisions += 1

            obs, _, done, _ = env.step(index)
            if done:
                break

    floor = int(env.gc.floor_num)
    return {"seed": seed, "floor": floor, "correction": correction,
            "corrected_floor": floor - correction,
            "corrected_decisions": corrected_decisions,
            "total_decisions": total_decisions,
            "outcome": env.gc.outcome.name,
            "seconds": time.perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--floor-critic", default=FLOOR_CRITIC)
    parser.add_argument("--value-data", default=VALUE_DATA)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--seed-base", type=int, default=1_003_000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default="runs/aivat_eval.jsonl")
    args = parser.parse_args()

    torch.set_num_threads(1)
    if args.refit or not os.path.exists(args.floor_critic):
        torch.set_num_threads(6)
        fit_floor_critic(args.value_data, args.floor_critic)
        torch.set_num_threads(1)

    seeds = [args.seed_base + i for i in range(args.runs)]
    policy_seeds = [7_000_003 + i for i in range(args.runs)]

    results = {}
    for correct in (False, True):
        label = "corrected" if correct else "plain"
        print(f"\n{label}: {args.runs} runs at T={args.temperature}, "
              f"{args.sims} sims, {args.workers} workers", flush=True)
        started = time.perf_counter()
        rows = []
        with ProcessPoolExecutor(
                max_workers=args.workers, initializer=_worker_init,
                initargs=(args.policy, args.floor_critic, args.sims,
                          args.ascension, args.temperature, correct)) as pool:
            for row in pool.map(_play, list(zip(seeds, policy_seeds)),
                                chunksize=4):
                rows.append(row)
        results[label] = {"rows": rows, "wall": time.perf_counter() - started}
        print(f"  {len(rows)} runs in {results[label]['wall']:.0f}s", flush=True)

    with open(args.out, "w", encoding="utf-8") as handle:
        for label, payload in results.items():
            for row in payload["rows"]:
                handle.write(json.dumps({"arm": label, **row}) + "\n")

    plain = np.array([r["floor"] for r in results["plain"]["rows"]], dtype=float)
    same_runs = np.array([r["floor"] for r in results["corrected"]["rows"]],
                         dtype=float)
    corrected = np.array([r["corrected_floor"]
                          for r in results["corrected"]["rows"]], dtype=float)
    coverage = np.mean([r["corrected_decisions"] / max(1, r["total_decisions"])
                        for r in results["corrected"]["rows"]])

    def sem(values):
        return float(values.std(ddof=1) / math.sqrt(len(values)))

    print(f"\n{'estimator':>26}  {'mean':>8}  {'sem':>7}  {'sd':>7}")
    print(f"{'plain floor':>26}  {plain.mean():>8.3f}  {sem(plain):>7.3f}  "
          f"{plain.std(ddof=1):>7.3f}")
    print(f"{'plain floor (corrected arm)':>26}  {same_runs.mean():>8.3f}  "
          f"{sem(same_runs):>7.3f}  {same_runs.std(ddof=1):>7.3f}")
    print(f"{'AIVAT-corrected floor':>26}  {corrected.mean():>8.3f}  "
          f"{sem(corrected):>7.3f}  {corrected.std(ddof=1):>7.3f}")

    # Unbiasedness: the correction has mean zero by construction, so the two
    # estimators must agree. A gap larger than its own standard error means the
    # implementation is wrong, not that the estimator is better.
    difference = corrected.mean() - same_runs.mean()
    diff_sem = float((corrected - same_runs).std(ddof=1) / math.sqrt(len(corrected)))
    status = "OK" if abs(difference) <= 3 * diff_sem else "BIASED"
    print(f"\nunbiasedness: corrected - plain = {difference:+.4f} "
          f"+/- {diff_sem:.4f}  {status}")

    ratio = (same_runs.std(ddof=1) / corrected.std(ddof=1)) ** 2
    cost = results["corrected"]["wall"] / results["plain"]["wall"]
    print(f"variance reduction: sd {same_runs.std(ddof=1):.3f} -> "
          f"{corrected.std(ddof=1):.3f}  "
          f"(effective sample x{ratio:.2f}, cost x{cost:.2f}, "
          f"net x{ratio / cost:.2f})")
    print(f"decisions corrected: {100 * coverage:.1f}% "
          f"(rest trigger combat when priced)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
