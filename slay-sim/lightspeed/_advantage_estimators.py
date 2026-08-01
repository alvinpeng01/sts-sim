"""Which advantage estimator actually tracks the truth on this task?

Three separate literatures say the same thing about a value network that cannot
rank states -- which is what this project measured its critic to be (val R2
0.32, and a lookup table on floor alone scores 0.25):

* VinePPO (arXiv 2410.01679) deletes the critic and estimates V(s) by branching
  Monte-Carlo rollouts from the visited state;
* GRPO drops the value function for a group-relative baseline over several
  rollouts of the same problem;
* GAZ Play-to-Plan (arXiv 2306.04403) reshapes a single-player task into
  competition against a greedy historical copy of the policy.

They differ only in what gets subtracted from the return. So they can be
compared in one harness, against a vine Monte-Carlo estimate of the true
advantage as the reference:

    A_vine(s,a)   = Q_mc(s,a) - V_mc(s)          <- reference, K rollouts each
    A_critic(s,a) = G_t - V_critic(s)            <- what ppo_update does today
    A_group       = G_episode - mean(G of episodes on the SAME seed)
    A_self        = G_episode - G_greedy(same seed)

`A_group` and `A_self` are episode-level by construction: they trade per-step
credit assignment for an unbiased, low-variance episode signal, and that trade
is exactly what is being measured here.

The seed is the group: this game's map, rewards and shuffles are all functions
of it, so several episodes at one seed differ only by the policy's own sampling
-- the closest thing to "the same problem attempted several times" the task
admits, and the reason a group baseline should be unusually strong here.

Returns use `WholeRunEnv`'s own reward, the same quantity `collect_run_value_data`
fits and `ppo_collect` accumulates, so the numbers are commensurable with the
rest of the RL stack.

Run from slay-sim/:
    python -m lightspeed._advantage_estimators --seeds 40 --workers 6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

DEFAULT_POLICY = "runs/whole_run_transformer_v37_critic.pt"
DEFAULT_CRITIC = "runs/run_critic_v37.pt"

_STATE: dict = {}


def _worker_init(policy_path: str, critic_path: str, sims: int, ascension: int,
                 temperature: float) -> None:
    from .eval_whole_run_policy import load_policy
    from .run_critic import load as load_critic
    from .search_config import DEFAULT_SEARCH_CONFIG_PATH

    torch.set_num_threads(1)
    _STATE.update(
        policy=load_policy(policy_path, torch.device("cpu")),
        critic=load_critic(critic_path), sims=sims, ascension=ascension,
        temperature=temperature, search_config=DEFAULT_SEARCH_CONFIG_PATH)


def _new_env():
    from .whole_run_env import RunConfig, WholeRunEnv

    return WholeRunEnv(RunConfig(
        ascension=_STATE["ascension"], combat_sims=_STATE["sims"],
        deterministic_combat=True, search_config_path=_STATE["search_config"]))


def env_return(gc, floor0: int, hp0: int) -> float:
    """The undiscounted sum of WholeRunEnv step rewards from a start state."""
    terminal = (1.0 if gc.outcome.name == "PLAYER_VICTORY"
                else -1.0 if gc.outcome.name == "PLAYER_LOSS" else 0.0)
    return (0.01 * (int(gc.floor_num) - floor0)
            + 0.002 * (int(gc.cur_hp) - hp0) + terminal)


def _playout(env, generator, greedy: bool) -> None:
    """Continue an env in place under the policy until the run ends."""
    policy = _STATE["policy"]
    temperature = _STATE["temperature"]
    with torch.inference_mode():
        while (env.gc.outcome.name == "UNDECIDED"
               and env.steps < env.config.max_decisions):
            obs = env.observation()
            logits, _ = policy(obs)
            if greedy:
                index = int(torch.argmax(logits))
            else:
                probs = torch.softmax(logits / max(1e-6, temperature), dim=-1)
                index = int(torch.multinomial(probs, 1, generator=generator))
            _, _, done, _ = env.step(index)
            if done:
                break


def _vine(env, action_index: int | None, rollouts: int, seed: int) -> float:
    """Mean return over `rollouts` continuations, optionally forcing an action.

    `GameContext.copy()` is what the label generator already relies on, so
    branching costs a copy rather than a replay from the start of the run.
    """
    floor0, hp0 = int(env.gc.floor_num), int(env.gc.cur_hp)
    total = 0.0
    for index in range(rollouts):
        branch = _new_env()
        # A branch skips reset(), which is what would normally initialize the
        # audit counters and the search seed, so both are set by hand here.
        branch._reset_combat_audit()
        branch.gc = env.gc.copy()
        branch.steps = env.steps
        branch.search_seed_base = seed + index
        generator = torch.Generator().manual_seed(seed + index)
        if action_index is not None:
            actions = branch.legal_actions()
            if action_index >= len(actions):
                return float("nan")
            _, _, done, _ = branch.step(action_index)
            if not done:
                _playout(branch, generator, greedy=False)
        else:
            _playout(branch, generator, greedy=False)
        total += env_return(branch.gc, floor0, hp0)
    return total / rollouts


def _play(job: dict) -> dict:
    seed = job["seed"]
    generator = torch.Generator().manual_seed(job["policy_seed"])
    policy, critic = _STATE["policy"], _STATE["critic"]

    from .run_critic import scalars_from_obs
    from .train_value_from_harvest import _state_features

    env = _new_env()
    obs = env.reset(seed)
    floor0, hp0 = int(env.gc.floor_num), int(env.gc.cur_hp)
    samples: list[dict] = []
    rewards: list[float] = []
    step = 0
    started = time.perf_counter()
    with torch.inference_mode():
        while (env.gc.outcome.name == "UNDECIDED"
               and env.steps < env.config.max_decisions):
            logits, _ = policy(obs)
            if job["greedy"]:
                index = int(torch.argmax(logits))
            else:
                probs = torch.softmax(
                    logits / max(1e-6, _STATE["temperature"]), dim=-1)
                index = int(torch.multinomial(probs, 1, generator=generator))

            if job["vine_states"] and step in job["vine_states"] and len(logits) > 1:
                state = _state_features(policy, obs)
                samples.append({
                    "step": step,
                    "floor": int(env.gc.floor_num),
                    "hp": int(env.gc.cur_hp),
                    "critic_value": critic.value_of(state, obs),
                    # TWO independent V estimates from disjoint rollout sets.
                    # The reference is Q_mc - V_mc, and the VinePPO arm is
                    # G - V_mc; sharing one V estimate would put the same
                    # estimation noise in both and manufacture correlation out
                    # of nothing. `v_mc_a` scores the reference, `v_mc_b` the
                    # arm, so the comparison is honest.
                    "v_mc_a": _vine(env, None, job["vine_rollouts"],
                                    job["policy_seed"] * 7919 + step),
                    "v_mc_b": _vine(env, None, job["vine_rollouts"],
                                    job["policy_seed"] * 5171 + step * 97 + 3),
                    "q_mc": _vine(env, index, job["vine_rollouts"],
                                  job["policy_seed"] * 6421 + step),
                    "reward_index": len(rewards),
                })

            obs, reward, done, _ = env.step(index)
            rewards.append(float(reward))
            step += 1
            if done:
                break

    total = env_return(env.gc, floor0, hp0)
    for sample in samples:
        # Return-to-go on the realized trajectory, the quantity PPO differences
        # against a baseline.
        sample["return_to_go"] = float(sum(rewards[sample["reward_index"]:]))
    return {"seed": seed, "policy_seed": job["policy_seed"],
            "greedy": job["greedy"], "role": job["role"],
            "floor": int(env.gc.floor_num), "episode_return": total,
            "samples": samples, "seconds": time.perf_counter() - started}


def correlate(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Pearson, Spearman, and sign agreement against the reference."""
    xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 3 or xs.std() == 0 or ys.std() == 0:
        return float("nan"), float("nan"), float("nan")
    pearson = float(np.corrcoef(xs, ys)[0, 1])
    rank_x = np.argsort(np.argsort(xs)).astype(float)
    rank_y = np.argsort(np.argsort(ys)).astype(float)
    spearman = float(np.corrcoef(rank_x, rank_y)[0, 1])
    agreement = float(np.mean(np.sign(xs) == np.sign(ys)))
    return pearson, spearman, agreement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--critic", default=DEFAULT_CRITIC)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=6_000_000)
    parser.add_argument("--group-size", type=int, default=6,
                        help="episodes per seed forming the GRPO group")
    parser.add_argument("--vine-rollouts", type=int, default=6)
    parser.add_argument("--vine-states", type=int, default=4,
                        help="states per focus episode to vine-estimate")
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default="runs/advantage_estimators.json")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    jobs = []
    for offset in range(args.seeds):
        seed = args.seed_base + offset
        # Vine states are chosen before the episode exists, so they are picked
        # from an early band every run reaches -- otherwise deep picks would
        # silently only ever land on the runs that survive.
        vine_states = sorted(rng.choice(np.arange(2, 34), size=args.vine_states,
                                        replace=False).tolist())
        jobs.append({"seed": seed, "policy_seed": offset * 1013 + 1,
                     "greedy": False, "role": "focus",
                     "vine_states": vine_states,
                     "vine_rollouts": args.vine_rollouts})
        for member in range(1, args.group_size):
            jobs.append({"seed": seed, "policy_seed": offset * 1013 + 1 + member,
                         "greedy": False, "role": "group",
                         "vine_states": [], "vine_rollouts": 0})
        jobs.append({"seed": seed, "policy_seed": 0, "greedy": True,
                     "role": "greedy", "vine_states": [], "vine_rollouts": 0})

    print(f"{args.seeds} seeds x (1 focus + {args.group_size - 1} group + 1 greedy) "
          f"= {len(jobs)} episodes, {args.vine_states} vine states each at "
          f"{args.vine_rollouts} rollouts, {args.workers} workers", flush=True)

    results = []
    started = time.perf_counter()
    with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_worker_init,
            initargs=(args.policy, args.critic, args.sims, args.ascension,
                      args.temperature)) as pool:
        for done, row in enumerate(pool.map(_play, jobs, chunksize=1), start=1):
            results.append(row)
            if done % 40 == 0:
                print(f"  {done}/{len(jobs)} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)

    by_seed: dict[int, list[dict]] = {}
    for row in results:
        by_seed.setdefault(row["seed"], []).append(row)

    reference, critic_arm, group_arm, self_arm, mc_arm = [], [], [], [], []
    for seed, rows in by_seed.items():
        focus = next(r for r in rows if r["role"] == "focus")
        sampled = [r["episode_return"] for r in rows if r["role"] in ("focus", "group")]
        greedy = next((r["episode_return"] for r in rows if r["role"] == "greedy"), None)
        group_mean = statistics.mean(sampled)
        for sample in focus["samples"]:
            if not all(np.isfinite(sample[key])
                       for key in ("v_mc_a", "v_mc_b", "q_mc")):
                continue
            reference.append(sample["q_mc"] - sample["v_mc_a"])
            critic_arm.append(sample["return_to_go"] - sample["critic_value"])
            mc_arm.append(sample["return_to_go"] - sample["v_mc_b"])
            group_arm.append(focus["episode_return"] - group_mean)
            self_arm.append(focus["episode_return"] - greedy
                            if greedy is not None else float("nan"))

    print(f"\n{len(reference)} vine-estimated states from {len(by_seed)} seeds "
          f"({time.perf_counter() - started:.0f}s)\n")
    print(f"{'estimator':>34}  {'pearson':>8}  {'spearman':>9}  {'sign':>6}  {'sd':>7}")
    arms = [
        ("A_critic = G - V_critic (today)", critic_arm),
        ("A_vineV  = G - V_mc (VinePPO)", mc_arm),
        ("A_group  = G_ep - group mean (GRPO)", group_arm),
        ("A_self   = G_ep - greedy (self-comp)", self_arm),
    ]
    summary = {}
    for name, arm in arms:
        pearson, spearman, sign = correlate(arm, reference)
        sd = float(np.nanstd(np.asarray(arm, dtype=np.float64)))
        summary[name] = {"pearson": pearson, "spearman": spearman,
                         "sign_agreement": sign, "sd": sd}
        print(f"{name:>34}  {pearson:>8.3f}  {spearman:>9.3f}  "
              f"{sign:>6.3f}  {sd:>7.4f}")
    print(f"\nreference A_vine: sd {np.std(reference):.4f}, "
          f"mean {np.mean(reference):+.4f}")

    # How much of the reference's own spread is signal?  v_mc_a and v_mc_b are
    # independent estimates of the SAME V(s), so their difference isolates the
    # estimator's noise: sd(a - b) = sqrt(2) * SE_V.  With that in hand,
    # var(A_vine) = var(true advantage) + SE_Q^2 + SE_V^2 can be solved for the
    # only term that matters.
    pairs = [(s["v_mc_a"], s["v_mc_b"]) for r in results
             for s in r["samples"]
             if np.isfinite(s["v_mc_a"]) and np.isfinite(s["v_mc_b"])]
    diffs = np.asarray([a - b for a, b in pairs], dtype=np.float64)
    se_v = float(diffs.std() / math.sqrt(2))
    observed = float(np.var(reference))
    signal = observed - 2.0 * se_v ** 2
    print(f"\nreference decomposition ({len(pairs)} paired V estimates, "
          f"{args.vine_rollouts} rollouts each)")
    print(f"  SE of one V_mc estimate      : {se_v:.4f}")
    print(f"  observed sd of A_vine        : {math.sqrt(observed):.4f}")
    print(f"  implied sd of TRUE advantage : "
          f"{math.sqrt(signal) if signal > 0 else float('nan'):.4f}"
          f"{'  (noise exceeds total spread)' if signal <= 0 else ''}")
    if signal > 0:
        needed = 2.0 * (args.vine_rollouts * se_v ** 2) / signal
        print(f"  rollouts per state for SNR 1 : {needed:.0f}")
    summary["reference"] = {"se_v": se_v, "observed_var": observed,
                            "signal_var": signal}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "n_states": len(reference),
                   "samples": [{k: v for k, v in s.items()}
                               for r in results for s in r["samples"]],
                   "config": vars(args)}, handle, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
