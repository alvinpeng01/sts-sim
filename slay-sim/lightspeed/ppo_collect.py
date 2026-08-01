"""On-policy trajectory collection for run-level PPO.

One iteration = N complete A20 runs played by a frozen policy snapshot, with
every overworld decision recorded as a PPO transition and every completed run
converted to GAE advantages.  Combat stays inside the environment: it is native
MCTS at a fixed budget, resolved by `WholeRunEnv._resolve_battles`, so a whole
battle is ONE semi-Markov transition of the outer trajectory and hundreds of
combat steps never enter the run-level sequence.

Decisions this file makes, and why:

* **Sampling, not argmax.**  PPO's behaviour policy must be the one whose
  log-probabilities it stores.  This costs floors up front -- the campfire head
  emits close to the label marginal, and 07-known-issues.md measures sampling at
  26.30 -> 14.82 floors at T=0.5 -- so `--temperature` is exposed and the
  sampled starting point is reported by this script rather than assumed.  That
  number, not the argmax number, is what an RL curve has to climb from.
* **gamma = 1.0, lambda = 0.97.**  Runs are finite and episodic, so undiscounted
  return is the true objective and potential-based shaping telescopes cleanly
  (FULL_RUN_RL_DESIGN.md sections 9-10).
* **Truncation is bootstrapped, termination is not.**  `WholeRunEnv` ends an
  episode either because the run is over or because it hit `max_decisions`, and
  it returns `None` for the observation in both cases.  Treating the second as
  terminal would teach the critic that surviving deep is worth zero.  Runs at
  this strength reach at most 204 of 256 decisions, but a stronger policy goes
  deeper, so the distinction is handled now.
* **Game seed and sampling seed are separate.**  Paired evaluation needs the map
  held fixed while the policy explores; sharing one RNG would couple them.
* **The critic is `run_critic.RunCritic`, not the policy's `value` head.**  The
  head cannot read the run-position scalars that are worth +0.10 R2, and as
  shipped in v37 it scores R2 -172 (see 02-training-pipeline.md).

`--verify` re-runs the stored observations through the same snapshot and checks
the recorded log-probabilities and values reproduce.  This catches the classic
collection bugs -- a mutated observation, a stale snapshot, a policy left in
train mode -- which otherwise surface as a silently wrong ratio in the PPO
update, where they are very hard to find.

Run from slay-sim/:
    python -m lightspeed.ppo_collect --episodes 96 --workers 6 --verify 200
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict

import numpy as np
import torch

DEFAULT_POLICY = "runs/whole_run_transformer_v37_critic.pt"
DEFAULT_CRITIC = "runs/run_critic_v37.pt"

_STATE: dict = {}


@dataclass
class CollectConfig:
    policy: str = DEFAULT_POLICY
    critic: str = DEFAULT_CRITIC
    sims: int = 100
    ascension: int = 20
    temperature: float = 1.0
    gamma: float = 1.0
    lam: float = 0.97
    keep_observations: bool = True


def _worker_init(config_json: str) -> None:
    from .eval_whole_run_policy import load_policy
    from .run_critic import load as load_critic
    from .search_config import DEFAULT_SEARCH_CONFIG_PATH

    torch.set_num_threads(1)
    config = CollectConfig(**json.loads(config_json))
    _STATE["config"] = config
    _STATE["policy"] = load_policy(config.policy, torch.device("cpu"))
    _STATE["critic"] = load_critic(config.critic)
    _STATE["search_config"] = DEFAULT_SEARCH_CONFIG_PATH


def compute_gae(rewards: np.ndarray, values: np.ndarray, last_value: float,
                gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and value targets for one episode.

    `last_value` is V(s_T) when the episode was cut short and 0.0 when the run
    actually ended -- the bootstrap that keeps a truncated deep run from being
    scored as if it had died there.
    """
    steps = len(rewards)
    advantages = np.zeros(steps, dtype=np.float32)
    running = 0.0
    for t in reversed(range(steps)):
        next_value = last_value if t == steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
    return advantages, advantages + values


def _play(job: tuple[int, int]) -> dict:
    from .run_critic import scalars_from_obs
    from .train_value_from_harvest import _state_features
    from .whole_run_env import RunConfig, WholeRunEnv

    seed, policy_seed = job
    config: CollectConfig = _STATE["config"]
    policy, critic = _STATE["policy"], _STATE["critic"]
    generator = torch.Generator().manual_seed(policy_seed)

    env = WholeRunEnv(RunConfig(
        ascension=config.ascension, combat_sims=config.sims,
        deterministic_combat=True, search_config_path=_STATE["search_config"]))
    obs = env.reset(seed)

    observations, actions, log_probs, values, rewards = [], [], [], [], []
    entropies, num_actions, states = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        while (env.gc.outcome.name == "UNDECIDED"
               and env.steps < env.config.max_decisions):
            logits, _ = policy(obs)
            scaled = logits / max(1e-6, config.temperature)
            distribution = torch.distributions.Categorical(logits=scaled)
            # A generator keeps sampling reproducible per (seed, policy_seed)
            # without touching the global RNG a worker shares across episodes.
            index = int(torch.multinomial(distribution.probs, 1,
                                          generator=generator))
            state = _state_features(policy, obs)
            value = critic.value_of(state, obs)

            if config.keep_observations:
                observations.append({k: v for k, v in obs.items()
                                     if k != "action_text"})
            # The critic's input, kept because the trunk is frozen during the
            # update: 96 floats per transition against re-running the encoder
            # over every stored observation at refit time (~31 s per iteration).
            states.append(state)
            actions.append(index)
            log_probs.append(float(distribution.log_prob(torch.tensor(index))))
            values.append(value)
            entropies.append(float(distribution.entropy()))
            num_actions.append(int(logits.shape[0]))

            obs, reward, done, _ = env.step(index)
            rewards.append(float(reward))
            if done:
                break

        terminated = env.gc.outcome.name != "UNDECIDED"
        last_value = 0.0
        if not terminated:
            # Cut short by max_decisions: bootstrap rather than score it a loss.
            tail = env.observation()
            last_value = critic.value_of(_state_features(policy, tail), tail)

    rewards_a = np.asarray(rewards, dtype=np.float32)
    values_a = np.asarray(values, dtype=np.float32)
    advantages, returns = compute_gae(rewards_a, values_a, last_value,
                                      config.gamma, config.lam)
    return {
        "seed": seed, "policy_seed": policy_seed,
        "observations": observations,
        "states": (np.stack(states).astype(np.float32) if states
                   else np.zeros((0, 96), np.float32)),
        "actions": np.asarray(actions, dtype=np.int64),
        "log_probs": np.asarray(log_probs, dtype=np.float32),
        "values": values_a, "rewards": rewards_a,
        "advantages": advantages, "returns": returns,
        "entropies": np.asarray(entropies, dtype=np.float32),
        "num_actions": np.asarray(num_actions, dtype=np.int16),
        "final_floor": int(env.gc.floor_num), "act": int(env.gc.act),
        "outcome": env.gc.outcome.name, "terminated": terminated,
        "last_value": last_value,
        "seconds": time.perf_counter() - started,
    }


def collect(config: CollectConfig, seeds: list[int], policy_seeds: list[int],
            workers: int, progress_every: int = 0) -> list[dict]:
    started = time.perf_counter()
    episodes = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(json.dumps(asdict(config)),)) as pool:
        for done, episode in enumerate(
                pool.map(_play, list(zip(seeds, policy_seeds)), chunksize=1),
                start=1):
            episodes.append(episode)
            if progress_every and done % progress_every == 0:
                print(f"  {done}/{len(seeds)} runs "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
    return episodes


def verify(config: CollectConfig, episodes: list[dict], count: int) -> None:
    """Recompute stored log-probs and values from the same snapshot."""
    from .eval_whole_run_policy import load_policy
    from .run_critic import load as load_critic
    from .train_value_from_harvest import _state_features

    if not episodes[0]["observations"]:
        print("verification skipped: observations were not kept")
        return
    policy = load_policy(config.policy, torch.device("cpu"))
    critic = load_critic(config.critic)
    checked = 0
    worst_logp = worst_value = 0.0
    with torch.inference_mode():
        for episode in episodes:
            for step in range(len(episode["actions"])):
                if checked >= count:
                    break
                obs = episode["observations"][step]
                logits, _ = policy(obs)
                scaled = logits / max(1e-6, config.temperature)
                distribution = torch.distributions.Categorical(logits=scaled)
                logp = float(distribution.log_prob(
                    torch.tensor(int(episode["actions"][step]))))
                value = critic.value_of(_state_features(policy, obs), obs)
                worst_logp = max(worst_logp,
                                 abs(logp - float(episode["log_probs"][step])))
                worst_value = max(worst_value,
                                  abs(value - float(episode["values"][step])))
                checked += 1
            if checked >= count:
                break
    status = "OK" if max(worst_logp, worst_value) < 1e-4 else "MISMATCH"
    print(f"verify: {checked} transitions replayed, "
          f"max |dlogp| {worst_logp:.2e}, max |dV| {worst_value:.2e}  {status}")
    if status == "MISMATCH":
        raise SystemExit("stored log-probs/values do not reproduce")


def summarize(episodes: list[dict], wall: float, workers: int) -> None:
    steps = sum(len(e["actions"]) for e in episodes)
    floors = np.array([e["final_floor"] for e in episodes], dtype=np.float32)
    advantages = np.concatenate([e["advantages"] for e in episodes])
    values = np.concatenate([e["values"] for e in episodes])
    returns = np.concatenate([e["returns"] for e in episodes])
    entropies = np.concatenate([e["entropies"] for e in episodes])
    wins = sum(1 for e in episodes if e["outcome"] == "PLAYER_VICTORY")
    truncated = sum(1 for e in episodes if not e["terminated"])

    explained = 1.0 - float(((returns - values) ** 2).sum()
                            / max(1e-9, ((returns - returns.mean()) ** 2).sum()))
    print(f"\n{len(episodes)} episodes, {steps} transitions "
          f"({steps / len(episodes):.1f} per run)")
    print(f"floor {floors.mean():.2f} +/- {floors.std() / np.sqrt(len(floors)):.2f}"
          f"   victories {wins}   truncated {truncated}")
    print(f"advantage  mean {advantages.mean():+.4f}  sd {advantages.std():.4f}")
    print(f"entropy    mean {entropies.mean():.3f} nats")
    print(f"critic explained variance on this batch: {explained:+.4f}")
    print(f"wall {wall:.1f}s on {workers} workers "
          f"({len(episodes) / wall:.2f} episodes/s, "
          f"{steps / wall:.0f} transitions/s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--critic", default=DEFAULT_CRITIC)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed-base", type=int, default=9_000_000)
    parser.add_argument("--policy-seed", type=int, default=1)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--verify", type=int, default=0,
                        help="replay N stored transitions through the snapshot")
    parser.add_argument("--drop-observations", action="store_true",
                        help="collect statistics only; a PPO update needs them")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = CollectConfig(
        policy=args.policy, critic=args.critic, sims=args.sims,
        ascension=args.ascension, temperature=args.temperature,
        gamma=args.gamma, lam=args.lam,
        keep_observations=not args.drop_observations)

    seeds = [args.seed_base + i for i in range(args.episodes)]
    # Separate stream: the map must be reproducible while sampling varies.
    policy_seeds = [args.policy_seed * 1_000_003 + i for i in range(args.episodes)]

    print(f"collecting {args.episodes} episodes at {args.sims} sims, "
          f"T={args.temperature}, on {args.workers} workers", flush=True)
    started = time.perf_counter()
    episodes = collect(config, seeds, policy_seeds, args.workers,
                       progress_every=max(1, args.episodes // 4))
    wall = time.perf_counter() - started
    summarize(episodes, wall, args.workers)

    if args.verify:
        verify(config, episodes, args.verify)

    if args.out:
        torch.save({"config": asdict(config), "episodes": episodes}, args.out)
        size = os.path.getsize(args.out) / 1e6
        print(f"wrote {args.out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
