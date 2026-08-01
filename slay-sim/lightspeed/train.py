"""Train the action-scoring policy against IroncladFightEnv.

Run:  PYTHONPATH=. .venv/bin/python -m lightspeed.train

Same REINFORCE algorithm as rl/reinforce.py (discounted returns, batch-
standardized advantage) -- only the action representation changed, not the
training math. Proves Phases 1-5 work together end to end: sts_lightspeed
bindings -> env -> action-scoring policy -> training loop.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import torch
import torch.optim as optim

from .env import IroncladFightEnv
from .policy import ActionScoringPolicy


def run_episode(env: IroncladFightEnv, policy: ActionScoringPolicy, seed=None, sample=True):
    obs = env.reset(seed=seed)
    log_probs, rewards = [], []
    done = False
    while not done:
        idx, log_prob, _ = policy.act(obs, sample=sample)
        action = obs["actions"][idx]
        obs, reward, done, info = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
    return log_probs, rewards, info


def evaluate(env: IroncladFightEnv, policy: ActionScoringPolicy, n: int, seed_offset: int = 0,
             per_encounter: bool = False):
    import slaythespire as sts
    wins = hp_sum = reward_sum = 0
    breakdown = {}  # encounter -> [wins, fights]
    with torch.no_grad():
        for i in range(n):
            _, rewards, info = run_episode(env, policy, seed=seed_offset + i, sample=False)
            reward_sum += sum(rewards)
            won = info["outcome"] == sts.BattleOutcome.PLAYER_VICTORY
            if won:
                wins += 1
                hp_sum += info["player_hp"]
            if per_encounter:
                key = env.last_encounter
                w, total = breakdown.get(key, [0, 0])
                breakdown[key] = [w + (1 if won else 0), total + 1]

    result = (wins / n, hp_sum / max(wins, 1), reward_sum / n)
    if per_encounter:
        return result, {enc: w / total for enc, (w, total) in breakdown.items()}
    return result


def train(env: IroncladFightEnv, policy: ActionScoringPolicy, updates=150,
          episodes_per_update=16, lr=1e-3, gamma=0.99,
          checkpoint_every: int = 0, checkpoint_eval_n: int = 100):
    """REINFORCE has no trust region -- a bad batch can push the policy
    somewhere worse and it has no protection against that (empirically
    observed: Gremlin Gang's win rate dropped from 72% to 40% between
    updates 400-600 in one run before recovering). checkpoint_every > 0
    evaluates every N updates and keeps the best-seen state_dict, so a late
    regression doesn't waste the run -- the final result is "best seen",
    not "whatever the last update happened to land on".

    Returns (history, best_state_dict) if checkpoint_every > 0, else just
    history (unchanged signature otherwise, for existing callers)."""
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    history = []
    best_reward = float("-inf")
    best_state = None

    for update in range(updates):
        episode_data = []
        all_returns: List[float] = []
        batch_rewards = []

        for ep in range(episodes_per_update):
            log_probs, rewards, _info = run_episode(env, policy, seed=None, sample=True)
            G, returns = 0.0, []
            for r in reversed(rewards):
                G = r + gamma * G
                returns.append(G)
            returns.reverse()
            episode_data.append((log_probs, returns))
            all_returns.extend(returns)
            batch_rewards.append(sum(rewards))

        returns_t = torch.tensor(all_returns, dtype=torch.float32)
        mean, std = returns_t.mean(), returns_t.std() + 1e-8

        optimizer.zero_grad()
        loss = torch.zeros(1)
        n_steps = 0
        for log_probs, returns in episode_data:
            for lp, G in zip(log_probs, returns):
                adv = (G - mean.item()) / std.item()
                loss = loss - lp * adv
                n_steps += 1
        loss = loss / max(n_steps, 1)
        loss.backward()
        optimizer.step()

        history.append(float(np.mean(batch_rewards)))

        if checkpoint_every and (update + 1) % checkpoint_every == 0:
            _, _, eval_reward = evaluate(env, policy, n=checkpoint_eval_n)
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_state = {k: v.clone() for k, v in policy.state_dict().items()}

    if checkpoint_every:
        if best_state is not None:
            policy.load_state_dict(best_state)
        return history, best_state
    return history


def main():
    env = IroncladFightEnv()
    policy = ActionScoringPolicy()

    print("=== baseline (untrained policy) ===")
    wr0, hp0, rew0 = evaluate(env, policy, n=100)
    print(f"win {wr0:.1%}  avg HP {hp0:.1f}  avg reward {rew0:.2f}")

    print("\n=== training ===")
    t0 = time.perf_counter()
    history = train(env, policy, updates=150, episodes_per_update=16)
    dt = time.perf_counter() - t0
    print(f"trained 150 updates x 16 episodes = {150*16} episodes in {dt:.1f}s "
          f"({150*16/dt:.1f} episodes/sec)")
    for i in range(0, len(history), 15):
        window = history[max(0, i - 5):i + 5] or [history[i]]
        print(f"  update {i:3d}: avg batch reward {np.mean(window):6.2f}")

    print("\n=== trained policy ===")
    wr1, hp1, rew1 = evaluate(env, policy, n=100)
    print(f"win {wr1:.1%}  avg HP {hp1:.1f}  avg reward {rew1:.2f}")
    print(f"\nimprovement: win {wr1-wr0:+.1%}, reward {rew1-rew0:+.2f}")


if __name__ == "__main__":
    main()
