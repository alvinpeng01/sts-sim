"""PPO trainer for the native whole-run candidate-action environment."""
from __future__ import annotations

import argparse
import random
import os

import torch

from .whole_run_env import RunConfig, WholeRunEnv
from .whole_run_policy import WholeRunPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--combat-sims", type=int, default=300)
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("--seed", type=int, default=900000)
    p.add_argument("--out", default="whole_run_policy.pt")
    p.add_argument("--load", default=None, help="optional policy checkpoint to continue")
    p.add_argument("--epochs", type=int, default=4)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = WholeRunPolicy().to(device)
    if args.load:
        policy.load_state_dict(torch.load(args.load, map_location=device, weights_only=True))
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rng = random.Random(args.seed)
    for update in range(args.updates):
        records = []
        episode_returns = []
        for _ in range(args.episodes):
            env = WholeRunEnv(RunConfig(ascension=args.ascension, combat_sims=args.combat_sims))
            obs = env.reset(rng.randrange(1, 2**31))
            trajectory = []
            done = False
            while not done:
                idx, log_prob, value, _ = policy.act(obs)
                old_value = float(value.detach())
                next_obs, reward, done, _ = env.step(idx)
                trajectory.append((obs, idx, float(log_prob.detach()), old_value, float(reward), done))
                obs = next_obs if next_obs is not None else obs
            running, total = 0.0, 0.0
            for item in reversed(trajectory):
                running = item[4] + 0.995 * running
                records.append((item[0], item[1], item[2], item[3], running))
                total += item[4]
            episode_returns.append(total)

        advantages = torch.tensor([r[4] - r[3] for r in records], device=device)
        returns = torch.tensor([r[4] for r in records], device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        for _ in range(args.epochs):
            losses = []
            for obs, idx, old_logp, _, _ in records:
                logits, value = policy(obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(torch.tensor(idx, device=device))
                ratio = torch.exp(new_logp - old_logp)
                a = advantages[len(losses)]
                unclipped = ratio * a
                clipped = torch.clamp(ratio, 0.8, 1.2) * a
                losses.append(-torch.minimum(unclipped, clipped)
                              + 0.5 * (value - returns[len(losses)]).square()
                              - 0.01 * dist.entropy())
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); optimizer.step()
        if update % 10 == 0:
            print(f"update={update} loss={loss.item():.4f} mean_episode_return={sum(episode_returns)/len(episode_returns):.3f}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(policy.state_dict(), args.out)


if __name__ == "__main__":
    main()
