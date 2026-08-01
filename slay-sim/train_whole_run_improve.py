"""Fast native policy improvement from short-horizon value-guided rankings."""
from __future__ import annotations

import argparse
import copy
import os
import random

import torch
from torch.nn import functional as F
import slaythespire as sts

from lightspeed.whole_run_env import RunConfig, WholeRunEnv
from lightspeed.whole_run_oracle import rank_actions_with_value
from lightspeed.whole_run_transformer import WholeRunTransformerPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--labels-per-episode", type=int, default=3)
    p.add_argument("--prefix-actions", type=int, default=12)
    p.add_argument("--combat-sims", type=int, default=25)
    p.add_argument("--ascension", type=int, default=20)
    p.add_argument("--seed", type=int, default=996000)
    p.add_argument("--load", required=True)
    p.add_argument("--critic-load", default=None,
                   help="separate frozen critic checkpoint; keeps the policy start unchanged")
    p.add_argument("--out", required=True)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--target-update", type=int, default=50,
                   help="refresh the frozen value critic every N policy episodes")
    p.add_argument("--prefix-policy", default=None,
                   help="optional self-trained policy used to reach label states")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WholeRunTransformerPolicy().to(device)
    missing, unexpected = model.load_state_dict(
        torch.load(args.load, map_location=device, weights_only=True), strict=False)
    if missing or unexpected:
        print(f"loaded compatible weights; new={missing} unused={unexpected}", flush=True)
    # Ranking actions with the same critic that is changing under CE loss
    # makes targets drift mid-pass.  Hold a read-only target critic fixed for
    # a short block, then refresh it from the improved policy.
    target_model = copy.deepcopy(model).eval()
    if args.critic_load:
        target_model = WholeRunTransformerPolicy().to(device)
        missing, unexpected = target_model.load_state_dict(
            torch.load(args.critic_load, map_location=device, weights_only=True), strict=False)
        if missing or unexpected:
            print(f"loaded compatible critic; new={missing} unused={unexpected}", flush=True)
        target_model.eval()
    for parameter in target_model.parameters():
        parameter.requires_grad_(False)
    prefix_policy = None
    if args.prefix_policy:
        prefix_policy = WholeRunTransformerPolicy().to(device)
        prefix_policy.load_state_dict(
            torch.load(args.prefix_policy, map_location=device, weights_only=True), strict=False)
        prefix_policy.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(args.seed)
    labels = 0
    for episode in range(args.episodes):
        env = WholeRunEnv(RunConfig(ascension=args.ascension, combat_sims=args.combat_sims))
        env.reset(rng.randrange(1, 2**31))
        prefix = sts.Agent(); prefix.pause_on_card_reward = False
        for _ in range(rng.randrange(args.prefix_actions + 1)):
            if env.gc.outcome != sts.GameOutcome.UNDECIDED: break
            if env.gc.screen_state == sts.ScreenState.BATTLE:
                env._resolve_battles(); continue
            if prefix_policy is None:
                prefix.step_out_of_combat_policy(env.gc); env.steps += 1; env._resolve_battles()
            else:
                obs = env.observation()
                with torch.no_grad():
                    action, _, _, _ = prefix_policy.act(obs, sample=False)
                _, _, done, _ = env.step(action)
                if done: break
        losses = []
        for _ in range(args.labels_per_episode):
            if env.gc.outcome != sts.GameOutcome.UNDECIDED: break
            ranked = rank_actions_with_value(env.gc, target_model, args.combat_sims)
            if not ranked: break
            obs = env.observation(); target = ranked[0][0]
            logits, _ = model(obs)
            losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=device)))
            labels += 1
            _, _, done, _ = env.step(target)
            if done: break
        if not losses: continue
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if not args.critic_load and args.target_update > 0 and (episode + 1) % args.target_update == 0:
            target_model.load_state_dict(model.state_dict())
        if episode % 20 == 0:
            print(f"episode={episode} labels={labels} loss={loss.item():.4f}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)


if __name__ == "__main__":
    main()
