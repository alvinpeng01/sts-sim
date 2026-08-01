"""Pretrain the native transformer value head from our own full-run rollouts."""
from __future__ import annotations

import argparse
import os
import random

import torch
import slaythespire as sts

from .whole_run_env import RunConfig, WholeRunEnv
from .whole_run_transformer import WholeRunTransformerPolicy


def run_return(gc) -> float:
    keys = int(gc.red_key) + int(gc.green_key) + int(gc.blue_key)
    progress = 0.4 * min(gc.floor_num, 56) / 56.0
    # A key collected on an Act 1/2 death is not a successful run resource.
    # Only credit keys after clearing Act 3 (or on a Heart victory), otherwise
    # the critic learns to value suicidal key routes too highly.
    key_credit = keys if gc.act >= 4 or gc.outcome == sts.GameOutcome.PLAYER_VICTORY else 0
    resources = 0.05 * key_credit + 0.02 * gc.cur_hp / max(1, gc.max_hp)
    heart = 0.6 if gc.outcome == sts.GameOutcome.PLAYER_VICTORY else 0.0
    return progress + resources + heart


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--combat-sims", type=int, default=25)
    p.add_argument("--ascension", type=int, default=20)
    p.add_argument("--seed", type=int, default=995000)
    p.add_argument("--load", default=None)
    p.add_argument("--rollout-policy", default=None,
                   help="optional self-trained policy used only to collect deeper states")
    p.add_argument("--out", default="whole_run_transformer_value.pt")
    p.add_argument("--save-every", type=int, default=100,
                   help="write <out>.partial.pt periodically (0 disables)")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WholeRunTransformerPolicy().to(device)
    if args.load:
        missing, unexpected = model.load_state_dict(
            torch.load(args.load, map_location=device, weights_only=True), strict=False)
        if missing or unexpected:
            print(f"loaded compatible weights; new={missing} unused={unexpected}", flush=True)
    rollout_policy = None
    if args.rollout_policy:
        rollout_policy = WholeRunTransformerPolicy().to(device)
        rollout_policy.load_state_dict(
            torch.load(args.rollout_policy, map_location=device, weights_only=True), strict=False)
        rollout_policy.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    rng = random.Random(args.seed)
    for episode in range(args.episodes):
        env = WholeRunEnv(RunConfig(ascension=args.ascension, combat_sims=args.combat_sims))
        env.reset(rng.randrange(1, 2**31))
        teacher = sts.Agent(); teacher.pause_on_card_reward = False
        states = []
        while env.gc.outcome == sts.GameOutcome.UNDECIDED and env.steps < env.config.max_decisions:
            if env.gc.screen_state == sts.ScreenState.BATTLE:
                env._resolve_battles(); continue
            if env.legal_actions():
                states.append(env.observation())
            if rollout_policy is None:
                teacher.step_out_of_combat_policy(env.gc)
            else:
                obs = env.observation()
                with torch.no_grad():
                    action, _, _, _ = rollout_policy.act(obs, sample=False)
                env.step(action)
                # step() already advances the counter and resolves battles.
                continue
            env.steps += 1
            env._resolve_battles()
        target = torch.tensor(run_return(env.gc), dtype=torch.float32, device=device)
        if states:
            values = torch.stack([model(state)[1] for state in states])
            loss = (values - target).square().mean()
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if episode % 10 == 0:
            print(f"episode={episode} states={len(states)} target={target.item():.3f} floor={env.gc.floor_num}", flush=True)
        if args.save_every and (episode + 1) % args.save_every == 0:
            torch.save(model.state_dict(), args.out + ".partial.pt")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)


if __name__ == "__main__":
    main()
