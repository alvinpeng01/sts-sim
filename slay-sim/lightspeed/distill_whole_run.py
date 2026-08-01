"""Behavior-clone our established overworld heuristic into WholeRunPolicy.

Combat never comes from the teacher: every BATTLE is resolved by the native
expectimax MCTS configured on WholeRunEnv.
"""
from __future__ import annotations

import argparse
import os
import random

import torch
from torch.nn import functional as F

from .whole_run_env import RunConfig, WholeRunEnv
from .whole_run_policy import WholeRunPolicy
from .whole_run_transformer import WholeRunTransformerPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--combat-sims", type=int, default=300)
    p.add_argument("--ascension", type=int, default=20)
    p.add_argument("--seed", type=int, default=940000)
    p.add_argument("--load", default=None)
    p.add_argument("--out", default="whole_run_policy_bc.pt")
    p.add_argument("--model", choices=("compact", "transformer"), default="transformer")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = (WholeRunTransformerPolicy() if args.model == "transformer" else WholeRunPolicy()).to(device)
    if args.load:
        policy.load_state_dict(torch.load(args.load, map_location=device, weights_only=True))
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rng = random.Random(args.seed)
    labels = 0
    for episode in range(args.episodes):
        env = WholeRunEnv(RunConfig(ascension=args.ascension, combat_sims=args.combat_sims))
        env.reset(rng.randrange(1, 2**31))
        teacher = __import__("slaythespire").Agent()
        teacher.pause_on_card_reward = False
        teacher.record_actions = True
        losses = []
        while env.gc.outcome.name == "UNDECIDED" and env.steps < env.config.max_decisions:
            if env.gc.screen_state.name == "BATTLE":
                env._resolve_battles()
                continue
            obs = env.observation()
            actions = env.legal_actions()
            before = len(teacher.game_action_history)
            teacher.step_out_of_combat_policy(env.gc)
            history = teacher.game_action_history
            if len(history) <= before:
                raise RuntimeError("teacher made no overworld action")
            chosen_bits = history[before]
            target = next((i for i, action in enumerate(actions) if action.bits == chosen_bits), None)
            if target is None:
                # The legacy reward helper can consume nested card rewards
                # through an action not surfaced by getAllActionsInState at
                # that intermediate screen.  It is safe to let the teacher
                # advance the run but exclude that ambiguous label.
                pass
            else:
                logits, _ = policy(obs)
                losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=device)))
                labels += 1
            env.steps += 1
            env._resolve_battles()
        if losses:
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); optimizer.step()
        if episode % 10 == 0:
            print(f"episode={episode} labels={labels} loss={loss.item():.4f} floor={env.gc.floor_num}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(policy.state_dict(), args.out)


if __name__ == "__main__":
    main()
