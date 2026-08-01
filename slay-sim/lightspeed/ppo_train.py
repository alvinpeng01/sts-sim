"""The run-level PPO loop: collect, update, snapshot, repeat.

Ties `ppo_collect.py` and `ppo_update.py` into a training run, and keeps the two
metrics that matter separate:

* **sampled floor** -- what the behaviour policy scores, the quantity PPO
  actually optimizes and the one that moves first;
* **greedy floor** -- what the checkpoint is worth deployed, measured
  periodically with argmax on a FIXED held-out seed set that no iteration
  collects on.

They are far apart at the start (18.32 sampled against 22.89 greedy) because the
policy's decision margins are ~0.13 nats, so reporting only one of them would be
misleading in either direction. The eval seed set is fixed so the greedy series
is paired across iterations; 04-evaluation.md's power warning applies to it --
at n=200 the paired sem is ~0.55 floors, so single-iteration wiggles are noise
and only the trend over many iterations means anything.

Each iteration writes a snapshot, and collection loads the policy from that file
rather than sharing memory with the trainer, so a worker can never mix weights
from two iterations.

Run from slay-sim/:
    python -m lightspeed.ppo_train --iterations 50 --episodes 256 --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .ppo_collect import CollectConfig, collect, summarize
from .ppo_update import UpdateConfig, ppo_update


def greedy_evaluation(policy_path: str, critic_path: str, seeds: list[int],
                      sims: int, ascension: int, workers: int) -> dict:
    """Argmax on held-out seeds -- the deployable number, not the training one."""
    config = CollectConfig(
        policy=policy_path, critic=critic_path, sims=sims, ascension=ascension,
        temperature=1e-6, keep_observations=False)
    episodes = collect(config, seeds, [0] * len(seeds), workers)
    floors = np.array([e["final_floor"] for e in episodes], dtype=np.float32)
    return {
        "greedy_floor": float(floors.mean()),
        "greedy_sem": float(floors.std() / np.sqrt(len(floors))),
        "greedy_wins": sum(1 for e in episodes if e["outcome"] == "PLAYER_VICTORY"),
        "greedy_floors": floors.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="ppo1")
    parser.add_argument("--policy", default="runs/whole_run_transformer_v37_critic.pt")
    parser.add_argument("--critic", default="runs/run_critic_v37.pt")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=20_000_000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--temperature-final", type=float, default=None,
                        help="linearly anneal collection temperature to this")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=2048,
                        help="larger than the single-process default: "
                             "transfer cost is per step, not per transition")
    parser.add_argument("--update-workers", type=int, default=6,
                        help="processes summing gradients; 1 keeps the "
                             "single-process update path")
    parser.add_argument("--train-trunk", action="store_true")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--eval-seed-base", type=int, default=1_003_000)
    parser.add_argument("--keep-batches", action="store_true",
                        help="retain each iteration's transitions (~57 MB each)")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(0)

    from .eval_whole_run_policy import load_policy
    from .run_critic import load as load_critic

    directory = os.path.join("runs", "ppo", args.tag)
    os.makedirs(directory, exist_ok=True)
    metrics_path = os.path.join(directory, "metrics.jsonl")
    policy_path = os.path.join(directory, "policy_current.pt")
    critic_path = os.path.join(directory, "critic_current.pt")

    policy = load_policy(args.policy, torch.device("cpu"))
    critic = load_critic(args.critic)
    torch.save(policy.state_dict(), policy_path)
    torch.save({"state_dict": critic.state_dict(), "dim": critic.dim,
                "hidden": 96}, critic_path)

    eval_seeds = [args.eval_seed_base + i for i in range(args.eval_episodes)]
    print(f"tag={args.tag}  {args.iterations} iterations x {args.episodes} "
          f"episodes at {args.sims} sims\nwriting to {directory}\n", flush=True)

    started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        fraction = (iteration - 1) / max(1, args.iterations - 1)
        temperature = args.temperature
        if args.temperature_final is not None:
            temperature += fraction * (args.temperature_final - args.temperature)

        seeds = [args.seed_base + iteration * 100_000 + i
                 for i in range(args.episodes)]
        policy_seeds = [iteration * 1_000_003 + i for i in range(args.episodes)]

        collect_config = CollectConfig(
            policy=policy_path, critic=critic_path, sims=args.sims,
            ascension=args.ascension, temperature=temperature)
        collect_started = time.perf_counter()
        episodes = collect(collect_config, seeds, policy_seeds, args.workers)
        collect_seconds = time.perf_counter() - collect_started

        floors = np.array([e["final_floor"] for e in episodes], dtype=np.float32)
        record = {
            "iteration": iteration,
            "temperature": temperature,
            "sampled_floor": float(floors.mean()),
            "sampled_sem": float(floors.std() / np.sqrt(len(floors))),
            "sampled_wins": sum(1 for e in episodes
                                if e["outcome"] == "PLAYER_VICTORY"),
            "transitions": int(sum(len(e["actions"]) for e in episodes)),
            "collect_seconds": collect_seconds,
        }

        update_config = UpdateConfig(
            epochs=args.epochs, minibatch=args.minibatch, clip=args.clip,
            target_kl=args.target_kl, entropy_coef=args.entropy_coef,
            lr=args.lr, temperature=temperature, train_trunk=args.train_trunk)
        print(f"iteration {iteration}: sampled floor "
              f"{record['sampled_floor']:.2f} +/- {record['sampled_sem']:.2f}, "
              f"{record['transitions']} transitions, "
              f"collect {collect_seconds:.0f}s", flush=True)

        pool = None
        if args.update_workers > 1:
            from .ppo_parallel import GradientPool

            # Workers hold the batch for the whole update, so it goes to disk
            # once per iteration rather than into every per-step message.
            batch_path = os.path.join(directory, "batch_current.pt")
            torch.save({"config": collect_config.__dict__,
                        "episodes": episodes}, batch_path)
            # ProcessPoolExecutor spawns lazily, so worker start-up and the
            # first batch load land inside the first minibatch rather than
            # here; update_seconds carries them.
            pool = GradientPool(policy_path, batch_path, update_config,
                                args.update_workers)
        try:
            record.update(ppo_update(
                policy, critic, episodes, update_config, pool=pool,
                # Once per run is enough: the reduction is either right or it
                # is not, and it does not depend on the iteration.
                verify_gradients=(iteration == 1 and pool is not None)))
        finally:
            if pool is not None:
                pool.shutdown()

        torch.save(policy.state_dict(), policy_path)
        torch.save({"state_dict": critic.state_dict(), "dim": critic.dim,
                    "hidden": 96}, critic_path)
        if args.keep_batches:
            torch.save({"config": collect_config.__dict__, "episodes": episodes},
                       os.path.join(directory, f"batch_iter{iteration}.pt"))

        if args.eval_every and (iteration % args.eval_every == 0
                                or iteration == args.iterations):
            evaluation = greedy_evaluation(
                policy_path, critic_path, eval_seeds, args.sims,
                args.ascension, args.workers)
            floors_list = evaluation.pop("greedy_floors")
            record.update(evaluation)
            torch.save(policy.state_dict(),
                       os.path.join(directory, f"policy_iter{iteration}.pt"))
            np.save(os.path.join(directory, f"greedy_floors_iter{iteration}.npy"),
                    np.asarray(floors_list, dtype=np.float32))
            print(f"  greedy eval: floor {evaluation['greedy_floor']:.2f} "
                  f"+/- {evaluation['greedy_sem']:.2f}, "
                  f"wins {evaluation['greedy_wins']}/{len(eval_seeds)}",
                  flush=True)

        record["elapsed_seconds"] = time.perf_counter() - started
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"  iteration {iteration} done in "
              f"{record['elapsed_seconds']:.0f}s total\n", flush=True)


if __name__ == "__main__":
    main()
