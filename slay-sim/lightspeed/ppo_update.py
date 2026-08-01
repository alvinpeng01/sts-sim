"""The PPO update: clipped policy surrogate, entropy bonus, critic refit.

Consumes a batch from `ppo_collect.py` and returns an updated policy and critic
plus the diagnostics an RL run is steered by.

Choices this file makes, each forced by something measured:

* **The trunk is frozen by default.**  Two reasons, not one.  It is 40% cheaper
  per transition (7.77 ms against 12.89 on this machine), and it keeps the
  critic's cached `state` features valid -- collection stores them, so a moving
  trunk would silently invalidate every stored feature mid-update.  v32 is also
  on record unfreezing the trunk and overfitting 4,008 rows; that argument is
  weaker here, since each iteration brings ~15,700 fresh transitions, so
  `--train-trunk` exists for when it is worth re-testing.
* **Single-threaded torch.**  Measured faster than six threads (7.77 ms against
  9.75) -- the model is small enough that intra-op parallelism is pure overhead.
  Parallelism belongs at the process level, which is where collection puts it.
* **A tight KL budget.**  95% of this policy's decisions are settled by under
  half a logit and the median margin is 0.129 nats
  ([07-known-issues.md](../../docs/07-known-issues.md)), so a parameter step
  that a normally-configured PPO would consider small can reorder the argmax
  everywhere.  `--target-kl` defaults to 0.01 and stops the epoch loop early
  rather than clipping alone.
* **Advantages are normalized over the whole iteration**, never per episode --
  per-episode normalization would erase exactly the between-run differences the
  advantage is supposed to express.

Run from slay-sim/ on a saved batch:
    python -m lightspeed.ppo_update --batch runs/ppo_batch_iter0.pt --dry-run
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

TRUNK_PREFIXES = (
    "card.", "relic.", "potion.", "room.", "coord.", "fixed.", "upgrade.",
    "counter.", "map_edges.", "encoder.",
)


@dataclass
class UpdateConfig:
    epochs: int = 4
    minibatch: int = 256
    clip: float = 0.2
    target_kl: float = 0.01
    entropy_coef: float = 0.005
    lr: float = 1e-4
    critic_lr: float = 3e-4
    critic_epochs: int = 4
    max_grad_norm: float = 1.0
    temperature: float = 0.2
    train_trunk: bool = False


def flatten(batch: list[dict]) -> dict:
    """One iteration's transitions, with advantages normalized across all runs."""
    observations = [o for episode in batch for o in episode["observations"]]
    stacked = {
        key: np.concatenate([episode[key] for episode in batch])
        for key in ("states", "actions", "log_probs", "values", "returns",
                    "advantages")
    }
    advantages = stacked["advantages"]
    stacked["advantages"] = (
        (advantages - advantages.mean()) / (advantages.std() + 1e-8))
    stacked["observations"] = observations
    return stacked


def assign_gradient(parameters, vector: torch.Tensor) -> None:
    """Write a flattened gradient back onto `.grad`, in `parameters` order.

    The same order `ppo_parallel.trainable_parameters` builds the vector with;
    a mismatch here would scramble gradients across tensors silently, which is
    what the gradient check exists to catch.
    """
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = vector[offset:offset + count].view_as(parameter).clone()
        offset += count
    if offset != vector.numel():
        raise RuntimeError(
            f"gradient vector has {vector.numel()} elements, "
            f"parameters need {offset}")


def set_trainable(policy: nn.Module, train_trunk: bool) -> tuple[int, int]:
    trainable = total = 0
    for name, parameter in policy.named_parameters():
        if not train_trunk and name.startswith(TRUNK_PREFIXES):
            parameter.requires_grad_(False)
        total += parameter.numel()
        trainable += parameter.numel() if parameter.requires_grad else 0
    return trainable, total


def update_critic(critic, states: np.ndarray, scalars: np.ndarray,
                  returns: np.ndarray, config: UpdateConfig,
                  holdout: float = 0.1) -> dict:
    """Refit V(s), keeping the update only if held-out MSE actually improves.

    A critic that degrades does not announce itself: it quietly biases every
    advantage in every later iteration. So the incumbent weights are one of the
    candidates, the comparison is on transitions the refit never saw, and the
    best of them is what survives. Weight decay is off -- targets here sit near
    -0.99 with a spread of ~0.04, and decay pulls the output bias toward zero,
    which is most of what went wrong the first time this was tried.
    """
    state_t = torch.from_numpy(states)
    scalar_t = torch.from_numpy(scalars)
    return_t = torch.from_numpy(returns)

    cut = max(1, int(len(state_t) * (1.0 - holdout)))
    permutation = torch.randperm(len(state_t))
    train_index, val_index = permutation[:cut], permutation[cut:]

    def val_mse(module) -> float:
        with torch.inference_mode():
            return float(nn.functional.mse_loss(
                module(state_t[val_index], scalar_t[val_index]),
                return_t[val_index]))

    before = val_mse(critic)
    best = before
    best_state = {k: v.clone() for k, v in critic.state_dict().items()}
    optimizer = torch.optim.AdamW(critic.parameters(), lr=config.critic_lr,
                                  weight_decay=0.0)
    for _ in range(config.critic_epochs):
        order = train_index[torch.randperm(len(train_index))]
        for start in range(0, len(order), 1024):
            index = order[start:start + 1024]
            loss = nn.functional.mse_loss(
                critic(state_t[index], scalar_t[index]), return_t[index])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(),
                                           config.max_grad_norm)
            optimizer.step()
        score = val_mse(critic)
        if score < best:
            best = score
            best_state = {k: v.clone() for k, v in critic.state_dict().items()}
    critic.load_state_dict(best_state)

    with torch.inference_mode():
        prediction = critic(state_t, scalar_t)
        residual = float(((return_t - prediction) ** 2).sum())
        total = float(((return_t - return_t.mean()) ** 2).sum())
    return {"critic_val_mse_before": before, "critic_val_mse_after": best,
            "critic_kept_update": best < before,
            "critic_explained_variance": 1.0 - residual / max(1e-9, total)}


def ppo_update(policy, critic, batch: list[dict], config: UpdateConfig,
               pool=None, verify_gradients: bool = False) -> dict:
    """One PPO update. With `pool`, gradients are summed across processes.

    The parallel path is numerically the same computation, not an approximation:
    each shard divides its loss by the FULL minibatch size, so summing shards
    reproduces the single-process mean. `verify_gradients` proves that on the
    first minibatch rather than trusting it.
    """
    from .run_critic import scalars_from_obs

    data = flatten(batch)
    observations = data["observations"]
    actions = torch.from_numpy(data["actions"])
    old_log_probs = torch.from_numpy(data["log_probs"])
    advantages = torch.from_numpy(data["advantages"])
    steps = len(observations)

    trainable, total = set_trainable(policy, config.train_trunk)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=1e-4)
    print(f"  policy trainable {trainable}/{total} "
          f"({100.0 * trainable / total:.1f}%), {steps} transitions", flush=True)

    if verify_gradients and pool is not None:
        from .ppo_parallel import verify_against_single_process

        verify_against_single_process(
            policy, pool, config, data,
            torch.randperm(steps)[:min(64, steps)].tolist())

    metrics = {"kl": 0.0, "clip_fraction": 0.0, "policy_loss": 0.0,
               "entropy": 0.0, "epochs_run": 0, "stopped_early": False}
    started = time.perf_counter()
    for epoch in range(config.epochs):
        order = torch.randperm(steps)
        epoch_kl, epoch_clip, epoch_loss, epoch_entropy, batches = 0.0, 0.0, 0.0, 0.0, 0
        for start in range(0, steps, config.minibatch):
            index = order[start:start + config.minibatch]
            optimizer.zero_grad()
            if pool is not None:
                weights = torch.nn.utils.parameters_to_vector(parameters)
                gradient, shard = pool.gradient(weights.detach(),
                                                index.tolist())
                assign_gradient(parameters, gradient)
                kl_sum, clip_sum = shard["kl"], shard["clip"]
                loss_sum, entropy_sum = shard["loss"], shard["entropy"]
            else:
                kl_sum = clip_sum = loss_sum = entropy_sum = 0.0
                for position in index.tolist():
                    logits, _ = policy(observations[position])
                    scaled = logits / max(1e-6, config.temperature)
                    distribution = torch.distributions.Categorical(logits=scaled)
                    log_prob = distribution.log_prob(actions[position])
                    ratio = torch.exp(log_prob - old_log_probs[position])
                    advantage = advantages[position]
                    unclipped = ratio * advantage
                    clipped = torch.clamp(
                        ratio, 1.0 - config.clip, 1.0 + config.clip) * advantage
                    entropy = distribution.entropy()
                    # Mean over the minibatch, accumulated one transition at a
                    # time because the action count varies per state and cannot
                    # be padded into one tensor without changing the softmax.
                    loss = (-torch.min(unclipped, clipped)
                            - config.entropy_coef * entropy) / len(index)
                    loss.backward()
                    with torch.no_grad():
                        # Schulman's low-variance KL estimator; always >= 0.
                        log_ratio = log_prob - old_log_probs[position]
                        kl_sum += float(torch.exp(log_ratio) - 1.0 - log_ratio)
                        clip_sum += float(
                            (ratio - 1.0).abs() > config.clip)
                        loss_sum += float(-torch.min(unclipped, clipped))
                        entropy_sum += float(entropy)
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            optimizer.step()
            epoch_kl += kl_sum / len(index)
            epoch_clip += clip_sum / len(index)
            epoch_loss += loss_sum / len(index)
            epoch_entropy += entropy_sum / len(index)
            batches += 1

        metrics.update(kl=epoch_kl / batches, clip_fraction=epoch_clip / batches,
                       policy_loss=epoch_loss / batches,
                       entropy=epoch_entropy / batches, epochs_run=epoch + 1)
        print(f"  epoch {epoch + 1}: KL {metrics['kl']:.5f}  "
              f"clip {metrics['clip_fraction']:.3f}  "
              f"entropy {metrics['entropy']:.3f}  "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
        if metrics["kl"] > config.target_kl:
            metrics["stopped_early"] = True
            print(f"  early stop: KL {metrics['kl']:.5f} "
                  f"exceeded target {config.target_kl}", flush=True)
            break

    scalars = np.stack([scalars_from_obs(o) for o in observations])
    metrics.update(update_critic(critic, data["states"], scalars,
                                 data["returns"], config))
    metrics["update_seconds"] = time.perf_counter() - started
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="runs/ppo_batch_iter0.pt")
    parser.add_argument("--policy", default=None,
                        help="defaults to the batch's own collection policy")
    parser.add_argument("--critic", default=None)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-trunk", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes summing gradients per minibatch; "
                             "1 keeps the single-process path")
    parser.add_argument("--verify-gradients", action="store_true",
                        help="check one minibatch against the "
                             "single-process gradient before training")
    parser.add_argument("--episodes", type=int, default=0,
                        help="use only the first N episodes of the batch")
    parser.add_argument("--dry-run", action="store_true",
                        help="run the update but write nothing")
    parser.add_argument("--out-policy", default=None)
    parser.add_argument("--out-critic", default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(0)

    from .eval_whole_run_policy import load_policy
    from .run_critic import load as load_critic

    payload = torch.load(args.batch, weights_only=False, map_location="cpu")
    episodes = payload["episodes"]
    if args.episodes:
        episodes = episodes[:args.episodes]
    collection = payload["config"]
    policy = load_policy(args.policy or collection["policy"], torch.device("cpu"))
    critic = load_critic(args.critic or collection["critic"])

    config = UpdateConfig(
        epochs=args.epochs, minibatch=args.minibatch, clip=args.clip,
        target_kl=args.target_kl, entropy_coef=args.entropy_coef, lr=args.lr,
        temperature=collection["temperature"], train_trunk=args.train_trunk)

    pool = None
    if args.workers > 1:
        from .ppo_parallel import GradientPool

        # Workers load the batch once from disk, so a step ships only weights
        # out and gradients back. When --episodes trims the batch, they must
        # see the same trimmed batch the parent flattened.
        batch_path = args.batch
        if args.episodes:
            batch_path = args.batch + f".first{args.episodes}.pt"
            torch.save({"config": collection, "episodes": episodes}, batch_path)
        pool = GradientPool(args.policy or collection["policy"], batch_path,
                            config, args.workers)

    try:
        metrics = ppo_update(policy, critic, episodes, config, pool=pool,
                             verify_gradients=args.verify_gradients)
    finally:
        if pool is not None:
            pool.shutdown()
    print("\n" + "\n".join(f"{k:>28}: {v}" for k, v in metrics.items()))

    if not args.dry_run and args.out_policy:
        torch.save(policy.state_dict(), args.out_policy)
        print(f"wrote {args.out_policy}")
    if not args.dry_run and args.out_critic:
        torch.save({"state_dict": critic.state_dict(), "dim": critic.dim,
                    "hidden": 96}, args.out_critic)
        print(f"wrote {args.out_critic}")


if __name__ == "__main__":
    main()
