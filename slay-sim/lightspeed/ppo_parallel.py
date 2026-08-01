"""Process-parallel gradient accumulation for the PPO update.

The update costs ~7.8 ms per transition per epoch against 3.3 ms to generate
one, so with collection already parallel the update is what sets the wall clock:
122 s per epoch over a 15,700-transition iteration, single process.

Structure.  Each worker loads the iteration's batch ONCE at start-up and keeps
it, so a step ships only the flattened trainable weights out and the summed
gradient back -- 5.2 MB each way per worker, against gigabytes if observations
moved per step.  The parent sums the shards, applies one optimizer step, and
broadcasts the new weights on the next step.

Why the minibatch default rises to 2048 here.  Transfer cost is per *step*, not
per transition, so 61 small steps per epoch would spend more time pickling than
computing.  Eight larger steps keep the ratio right, and PPO is routinely run
with 4-8 minibatches per epoch.

**A wrong gradient reduction is invisible.**  It does not crash; it produces a
training curve that quietly does not move, which is indistinguishable from the
task being hard.  So `--verify-gradients` computes one minibatch both ways from
identical weights and compares the flattened vectors elementwise, and the loop
driver runs it once before the first real step.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch import nn

_STATE: dict = {}


def trainable_parameters(policy: nn.Module) -> list[nn.Parameter]:
    """Deterministic order, so a flattened vector means the same thing to all."""
    return [p for _, p in policy.named_parameters() if p.requires_grad]


def _init(policy_path: str, batch_path: str, config_json: str) -> None:
    import json

    from .eval_whole_run_policy import load_policy
    from .ppo_update import UpdateConfig, flatten, set_trainable

    torch.set_num_threads(1)
    config = UpdateConfig(**json.loads(config_json))
    policy = load_policy(policy_path, torch.device("cpu"))
    set_trainable(policy, config.train_trunk)
    payload = torch.load(batch_path, weights_only=False, map_location="cpu")
    data = flatten(payload["episodes"])
    _STATE.update(
        policy=policy, config=config, data=data,
        parameters=trainable_parameters(policy),
        actions=torch.from_numpy(data["actions"]),
        old_log_probs=torch.from_numpy(data["log_probs"]),
        advantages=torch.from_numpy(data["advantages"]))


def _grad_shard(job: tuple[np.ndarray, list[int], int]) -> tuple[np.ndarray, dict]:
    weights, indices, total = job
    policy = _STATE["policy"]
    config = _STATE["config"]
    parameters = _STATE["parameters"]
    observations = _STATE["data"]["observations"]
    actions, old_log_probs = _STATE["actions"], _STATE["old_log_probs"]
    advantages = _STATE["advantages"]

    torch.nn.utils.vector_to_parameters(
        torch.from_numpy(weights), parameters)
    for parameter in parameters:
        parameter.grad = None

    kl = clip = loss_total = entropy_total = 0.0
    for position in indices:
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
        # Divided by the FULL minibatch, not this shard, so summing the shards
        # reproduces the single-process mean exactly.
        loss = (-torch.min(unclipped, clipped)
                - config.entropy_coef * entropy) / total
        loss.backward()
        with torch.no_grad():
            log_ratio = log_prob - old_log_probs[position]
            kl += float(torch.exp(log_ratio) - 1.0 - log_ratio)
            clip += float((ratio - 1.0).abs() > config.clip)
            loss_total += float(-torch.min(unclipped, clipped))
            entropy_total += float(entropy)

    gradient = torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        for p in parameters]).numpy()
    return gradient, {"kl": kl, "clip": clip, "loss": loss_total,
                      "entropy": entropy_total, "count": len(indices)}


class GradientPool:
    """Sums per-shard gradients for one minibatch across worker processes."""

    def __init__(self, policy_path: str, batch_path: str, config,
                 workers: int):
        import json
        from dataclasses import asdict

        self.workers = workers
        self.executor = ProcessPoolExecutor(
            max_workers=workers, initializer=_init,
            initargs=(policy_path, batch_path, json.dumps(asdict(config))))

    def gradient(self, weights: torch.Tensor,
                 indices: list[int]) -> tuple[torch.Tensor, dict]:
        total = len(indices)
        shards = [s.tolist() for s in np.array_split(np.asarray(indices),
                                                     self.workers) if len(s)]
        payload = weights.detach().numpy()
        jobs = [(payload, shard, total) for shard in shards]
        summed = None
        metrics = {"kl": 0.0, "clip": 0.0, "loss": 0.0, "entropy": 0.0,
                   "count": 0}
        for gradient, shard_metrics in self.executor.map(_grad_shard, jobs):
            tensor = torch.from_numpy(gradient)
            summed = tensor if summed is None else summed + tensor
            for key in metrics:
                metrics[key] += shard_metrics[key]
        return summed, metrics

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)


def verify_against_single_process(policy, pool: "GradientPool", config,
                                  data: dict, indices: list[int],
                                  tolerance: float = 1e-5) -> float:
    """One minibatch, both ways, from identical weights."""
    parameters = trainable_parameters(policy)
    weights = torch.nn.utils.parameters_to_vector(parameters).detach().clone()

    actions = torch.from_numpy(data["actions"])
    old_log_probs = torch.from_numpy(data["log_probs"])
    advantages = torch.from_numpy(data["advantages"])
    for parameter in parameters:
        parameter.grad = None
    for position in indices:
        logits, _ = policy(data["observations"][position])
        scaled = logits / max(1e-6, config.temperature)
        distribution = torch.distributions.Categorical(logits=scaled)
        log_prob = distribution.log_prob(actions[position])
        ratio = torch.exp(log_prob - old_log_probs[position])
        advantage = advantages[position]
        clipped = torch.clamp(
            ratio, 1.0 - config.clip, 1.0 + config.clip) * advantage
        loss = (-torch.min(ratio * advantage, clipped)
                - config.entropy_coef * distribution.entropy()) / len(indices)
        loss.backward()
    serial = torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        for p in parameters])

    parallel, _ = pool.gradient(weights, indices)
    difference = float((serial - parallel).abs().max())
    scale = float(serial.abs().max())
    print(f"  gradient check: max |serial - parallel| {difference:.3e} "
          f"(gradient scale {scale:.3e})", flush=True)
    if difference > tolerance * max(1.0, scale):
        raise SystemExit(
            f"parallel gradient disagrees with single process by {difference:.3e}")
    for parameter in parameters:
        parameter.grad = None
    return difference
