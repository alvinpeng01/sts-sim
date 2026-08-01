"""PPO training for ActionScoringPolicy -- same environment/reward as
REINFORCE (train.py); the algorithmic difference is what actually targets
the instability REINFORCE showed empirically (Gremlin Gang's win rate
dropped from 72% to 40% between updates 400-600 in one extended run, with
nothing preventing a bad batch from pushing the policy somewhere worse):

  1. A value-function baseline (advantage = return - value(state)) instead
     of a batch-standardized raw return, reducing advantage variance.
  2. A clipped surrogate objective with multiple gradient epochs per
     collected batch, which bounds how far any single update can move the
     policy away from the one that collected the data.

Simplification: advantage = return - value(state) (Monte Carlo advantage),
not full GAE(lambda) -- skips lambda's bias/variance tradeoff tuning for a
simpler, still-correct implementation.

The update step is fully vectorized (see policy.score_actions_batched):
state encoding batches trivially (every step has one fixed-size state
vector); the variable-length legal-action set per step is handled by
padding to the batch's max action count and masking. An earlier version did
this with a per-step Python loop instead (the same reasoning score_actions()
already applies to a single state), which measured 21.7 vs REINFORCE's 103.9
eps/sec (~4.8x slower) -- purely that implementation choice, not anything
inherent to PPO. This version closes that gap.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.optim as optim

from .env import IroncladFightEnv
from .policy import ActionScoringPolicy, prep_obs
from .train import evaluate


@dataclass
class Step:
    state: torch.Tensor
    action_features: torch.Tensor
    card_idxs: torch.Tensor
    monster_idxs: torch.Tensor
    action_potion_idxs: torch.Tensor
    relic_idxs: torch.Tensor
    relic_mask: torch.Tensor
    potion_idxs: torch.Tensor
    potion_mask: torch.Tensor
    action_idx: int
    old_log_prob: float
    value: float
    # NOTE on why the parallel path doesn't just pickle Step objects directly:
    # measured torch.Tensor pickling at ~34x the cost of pickling the
    # equivalent numpy array (792ms vs 23ms for 2000 small-tensor triples --
    # isolated by profiling collect_batch_parallel: worker compute itself
    # was fast (~300ms), but pool.map()'s total wall time was ~4s, and a
    # no-op worker returning empty results confirmed pure dispatch is fast
    # (32ms) -- so the ~3.7s of unexplained overhead was RESULT transfer,
    # and swapping in a torch-vs-numpy pickling microbenchmark confirmed
    # why). See _step_to_numpy/_step_from_numpy below -- workers convert to
    # numpy before returning, the main process converts back after
    # collecting, so the expensive-to-pickle representation never actually
    # crosses the process boundary.
    reward: float
    return_: float = 0.0


def _step_to_numpy(s: Step) -> tuple:
    """Worker-side: strip a Step down to plain numpy/python types before it
    crosses the process boundary -- see the module-level NOTE on Step for
    why (torch.Tensor pickling measured ~34x slower than numpy for the
    same data)."""
    return (s.state.numpy(), s.action_features.numpy(), s.card_idxs.numpy(),
            s.monster_idxs.numpy(), s.action_potion_idxs.numpy(),
            s.relic_idxs.numpy(), s.relic_mask.numpy(),
            s.potion_idxs.numpy(), s.potion_mask.numpy(),
            s.action_idx, s.old_log_prob, s.value, s.reward, s.return_)


def _step_from_numpy(t: tuple) -> Step:
    """Main-process side: rebuild a real Step (torch tensors restored) from
    what a worker sent back. torch.from_numpy shares memory with the numpy
    array rather than copying -- fine here since nothing else holds a
    reference to that array afterward."""
    (state_np, af_np, ci_np, mi_np, api_np, ri_np, rm_np, pi_np, pm_np,
     action_idx, old_log_prob, value, reward, return_) = t
    return Step(
        state=torch.from_numpy(state_np), action_features=torch.from_numpy(af_np),
        card_idxs=torch.from_numpy(ci_np), monster_idxs=torch.from_numpy(mi_np),
        action_potion_idxs=torch.from_numpy(api_np),
        relic_idxs=torch.from_numpy(ri_np), relic_mask=torch.from_numpy(rm_np),
        potion_idxs=torch.from_numpy(pi_np), potion_mask=torch.from_numpy(pm_np),
        action_idx=action_idx,
        old_log_prob=old_log_prob, value=value, reward=reward, return_=return_,
    )


def _collect_one_episode(env: IroncladFightEnv, policy: ActionScoringPolicy,
                          gamma: float) -> Step:
    """One episode's worth of Steps with returns already filled in.
    Extracted so the sequential path (collect_batch) and the parallel
    worker path (_worker_collect_chunk) run the EXACT same collection
    logic -- a second hand-copied implementation for the parallel path
    would be exactly the kind of drift that silently changes what's being
    trained without anyone noticing."""
    obs = env.reset(seed=None)
    episode_steps: List[Step] = []
    done = False
    while not done:
        state, af, ci, mi, ri, rm, api, pi, pm = prep_obs(obs)
        state_emb = policy.encode_state(state, ri, rm, pi, pm)
        scores = policy.score_actions(state, af, ci, mi, api, ri, rm, pi, pm, state_emb=state_emb)
        probs = torch.softmax(scores, dim=-1)
        # validate_args=False: skips Categorical's constraint-checking on
        # every construction (profiled at a real ~14% of collect_batch's
        # total wall time -- constructions happen once per DECISION, i.e.
        # thousands of times per training update). Safe here specifically
        # because `probs` is always fresh softmax output (sums to 1,
        # non-negative by construction), never hand-assembled or subject to
        # the masked-fill -1e9 path (that's the logits-based construction
        # in ppo_update, same reasoning, see below).
        dist = torch.distributions.Categorical(probs=probs, validate_args=False)
        idx = dist.sample()
        log_prob = dist.log_prob(idx)
        value = policy.value_head(state_emb).squeeze(-1)

        action = obs["actions"][int(idx.item())]
        obs, reward, done, info = env.step(action)

        episode_steps.append(Step(
            state=state, action_features=af, card_idxs=ci, monster_idxs=mi, action_potion_idxs=api,
            relic_idxs=ri, relic_mask=rm, potion_idxs=pi, potion_mask=pm,
            action_idx=int(idx.item()), old_log_prob=float(log_prob.item()),
            value=float(value.item()), reward=reward,
        ))

    G = 0.0
    for s in reversed(episode_steps):
        G = s.reward + gamma * G
        s.return_ = G
    return episode_steps


def collect_batch(env: IroncladFightEnv, policy: ActionScoringPolicy,
                   episodes_per_update: int, gamma: float):
    steps: List[Step] = []
    batch_rewards = []
    with torch.no_grad():
        for _ in range(episodes_per_update):
            episode_steps = _collect_one_episode(env, policy, gamma)
            steps.extend(episode_steps)
            batch_rewards.append(sum(s.reward for s in episode_steps))
    return steps, batch_rewards


def _env_kwargs_from(env: IroncladFightEnv) -> dict:
    """Reverse-engineers constructor kwargs from a live env's own attributes
    (every one of these is set verbatim from the matching __init__ param),
    so a worker process can build an independent env that behaves
    identically without env.py needing to expose a separate 'give me my own
    config back' method just for this."""
    return dict(
        encounter=env.encounter_pool,
        extra_deck_cards=env.extra_deck_cards,
        player_hp=env.player_hp,
        deck_exclude=env.deck_exclude,
        deck_force_include=env.deck_force_include,
        encounter_resources=env.encounter_resources,
        upgrade_chance=env.upgrade_chance,
        starter_removals=env.starter_removals,
        ascension=env.ascension,
        encounter_weights=env.encounter_weights,
        deck_generator=env.deck_generator,
        relic_generator=env.relic_generator,
        relic_count=env.relic_count,
        potion_generator=env.potion_generator,
        potion_count=env.potion_count,
    )


# Per-process worker state, set up ONCE by _worker_init (a Pool
# initializer, runs once when each worker process starts, not once per
# task). Module-level globals are safe here specifically BECAUSE each
# worker is a separate OS process -- there's no cross-worker sharing to
# worry about, each process gets its own copy of this module's globals.
# Before this, _worker_collect_chunk rebuilt IroncladFightEnv AND
# ActionScoringPolicy from scratch on every single pool.map call (every
# training update, not just once per worker) -- avoidable overhead now
# that env construction doesn't need to happen again each round; only the
# policy's WEIGHTS actually change between updates.
_worker_env: Optional[IroncladFightEnv] = None
_worker_policy: Optional[ActionScoringPolicy] = None


def _worker_init(env_kwargs: dict) -> None:
    global _worker_env, _worker_policy
    # torch defaults to multi-threaded intra-op parallelism (6 threads on
    # this machine) PER PROCESS -- with N_WORKERS processes each doing that,
    # you get up to N_WORKERS*6 threads fighting over the actual core count,
    # which measured as the parallel path being dramatically SLOWER than
    # sequential (not a hang, just severe oversubscription) before this
    # fix. Each worker's forward passes are tiny (a handful of legal
    # actions at a time), so there's nothing to gain from intra-op
    # parallelism here anyway -- the real parallelism is having many
    # independent single-threaded processes, not multi-threaded ones.
    torch.set_num_threads(1)
    _worker_env = IroncladFightEnv(**env_kwargs)
    _worker_policy = ActionScoringPolicy()


def _worker_collect_chunk(args):
    """Runs in a worker process (module-level, not a closure/lambda --
    multiprocessing needs to pickle a reference to this function itself).
    Reuses the persistent env+policy _worker_init already built for this
    process -- only the policy's weights need updating each round."""
    state_dict, n_episodes, gamma, seed = args
    torch.manual_seed(seed)  # otherwise every worker samples identically each round
    _worker_policy.load_state_dict(state_dict)
    steps: List[Step] = []
    batch_rewards = []
    with torch.no_grad():
        for _ in range(n_episodes):
            episode_steps = _collect_one_episode(_worker_env, _worker_policy, gamma)
            steps.extend(episode_steps)
            batch_rewards.append(sum(s.reward for s in episode_steps))
    return [_step_to_numpy(s) for s in steps], batch_rewards


def collect_batch_parallel(env: IroncladFightEnv, policy: ActionScoringPolicy,
                            episodes_per_update: int, gamma: float, pool: "mp.pool.Pool",
                            n_workers: int, seed_counter: List[int]):
    """Same contract as collect_batch (returns (steps, batch_rewards) for
    the full requested episode count), but splits the episodes across
    `n_workers` OS processes. `pool` is created once by the caller and
    reused across every update -- spinning up a fresh process pool each
    update would burn most of the parallelism gain on process-startup
    overhead instead of actual simulation. `seed_counter` is a 1-element
    list used as a mutable int (so each call advances it) purely to give
    each worker-call a different torch random seed across updates -- without
    this, every update's workers would resample the exact same action
    sequences as the previous update's workers did."""
    per_worker = [episodes_per_update // n_workers] * n_workers
    for i in range(episodes_per_update % n_workers):
        per_worker[i] += 1
    per_worker = [n for n in per_worker if n > 0]

    # env_kwargs is NOT sent per-call anymore -- each worker built its own
    # persistent env once via _worker_init (see train_ppo's pool creation),
    # so only the policy's weights (which DO change every update) need to
    # cross the process boundary each round.
    state_dict = {k: v.clone() for k, v in policy.state_dict().items()}

    args = []
    for n_eps in per_worker:
        seed_counter[0] += 1
        args.append((state_dict, n_eps, gamma, seed_counter[0]))

    results = pool.map(_worker_collect_chunk, args)
    steps: List[Step] = []
    batch_rewards = []
    for worker_steps_np, worker_rewards in results:
        steps.extend(_step_from_numpy(t) for t in worker_steps_np)
        batch_rewards.extend(worker_rewards)
    return steps, batch_rewards


def _pad_batch(steps: List[Step]):
    """Collate a list of Steps (each with its own legal-action count) into
    padded/masked tensors for score_actions_batched()."""
    n = len(steps)
    max_actions = max(s.action_features.shape[0] for s in steps)
    action_dim = steps[0].action_features.shape[1]

    state_batch = torch.stack([s.state for s in steps])
    action_features_padded = torch.zeros(n, max_actions, action_dim)
    card_idxs_padded = torch.zeros(n, max_actions, dtype=torch.long)
    monster_idxs_padded = torch.zeros(n, max_actions, dtype=torch.long)
    action_potion_idxs_padded = torch.zeros(n, max_actions, dtype=torch.long)
    mask = torch.zeros(n, max_actions, dtype=torch.bool)
    chosen_idx = torch.zeros(n, dtype=torch.long)

    for i, s in enumerate(steps):
        n_a = s.action_features.shape[0]
        action_features_padded[i, :n_a] = s.action_features
        card_idxs_padded[i, :n_a] = s.card_idxs
        monster_idxs_padded[i, :n_a] = s.monster_idxs
        action_potion_idxs_padded[i, :n_a] = s.action_potion_idxs
        mask[i, :n_a] = True
        chosen_idx[i] = s.action_idx

    # relic_idxs/relic_mask/potion_idxs/potion_mask are already fixed-width
    # per step -- a plain stack, no padding needed, unlike the variable-
    # length per-step legal-action set above.
    relic_idxs_batch = torch.stack([s.relic_idxs for s in steps])
    relic_mask_batch = torch.stack([s.relic_mask for s in steps])
    potion_idxs_batch = torch.stack([s.potion_idxs for s in steps])
    potion_mask_batch = torch.stack([s.potion_mask for s in steps])

    return (state_batch, action_features_padded, card_idxs_padded, monster_idxs_padded, action_potion_idxs_padded,
            relic_idxs_batch, relic_mask_batch, potion_idxs_batch, potion_mask_batch, mask, chosen_idx)


def ppo_update(policy: ActionScoringPolicy, optimizer, steps: List[Step],
                epochs: int = 4, clip_eps: float = 0.2,
                value_coef: float = 0.5, entropy_coef: float = 0.01):
    (state_batch, af_padded, ci_padded, mi_padded, api_padded,
     relic_idxs_batch, relic_mask_batch, potion_idxs_batch, potion_mask_batch, mask, chosen_idx) = _pad_batch(steps)
    old_log_probs = torch.tensor([s.old_log_prob for s in steps], dtype=torch.float32)
    returns = torch.tensor([s.return_ for s in steps], dtype=torch.float32)
    old_values = torch.tensor([s.value for s in steps], dtype=torch.float32)

    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _epoch in range(epochs):
        optimizer.zero_grad()
        scores, state_emb = policy.score_actions_batched(
            state_batch, af_padded, ci_padded, mi_padded, api_padded,
            relic_idxs_batch, relic_mask_batch, potion_idxs_batch, potion_mask_batch, mask)
        # logits, not probs: softmax(-1e9) correctly underflows to exact 0.0
        # at float32 without ever computing 0*inf, which is what produces
        # NaN -- verified empirically before trusting this in a real run.
        # validate_args=False for the same reason as collect_batch's
        # Categorical -- real profiled overhead, safe since `scores` is
        # always finite by construction (masked-fill uses -1e9, never -inf).
        dist = torch.distributions.Categorical(logits=scores, validate_args=False)
        new_log_probs = dist.log_prob(chosen_idx)
        entropy = dist.entropy()
        values = policy.value_head(state_emb).squeeze(-1)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = ((values - returns) ** 2).mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

        loss.backward()
        optimizer.step()


def train_ppo(env: IroncladFightEnv, policy: ActionScoringPolicy, updates: int = 150,
              episodes_per_update: int = 16, lr: float = 3e-4, gamma: float = 0.99,
              epochs_per_update: int = 4, clip_eps: float = 0.2,
              checkpoint_every: int = 0, checkpoint_eval_n: int = 100,
              n_workers: int = 1, pool: Optional["mp.pool.Pool"] = None):
    """Returns (history, best_state_dict) if checkpoint_every > 0, else
    just history -- same convention as train.train().

    n_workers > 1 switches rollout collection to collect_batch_parallel
    (see its docstring) -- everything else (the PPO update itself, eval,
    checkpointing) stays single-process, since only collection was ever the
    bottleneck (the update step was already vectorized). If `pool` isn't
    passed, one is created here (with _worker_init as its initializer, so
    each worker builds its persistent env+policy once -- see
    _worker_init/_worker_collect_chunk) and closed at the end of this call;
    pass your own (already-open) Pool if you're calling train_ppo repeatedly
    (e.g. from a driver script doing its own chunked reporting loop) so the
    worker processes aren't torn down and respawned between calls -- that
    respawn cost is exactly the overhead this feature exists to avoid. If
    you do pass your own Pool, YOU'RE responsible for having created it
    with initializer=_worker_init, initargs=(env_kwargs,) -- workers no
    longer accept env_kwargs per-task, only via that initializer."""
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    history = []
    best_reward = float("-inf")
    best_state = None
    seed_counter = [0]

    owns_pool = n_workers > 1 and pool is None
    if owns_pool:
        pool = mp.Pool(n_workers, initializer=_worker_init, initargs=(_env_kwargs_from(env),))

    try:
        for update in range(updates):
            if n_workers > 1:
                steps, batch_rewards = collect_batch_parallel(
                    env, policy, episodes_per_update, gamma, pool, n_workers, seed_counter)
            else:
                steps, batch_rewards = collect_batch(env, policy, episodes_per_update, gamma)
            ppo_update(policy, optimizer, steps, epochs=epochs_per_update, clip_eps=clip_eps)
            history.append(float(np.mean(batch_rewards)))

            if checkpoint_every and (update + 1) % checkpoint_every == 0:
                _, _, eval_reward = evaluate(env, policy, n=checkpoint_eval_n)
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    best_state = {k: v.clone() for k, v in policy.state_dict().items()}
    finally:
        if owns_pool:
            pool.close()
            pool.join()

    if checkpoint_every:
        if best_state is not None:
            policy.load_state_dict(best_state)
        return history, best_state
    return history
