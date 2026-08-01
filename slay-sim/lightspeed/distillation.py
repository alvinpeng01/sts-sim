"""Bounded, periodic search-distillation: trains the fast blind policy to
imitate what a search actually decided, instead of only ever learning from
raw PPO policy-gradient on environment reward.

TEACHER: expectimax_search.py's native, network-free MCTS (sts.run_mcts_search),
NOT az_search.py's NN-guided PUCT search this module originally used. Switched
after a real, measured comparison (this session, see expectimax_search.py's own
module docstring) found the no-NN heuristic-rollout search beating the PUCT
search 100% vs 0-20% on Time Eater/Donu & Deca at matched compute -- unsurprising
given the network is trained on a small fraction of the compute AlphaZero-style
methods assume, so guiding search with an undertrained network's priors wasn't
paying for itself. Several further correctness/quality fixes landed on the
expectimax search this same session (defense-vs-attack scoring, Strength/Weak/
Vulnerable damage prediction, a Haste-penalty gating bug, and -- the largest
single lever -- a loss-progress-credit terminal evaluation found by reading
Silver Automaton's actual source) that measurably improved it further (Time
Eater roughly 0%->30-40%, Donu & Deca roughly 40%->75-80% on the session's own
test deck). Because expectimax's search never reads the policy network at all
(no priors, no learned value head), collecting distillation targets needs no
`policy` argument and no state_dict IPC to worker processes -- a genuine
simplification, not just a drop-in swap.

This is deliberately NOT full AlphaZero self-play (search driving every
training rollout, continuously) -- measured cost ruled that out. Search is
far slower per fight than the blind policy; using it for every rollout would
turn an hours-long training run into weeks. Instead this collects a SMALL,
BOUNDED batch of search-driven episodes periodically (see
train_distillation_v5.py's DISTILL_EVERY_CHUNKS), and does a supervised
fine-tuning step against just that batch: the policy head is trained to
match search's visit-count distribution (the AlphaZero-standard training
target), and the value head against the real episode outcome, same target
ppo_update already uses.

The point of this: bake whatever tactical improvement search demonstrates
directly into the policy's WEIGHTS, so the deployed autonomous bot never
needs to run search at inference time -- it plays at the fast blind-policy
speed, having already learned from search's judgment during training. Known
caveat, carried over from the PUCT-teacher version and not yet re-measured
against the new teacher: search may still have blind spots on some long/
multi-phase fights, in which case distillation rounds would occasionally
include imitation targets from a decision-maker that's weak on exactly
that fight. Mitigated by keeping distillation a small, occasional nudge
(DISTILL_LR is low, blended with the much larger volume of ordinary PPO
updates on real environment reward, never a replacement for them).
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from . import expectimax_search as ex
from .cards import OTHER_CARD_INDEX
from .monsters import OTHER_MONSTER_INDEX
from .policy import ActionScoringPolicy
from .potion_features import EMPTY_POTION_INDEX
from .search_config import apply_search_config


@dataclass
class DistillStep:
    state: torch.Tensor
    action_features: torch.Tensor
    card_idxs: torch.Tensor
    monster_idxs: torch.Tensor
    action_potion_idxs: torch.Tensor
    relic_idxs: torch.Tensor
    relic_mask: torch.Tensor
    potion_idxs: torch.Tensor
    potion_mask: torch.Tensor
    pi_target: torch.Tensor  # search's visit-count distribution over legal actions, sums to 1
    reward: float
    return_: float = 0.0


def collect_distillation_episode(env, n_simulations: int, gamma: float,
                                  seed=None, n_trees: int = 1) -> List[DistillStep]:
    """One episode driven entirely by expectimax_search's native, network-
    free MCTS, recording its own visit distribution at each decision as the
    imitation target -- not just the final outcome, unlike ppo.py's rollout
    collection. `obs` (from env.reset()/env.step()) and the search's
    internal visit counts are guaranteed to index the SAME legal-action
    ordering: both ultimately call bc.get_legal_actions() -- respectively
    on env.bc directly and on a fresh copy of it inside the search -- a
    pure, deterministic function of game state, so a copy and its original
    always enumerate identically. No `policy` argument, unlike this
    function's az_search.py-PUCT-teacher predecessor -- see this module's
    own docstring for why (expectimax's search never reads the network).
    n_trees > 1 uses root_parallel_search (real OS threads, ~1.9x faster
    wall-clock measured this session with no accuracy cost) instead of a
    single tree; safe to use here since it doesn't touch policy state at
    all, so it composes cleanly with this function's own multiprocess
    worker parallelism (see collect_distillation_batch_parallel) -- each
    worker PROCESS can still spawn its own root-parallel THREADS."""
    obs = env.reset(seed=seed)
    steps: List[DistillStep] = []
    done = False
    while not done:
        relic_idxs = torch.as_tensor(obs["relic_idxs"], dtype=torch.long)
        relic_mask = torch.as_tensor(obs["relic_mask"], dtype=torch.bool)
        if n_trees > 1:
            action, visits = ex.root_parallel_search(env.bc, n_simulations=n_simulations, n_trees=n_trees)
        else:
            action, visits = ex.choose_action_native(env.bc, n_simulations=n_simulations)

        state = torch.as_tensor(obs["state"], dtype=torch.float32)
        action_features = torch.as_tensor(np.stack(obs["action_features"]), dtype=torch.float32)
        card_idxs = torch.as_tensor(
            [OTHER_CARD_INDEX if c is None else c for c in obs["action_card_idx"]], dtype=torch.long,
        )
        monster_idxs = torch.as_tensor(
            [OTHER_MONSTER_INDEX if m is None else m for m in obs["action_monster_idx"]], dtype=torch.long,
        )
        action_potion_idxs = torch.as_tensor(
            [EMPTY_POTION_INDEX if p is None else p for p in obs["action_potion_idx"]], dtype=torch.long,
        )
        potion_idxs = torch.as_tensor(obs["potion_idxs"], dtype=torch.long)
        potion_mask = torch.as_tensor(obs["potion_mask"], dtype=torch.bool)
        pi_target = torch.as_tensor(visits, dtype=torch.float32)
        pi_target = pi_target / pi_target.sum().clamp(min=1e-8)

        obs, reward, done, info = env.step(action)
        steps.append(DistillStep(
            state=state, action_features=action_features, card_idxs=card_idxs,
            monster_idxs=monster_idxs, action_potion_idxs=action_potion_idxs,
            relic_idxs=relic_idxs, relic_mask=relic_mask,
            potion_idxs=potion_idxs, potion_mask=potion_mask,
            pi_target=pi_target, reward=reward,
        ))

    G = 0.0
    for s in reversed(steps):
        G = s.reward + gamma * G
        s.return_ = G
    return steps


def collect_distillation_batch(env, n_episodes: int, n_simulations: int,
                                gamma: float = 0.99, seed_offset: int = 0, n_trees: int = 1) -> List[DistillStep]:
    """Sequential (not parallelized) -- this batch is meant to be small and
    occasional (see train_distillation_v5.py), so the added complexity of
    a worker pool for search specifically wasn't worth it for a periodic,
    bounded cost rather than the continuous cost full self-play would be.
    No `policy` argument -- see collect_distillation_episode's docstring."""
    all_steps: List[DistillStep] = []
    for i in range(n_episodes):
        all_steps.extend(collect_distillation_episode(env, n_simulations, gamma, seed=seed_offset + i, n_trees=n_trees))
    return all_steps


# --- multiprocessing-parallel collection -----------------------------------
#
# Episode-level parallelism, same shape as az_search.py's
# evaluate_with_search_parallel (each episode's search-driven collection is
# fully independent of every other episode's) -- this is exactly the
# "worker pool for search" the docstring above judged not worth it when
# distillation batches were small and occasional; now that az_search.py's
# own evaluate_with_search_parallel proved the pattern out (4.61x measured
# on this machine, n=60), the marginal cost of adding it here is just this
# function, not a new mechanism.

_distill_worker_env = None


def _distill_worker_init(env_kwargs: dict, search_params: dict = None) -> None:
    global _distill_worker_env
    # Same oversubscription fix as ppo.py's _worker_init -- N_WORKERS
    # processes each defaulting to multi-threaded intra-op parallelism would
    # fight over the actual core count. No policy instance needed in workers
    # any more (see this module's own docstring): expectimax's search never
    # reads the network, so there's nothing to load a state_dict into here
    # -- a genuine simplification over the az_search.py-PUCT-teacher
    # version, not just a smaller torch.set_num_threads(1) concern.
    torch.set_num_threads(1)
    from .env import IroncladFightEnv
    _distill_worker_env = IroncladFightEnv(**env_kwargs)
    if search_params is not None:
        # Each worker is its OWN process with its own independent copy of
        # the C++ module's g_params (see slaythespire.cpp's
        # set_search_params docstring on why this matters: g_params is
        # unlocked global mutable state, so setting it ONCE here at
        # process init -- never touched again for this worker's lifetime
        # -- is the safe pattern, not calling it repeatedly per-task or
        # from multiple threads).
        apply_search_config(search_params)


def _distill_step_to_numpy(s: DistillStep) -> tuple:
    """Worker-side: strip a DistillStep to plain numpy/python types before
    it crosses the process boundary -- same reasoning as ppo.py's
    _step_to_numpy (torch.Tensor pickling measured ~34x slower than numpy
    for the same data), just with pi_target/relic_idxs/relic_mask as the
    extra fields DistillStep has that Step doesn't."""
    return (s.state.numpy(), s.action_features.numpy(), s.card_idxs.numpy(),
            s.monster_idxs.numpy(), s.action_potion_idxs.numpy(),
            s.relic_idxs.numpy(), s.relic_mask.numpy(),
            s.potion_idxs.numpy(), s.potion_mask.numpy(),
            s.pi_target.numpy(), s.reward, s.return_)


def _distill_step_from_numpy(t: tuple) -> DistillStep:
    """Main-process side: rebuild a real DistillStep (torch tensors
    restored) from what a worker sent back."""
    (state_np, af_np, ci_np, mi_np, api_np, ri_np, rm_np, pi_slots_np, pm_np,
     pi_target_np, reward, return_) = t
    return DistillStep(
        state=torch.from_numpy(state_np), action_features=torch.from_numpy(af_np),
        card_idxs=torch.from_numpy(ci_np), monster_idxs=torch.from_numpy(mi_np),
        action_potion_idxs=torch.from_numpy(api_np),
        relic_idxs=torch.from_numpy(ri_np), relic_mask=torch.from_numpy(rm_np),
        potion_idxs=torch.from_numpy(pi_slots_np), potion_mask=torch.from_numpy(pm_np),
        pi_target=torch.from_numpy(pi_target_np), reward=reward, return_=return_,
    )


def _distill_worker_collect(args):
    n_episodes, n_simulations, gamma, seed_offset, n_trees = args
    steps = collect_distillation_batch(_distill_worker_env, n_episodes, n_simulations, gamma, seed_offset, n_trees)
    return [_distill_step_to_numpy(s) for s in steps]


def collect_distillation_batch_parallel(env, n_episodes: int, n_simulations: int,
                                         gamma: float = 0.99, seed_offset: int = 0,
                                         n_workers: int = 6, n_trees: int = 1,
                                         search_params: dict = None) -> List[DistillStep]:
    """Same contract/result as collect_distillation_batch, split across
    n_workers processes by episode. A fresh pool per call (not reused across
    the training run the way ppo.py's collect_batch_parallel reuses one) --
    matches evaluate_with_search_parallel's choice for the same reason:
    distillation batches are periodic/occasional (see DISTILL_EVERY_CHUNKS),
    not called every single update, so pool-creation overhead is negligible
    relative to the search cost itself, and not worth the added bookkeeping
    of threading a persistent pool through train_distillation_v5.py's loop.
    No `policy`/state_dict IPC any more -- see this module's own docstring.
    n_trees > 1 has each of the n_workers PROCESSES additionally spawn its
    own root-parallel THREADS per decision (process-level parallelism
    across episodes, thread-level parallelism within each episode's search)
    -- composes cleanly since root_parallel_search touches no shared state.
    search_params, if given, is applied via set_search_params in EACH
    worker process at init (see _distill_worker_init) -- e.g. the CMA-ES-
    tuned values from tune_search_cma.py/tuned_search_params.json, instead
    of leaving every worker on the C++ module's own compiled-in defaults."""
    from .ppo import _env_kwargs_from
    env_kwargs = _env_kwargs_from(env)

    counts = [n_episodes // n_workers + (1 if i < n_episodes % n_workers else 0) for i in range(n_workers)]
    args_list = []
    offset = seed_offset
    for c in counts:
        if c > 0:
            args_list.append((c, n_simulations, gamma, offset, n_trees))
        offset += c

    with mp.Pool(len(args_list), initializer=_distill_worker_init, initargs=(env_kwargs, search_params)) as pool:
        results = pool.map(_distill_worker_collect, args_list)

    all_steps: List[DistillStep] = []
    for worker_steps in results:
        all_steps.extend(_distill_step_from_numpy(t) for t in worker_steps)
    return all_steps


def _pad_distill_batch(steps: List[DistillStep]):
    """Same collate pattern as ppo.py's _pad_batch, plus padding pi_target
    (zeros at padded positions, so they contribute nothing to the
    cross-entropy regardless of the corresponding logits)."""
    n = len(steps)
    max_actions = max(s.action_features.shape[0] for s in steps)
    action_dim = steps[0].action_features.shape[1]

    state_batch = torch.stack([s.state for s in steps])
    action_features_padded = torch.zeros(n, max_actions, action_dim)
    card_idxs_padded = torch.zeros(n, max_actions, dtype=torch.long)
    monster_idxs_padded = torch.zeros(n, max_actions, dtype=torch.long)
    action_potion_idxs_padded = torch.zeros(n, max_actions, dtype=torch.long)
    pi_padded = torch.zeros(n, max_actions)
    mask = torch.zeros(n, max_actions, dtype=torch.bool)

    for i, s in enumerate(steps):
        n_a = s.action_features.shape[0]
        action_features_padded[i, :n_a] = s.action_features
        card_idxs_padded[i, :n_a] = s.card_idxs
        monster_idxs_padded[i, :n_a] = s.monster_idxs
        action_potion_idxs_padded[i, :n_a] = s.action_potion_idxs
        pi_padded[i, :n_a] = s.pi_target
        mask[i, :n_a] = True

    # relic_idxs/relic_mask/potion_idxs/potion_mask are already fixed-width
    # per step -- a plain stack, no padding needed (same as ppo.py's _pad_batch).
    relic_idxs_batch = torch.stack([s.relic_idxs for s in steps])
    relic_mask_batch = torch.stack([s.relic_mask for s in steps])
    potion_idxs_batch = torch.stack([s.potion_idxs for s in steps])
    potion_mask_batch = torch.stack([s.potion_mask for s in steps])

    returns = torch.tensor([s.return_ for s in steps], dtype=torch.float32)
    return (state_batch, action_features_padded, card_idxs_padded, monster_idxs_padded, action_potion_idxs_padded,
            relic_idxs_batch, relic_mask_batch, potion_idxs_batch, potion_mask_batch, pi_padded, mask, returns)


def distillation_update(policy: ActionScoringPolicy, optimizer, steps: List[DistillStep],
                         epochs: int = 2, value_coef: float = 0.5) -> dict:
    """Supervised fine-tune step against a collected search-driven batch:
    policy head -> soft cross-entropy against search's own visit
    distribution (a distribution over legal actions, not a single hard
    label -- search's belief is itself uncertain/spread out, and that's
    worth preserving as the target rather than collapsing to argmax);
    value head -> MSE against the real episode return, the same target
    ppo_update already trains against. Meant to be called occasionally,
    blended alongside normal PPO updates -- see train_distillation_v5.py."""
    (state_batch, af_padded, ci_padded, mi_padded, api_padded,
     relic_idxs_batch, relic_mask_batch, potion_idxs_batch, potion_mask_batch,
     pi_padded, mask, returns) = _pad_distill_batch(steps)

    last_policy_loss = last_value_loss = 0.0
    for _epoch in range(epochs):
        optimizer.zero_grad()
        scores, state_emb = policy.score_actions_batched(
            state_batch, af_padded, ci_padded, mi_padded, api_padded,
            relic_idxs_batch, relic_mask_batch, potion_idxs_batch, potion_mask_batch, mask)
        log_probs = F.log_softmax(scores, dim=-1)
        policy_loss = -(pi_padded * log_probs).sum(dim=-1).mean()
        values = policy.value_head(state_emb).squeeze(-1)
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + value_coef * value_loss
        loss.backward()
        optimizer.step()
        last_policy_loss = float(policy_loss.item())
        last_value_loss = float(value_loss.item())

    return {"policy_loss": last_policy_loss, "value_loss": last_value_loss, "n_steps": len(steps)}
