"""8-hour from-scratch run against the now-live relic-aware architecture
(relic_embedding + sum-pooled relic_idxs/relic_mask threaded through
policy.py/env.py/ppo.py/az_search.py/distillation.py -- promoted to live
code this session, verified consistent end-to-end via a smoke test before
this run). From-scratch, not fine-tuned from checkpoint_5hour_v4.pt/
checkpoint_relics_v1.pt: relic_embedding is a genuinely new layer and
state_encoder's input width changed, so those checkpoints' state_dicts no
longer match this architecture's shapes at all -- same "architecture change
invalidates existing checkpoints" rule this project has followed for every
prior one (reward-v2, observation-v2, monster embeddings, ...).

Also bakes in this session's two measured training-speed fixes, neither of
which were in any prior train_*.py script:
  - persistent worker pool, created once and passed to every train_ppo call
    instead of letting train_ppo spin up a fresh one per chunk (Windows
    process spawn is expensive) -- measured 2.56x on a short A/B, and closer
    to 4.7x once cold-start noise is excluded from the comparison (~22 ->
    ~104 eps/sec steady-state).
  - episodes_per_update=64 instead of 32 -- a further ~14% (104 -> 118
    eps/sec measured), amortizing the per-round policy state_dict IPC
    dispatch and the (inherently sequential, unparallelizable) PPO update
    step over more collected episodes per round. 128 measured flat/worse,
    so 64 is the actual sweet spot, not "bigger is better".
  - n_workers stays at 6 (this machine has 12 cores) -- measured WORSE
    throughput at 10-11 workers (56.5 -> 51.3 eps/sec), so more workers
    isn't free; 6 is the measured sweet spot, not a guess.

Run:  PYTHONPATH=. python -m lightspeed.train_relics_v2_8hour
"""

from __future__ import annotations

import multiprocessing as mp
import time

import torch

from .env import IroncladFightEnv, ALL_ENCOUNTERS, ALL_ACT_TIER_GROUPS, build_full_encounter_resources
from .cards import weighted_ironclad_deck
from .relics import weighted_ironclad_relics
from .policy import ActionScoringPolicy
from .ppo import train_ppo, _env_kwargs_from, _worker_init
from .train import evaluate

TIME_BUDGET_SECONDS = 8 * 60 * 60
EPISODES_PER_UPDATE = 64  # measured sweet spot -- see module docstring
N_WORKERS = 6             # measured sweet spot -- see module docstring
CHUNK_UPDATES = 25
CHECKPOINT_EVAL_N = 100
FINAL_EVAL_N = 500
CURRICULUM_K = 4.0
EMA_ALPHA = 0.3
OUT_CHECKPOINT = "lightspeed/checkpoint_relics_v2_8hour.pt"
LOG_PATH = "lightspeed/train_relics_v2_8hour_progress.log"


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def per_act_breakdown(breakdown):
    by_act = {}
    for act, tier, encs in ALL_ACT_TIER_GROUPS:
        ws = [breakdown[e] for e in encs if e in breakdown]
        if ws:
            by_act.setdefault(act, []).extend(ws)
    return {a: sum(v) / len(v) for a, v in sorted(by_act.items())}


def main():
    resources = build_full_encounter_resources()
    env = IroncladFightEnv(encounter=ALL_ENCOUNTERS, encounter_resources=resources,
                            deck_generator=weighted_ironclad_deck,
                            relic_generator=weighted_ironclad_relics)
    policy = ActionScoringPolicy()
    _log("=== relics-v2 8-hour run: from-scratch against the live relic-embedding architecture, "
         "persistent-pool + episodes_per_update=64 speed fixes ===")
    _log(f"n_workers={N_WORKERS}, episodes_per_update={EPISODES_PER_UPDATE}, K={CURRICULUM_K}, ema_alpha={EMA_ALPHA}")

    win_rate_estimate = {e: 0.5 for e in ALL_ENCOUNTERS}

    start = time.time()
    best_reward = float("-inf")
    best_state = None
    total_updates = 0
    chunk_num = 0
    batch_reward_history = []

    pool = mp.Pool(N_WORKERS, initializer=_worker_init, initargs=(_env_kwargs_from(env),))
    try:
        while time.time() - start < TIME_BUDGET_SECONDS:
            chunk_num += 1
            chunk_t0 = time.time()
            # checkpoint_every/checkpoint_eval_n deliberately NOT passed here:
            # train_ppo's own internal eval (when checkpoint_every==updates,
            # as every prior train_*.py script in this project set it) fires
            # exactly once per chunk, but its result was never used here --
            # this driver always ran its OWN separate per_encounter=True eval
            # right below regardless, so every chunk was silently paying for
            # 200 eval episodes (100 unused + 100 used) instead of 100.
            history = train_ppo(
                env, policy, updates=CHUNK_UPDATES, episodes_per_update=EPISODES_PER_UPDATE,
                n_workers=N_WORKERS, pool=pool,
            )
            batch_reward_history.extend(history)
            total_updates += CHUNK_UPDATES
            elapsed = time.time() - start
            chunk_dt = time.time() - chunk_t0
            eps_per_sec = (CHUNK_UPDATES * EPISODES_PER_UPDATE) / chunk_dt

            if any(r != r for r in history):
                _log(f"[FATAL] NaN batch reward detected in chunk {chunk_num}, history: {history}")
                return

            (win, hp, reward), breakdown = evaluate(env, policy, n=CHECKPOINT_EVAL_N, per_encounter=True)
            by_act = per_act_breakdown(breakdown)
            act_str = "  ".join(f"{a} {w*100:.1f}%" for a, w in by_act.items())
            _log(f"chunk {chunk_num:4d} (update {total_updates:6d}, {elapsed/3600:4.2f}h, "
                 f"{eps_per_sec:6.1f} eps/sec): win {win*100:.1f}%  avg HP {hp:.1f}  reward {reward:.2f}  |  {act_str}")

            for enc, w in breakdown.items():
                prior = win_rate_estimate.get(enc, 0.5)
                win_rate_estimate[enc] = EMA_ALPHA * w + (1 - EMA_ALPHA) * prior
            env.encounter_weights = [
                1.0 + CURRICULUM_K * (1.0 - win_rate_estimate.get(e, 0.5))
                for e in ALL_ENCOUNTERS
            ]

            if reward > best_reward:
                best_reward = reward
                best_state = {k: v.clone() for k, v in policy.state_dict().items()}
                torch.save(best_state, OUT_CHECKPOINT)

            if chunk_num % 40 == 0:
                worst = sorted(win_rate_estimate.items(), key=lambda kv: kv[1])[:5]
                _log("  curriculum lowest-5: " + ", ".join(f"{e}={r*100:.0f}%" for e, r in worst))
    except Exception:
        import traceback
        _log(f"[FATAL] crashed after {total_updates} updates:\n{traceback.format_exc()}")
        return
    finally:
        pool.close()
        pool.join()

    _log(f"=== budget reached after {total_updates} updates ===")
    _log(f"batch reward trend (first 3 vs last 3 chunks' mean): "
         f"{sum(batch_reward_history[:3])/max(1,len(batch_reward_history[:3])):.2f} -> "
         f"{sum(batch_reward_history[-3:])/max(1,len(batch_reward_history[-3:])):.2f}")
    if best_state is not None:
        policy.load_state_dict(best_state)

    env.encounter_weights = None  # uniform encounters for the comparable final eval
    _log(f"=== final per-encounter breakdown (best checkpoint, reward {best_reward:.2f}), n={FINAL_EVAL_N} ===")
    (win, hp, reward), breakdown = evaluate(env, policy, n=FINAL_EVAL_N, per_encounter=True)
    _log(f"overall: win {win*100:.1f}%  avg HP {hp:.1f}  avg reward {reward:.2f}")
    for act, tier, encs in ALL_ACT_TIER_GROUPS:
        for e in encs:
            if e in breakdown:
                _log(f"  [{act}/{tier:5s}] {e}: {breakdown[e]*100:.1f}%")
    _log("=== done ===")


if __name__ == "__main__":
    main()
