"""Failure-weighted encounter curriculum: same chunked training loop as
train_2hour_v2.py, but after each chunk's per-encounter eval, biases the
NEXT chunk's encounter sampling toward whatever the policy is currently
losing, instead of every encounter getting equal exposure regardless of
difficulty. Time Eater/Awakened One/Donu & Deca are each ~1/35 of the pool
under uniform sampling -- exactly the ones that need the most training
volume to read as anything but noise (see overnight_v2_progress.log) get
the least under a flat curriculum.

win_rate_estimate is a persistent per-encounter EMA (alpha=0.3), not just
"whatever this chunk's eval said" -- CHECKPOINT_EVAL_N=100 spread over ~35
encounters means most single-chunk samples per encounter are single-digit,
so using the raw per-chunk number directly would make the curriculum chase
noise. Encounters not seen in the eval this chunk keep their prior estimate
rather than resetting.

weight = 1 + K*(1 - win_rate), K=4 -- a 0% win rate encounter gets ~5x a
100% one's sampling weight, not so extreme that it starves the easy
encounters entirely (some continued exposure to what's already working
guards against catastrophic forgetting on those).

Combines with the reward-v2 + observation-v2 changes already validated
(checkpoint_10min_v2.pt / checkpoint_2hour_v2.pt) -- this is the next
variable in the same investigation, not a replacement for them.

Run:  PYTHONPATH=. python -m lightspeed.train_curriculum
"""

from __future__ import annotations

import time

import torch

from .env import IroncladFightEnv, ALL_ENCOUNTERS, ALL_ACT_TIER_GROUPS, build_full_encounter_resources
from .policy import ActionScoringPolicy
from .ppo import train_ppo
from .train import evaluate

TIME_BUDGET_SECONDS = 10 * 60
EPISODES_PER_UPDATE = 32
N_WORKERS = 6
CHUNK_UPDATES = 25
CHECKPOINT_EVAL_N = 100
FINAL_EVAL_N = 500
CURRICULUM_K = 4.0
EMA_ALPHA = 0.3
OUT_CHECKPOINT = "lightspeed/checkpoint_curriculum.pt"
LOG_PATH = "lightspeed/train_curriculum_progress.log"


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
    env = IroncladFightEnv(encounter=ALL_ENCOUNTERS, encounter_resources=resources)
    policy = ActionScoringPolicy()
    _log("=== curriculum run: reward-v2 + observation-v2 + failure-weighted sampling (from scratch, A0) ===")
    _log(f"n_workers={N_WORKERS}, episodes_per_update={EPISODES_PER_UPDATE}, K={CURRICULUM_K}, ema_alpha={EMA_ALPHA}")

    # Start every encounter at an assumed 50% win rate (neutral -- neither
    # oversampled nor undersampled) until the first real eval updates it;
    # uniform weights for chunk 1 since there's no data yet.
    win_rate_estimate = {e: 0.5 for e in ALL_ENCOUNTERS}

    start = time.time()
    best_reward = float("-inf")
    best_state = None
    total_updates = 0
    chunk_num = 0
    batch_reward_history = []

    try:
        while time.time() - start < TIME_BUDGET_SECONDS:
            chunk_num += 1
            chunk_t0 = time.time()
            history, chunk_best = train_ppo(
                env, policy, updates=CHUNK_UPDATES, episodes_per_update=EPISODES_PER_UPDATE,
                checkpoint_every=CHUNK_UPDATES, checkpoint_eval_n=CHECKPOINT_EVAL_N,
                n_workers=N_WORKERS,
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
            _log(f"chunk {chunk_num:4d} (update {total_updates:5d}, {elapsed/60:5.1f}min, "
                 f"{eps_per_sec:6.1f} eps/sec): win {win*100:.1f}%  avg HP {hp:.1f}  reward {reward:.2f}  |  {act_str}")

            for enc, w in breakdown.items():
                prior = win_rate_estimate.get(enc, 0.5)
                win_rate_estimate[enc] = EMA_ALPHA * w + (1 - EMA_ALPHA) * prior

            # Applies to the NEXT chunk's rollouts (this chunk's env.reset()
            # calls already happened inside train_ppo above), computed off
            # the estimate this chunk's eval just updated.
            env.encounter_weights = [
                1.0 + CURRICULUM_K * (1.0 - win_rate_estimate.get(e, 0.5))
                for e in ALL_ENCOUNTERS
            ]
            worst = sorted(win_rate_estimate.items(), key=lambda kv: kv[1])[:3]
            _log(f"  curriculum: lowest-estimate encounters now "
                 + ", ".join(f"{e}={r*100:.0f}%" for e, r in worst))

            if reward > best_reward:
                best_reward = reward
                best_state = {k: v.clone() for k, v in policy.state_dict().items()}
                torch.save(best_state, OUT_CHECKPOINT)
    except Exception:
        import traceback
        _log(f"[FATAL] crashed after {total_updates} updates:\n{traceback.format_exc()}")
        return

    _log(f"=== budget reached after {total_updates} updates ===")
    _log(f"batch reward trend (first 3 vs last 3 chunks' mean): "
         f"{sum(batch_reward_history[:3])/max(1,len(batch_reward_history[:3])):.2f} -> "
         f"{sum(batch_reward_history[-3:])/max(1,len(batch_reward_history[-3:])):.2f}")
    if best_state is not None:
        policy.load_state_dict(best_state)

    # Final breakdown evaluated WITHOUT curriculum weighting (uniform) --
    # this is the number that's actually comparable to overnight_v2/
    # checkpoint_2hour_v2's own final evals, all of which measured uniform
    # per-encounter win rate, not the (deliberately skewed) training mix.
    env.encounter_weights = None
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
