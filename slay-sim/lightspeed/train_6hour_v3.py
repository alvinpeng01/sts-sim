"""6-hour from-scratch run: the real comparison against overnight_v2's
190,500-update / 8-hour baseline (86.8% overall; Collector 41.7%,
Reptomancer 25%, Automaton 66.7%, Champ 66.7%, Awakened One 12.5%, Time
Eater 11.1%, Donu & Deca 10.0%).

Everything from this session's investigation is now in place and each piece
was individually verified before this run:
  - reward-v2: BETA-weighted incoming damage in the potential + widened
    W_WIN/W_DEATH terminal gap;
  - potential() half-dead fix: a reviving Awakened One no longer reads as a
    near-win (verified: contributes -max_hp, not 0);
  - observation-v2: enemy block, 19 player power stacks, 5 enemy statuses
    (Poison/Plated Armor/Artifact/Metallicize/Mode Shift), and the REAL
    Time Warp counter (verified tracking 0.25->0.50->0.75->0.92->reset
    across turns, replacing the per-turn counter that was the wrong signal);
  - monster identity embedding (47 monsters);
  - saturating normalization on unbounded magnitudes (block/damage/strength/
    stacks), so Barricade/Demon Form builds don't live in a poorly-resolved
    linear tail;
  - failure-weighted curriculum sampling: oversamples whatever the policy
    is currently losing, so the rare hard bosses (each ~1/35 of the pool
    under uniform sampling) get the training volume they need.

STATE_FEATURES is now 40 (was 13 at the start of this investigation) plus a
monster embedding, so the network is substantially larger -- this run needs
its full budget to be a fair comparison, and even 6h is shorter than the
baseline's 8h. The final per-encounter breakdown is evaluated WITHOUT
curriculum weighting (uniform), so it's directly comparable to the baseline.

Run:  PYTHONPATH=. python -m lightspeed.train_6hour_v3
"""

from __future__ import annotations

import time

import torch

from .env import IroncladFightEnv, ALL_ENCOUNTERS, ALL_ACT_TIER_GROUPS, build_full_encounter_resources
from .policy import ActionScoringPolicy
from .ppo import train_ppo
from .train import evaluate

TIME_BUDGET_SECONDS = 6 * 60 * 60
EPISODES_PER_UPDATE = 32
N_WORKERS = 6
CHUNK_UPDATES = 25
CHECKPOINT_EVAL_N = 100
FINAL_EVAL_N = 500
CURRICULUM_K = 4.0
EMA_ALPHA = 0.3
OUT_CHECKPOINT = "lightspeed/checkpoint_6hour_v3.pt"
LOG_PATH = "lightspeed/train_6hour_v3_progress.log"


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
    _log("=== 6-hour v3 run: reward-v2 + obs-v2 + monster-embed + saturating-norm + curriculum (from scratch, A0) ===")
    _log(f"n_workers={N_WORKERS}, episodes_per_update={EPISODES_PER_UPDATE}, K={CURRICULUM_K}, ema_alpha={EMA_ALPHA}")

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

            # Periodic full uniform breakdown so a mid-run snapshot exists if
            # the machine is interrupted before the 6h mark (every ~40 chunks
            # = ~1000 updates), logged but not acted on.
            if chunk_num % 40 == 0:
                worst = sorted(win_rate_estimate.items(), key=lambda kv: kv[1])[:5]
                _log("  curriculum lowest-5: " + ", ".join(f"{e}={r*100:.0f}%" for e, r in worst))
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

    env.encounter_weights = None  # uniform final eval, comparable to the baseline
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
