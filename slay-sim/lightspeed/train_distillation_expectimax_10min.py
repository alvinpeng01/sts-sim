"""10-minute validation run of train_distillation_expectimax.py's pipeline
before committing to the full 8-hour budget -- same everything, just a much
shorter TIME_BUDGET_SECONDS and a smaller CHUNK_UPDATES/DISTILL_EVERY_CHUNKS
so at least a couple of real distillation rounds actually happen inside the
window (the production script's DISTILL_EVERY_CHUNKS=40 wouldn't fire even
once in 10 minutes at CHUNK_UPDATES=25). DISTILL_EPISODES/SIMULATIONS/
WORKERS/TREES are left at production values so the distillation round's own
timing is a realistic preview, not scaled down along with everything else.

Run:  PYTHONPATH=. python -m lightspeed.train_distillation_expectimax_10min
"""

from __future__ import annotations

import time

import torch

from .env import IroncladFightEnv, ALL_ENCOUNTERS, ALL_ACT_TIER_GROUPS, build_full_encounter_resources
from .cards import weighted_ironclad_deck
from .policy import ActionScoringPolicy
from .ppo import train_ppo
from .train import evaluate
from .distillation import collect_distillation_batch_parallel, distillation_update

TIME_BUDGET_SECONDS = 10 * 60
EPISODES_PER_UPDATE = 32
N_WORKERS = 6
CHUNK_UPDATES = 5
CHECKPOINT_EVAL_N = 50
FINAL_EVAL_N = 100
CURRICULUM_K = 4.0
EMA_ALPHA = 0.3

DISTILL_EVERY_CHUNKS = 2
DISTILL_EPISODES = 64
DISTILL_SIMULATIONS = 200
DISTILL_WORKERS = 6
DISTILL_TREES = 2
DISTILL_LR = 3e-5
DISTILL_EPOCHS = 2

OUT_CHECKPOINT = "lightspeed/checkpoint_distillation_expectimax_10min.pt"
LOG_PATH = "lightspeed/train_distillation_expectimax_10min_progress.log"


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
                            deck_generator=weighted_ironclad_deck)
    policy = ActionScoringPolicy()
    distill_optimizer = torch.optim.Adam(policy.parameters(), lr=DISTILL_LR)

    _log("=== 10-MIN VALIDATION: expectimax-distillation pipeline preview ===")
    _log(f"n_workers={N_WORKERS}, episodes_per_update={EPISODES_PER_UPDATE}, K={CURRICULUM_K}, ema_alpha={EMA_ALPHA}")
    _log(f"distillation: every {DISTILL_EVERY_CHUNKS} chunks, {DISTILL_EPISODES} episodes, "
         f"{DISTILL_SIMULATIONS} sims/decision, {DISTILL_WORKERS} workers x {DISTILL_TREES} trees, lr={DISTILL_LR}")

    win_rate_estimate = {e: 0.5 for e in ALL_ENCOUNTERS}

    start = time.time()
    best_reward = float("-inf")
    best_state = None
    total_updates = 0
    chunk_num = 0
    batch_reward_history = []
    active_time = 0.0

    try:
        while active_time < TIME_BUDGET_SECONDS:
            chunk_num += 1
            chunk_t0 = time.time()
            history, chunk_best = train_ppo(
                env, policy, updates=CHUNK_UPDATES, episodes_per_update=EPISODES_PER_UPDATE,
                checkpoint_every=CHUNK_UPDATES, checkpoint_eval_n=CHECKPOINT_EVAL_N,
                n_workers=N_WORKERS,
            )
            batch_reward_history.extend(history)
            total_updates += CHUNK_UPDATES
            chunk_dt = time.time() - chunk_t0
            active_time += chunk_dt
            elapsed = time.time() - start
            eps_per_sec = (CHUNK_UPDATES * EPISODES_PER_UPDATE) / chunk_dt

            if any(r != r for r in history):
                _log(f"[FATAL] NaN batch reward detected in chunk {chunk_num}, history: {history}")
                return

            (win, hp, reward), breakdown = evaluate(env, policy, n=CHECKPOINT_EVAL_N, per_encounter=True)
            by_act = per_act_breakdown(breakdown)
            act_str = "  ".join(f"{a} {w*100:.1f}%" for a, w in by_act.items())
            _log(f"chunk {chunk_num:4d} (update {total_updates:6d}, active {active_time/60:5.1f}m / "
                 f"wall {elapsed/60:5.1f}m, {eps_per_sec:6.1f} eps/sec): win {win*100:.1f}%  avg HP {hp:.1f}  "
                 f"reward {reward:.2f}  |  {act_str}")

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

            if chunk_num % DISTILL_EVERY_CHUNKS == 0:
                distill_t0 = time.time()
                steps = collect_distillation_batch_parallel(
                    env, n_episodes=DISTILL_EPISODES, n_simulations=DISTILL_SIMULATIONS,
                    n_workers=DISTILL_WORKERS, n_trees=DISTILL_TREES,
                )
                stats = distillation_update(policy, distill_optimizer, steps, epochs=DISTILL_EPOCHS)
                distill_dt = time.time() - distill_t0
                active_time += distill_dt
                _log(f"  [distill] {stats['n_steps']} steps from {DISTILL_EPISODES} search-driven episodes "
                     f"({distill_dt:.1f}s): policy_loss={stats['policy_loss']:.3f}  value_loss={stats['value_loss']:.1f}")
    except Exception:
        import traceback
        _log(f"[FATAL] crashed after {total_updates} updates:\n{traceback.format_exc()}")
        return

    _log(f"=== budget reached after {total_updates} updates ({active_time/60:.1f}m active / "
         f"{(time.time()-start)/60:.1f}m wall) ===")
    if best_state is not None:
        policy.load_state_dict(best_state)

    env.encounter_weights = None
    _log(f"=== final per-encounter breakdown (best checkpoint, reward {best_reward:.2f}), n={FINAL_EVAL_N} ===")
    (win, hp, reward), breakdown = evaluate(env, policy, n=FINAL_EVAL_N, per_encounter=True)
    _log(f"overall: win {win*100:.1f}%  avg HP {hp:.1f}  avg reward {reward:.2f}")
    _log("=== done ===")


if __name__ == "__main__":
    main()
