"""20-minute validation run: continues from checkpoint_overnight.pt (2250
updates in, reward 16.07, target_threat_share encoding) using the NOW-FIXED
parallel collection path (n_workers=6, matches physical core count -- see
ppo.py's Step/_step_to_numpy for the torch-tensor-pickling fix that made
parallel actually faster than sequential). Two things this answers:
  1. Does the fixed parallel path sustain a real speedup over a full 20
     minutes uncontended (the 5-minute test was contended by the overnight
     run for its first ~230s)?
  2. Does more training with the target_threat_share feature start pulling
     ahead of where the old 4-feature encoding was at the same update
     count (it was slightly behind at 2250 updates -- too early to tell if
     that's just early-training noise or a real cost).

Saves to a NEW checkpoint file (not overwriting checkpoint_overnight.pt),
so the already-good starting point is never at risk.

Run:  PYTHONPATH=. .venv/bin/python -m lightspeed.train_20min_parallel
"""

from __future__ import annotations

import time

import torch

from .env import IroncladFightEnv, ALL_ENCOUNTERS, ALL_ACT_TIER_GROUPS, build_full_encounter_resources
from .policy import ActionScoringPolicy
from .ppo import train_ppo
from .train import evaluate

TIME_BUDGET_SECONDS = 20 * 60
EPISODES_PER_UPDATE = 32
N_WORKERS = 6
CHUNK_UPDATES = 25
CHECKPOINT_EVAL_N = 100
FINAL_EVAL_N = 400
START_CHECKPOINT = "lightspeed/checkpoint_overnight.pt"
OUT_CHECKPOINT = "lightspeed/checkpoint_20min_parallel.pt"
LOG_PATH = "lightspeed/train_20min_parallel_progress.log"


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
    policy.load_state_dict(torch.load(START_CHECKPOINT))
    _log(f"loaded {START_CHECKPOINT} (2250 updates, reward 16.07) as starting point")
    _log(f"20-minute run: n_workers={N_WORKERS} (parallel, fixed), "
         f"episodes_per_update={EPISODES_PER_UPDATE}")

    start = time.time()
    best_reward = float("-inf")
    best_state = None
    total_updates = 0
    chunk_num = 0

    try:
        while time.time() - start < TIME_BUDGET_SECONDS:
            chunk_num += 1
            chunk_t0 = time.time()
            history, chunk_best = train_ppo(
                env, policy, updates=CHUNK_UPDATES, episodes_per_update=EPISODES_PER_UPDATE,
                checkpoint_every=CHUNK_UPDATES, checkpoint_eval_n=CHECKPOINT_EVAL_N,
                n_workers=N_WORKERS,
            )
            total_updates += CHUNK_UPDATES
            elapsed = time.time() - start
            chunk_dt = time.time() - chunk_t0
            eps_per_sec = (CHUNK_UPDATES * EPISODES_PER_UPDATE) / chunk_dt

            (win, hp, reward), breakdown = evaluate(env, policy, n=CHECKPOINT_EVAL_N, per_encounter=True)
            by_act = per_act_breakdown(breakdown)
            act_str = "  ".join(f"{a} {w*100:.1f}%" for a, w in by_act.items())
            _log(f"chunk {chunk_num:4d} (+{total_updates} updates this run, {elapsed:5.1f}s elapsed, "
                 f"{eps_per_sec:6.1f} eps/sec): win {win*100:.1f}%  avg HP {hp:.1f}  reward {reward:.2f}  |  {act_str}")

            if reward > best_reward:
                best_reward = reward
                best_state = {k: v.clone() for k, v in policy.state_dict().items()}
                torch.save(best_state, OUT_CHECKPOINT)
                _log(f"  new best (reward {reward:.2f}), checkpoint saved to {OUT_CHECKPOINT}")
    except Exception:
        import traceback
        _log(f"[FATAL] crashed after {total_updates} updates:\n{traceback.format_exc()}")
        return

    _log(f"=== 20-minute budget reached after {total_updates} more updates "
         f"(total {2250 + total_updates}) ===")

    if best_state is not None:
        policy.load_state_dict(best_state)
    else:
        torch.save(policy.state_dict(), OUT_CHECKPOINT)

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
