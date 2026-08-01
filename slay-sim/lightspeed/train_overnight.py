"""Overnight training run: full 42-encounter roster, the target-threat-share
action feature (fixes the minion-targeting blind spot diagnosed via the
expectimax-vs-greedy comparison earlier), PARALLEL rollout collection
(n_workers=6). Parallel was sequential-only in the original version of this
script because the parallel path was genuinely slower at the time -- root
cause (torch.Tensor pickling ~34x slower than numpy for the same result
data) has since been found and fixed, plus a persistent-per-worker-state
fix on top of that, plus several redundant-C++-binding-fetch fixes in
env.py's observation encoding. Parallel is now the fast path, confirmed via
repeated measurement (~180-210 eps/sec sustained vs sequential's ~80).
Ascension defaults to A0 (reverted from a brief A20 default -- see env.py's
own comment on `ascension` for why).

Time-budgeted, not update-count-budgeted: throughput has enough real
variance (fight length varies by encounter mix each batch) that a fixed
update count is a worse proxy for "how long will this actually run" than
just checking the clock. Checkpoints the best-eval-reward policy every
CHECKPOINT_EVERY updates (same convention as every other training script
this project uses) specifically so an unattended run that gets interrupted
--  crash, host sleep, anything -- still leaves a usable, good checkpoint
behind, not just whatever the last update happened to land on.

Run:  PYTHONPATH=. nohup .venv/bin/python -m lightspeed.train_overnight > lightspeed/overnight.log 2>&1 &
"""

from __future__ import annotations

import time
import traceback

import torch

from .env import IroncladFightEnv, ALL_ENCOUNTERS, ALL_ACT_TIER_GROUPS, build_full_encounter_resources
from .policy import ActionScoringPolicy
from .ppo import train_ppo
from .train import evaluate

TIME_BUDGET_SECONDS = 8.0 * 3600  # ~8 hours; stops after the chunk that crosses this, not exactly at it
EPISODES_PER_UPDATE = 32     # bigger than the original sequential run's 16 -- parallel amortizes dispatch overhead better at larger batches (measured)
N_WORKERS = 6                # matches physical core count; parallel is now the fast path, see module docstring
CHECKPOINT_EVERY = 250       # updates per chunk; also the checkpoint/report cadence
CHECKPOINT_EVAL_N = 150
FINAL_EVAL_N = 500
CHECKPOINT_PATH = "lightspeed/checkpoint_overnight_v2.pt"  # v2: the original checkpoint_overnight.pt (2250 updates, old 4-feature encoding) stays untouched as a reference point
LOG_PATH = "lightspeed/overnight_v2_progress.log"  # separate from stdout log -- easy to tail, survives stdout redirection weirdness


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

    _log(f"=== overnight run v2 starting ===")
    _log(f"pool: {len(ALL_ENCOUNTERS)} encounters (full Act1-3 roster, ascension={env.ascension})")
    _log(f"action encoding: 5 features incl. target_threat_share (the minion-targeting fix)")
    _log(f"n_workers={N_WORKERS} (parallel, fixed -- see module docstring)")
    _log(f"time budget: {TIME_BUDGET_SECONDS/3600:.1f}h, episodes_per_update={EPISODES_PER_UPDATE}, "
         f"checkpoint every {CHECKPOINT_EVERY} updates")

    start = time.time()
    best_reward = float("-inf")
    best_state = None
    total_updates = 0
    chunk_num = 0

    try:
        while time.time() - start < TIME_BUDGET_SECONDS:
            chunk_num += 1
            chunk_t0 = time.time()
            history, chunk_best_state = train_ppo(
                env, policy, updates=CHECKPOINT_EVERY, episodes_per_update=EPISODES_PER_UPDATE,
                checkpoint_every=CHECKPOINT_EVERY, checkpoint_eval_n=CHECKPOINT_EVAL_N,
                n_workers=N_WORKERS,
            )
            total_updates += CHECKPOINT_EVERY
            elapsed = time.time() - start
            chunk_dt = time.time() - chunk_t0
            eps_per_sec = (CHECKPOINT_EVERY * EPISODES_PER_UPDATE) / chunk_dt

            (win, hp, reward), breakdown = evaluate(env, policy, n=CHECKPOINT_EVAL_N, per_encounter=True)
            by_act = per_act_breakdown(breakdown)
            act_str = "  ".join(f"{a} {w*100:.1f}%" for a, w in by_act.items())
            _log(f"chunk {chunk_num:4d} (update {total_updates:6d}, {elapsed/3600:.2f}h elapsed, "
                 f"{eps_per_sec:.1f} eps/sec): win {win*100:.1f}%  avg HP {hp:.1f}  reward {reward:.2f}  |  {act_str}")

            if reward > best_reward:
                best_reward = reward
                best_state = {k: v.clone() for k, v in policy.state_dict().items()}
                torch.save(best_state, CHECKPOINT_PATH)
                _log(f"  new best (reward {reward:.2f}), checkpoint saved to {CHECKPOINT_PATH}")

    except Exception:
        _log(f"[FATAL] training loop crashed after {total_updates} updates:\n{traceback.format_exc()}")
        _log(f"best checkpoint so far (reward {best_reward:.2f}) is still saved at {CHECKPOINT_PATH} -- not lost.")
        return

    _log(f"=== time budget reached after {total_updates} updates, {(time.time()-start)/3600:.2f}h ===")

    if best_state is not None:
        policy.load_state_dict(best_state)

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
