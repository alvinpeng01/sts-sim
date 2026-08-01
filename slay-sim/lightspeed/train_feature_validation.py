"""Quick validation run for the enemy-status/card-type feature fix (see
env.py's _encode_state/_encode_action_and_card_idx): trains from scratch on
the full roster (avoids overfitting to just the two fights we care about),
but reports Gremlin Nob and Lagavulin win-rate/HP specifically at every
checkpoint -- those are the fights diagnosed as blind to enemy Strength
(Gremlin Nob's Enrage) and card type, so they're the direct test of whether
the new features actually help, not just a general roster-wide number.

Baseline to beat (old checkpoint_overnight_v2.pt, old 10/5-feature
architecture, matched-start vs native MCTS):
  Gremlin Nob: 97.0% win, 39.9 avg HP
  Lagavulin:   100.0% win, 37.1 avg HP

Second baseline (25-min run, fixed 13/8-feature architecture, OLD reward --
i.e. isolates the turn-efficiency reward's own effect on top of the feature
fix, same eval methodology, n=100 during training / n=200 final):
  Gremlin Nob: 88.5% win, 39.7 avg HP
  Lagavulin:   99.0% win, 36.0 avg HP

Run:  PYTHONPATH=. nohup .venv/bin/python -m lightspeed.train_feature_validation > lightspeed/feature_validation.log 2>&1 &
"""

from __future__ import annotations

import time

import torch

from .env import IroncladFightEnv, ALL_ENCOUNTERS, build_full_encounter_resources
from .policy import ActionScoringPolicy
from .ppo import train_ppo
from .train import evaluate

import slaythespire as sts

TIME_BUDGET_SECONDS = 25 * 60
EPISODES_PER_UPDATE = 32
N_WORKERS = 6
CHECKPOINT_EVERY = 100
CHECKPOINT_EVAL_N = 150
CHECKPOINT_PATH = "lightspeed/checkpoint_turn_efficiency_validation.pt"
LOG_PATH = "lightspeed/turn_efficiency_validation_progress.log"

FOCUS_ENCOUNTERS = [
    ("Gremlin Nob", sts.MonsterEncounter.GREMLIN_NOB),
    ("Lagavulin", sts.MonsterEncounter.LAGAVULIN),
]


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def eval_focus_encounters(policy, resources, n):
    results = {}
    for name, enc in FOCUS_ENCOUNTERS:
        env = IroncladFightEnv(encounter=[enc], encounter_resources=resources)
        win, hp, _reward = evaluate(env, policy, n=n)
        results[name] = (win, hp)
    return results


def main():
    resources = build_full_encounter_resources()
    env = IroncladFightEnv(encounter=ALL_ENCOUNTERS, encounter_resources=resources)
    policy = ActionScoringPolicy()

    _log("=== turn-efficiency-validation run starting (features fixed + turn-efficiency reward) ===")
    _log(f"baseline 1 (old arch): Gremlin Nob 97.0%/39.9hp, Lagavulin 100.0%/37.1hp")
    _log(f"baseline 2 (features fixed, old reward): Gremlin Nob 88.5%/39.7hp, Lagavulin 99.0%/36.0hp")

    start = time.time()
    best_reward = float("-inf")
    best_state = None
    total_updates = 0
    chunk_num = 0

    try:
        while time.time() - start < TIME_BUDGET_SECONDS:
            chunk_num += 1
            history, chunk_best = train_ppo(
                env, policy, updates=CHECKPOINT_EVERY, episodes_per_update=EPISODES_PER_UPDATE,
                checkpoint_every=CHECKPOINT_EVERY, checkpoint_eval_n=CHECKPOINT_EVAL_N,
                n_workers=N_WORKERS,
            )
            total_updates += CHECKPOINT_EVERY
            elapsed = time.time() - start

            (win, hp, reward) = evaluate(env, policy, n=CHECKPOINT_EVAL_N)
            focus = eval_focus_encounters(policy, resources, n=100)
            focus_str = "  ".join(f"{name} {w*100:.1f}%/{h:.1f}hp" for name, (w, h) in focus.items())
            _log(f"chunk {chunk_num:3d} (update {total_updates:5d}, {elapsed/60:.1f}min): "
                 f"overall win {win*100:.1f}%  hp {hp:.1f}  reward {reward:.2f}  |  {focus_str}")

            if reward > best_reward:
                best_reward = reward
                best_state = {k: v.clone() for k, v in policy.state_dict().items()}
                torch.save(best_state, CHECKPOINT_PATH)
                _log(f"  new best (reward {reward:.2f}), checkpoint saved")

    except Exception:
        import traceback
        _log(f"[FATAL] crashed after {total_updates} updates:\n{traceback.format_exc()}")
        return

    _log(f"=== budget reached after {total_updates} updates ===")
    if best_state is not None:
        policy.load_state_dict(best_state)
    focus = eval_focus_encounters(policy, resources, n=200)
    for name, (w, h) in focus.items():
        _log(f"  final {name}: win {w*100:.1f}%  avg HP {h:.1f}")
    _log("=== done ===")


if __name__ == "__main__":
    main()
