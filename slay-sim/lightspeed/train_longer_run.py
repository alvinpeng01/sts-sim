"""Longer follow-on to train_feature_validation.py's 25-minute run: warm-
starts from checkpoint_turn_efficiency_validation.pt (same 13/8-feature
architecture + turn-efficiency reward, so the weights load directly) and
keeps training, to test whether AZ search's remaining gap to native MCTS
(Gremlin Nob 42.2 vs 54.0 avg HP, Lagavulin 38.4 vs 54.5, both at 50
simulations) is a value-net-maturity problem rather than a search-budget one
-- confirmed via a sims=50/200/500 sweep on the 25-min checkpoint showing
HP retention flat regardless of simulation count (43.4/42.4/42.4), i.e. the
search already converges to what the value net believes almost immediately;
more compute doesn't change that belief, more training should.

Baselines (matched-start vs native MCTS):
  Old arch, ~8h training:            Gremlin Nob 97.0%/39.9hp, Lagavulin 100.0%/37.1hp
  New arch (features), 25min:        Gremlin Nob 88.5%/39.7hp, Lagavulin 99.0%/36.0hp
  New arch (+turn-eff), 25min:       Gremlin Nob 94.0%/40.0hp, Lagavulin 99.0%/38.1hp
  AZ search (50 sims) on the above:  Gremlin Nob 93.3%/42.2hp, Lagavulin 100.0%/38.4hp
  Native MCTS (2000 sims):           Gremlin Nob 100.0%/54.0hp, Lagavulin 100.0%/54.5hp

Run:  PYTHONPATH=. nohup .venv/bin/python -m lightspeed.train_longer_run > lightspeed/longer_run.log 2>&1 &
"""

from __future__ import annotations

import time

import torch

from .env import IroncladFightEnv, ALL_ENCOUNTERS, build_full_encounter_resources
from .policy import ActionScoringPolicy
from .ppo import train_ppo
from .train import evaluate

import slaythespire as sts

TIME_BUDGET_SECONDS = 2.0 * 3600
EPISODES_PER_UPDATE = 32
N_WORKERS = 6
CHECKPOINT_EVERY = 250
CHECKPOINT_EVAL_N = 150
WARM_START_PATH = "lightspeed/checkpoint_turn_efficiency_validation.pt"
CHECKPOINT_PATH = "lightspeed/checkpoint_longer_run.pt"
LOG_PATH = "lightspeed/longer_run_progress.log"

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
    policy.load_state_dict(torch.load(WARM_START_PATH, map_location="cpu"))

    _log("=== longer run starting (warm-started from checkpoint_turn_efficiency_validation.pt) ===")
    focus = eval_focus_encounters(policy, resources, n=150)
    focus_str = "  ".join(f"{name} {w*100:.1f}%/{h:.1f}hp" for name, (w, h) in focus.items())
    _log(f"warm-start eval: {focus_str}")

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
            _log(f"chunk {chunk_num:3d} (update {total_updates:6d}, {elapsed/60:.1f}min): "
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
