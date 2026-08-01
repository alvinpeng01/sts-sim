"""Short (~20-min) from-scratch run against the current live architecture
(relics + potions both wired into policy.py/env.py). Purpose: get a real
trained checkpoint compatible with the CURRENT policy shape for a fair
comparison against Silver Automaton's combat search on Time Eater/Donu &
Deca -- neither checkpoint_5hour_v4.pt (pre-relics) nor
checkpoint_relics_v2_8hour.pt (relics, no potions) loads into the current
architecture at all (confirmed: state_dict shape mismatch), and a random-
init network would make our search look artificially bad in that
comparison, since our search's priors/value depend on a trained network in
a way Silver Automaton's heuristic-rollout search doesn't. Not meant to
reach the quality of the 8-hour run -- just needs to be meaningfully better
than random for a fair reading.

Run:  PYTHONPATH=. python -m lightspeed.train_quick_v3
"""

from __future__ import annotations

import multiprocessing as mp
import time

import torch

from .env import IroncladFightEnv, ALL_ENCOUNTERS, ALL_ACT_TIER_GROUPS, build_full_encounter_resources
from .cards import weighted_ironclad_deck
from .relics import weighted_ironclad_relics
from .potions import uniform_ironclad_potions
from .policy import ActionScoringPolicy
from .ppo import train_ppo, _env_kwargs_from, _worker_init
from .train import evaluate

TIME_BUDGET_SECONDS = 20 * 60
EPISODES_PER_UPDATE = 64
N_WORKERS = 6
CHUNK_UPDATES = 25
CHECKPOINT_EVAL_N = 60
OUT_CHECKPOINT = "lightspeed/checkpoint_quick_v3.pt"
LOG_PATH = "lightspeed/train_quick_v3_progress.log"


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    resources = build_full_encounter_resources()
    env = IroncladFightEnv(encounter=ALL_ENCOUNTERS, encounter_resources=resources,
                            deck_generator=weighted_ironclad_deck,
                            relic_generator=weighted_ironclad_relics,
                            potion_generator=uniform_ironclad_potions, potion_count=2)
    policy = ActionScoringPolicy()
    _log("=== quick-v3: short run for a fair Silver-Automaton-comparison baseline ===")

    start = time.time()
    best_reward = float("-inf")
    best_state = None
    total_updates = 0
    chunk_num = 0

    pool = mp.Pool(N_WORKERS, initializer=_worker_init, initargs=(_env_kwargs_from(env),))
    try:
        while time.time() - start < TIME_BUDGET_SECONDS:
            chunk_num += 1
            history = train_ppo(
                env, policy, updates=CHUNK_UPDATES, episodes_per_update=EPISODES_PER_UPDATE,
                n_workers=N_WORKERS, pool=pool,
            )
            total_updates += CHUNK_UPDATES
            elapsed = time.time() - start

            (win, hp, reward) = evaluate(env, policy, n=CHECKPOINT_EVAL_N)
            _log(f"chunk {chunk_num:4d} (update {total_updates:6d}, {elapsed/60:4.1f}min): "
                 f"win {win*100:.1f}%  avg HP {hp:.1f}  reward {reward:.2f}")

            if reward > best_reward:
                best_reward = reward
                best_state = {k: v.clone() for k, v in policy.state_dict().items()}
                torch.save(best_state, OUT_CHECKPOINT)
    except Exception:
        import traceback
        _log(f"[FATAL] crashed after {total_updates} updates:\n{traceback.format_exc()}")
        return
    finally:
        pool.close()
        pool.join()

    _log(f"=== budget reached after {total_updates} updates, best_reward={best_reward:.2f} ===")


if __name__ == "__main__":
    main()
