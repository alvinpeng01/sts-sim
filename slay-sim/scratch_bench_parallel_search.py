import time

import torch

import slaythespire as sts
from lightspeed.env import IroncladFightEnv, build_full_encounter_resources
from lightspeed.policy import ActionScoringPolicy
from lightspeed import az_search

if __name__ == "__main__":
    resources = build_full_encounter_resources()
    policy = ActionScoringPolicy()
    policy.load_state_dict(torch.load("lightspeed/checkpoint_5hour_v4.pt", map_location="cpu", weights_only=False))
    policy.eval()

    env = IroncladFightEnv(encounter=sts.MonsterEncounter.AWAKENED_ONE, encounter_resources=resources)

    N = 60
    N_SIM = 40
    N_WORKERS = 4  # kept modest since train_relics_v1 is concurrently using 6 workers on this same 12-core machine

    t0 = time.time()
    serial_result = az_search.evaluate_with_search(env, policy, n=N, n_simulations=N_SIM)
    dt_serial = time.time() - t0
    print(f"serial:   n={N} in {dt_serial:.1f}s ({dt_serial/N:.3f}s/ep) win={serial_result[0]*100:.1f}%")

    t0 = time.time()
    parallel_result = az_search.evaluate_with_search_parallel(env, policy, n=N, n_simulations=N_SIM, n_workers=N_WORKERS)
    dt_parallel = time.time() - t0
    print(f"parallel: n={N} in {dt_parallel:.1f}s ({dt_parallel/N:.3f}s/ep) win={parallel_result[0]*100:.1f}% n_workers={N_WORKERS}")

    print(f"speedup: {dt_serial/dt_parallel:.2f}x (vs {N_WORKERS} workers, {N_WORKERS} of 12 cores, "
          f"train_relics_v1 concurrently using ~6)")
