"""CMA-ES tuning of the enriched leaf value-function weights (g_params.vf_*,
see nativeLeafValueEstimate in slaythespire.cpp) so that leaf_eval_mode="value"
-- static leaf evaluation, ~5x faster than full rollouts -- can STAND IN for a
rollout-to-terminal without the win-rate collapse the un-tuned potential showed
(28% win vs rollout's 85% this session).

This is the direct analog of how Silverbot built its fast evaluateEndState: a
weighted sum of state features, tuned (they used Optuna) specifically as a
standalone leaf estimator. We reuse the same CMA-ES machinery as
tune_search_cma.py, but here:
  - leaf_eval_mode is forced to "value" in every worker,
  - the BASE search/heuristic params (c_ucb, attack_base, ... per_card_weight_
    scale) are held FIXED at their already-tuned values (tuned_search_params.
    json) -- we're tuning ONLY the 10 value weights, isolating the question
    "can a tuned static value function recover rollout-level win rate?",
  - all 10 weights are searched in RAW units (additive), lower-bounded at 0
    (every vf_* weight's sign is baked into nativeLeafValueEstimate's formula,
    so weights are magnitudes and must stay non-negative to keep their meaning).

Run:  PYTHONPATH=".;../sts_lightspeed/build" python -m lightspeed.tune_value_leaf
"""
from __future__ import annotations

import multiprocessing as mp
import time

import cma
import numpy as np

from .search_config import apply_search_config, load_search_config

TIME_BUDGET_SECONDS = 2 * 60 * 60
N_EPISODES_PER_ENCOUNTER = 15
SIMS = 150
ENCOUNTERS = ["TIME_EATER", "DONU_AND_DECA", "GREMLIN_NOB", "HEXAGHOST", "CHAMP", "AUTOMATON"]
SIGMA0 = 0.25
N_WORKERS = 12
BASE_PARAMS_PATH = "lightspeed/tuned_search_params.json"  # fixed base search params (NOT tuned here)
OUT_PATH = "lightspeed/tuned_value_params.json"
LOG_PATH = "lightspeed/tune_value_leaf_progress.log"

# (name, lower, upper) -- all additive/raw-unit, all >= 0. Defaults come from
# get_search_params() (vf_hp=1.5, vf_monster_hp=1.0, vf_incoming=3.0, rest 0).
VALUE_PARAMS = [
    ("vf_hp", 0.0, 10.0),
    ("vf_monster_hp", 0.0, 10.0),
    ("vf_incoming", 0.0, 12.0),
    ("vf_block", 0.0, 6.0),
    ("vf_energy", 0.0, 20.0),
    ("vf_strength", 0.0, 30.0),
    ("vf_dexterity", 0.0, 20.0),
    ("vf_alive", 0.0, 30.0),
    ("vf_turn", 0.0, 10.0),
    ("vf_metallicize", 0.0, 15.0),
]
PARAM_NAMES = [n for n, _, _ in VALUE_PARAMS]


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# --- worker-side: each candidate evaluated in its own process ---------------
_worker_envs = None
_base_params = None


def _worker_init(base_config) -> None:
    global _worker_envs, _base_params
    import slaythespire as sts
    from lightspeed.env import IroncladFightEnv, build_full_encounter_resources
    from lightspeed.cards import weighted_ironclad_deck

    _base_params = base_config["params"]
    # Fix the base search params + value leaf mode ONCE per process (both are
    # process-global mutable state -- safe because each candidate below only
    # overwrites the vf_* subset, never the base, and processes never share).
    apply_search_config(base_config)
    sts.set_leaf_eval_mode("value")

    resources = build_full_encounter_resources()
    _worker_envs = {}
    for enc_idx, enc_name in enumerate(ENCOUNTERS):
        enc = getattr(sts.MonsterEncounter, enc_name)
        _worker_envs[enc_name] = IroncladFightEnv(
            encounter=enc, encounter_resources=resources, deck_generator=weighted_ironclad_deck,
        )


def _evaluate_candidate(args) -> float:
    x, seed_base = args
    import slaythespire as sts

    vf = {name: max(0.0, float(x[i])) for i, name in enumerate(PARAM_NAMES)}
    sts.set_search_params(vf)  # partial update -- keeps the base params set in _worker_init

    total_score = 0.0
    total_n = 0
    for enc_name in ENCOUNTERS:
        env = _worker_envs[enc_name]
        for seed in range(N_EPISODES_PER_ENCOUNTER):
            obs = env.reset(seed=seed_base + seed)
            done = False
            steps = 0
            info = None
            while not done and steps < 150:
                search_seed = ((seed_base + seed) << 32) ^ (enc_idx << 16) ^ steps
                action, _ = sts.run_mcts_search(env.bc, SIMS, None, search_seed)
                obs, reward, done, info = env.step(action)
                steps += 1
            won = info["outcome"] == sts.BattleOutcome.PLAYER_VICTORY
            score = 1.0 + (0.3 * info["player_hp"] / env.bc.player_max_hp if won else 0.0) if won else 0.0
            total_score += score
            total_n += 1
    return -(total_score / total_n)


def main():
    import sys
    sys.path.insert(0, r"C:\Users\Alvin\grok\sts-project\sts_lightspeed\build")
    import slaythespire as sts

    defaults = sts.get_search_params()

    base_config = load_search_config(BASE_PARAMS_PATH)

    x0 = [defaults[name] for name in PARAM_NAMES]
    lower = [lo for _, lo, _ in VALUE_PARAMS]
    upper = [hi for _, _, hi in VALUE_PARAMS]
    # Per-coordinate step scaling: SIGMA0 is calibrated relative to CMA_stds,
    # so give each raw-unit dimension an initial std of ~(range/4) (see
    # tune_search_cma.py's identical CMA_stds rationale for the additive dim).
    cma_stds = [((hi - lo) / 4.0) / SIGMA0 for _, lo, hi in VALUE_PARAMS]

    _log("=== CMA-ES leaf value-function tuning (leaf_eval_mode='value') ===")
    _log(f"encounters={ENCOUNTERS}, episodes/enc={N_EPISODES_PER_ENCOUNTER}, sims={SIMS}, "
         f"n_workers={N_WORKERS}, sigma0={SIGMA0}")
    _log(f"tuning {len(PARAM_NAMES)} value weights (base search params held fixed from {BASE_PARAMS_PATH})")
    _log(f"x0 (defaults, == un-enriched potential shape): {dict(zip(PARAM_NAMES, x0))}")

    es = cma.CMAEvolutionStrategy(
        x0, SIGMA0,
        {"popsize": N_WORKERS, "bounds": [lower, upper], "CMA_stds": cma_stds, "verbose": -9},
    )

    best_score = float("-inf")
    best_params = dict(zip(PARAM_NAMES, x0))
    start = time.time()
    gen = 0
    seed_base = 0

    with mp.Pool(N_WORKERS, initializer=_worker_init, initargs=(base_config,)) as pool:
        while time.time() - start < TIME_BUDGET_SECONDS and not es.stop():
            gen += 1
            candidates = es.ask()
            args_list = [(np.array(x), seed_base) for x in candidates]
            seed_base += N_EPISODES_PER_ENCOUNTER
            fitnesses = pool.map(_evaluate_candidate, args_list)
            es.tell(candidates, fitnesses)

            gen_best_idx = int(np.argmin(fitnesses))
            gen_best_score = -fitnesses[gen_best_idx]
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_x = candidates[gen_best_idx]
                best_params = {name: max(0.0, float(best_x[i])) for i, name in enumerate(PARAM_NAMES)}
                with open(OUT_PATH, "w") as f:
                    json.dump({"score": best_score, "value_params": best_params,
                               "base_params_from": BASE_PARAMS_PATH}, f, indent=2)

            elapsed = time.time() - start
            mean_score = -float(np.mean(fitnesses))
            _log(f"gen {gen:3d} (t={elapsed/60:4.1f}m): mean={mean_score:.3f} "
                 f"gen_best={gen_best_score:.3f} all_time_best={best_score:.3f}")

    _log(f"=== done after {gen} generations ({(time.time()-start)/60:.1f}m) ===")
    _log(f"best score: {best_score:.3f}")
    _log(f"best value weights: {best_params}")
    _log(f"saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
