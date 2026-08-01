"""Same chunked PPO training loop as train_distillation_v5.py, but the
periodic search-distillation round now imitates expectimax_search.py's
native, network-free MCTS instead of az_search.py's NN-guided PUCT search
-- see distillation.py's own module docstring for the full story (a real,
measured comparison this session found the no-NN search beating PUCT search
100% vs 0-20% on Time Eater/Donu & Deca at matched compute, plus several
further correctness/quality fixes landed on it the same session: defense-
vs-attack scoring, Strength/Weak/Vulnerable damage prediction, a Haste-
penalty gating bug, and a loss-progress-credit terminal evaluation found by
reading Silver Automaton's own source -- together roughly Time Eater 0%->
30-40%, Donu & Deca 40%->75-80% on the session's own test deck).

Two knock-on changes from the teacher swap:
  - DISTILL_SIMULATIONS is much higher than v5's (200 vs 20) -- affordable
    because the native search measured ~9x faster than the old pure-Python
    MCTS v5's number was tuned around, on top of which root parallelization
    (DISTILL_TREES real OS threads per decision, see
    expectimax_search.root_parallel_search) buys a further ~1.9x wall-clock
    with no measured accuracy cost. DISTILL_EPISODES is correspondingly
    larger (64 vs 40) since the same wall-clock buys a bigger batch now.
  - collect_distillation_batch_parallel no longer needs a policy/state_dict
    argument at all (expectimax's search never reads the network) -- one
    less thing to pickle across the process boundary every round.

DISTILL_WORKERS * DISTILL_TREES = 6 * 2 = 12 concurrent units (processes *
threads-per-process), matched to this machine's core count -- process-level
parallelism across episodes, thread-level within each episode's own search.

Run:  PYTHONPATH=. python -m lightspeed.train_distillation_expectimax
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
from .search_config import load_search_config

# CMA-ES-tuned search constants (tune_search_cma.py), applied to every
# distillation round's worker processes -- validated on held-out seeds vs
# the hand-tuned defaults: Time Eater 32%->80% win rate, Donu & Deca
# 68%->88%, and HP-on-win margin improved across EVERY encounter tested
# (e.g. Spheric Guardian 47.4%->70.6% even though both were already 100%
# win rate) -- a broad, real improvement, not just overfitting to the
# tuning run's own small evaluation set. Falls back to None (the C++
# module's own compiled-in defaults) if the tuned configuration file doesn't
# exist yet, so this script still runs standalone.
TUNED_SEARCH_PARAMS_PATH = "lightspeed/tuned_search_params.json"
try:
    SEARCH_CONFIG = load_search_config(TUNED_SEARCH_PARAMS_PATH)
except FileNotFoundError:
    SEARCH_CONFIG = None

TIME_BUDGET_SECONDS = 8 * 60 * 60
EPISODES_PER_UPDATE = 32
N_WORKERS = 6
CHUNK_UPDATES = 25
CHECKPOINT_EVAL_N = 100
FINAL_EVAL_N = 500
CURRICULUM_K = 4.0
EMA_ALPHA = 0.3

DISTILL_EVERY_CHUNKS = 40
DISTILL_EPISODES = 64
DISTILL_SIMULATIONS = 200
DISTILL_WORKERS = 6
DISTILL_TREES = 2
DISTILL_LR = 3e-5
DISTILL_EPOCHS = 2

OUT_CHECKPOINT = "lightspeed/checkpoint_distillation_expectimax.pt"
LOG_PATH = "lightspeed/train_distillation_expectimax_progress.log"


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
    # One distillation optimizer for the WHOLE run (not recreated per
    # round) -- unlike train_ppo's own per-chunk fresh Adam instance, this
    # one runs infrequently enough that keeping its momentum state across
    # rounds is the more sensible default, not an established convention
    # to match.
    distill_optimizer = torch.optim.Adam(policy.parameters(), lr=DISTILL_LR)

    _log("=== expectimax-distillation run: v5's PPO stack + periodic bounded "
         "search-distillation from the native expectimax teacher (from scratch, A0) ===")
    _log(f"n_workers={N_WORKERS}, episodes_per_update={EPISODES_PER_UPDATE}, K={CURRICULUM_K}, ema_alpha={EMA_ALPHA}")
    _log(f"distillation: every {DISTILL_EVERY_CHUNKS} chunks, {DISTILL_EPISODES} episodes, "
         f"{DISTILL_SIMULATIONS} sims/decision, {DISTILL_WORKERS} workers x {DISTILL_TREES} trees, lr={DISTILL_LR}")
    _log(f"search config: {'CMA-ES-tuned (' + TUNED_SEARCH_PARAMS_PATH + ')' if SEARCH_CONFIG else 'C++ module defaults (no tuned-params file found)'}")

    win_rate_estimate = {e: 0.5 for e in ALL_ENCOUNTERS}

    start = time.time()
    best_reward = float("-inf")
    best_state = None
    total_updates = 0
    chunk_num = 0
    batch_reward_history = []
    # Tracks actual compute time (sum of chunk/distill durations), NOT
    # wall-clock time.time() - start -- see train_distillation_v5.py's own
    # comment on why (immune to the process sleeping mid-run).
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
            elapsed = time.time() - start  # wall-clock, logged for reference only
            eps_per_sec = (CHUNK_UPDATES * EPISODES_PER_UPDATE) / chunk_dt

            if any(r != r for r in history):
                _log(f"[FATAL] NaN batch reward detected in chunk {chunk_num}, history: {history}")
                return

            (win, hp, reward), breakdown = evaluate(env, policy, n=CHECKPOINT_EVAL_N, per_encounter=True)
            by_act = per_act_breakdown(breakdown)
            act_str = "  ".join(f"{a} {w*100:.1f}%" for a, w in by_act.items())
            _log(f"chunk {chunk_num:4d} (update {total_updates:6d}, active {active_time/3600:4.2f}h / "
                 f"wall {elapsed/3600:4.2f}h, {eps_per_sec:6.1f} eps/sec): win {win*100:.1f}%  avg HP {hp:.1f}  "
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
                    n_workers=DISTILL_WORKERS, n_trees=DISTILL_TREES, search_params=SEARCH_CONFIG,
                )
                stats = distillation_update(policy, distill_optimizer, steps, epochs=DISTILL_EPOCHS)
                distill_dt = time.time() - distill_t0
                active_time += distill_dt  # counts toward the budget too -- it's real training time
                _log(f"  [distill] {stats['n_steps']} steps from {DISTILL_EPISODES} search-driven episodes "
                     f"({distill_dt:.1f}s): policy_loss={stats['policy_loss']:.3f}  value_loss={stats['value_loss']:.1f}")

            if chunk_num % 40 == 0:
                worst = sorted(win_rate_estimate.items(), key=lambda kv: kv[1])[:5]
                _log("  curriculum lowest-5: " + ", ".join(f"{e}={r*100:.0f}%" for e, r in worst))
    except Exception:
        import traceback
        _log(f"[FATAL] crashed after {total_updates} updates:\n{traceback.format_exc()}")
        return

    _log(f"=== budget reached after {total_updates} updates ({active_time/3600:.2f}h active / "
         f"{(time.time()-start)/3600:.2f}h wall) ===")
    _log(f"batch reward trend (first 3 vs last 3 chunks' mean): "
         f"{sum(batch_reward_history[:3])/max(1,len(batch_reward_history[:3])):.2f} -> "
         f"{sum(batch_reward_history[-3:])/max(1,len(batch_reward_history[-3:])):.2f}")
    if best_state is not None:
        policy.load_state_dict(best_state)

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
