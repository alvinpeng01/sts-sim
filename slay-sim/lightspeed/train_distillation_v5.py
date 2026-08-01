"""Same chunked PPO training loop as train_5hour_v4.py (reward-v2 + obs-v2
+ monster-embed + saturating-norm + curriculum + weighted-deck), plus a
periodic BOUNDED search-distillation round (see distillation.py) every
DISTILL_EVERY_CHUNKS chunks: a small batch of search-driven self-play
episodes, then a supervised fine-tune step training the policy to match
search's own visit-count distribution and the value head against real
outcomes. The deployed/autonomous bot this is ultimately for never runs
search itself -- it plays at the fast blind-policy speed (13-50ms/fight,
measured), having absorbed whatever tactical improvement search
demonstrated (real, measured wins on Time Eater/Reptomancer/Automaton/
Champ) directly into its weights during training.

Known, deliberately accepted caveat: search itself has a demonstrated
blind spot on long/multi-phase fights (Awakened One: 28.0% blind vs 15.0%
search, n=100, statistically real; neither more simulations nor leaf-
averaging closed the gap -- see az_search.py's module docstring). This
means distillation rounds will occasionally include imitation targets from
a decision-maker that's known to be weak on exactly that kind of fight.
Mitigated by keeping distillation a small, occasional nudge (DISTILL_LR is
low, and it's blended with the much larger volume of ordinary PPO updates
on real environment reward, never a replacement for them) -- if this
still measurably hurts long-fight performance in practice, that's the
first thing to check in the per-encounter breakdown.

DISTILL_EPISODES/DISTILL_SIMULATIONS are deliberately modest (not the 40
sims used for evaluation) specifically to keep each round's wall-clock
cost bounded -- this is a periodic tax on training time, not something
that should dominate it.

Run:  PYTHONPATH=. python -m lightspeed.train_distillation_v5
"""

from __future__ import annotations

import time

import torch

from .env import IroncladFightEnv, ALL_ENCOUNTERS, ALL_ACT_TIER_GROUPS, build_full_encounter_resources
from .cards import weighted_ironclad_deck
from .policy import ActionScoringPolicy
from .ppo import train_ppo
from .train import evaluate
from .distillation import collect_distillation_batch, distillation_update

TIME_BUDGET_SECONDS = 8 * 60 * 60
EPISODES_PER_UPDATE = 32
N_WORKERS = 6
CHUNK_UPDATES = 25
CHECKPOINT_EVAL_N = 100
FINAL_EVAL_N = 500
CURRICULUM_K = 4.0
EMA_ALPHA = 0.3

DISTILL_EVERY_CHUNKS = 40
DISTILL_EPISODES = 40
DISTILL_SIMULATIONS = 20
DISTILL_LR = 3e-5
DISTILL_EPOCHS = 2

OUT_CHECKPOINT = "lightspeed/checkpoint_distillation_v5.pt"
LOG_PATH = "lightspeed/train_distillation_v5_progress.log"


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

    _log("=== 8-hour v5 run: v4's stack + periodic bounded search-distillation (from scratch, A0) ===")
    _log(f"n_workers={N_WORKERS}, episodes_per_update={EPISODES_PER_UPDATE}, K={CURRICULUM_K}, ema_alpha={EMA_ALPHA}")
    _log(f"distillation: every {DISTILL_EVERY_CHUNKS} chunks, {DISTILL_EPISODES} episodes, "
         f"{DISTILL_SIMULATIONS} sims/decision, lr={DISTILL_LR}")

    win_rate_estimate = {e: 0.5 for e in ALL_ENCOUNTERS}

    start = time.time()
    best_reward = float("-inf")
    best_state = None
    total_updates = 0
    chunk_num = 0
    batch_reward_history = []
    # Tracks actual compute time (sum of chunk/distill durations), NOT
    # wall-clock time.time() - start -- a prior 5-hour run's budget check
    # used wall-clock elapsed and got fooled by a ~4-hour machine sleep in
    # the middle, silently cutting real training down to ~1.6h while
    # reporting "budget reached" as if it had run the full 5. Summing each
    # chunk's own measured duration is immune to that: a sleeping process
    # doesn't accumulate any chunk_dt while suspended.
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
                steps = collect_distillation_batch(
                    env, policy, n_episodes=DISTILL_EPISODES, n_simulations=DISTILL_SIMULATIONS,
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
