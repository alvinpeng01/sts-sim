"""Fine-tune checkpoint_5hour_v4.pt with relics enabled -- the first real
training run since relic modeling (bindings, real-data-weighted sampling,
env.py wiring) was built and crash-verified this session (3000-trial fuzz
sweep across every encounter x relic-count tier, zero crashes/exceptions
after excluding NINJA_SCROLL/HOLY_WATER/THE_SPECIMEN/NEOWS_LAMENT).

Fine-tunes from checkpoint_5hour_v4.pt rather than starting from scratch:
relics are an additional per-episode variation on top of matchups the
policy already plays reasonably well (90.4% overall pre-relics), not a
change to the reward/observation architecture itself, so there's no
value-head-invalidation reason to restart (contrast with an actual reward
formula change, which DOES invalidate every existing checkpoint's value
head -- see env.py's own comment on that).

Known simplification, not addressed here: _encode_state/
_encode_action_and_card_idx have no explicit "which relics does the player
have" feature. Relic effects still reach the policy indirectly through
existing state features (block, strength, energy, HP deltas, etc.), same
as how a human infers an unfamiliar relic's presence from its downstream
effects -- just less sample-efficient than an explicit feature would be.
Adding one is a separate follow-up, not required for relics to train at all.

Run:  PYTHONPATH=. python -m lightspeed.train_relics_v1
"""

from __future__ import annotations

import time

import torch

from .env import IroncladFightEnv, ALL_ENCOUNTERS, ALL_ACT_TIER_GROUPS, build_full_encounter_resources
from .cards import weighted_ironclad_deck
from .relics import weighted_ironclad_relics
from .policy import ActionScoringPolicy
from .ppo import train_ppo
from .train import evaluate

TIME_BUDGET_SECONDS = 4 * 60 * 60
EPISODES_PER_UPDATE = 32
N_WORKERS = 6
CHUNK_UPDATES = 25
CHECKPOINT_EVAL_N = 100
FINAL_EVAL_N = 500
CURRICULUM_K = 4.0
EMA_ALPHA = 0.3
IN_CHECKPOINT = "lightspeed/checkpoint_5hour_v4.pt"
OUT_CHECKPOINT = "lightspeed/checkpoint_relics_v1.pt"
LOG_PATH = "lightspeed/train_relics_v1_progress.log"


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
    # 5th element of each ACT_TIER_RESOURCES entry is relic_count -- already
    # baked in from this session's relic work, just unused until a
    # relic_generator is actually supplied (see env.py's reset()).
    resources = build_full_encounter_resources()
    env = IroncladFightEnv(encounter=ALL_ENCOUNTERS, encounter_resources=resources,
                            deck_generator=weighted_ironclad_deck,
                            relic_generator=weighted_ironclad_relics)
    policy = ActionScoringPolicy()
    policy.load_state_dict(torch.load(IN_CHECKPOINT, map_location="cpu", weights_only=False))
    _log("=== relics-v1 run: fine-tuning checkpoint_5hour_v4.pt with relics enabled ===")
    _log(f"n_workers={N_WORKERS}, episodes_per_update={EPISODES_PER_UPDATE}, K={CURRICULUM_K}, ema_alpha={EMA_ALPHA}")

    win_rate_estimate = {e: 0.5 for e in ALL_ENCOUNTERS}

    start = time.time()
    best_reward = float("-inf")
    best_state = None
    total_updates = 0
    chunk_num = 0
    batch_reward_history = []

    try:
        while time.time() - start < TIME_BUDGET_SECONDS:
            chunk_num += 1
            chunk_t0 = time.time()
            history, chunk_best = train_ppo(
                env, policy, updates=CHUNK_UPDATES, episodes_per_update=EPISODES_PER_UPDATE,
                checkpoint_every=CHUNK_UPDATES, checkpoint_eval_n=CHECKPOINT_EVAL_N,
                n_workers=N_WORKERS,
            )
            batch_reward_history.extend(history)
            total_updates += CHUNK_UPDATES
            elapsed = time.time() - start
            chunk_dt = time.time() - chunk_t0
            eps_per_sec = (CHUNK_UPDATES * EPISODES_PER_UPDATE) / chunk_dt

            if any(r != r for r in history):
                _log(f"[FATAL] NaN batch reward detected in chunk {chunk_num}, history: {history}")
                return

            (win, hp, reward), breakdown = evaluate(env, policy, n=CHECKPOINT_EVAL_N, per_encounter=True)
            by_act = per_act_breakdown(breakdown)
            act_str = "  ".join(f"{a} {w*100:.1f}%" for a, w in by_act.items())
            _log(f"chunk {chunk_num:4d} (update {total_updates:6d}, {elapsed/3600:4.2f}h, "
                 f"{eps_per_sec:6.1f} eps/sec): win {win*100:.1f}%  avg HP {hp:.1f}  reward {reward:.2f}  |  {act_str}")

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

            if chunk_num % 40 == 0:
                worst = sorted(win_rate_estimate.items(), key=lambda kv: kv[1])[:5]
                _log("  curriculum lowest-5: " + ", ".join(f"{e}={r*100:.0f}%" for e, r in worst))
    except Exception:
        import traceback
        _log(f"[FATAL] crashed after {total_updates} updates:\n{traceback.format_exc()}")
        return

    _log(f"=== budget reached after {total_updates} updates ===")
    _log(f"batch reward trend (first 3 vs last 3 chunks' mean): "
         f"{sum(batch_reward_history[:3])/max(1,len(batch_reward_history[:3])):.2f} -> "
         f"{sum(batch_reward_history[-3:])/max(1,len(batch_reward_history[-3:])):.2f}")
    if best_state is not None:
        policy.load_state_dict(best_state)

    env.encounter_weights = None  # uniform encounters for the comparable final eval
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
