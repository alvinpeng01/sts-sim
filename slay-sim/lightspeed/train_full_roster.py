"""Train against the FULL verified encounter roster (Act1+2+3, all tiers),
per the methodology shift: encounters are a fixed/enumerable set (we've now
verified nearly all of them run cleanly through the bindings), so the
training pool should just cover all of them -- it's deck composition that's
the actually-unpredictable thing at deployment, which is what generalization
testing should stress (see eval_deck_generalization.py).

Run:  PYTHONPATH=. .venv/bin/python -m lightspeed.train_full_roster
"""

from __future__ import annotations

import time

from .env import IroncladFightEnv, ALL_ENCOUNTERS, ALL_ACT_TIER_GROUPS, build_full_encounter_resources
from .policy import ActionScoringPolicy
from .ppo import train_ppo
from .train import evaluate

import torch

UPDATES = 6000
EPISODES_PER_UPDATE = 16
CHECKPOINT_EVERY = 200
CHECKPOINT_EVAL_N = 150

if __name__ == "__main__":
    resources = build_full_encounter_resources()
    env = IroncladFightEnv(encounter=ALL_ENCOUNTERS, encounter_resources=resources)
    policy = ActionScoringPolicy()

    tier_counts = {}
    for act, tier, encs in ALL_ACT_TIER_GROUPS:
        tier_counts[f"{act}/{tier}"] = len(encs)
    print(f"pool: {len(ALL_ENCOUNTERS)} encounters -- {tier_counts}")
    print(f"total updates={UPDATES}")
    print()

    start = time.time()

    def eval_and_report(update):
        elapsed = time.time() - start
        eps_per_sec = (update + 1) * EPISODES_PER_UPDATE / elapsed
        (win, hp, reward), breakdown = evaluate(env, policy, n=CHECKPOINT_EVAL_N, per_encounter=True)
        by_act = {}
        for act, tier, encs in ALL_ACT_TIER_GROUPS:
            ws = [breakdown[e] for e in encs if e in breakdown]
            if ws:
                by_act.setdefault(act, []).extend(ws)
        act_str = "  |  " + "  ".join(f"{a} {sum(v)/len(v)*100:.1f}%" for a, v in sorted(by_act.items()))
        print(f"update {update+1:5d} ({elapsed:6.1f}s, {eps_per_sec:.1f} eps/sec): "
              f"overall win {win*100:.1f}%  avg HP {hp:.1f}  reward {reward:.2f}{act_str}")
        return reward

    # train_ppo doesn't call back mid-run, so drive it in CHECKPOINT_EVERY-sized
    # chunks from here to get periodic reporting identical in shape to the
    # Act1-full-pool run's log. Each chunk already returns its own
    # internal-best state (via checkpoint_every=chunk), so the eval below
    # both reports progress and drives the across-chunk best tracking --
    # one evaluate() call serves both purposes, not two.
    history = []
    best_reward = float("-inf")
    best_state = None
    done_updates = 0
    while done_updates < UPDATES:
        chunk = min(CHECKPOINT_EVERY, UPDATES - done_updates)
        h, state = train_ppo(env, policy, updates=chunk, episodes_per_update=EPISODES_PER_UPDATE,
                              checkpoint_every=chunk, checkpoint_eval_n=CHECKPOINT_EVAL_N)
        history.extend(h)
        done_updates += chunk
        r = eval_and_report(done_updates - 1)
        if r > best_reward:
            best_reward = r
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}

    if best_state is not None:
        policy.load_state_dict(best_state)
    print(f"\nbest checkpoint reward {best_reward:.2f}, saved to lightspeed/checkpoint_full_roster.pt")
    torch.save(policy.state_dict(), "lightspeed/checkpoint_full_roster.pt")

    print("\n=== final per-encounter breakdown (best checkpoint), large sample ===")
    (win, hp, reward), breakdown = evaluate(env, policy, n=400, per_encounter=True)
    print(f"overall: win {win*100:.1f}%  avg HP {hp:.1f}  avg reward {reward:.2f}")
    for act, tier, encs in ALL_ACT_TIER_GROUPS:
        for e in encs:
            if e in breakdown:
                print(f"  [{act}/{tier:5s}] {e}: {breakdown[e]*100:.1f}%")
