"""Shore up Act 2 elites/bosses (the checkpoint_full_roster.pt weak spot:
Gremlin Leader 50%, Automaton 60%, Collector 22.2%, Champ 16.7%) without
forgetting the rest of the roster.

Approach: continue training from checkpoint_full_roster.pt against the FULL
42-encounter pool (so nothing gets zero training signal and regresses via
catastrophic forgetting), but with Act2 elites+bosses duplicated into the
pool several times each so reset() samples them far more often than their
1/42 base share -- oversampling, not a narrowed pool. Evaluates against the
full roster afterward specifically to check both halves of the claim: did
the target encounters improve, and did anything else get worse.

Run:  PYTHONPATH=. .venv/bin/python -m lightspeed.finetune_act2_elite_boss
"""

from __future__ import annotations

import time

import torch

from .env import (
    IroncladFightEnv, ALL_ENCOUNTERS, ALL_ACT_TIER_GROUPS,
    ACT2_ELITE_ENCOUNTERS, ACT2_BOSS_ENCOUNTERS, build_full_encounter_resources,
)
from .policy import ActionScoringPolicy
from .ppo import train_ppo
from .train import evaluate

UPDATES = 2500
EPISODES_PER_UPDATE = 16
CHECKPOINT_EVERY = 200
CHECKPOINT_EVAL_N = 150
OVERSAMPLE_FACTOR = 4  # each Act2 elite/boss encounter appears 1 + this many extra times

if __name__ == "__main__":
    resources = build_full_encounter_resources()
    target = ACT2_ELITE_ENCOUNTERS + ACT2_BOSS_ENCOUNTERS
    pool = ALL_ENCOUNTERS + target * OVERSAMPLE_FACTOR

    env = IroncladFightEnv(encounter=pool, encounter_resources=resources)
    policy = ActionScoringPolicy()
    policy.load_state_dict(torch.load("lightspeed/checkpoint_full_roster.pt"))

    print(f"fine-tuning from checkpoint_full_roster.pt: {len(ALL_ENCOUNTERS)} encounters, "
          f"Act2 elites+bosses ({len(target)}) oversampled {OVERSAMPLE_FACTOR}x "
          f"(pool size {len(pool)})")
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
        target_str = "  ||  " + "  ".join(
            f"{str(e).split('.')[-1]} {breakdown[e]*100:.0f}%" for e in target if e in breakdown
        )
        print(f"update {update+1:5d} ({elapsed:6.1f}s, {eps_per_sec:.1f} eps/sec): "
              f"overall win {win*100:.1f}%  avg HP {hp:.1f}  reward {reward:.2f}{act_str}{target_str}")
        return reward

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
    print(f"\nbest checkpoint reward {best_reward:.2f}, saved to lightspeed/checkpoint_full_roster_act2boosted.pt")
    torch.save(policy.state_dict(), "lightspeed/checkpoint_full_roster_act2boosted.pt")

    print("\n=== final per-encounter breakdown (best checkpoint), large sample, FULL roster (uniform, not oversampled) ===")
    eval_env = IroncladFightEnv(encounter=ALL_ENCOUNTERS, encounter_resources=resources)
    (win, hp, reward), breakdown = evaluate(eval_env, policy, n=400, per_encounter=True)
    print(f"overall: win {win*100:.1f}%  avg HP {hp:.1f}  avg reward {reward:.2f}")
    for act, tier, encs in ALL_ACT_TIER_GROUPS:
        for e in encs:
            if e in breakdown:
                flag = "  <-- target" if e in target else ""
                print(f"  [{act}/{tier:5s}] {e}: {breakdown[e]*100:.1f}%{flag}")
