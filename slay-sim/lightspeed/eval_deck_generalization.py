"""Deck generalization test, scaled to the full verified encounter roster.

Same idea as eval_holdout.py (train with certain cards excluded from every
deck, then check whether the agent still performs when those cards show up),
but against ALL_ENCOUNTERS instead of a single fixed fight. This is the
right axis to stress: the encounter roster is fixed and enumerable (we train
on all of it), but the deck a run hands the agent is the actually
unpredictable thing at deployment -- so "does this generalize" should mean
"does it generalize across decks, given it already knows every matchup",
not "does it generalize across matchups it's never seen."

Run:  PYTHONPATH=. .venv/bin/python -m lightspeed.eval_deck_generalization
"""

from __future__ import annotations

import time

import numpy as np
import slaythespire as sts
import torch

from .cards import card_index
from .env import IroncladFightEnv, ALL_ENCOUNTERS, ALL_ACT_TIER_GROUPS, build_full_encounter_resources
from .policy import ActionScoringPolicy
from .ppo import train_ppo
from .train import evaluate

# Same 10 cards as eval_holdout.py -- common/uncommon/rare, attack/skill/power.
HELD_OUT_CARDS = [
    sts.CardId.TWIN_STRIKE,
    sts.CardId.IRON_WAVE,
    sts.CardId.FLEX,
    sts.CardId.METALLICIZE,
    sts.CardId.UPPERCUT,
    sts.CardId.WHIRLWIND,
    sts.CardId.BLUDGEON,
    sts.CardId.DEMON_FORM,
    sts.CardId.OFFERING,
    sts.CardId.FEED,
]

UPDATES = 3000
EPISODES_PER_UPDATE = 16


def held_out_action_rate(env, policy, n, seed_offset=0):
    held_out_idxs = {card_index(cid) for cid in HELD_OUT_CARDS}
    chosen, legal_total = 0, 0
    with torch.no_grad():
        for i in range(n):
            obs = env.reset(seed=seed_offset + i)
            done = False
            while not done:
                idx, _, _ = policy.act(obs, sample=False)
                for j, cidx in enumerate(obs["action_card_idx"]):
                    if cidx in held_out_idxs:
                        legal_total += 1
                        if j == idx:
                            chosen += 1
                action = obs["actions"][idx]
                obs, reward, done, info = env.step(action)
                if obs is None:
                    break
    return chosen / max(legal_total, 1), legal_total


def per_act_breakdown(breakdown):
    by_act = {}
    for act, tier, encs in ALL_ACT_TIER_GROUPS:
        ws = [breakdown[e] for e in encs if e in breakdown]
        if ws:
            by_act.setdefault(act, []).extend(ws)
    return {a: sum(v) / len(v) for a, v in by_act.items()}


if __name__ == "__main__":
    resources = build_full_encounter_resources()

    train_env = IroncladFightEnv(encounter=ALL_ENCOUNTERS, encounter_resources=resources,
                                  deck_exclude=HELD_OUT_CARDS)
    policy = ActionScoringPolicy()

    print(f"=== training on all {len(ALL_ENCOUNTERS)} encounters, "
          f"WITHOUT these {len(HELD_OUT_CARDS)} cards ever appearing in a deck ===")
    print([str(sts.Card(c)) for c in HELD_OUT_CARDS])
    print(f"total updates={UPDATES}")

    start = time.time()
    history, best_state = train_ppo(train_env, policy, updates=UPDATES,
                                     episodes_per_update=EPISODES_PER_UPDATE,
                                     checkpoint_every=200, checkpoint_eval_n=150)
    if best_state is not None:
        policy.load_state_dict(best_state)
    elapsed = time.time() - start
    print(f"done in {elapsed:.1f}s, final training batch reward: {np.mean(history[-10:]):.2f}")
    torch.save(policy.state_dict(), "lightspeed/checkpoint_full_roster_heldout.pt")

    control_env = IroncladFightEnv(encounter=ALL_ENCOUNTERS, encounter_resources=resources)
    holdout_env = IroncladFightEnv(encounter=ALL_ENCOUNTERS, encounter_resources=resources,
                                    deck_force_include=HELD_OUT_CARDS)

    print("\n=== eval: in-distribution control (random decks, matches training) ===")
    (wr_c, hp_c, rew_c), bd_c = evaluate(control_env, policy, n=400, seed_offset=10000, per_encounter=True)
    print(f"win {wr_c:.1%}  avg HP {hp_c:.1f}  avg reward {rew_c:.2f}")
    for act, wr in sorted(per_act_breakdown(bd_c).items()):
        print(f"  {act}: {wr:.1%}")

    print("\n=== eval: 10 held-out cards FORCED into every deck (never seen in training) ===")
    (wr_h, hp_h, rew_h), bd_h = evaluate(holdout_env, policy, n=400, seed_offset=20000, per_encounter=True)
    print(f"win {wr_h:.1%}  avg HP {hp_h:.1f}  avg reward {rew_h:.2f}")
    for act, wr in sorted(per_act_breakdown(bd_h).items()):
        print(f"  {act}: {wr:.1%}")

    print(f"\ngeneralization gap: win {wr_h-wr_c:+.1%}, reward {rew_h-rew_c:+.2f}")

    print("\n=== does it actually USE the held-out cards, or avoid them? ===")
    rate, n_legal = held_out_action_rate(holdout_env, policy, n=150, seed_offset=30000)
    print(f"held-out cards were legal {n_legal} times across 150 fights; "
          f"chosen {rate:.1%} of those times")
