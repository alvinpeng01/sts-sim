"""Held-out-card generalization test: train with a set of cards EXCLUDED
from every deck, then check whether the agent still plays reasonably when
those same cards show up at eval time. This is the actual test of whether
the action-scoring architecture generalizes across cards via shared weights
+ embeddings, versus just memorizing whatever specific cards it happened to
train on -- the thing "I want the bot to learn all cards" actually needs.

Run:  PYTHONPATH=. .venv/bin/python -m lightspeed.eval_holdout
"""

from __future__ import annotations

import numpy as np
import slaythespire as sts

from .cards import card_index
from .env import IroncladFightEnv
from .policy import ActionScoringPolicy
from .train import train, evaluate, run_episode

# Ten cards spanning common/uncommon/rare and attack/skill/power, held out
# of every training deck entirely.
HELD_OUT_CARDS = [
    sts.CardId.TWIN_STRIKE,     # common attack
    sts.CardId.IRON_WAVE,       # common attack+block hybrid
    sts.CardId.FLEX,            # common skill
    sts.CardId.METALLICIZE,     # uncommon power
    sts.CardId.UPPERCUT,        # uncommon attack+debuff
    sts.CardId.WHIRLWIND,       # uncommon X-cost AoE
    sts.CardId.BLUDGEON,        # rare big single hit
    sts.CardId.DEMON_FORM,      # rare power
    sts.CardId.OFFERING,        # rare skill
    sts.CardId.FEED,            # rare execute
]


def held_out_action_rate(env: IroncladFightEnv, policy, n: int, seed_offset=0):
    """Of the times a held-out card was a legal action, how often did the
    policy actually choose it (vs some other legal action)? A policy that's
    just afraid of unknown cards would pick them far less than their legal-
    action share; a policy that's learned real card semantics should pick
    them roughly in proportion to how good they are, not near-zero out of
    pure uncertainty."""
    held_out_idxs = {card_index(cid) for cid in HELD_OUT_CARDS}
    chosen, legal_total = 0, 0
    import torch
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


def main():
    HP, EXTRA = 30, 8

    train_env = IroncladFightEnv(player_hp=HP, extra_deck_cards=EXTRA,
                                 deck_exclude=HELD_OUT_CARDS)
    policy = ActionScoringPolicy()

    print(f"=== training WITHOUT these {len(HELD_OUT_CARDS)} cards ever appearing ===")
    print([str(sts.Card(c)) for c in HELD_OUT_CARDS])
    hist = train(train_env, policy, updates=200, episodes_per_update=16)
    print(f"final training batch reward: {np.mean(hist[-10:]):.2f}")

    control_env = IroncladFightEnv(player_hp=HP, extra_deck_cards=EXTRA)  # same distribution as training
    holdout_env = IroncladFightEnv(player_hp=HP, extra_deck_cards=EXTRA,
                                   deck_force_include=HELD_OUT_CARDS)  # every deck has ALL 10 held-out cards

    print("\n=== eval: in-distribution control (no held-out cards forced, matches training) ===")
    wr_c, hp_c, rew_c = evaluate(control_env, policy, n=150, seed_offset=10000)
    print(f"win {wr_c:.1%}  avg HP {hp_c:.1f}  avg reward {rew_c:.2f}")

    print("\n=== eval: held-out cards FORCED into every deck (never seen in training) ===")
    wr_h, hp_h, rew_h = evaluate(holdout_env, policy, n=150, seed_offset=20000)
    print(f"win {wr_h:.1%}  avg HP {hp_h:.1f}  avg reward {rew_h:.2f}")

    print(f"\ngeneralization gap: win {wr_h-wr_c:+.1%}, reward {rew_h-rew_c:+.2f}")

    print("\n=== does it actually USE the held-out cards, or avoid them? ===")
    rate, n_legal = held_out_action_rate(holdout_env, policy, n=80, seed_offset=30000)
    print(f"held-out cards were legal {n_legal} times across 80 fights; "
          f"chosen {rate:.1%} of those times")


if __name__ == "__main__":
    main()
