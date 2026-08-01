"""Record what the SEARCH picks, so the rollout policy can be trained to imitate it.

The rollout policy is the measured ceiling on combat. Search budget is flat from
43 to 1500 simulations (see `docs/03-combat-search.md`), so the tree is not the
constraint -- but running the same fights at 1 simulation, which is essentially
the bare rollout policy making the pick, scores **-31.83** against 100
simulations' **-1.28**. That +30.55 +/- 1.60 HP is what search buys over the
policy it rolls out with, and it is the signal this script exists to capture.

The idea is ordinary policy iteration: MCTS output is stronger than the rollout
that produced it, so training the rollout to imitate the search makes rollouts
stronger, which makes every leaf evaluation better, which makes the search
stronger. The engine already has the socket -- `load_policy_net` plus
`g_params.policyNetWeight`, which adds the net's score to `nativeScoreAction`.

States come from the human benchmark rather than synthetic decks, for the same
reason the tuner uses it: `ACT_TIER_RESOURCES` fights bosses at zero relics, a
regime the agent never occupies, and that was measured as the single largest
distortion in the old CMA-ES objective.

Features are the engine's own, not reconstructed here: `bc.leaf_features()` (10,
per state) and `bc.action_features(a)` (8, per action) are exactly the 18 inputs
`nativePolicyNetScore` sees at inference, so there is no train/serve skew.

**Known limitation, worth reading before interpreting results.** The 8 action
features are [is_attack, is_skill, is_power, is_other, target_hp_missing_frac,
target_block_frac, is_aoe_multi, card_pick_rate_weight] -- they do NOT identify
which card is being played. A net over these can learn state/action *interaction*
that the hand-tuned heuristic hardcodes, but it cannot learn card-specific policy
beyond the pick-rate prior. If distillation underperforms, this is the first
suspect, and widening the feature vector is an engine change.

    python -m lightspeed.collect_rollout_policy_data --split train --out runs/rollout_policy_data.pt
"""
from __future__ import annotations

import argparse
import json
import time
import zlib

import numpy as np
import torch

import slaythespire as sts

from ._human_deck_combat import build_battle
from .search_config import DEFAULT_SEARCH_CONFIG_PATH, ensure_search_config
from .paths import HUMAN_BENCHMARK

BENCHMARK_PATH = str(HUMAN_BENCHMARK)


def collect_fight(rec: dict, sims: int, ascension: int, max_steps: int = 600):
    """Play one fight with the search, recording every decision it makes.

    Returns (features, group_sizes, chosen) where `features` is
    (total_actions, 18), `group_sizes` gives the legal-action count per decision,
    and `chosen` is the index WITHIN each decision's action list that the search
    took. Decisions with a single legal action are skipped -- they carry no
    preference information and would dominate the dataset.
    """
    bc, _ = build_battle(rec["deck"], rec["relics"], rec["cur_hp"], rec["max_hp"],
                         getattr(sts.MonsterEncounter, rec["encounter"]),
                         ascension, rec["act"], rec.get("potions", ()))
    fight_key = zlib.crc32(f"{rec['run_id']}:{rec['floor']}".encode())

    features, group_sizes, chosen = [], [], []
    for step in range(max_steps):
        if bc.outcome != sts.BattleOutcome.UNDECIDED:
            break
        actions = bc.get_legal_actions()
        if not actions:
            break
        action, _ = sts.run_mcts_search(bc, sims, None, (fight_key << 20) ^ step)
        if len(actions) > 1:
            # leaf_features is per-STATE, so hoist it out of the action loop --
            # the same reason the engine precomputes it once per decision.
            state = list(bc.leaf_features())
            rows = [state + list(bc.action_features(a)) for a in actions]
            # The search returns an Action, not an index. Match on identity of
            # the encoded action rather than on object equality.
            keys = [str(a) for a in actions]
            try:
                pick = keys.index(str(action))
            except ValueError:
                # Search returned something not in the enumerated list (should
                # not happen); skip rather than mislabel.
                action.execute(bc)
                continue
            features.extend(rows)
            group_sizes.append(len(actions))
            chosen.append(pick)
        action.execute(bc)
    return features, group_sizes, chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--sims", type=int, default=100,
                        help="label quality; 100 is where the sim curve flattens")
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    ensure_search_config(DEFAULT_SEARCH_CONFIG_PATH)
    # A loaded net would make the search imitate ITSELF, which is a feedback loop
    # rather than distillation. Explicitly off for collection.
    sts.set_search_params({"policy_net_weight": 0.0})

    with open(BENCHMARK_PATH, encoding="utf-8") as handle:
        records = json.load(handle)
    fights = [r for r in records
              if args.split == "all" or r["split"] == args.split]
    if args.limit:
        fights = fights[: args.limit]
    print(f"collecting from {len(fights)} {args.split} fights at {args.sims} sims")

    all_features, all_groups, all_chosen = [], [], []
    start = time.time()
    for index, rec in enumerate(fights):
        try:
            features, groups, chosen = collect_fight(rec, args.sims, args.ascension)
        except Exception as error:  # noqa: BLE001 - reported, not hidden
            print(f"  skip {rec['run_id']}@{rec['floor']}: "
                  f"{type(error).__name__}: {error}")
            continue
        all_features.extend(features)
        all_groups.extend(groups)
        all_chosen.extend(chosen)
        if (index + 1) % 250 == 0:
            elapsed = time.time() - start
            print(f"  {index+1}/{len(fights)} fights, {len(all_groups)} decisions, "
                  f"{elapsed:.0f}s ({elapsed/(index+1)*1000:.0f} ms/fight)")

    x = torch.tensor(np.asarray(all_features, dtype=np.float32))
    groups = torch.tensor(np.asarray(all_groups, dtype=np.int64))
    chosen = torch.tensor(np.asarray(all_chosen, dtype=np.int64))
    torch.save({"x": x, "groups": groups, "chosen": chosen,
                "meta": {"split": args.split, "sims": args.sims,
                         "ascension": args.ascension, "fights": len(fights),
                         "feature_dim": int(x.shape[1])}}, args.out)
    print(f"\n{len(all_groups)} decisions, {x.shape[0]} scored actions, "
          f"dim {x.shape[1]}")
    print(f"mean legal actions per decision: {groups.float().mean():.1f}")
    print(f"wrote {args.out} ({time.time()-start:.0f}s)")


if __name__ == "__main__":
    main()
