"""Which boss relics does the policy actually end up holding?

Written to size the blast radius of a proposed Runic Dome fix, but the useful
output turned out to be the whole distribution. Any question of the form "is it
worth fixing how the engine models relic X" is bounded by how often X is taken,
and this measures that directly rather than assuming.

Run from slay-sim/:
    PYTHONPATH='../sts_lightspeed/build;.' python -m lightspeed._relic_uptake [runs] [checkpoint]

Measured 2026-07-30, v31, 100 A20 seeds: 68/100 runs acquired a boss relic and
the distribution across 21 relics is nearly flat (top relic 6%, Runic Dome 3%).
That flatness is itself a finding — the same near-uniform marginal the policy
shows at campfires — and it is why the Dome fix was dropped. See
docs/07-known-issues.md.
"""
from __future__ import annotations

import collections
import statistics
import sys

import slaythespire as sts
import torch

from lightspeed.eval_whole_run_policy import load_policy
from lightspeed.whole_run_env import RunConfig, WholeRunEnv


DEFAULT_CHECKPOINT = "runs/whole_run_transformer_yield10x_a20_v31.pt"
SEED_BASE = 18_900_000

# The Ironclad boss-relic pool (include/constants/RelicPools.h). Kept explicit so
# a relic acquired from any other source is not miscounted as a boss pick.
BOSS_POOL = {
    "FUSION_HAMMER", "VELVET_CHOKER", "RUNIC_DOME", "SLAVERS_COLLAR",
    "SNECKO_EYE", "PANDORAS_BOX", "CURSED_KEY", "BUSTED_CROWN", "ECTOPLASM",
    "TINY_HOUSE", "SOZU", "PHILOSOPHERS_STONE", "ASTROLABE", "BLACK_STAR",
    "SACRED_BARK", "EMPTY_CAGE", "RUNIC_PYRAMID", "CALLING_BELL",
    "COFFEE_DRIPPER", "BLACK_BLOOD", "MARK_OF_PAIN", "RUNIC_CUBE",
}


def relic_names(gc) -> set[str]:
    names = set()
    for raw in sts.getNNRepresentation(gc).relics.relics:
        try:
            names.add(sts.RelicId(int(raw)).name)
        except ValueError:
            pass
    return names


def main() -> None:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CHECKPOINT

    torch.set_num_threads(1)
    policy = load_policy(checkpoint, torch.device("cpu"))

    held: collections.Counter = collections.Counter()
    floors_with: dict[str, list[int]] = collections.defaultdict(list)
    runs_with_boss_relic = 0
    all_floors = []

    for offset in range(runs):
        env = WholeRunEnv(RunConfig(ascension=20, combat_sims=300,
                                    deterministic_combat=True))
        obs = env.reset(SEED_BASE + offset)
        with torch.inference_mode():
            while (env.gc.outcome == sts.GameOutcome.UNDECIDED
                   and env.steps < env.config.max_decisions):
                action, _, _, _ = policy.act(obs, sample=False)
                obs, _, done, _ = env.step(action)
                if done:
                    break
        floor = int(env.gc.floor_num)
        all_floors.append(floor)
        taken = relic_names(env.gc) & BOSS_POOL
        held.update(taken)
        for name in taken:
            floors_with[name].append(floor)
        if taken:
            runs_with_boss_relic += 1

    print()
    print("=" * 68)
    print(f"{checkpoint.split('/')[-1]}, {runs} seeds, A20, 300 sims")
    print("=" * 68)
    print(f"  mean floor                        : {statistics.mean(all_floors):.2f}")
    print(f"  runs that acquired any boss relic : {runs_with_boss_relic}/{runs}")
    print()
    print("  boss relics held, by frequency:")
    for name, count in held.most_common():
        floors = floors_with[name]
        print(f"    {name:22s} {count:3d}  ({100 * count / runs:4.1f}% of runs)  "
              f"mean floor when held {statistics.mean(floors):5.2f}")
    if held:
        top = held.most_common(1)[0][1]
        print()
        print(f"  spread: top relic {100 * top / runs:.1f}% of runs across "
              f"{len(held)} distinct relics — a near-uniform distribution here "
              f"means the policy is close to indifferent among boss relics.")


if __name__ == "__main__":
    main()
