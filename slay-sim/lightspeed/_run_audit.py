"""Audit the run-economy decisions the evaluations never record.

Floors and act reach say whether a run died; they say nothing about *why* the
deck was too weak to survive. This measures the resources a policy actually
accumulates — deck size, upgrades, relics, gold, max HP — so a weak deck shows
up directly instead of being inferred from a death histogram.

Motivated by two findings: the policy takes 4.6% of offered act-1 elites (the
main relic source) and chose REST over SMITH at 137 of 137 campfires, so it
should be arriving at act bosses both under-relic'd and un-upgraded. This checks
whether that shows up in the totals.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics

import torch

import slaythespire as sts
from .eval_whole_run_policy import load_policy
from .generate_whole_run_rollouts import decision_type
from .whole_run_env import RunConfig, WholeRunEnv
from .search_config import DEFAULT_SEARCH_CONFIG_PATH


def audit(policy, *, runs: int, seed_base: int, sims: int, ascension: int):
    out = []
    for offset in range(runs):
        env = WholeRunEnv(RunConfig(
            ascension=ascension, combat_sims=sims, deterministic_combat=True,
            search_config_path=DEFAULT_SEARCH_CONFIG_PATH))
        obs = env.reset(seed_base + offset)
        screens = collections.Counter()
        with torch.inference_mode():
            while (env.gc.outcome.name == "UNDECIDED"
                   and env.steps < env.config.max_decisions):
                screens[decision_type(env.gc)] += 1
                action, _, _, _ = policy.act(obs, sample=False)
                obs, _, done, _ = env.step(action)
                if done:
                    break
        rep = sts.getNNRepresentation(env.gc)
        upgrades = sum(int(u) > 0 for u in rep.deck.upgrades)
        out.append({
            "seed": seed_base + offset,
            "floor": int(env.gc.floor_num), "act": int(env.gc.act),
            "deck": len(rep.deck.cards), "upgrades": upgrades,
            "relics": len(rep.relics.relics), "gold": int(env.gc.gold),
            "max_hp": int(env.gc.max_hp),
            "screens": dict(screens),
        })
    return out


def summarise(tag: str, rows: list[dict]) -> None:
    n = len(rows)

    def mean(key):
        return statistics.mean(r[key] for r in rows)

    print(f"\n{tag}  ({n} runs, mean floor {mean('floor'):.1f})")
    print(f"  deck size      {mean('deck'):6.1f}   (Ironclad starts at 10)")
    print(f"  upgrades       {mean('upgrades'):6.1f}")
    print(f"  relics         {mean('relics'):6.1f}   (starts at 1)")
    print(f"  gold unspent   {mean('gold'):6.1f}")
    print(f"  max HP         {mean('max_hp'):6.1f}   (starts at 80 at A20)")
    deep = [r for r in rows if r["act"] >= 2]
    if deep:
        print(f"  -- runs reaching act 2 ({len(deep)}): deck "
              f"{statistics.mean(r['deck'] for r in deep):.1f}, upgrades "
              f"{statistics.mean(r['upgrades'] for r in deep):.1f}, relics "
              f"{statistics.mean(r['relics'] for r in deep):.1f}")
    agg = collections.Counter()
    for r in rows:
        agg.update(r["screens"])
    print("  decisions/run: " + " ".join(
        f"{k}={v/n:.1f}" for k, v in agg.most_common(8)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=18_900_000)
    parser.add_argument("--sims", type=int, default=300)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cpu")
    records = []
    for path in args.checkpoints:
        policy = load_policy(path, device)
        rows = audit(policy, runs=args.runs, seed_base=args.seed_base,
                     sims=args.sims, ascension=args.ascension)
        for r in rows:
            r["checkpoint"] = path.split("/")[-1]
        summarise(path.split("/")[-1], rows)
        records.extend(rows)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            for r in records:
                handle.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
