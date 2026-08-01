"""What deck does a policy actually build?

The decision ablation put drafting at **-5.85 floors** if randomized -- the
single largest thing the network does -- and 07-known-issues.md documents it as
the largest known defect: v31 drafts raw attacks (Perfected Strike 42, Clash 25)
where a top human drafts an exhaust engine (Shrug It Off 103, Feel No Pain 97,
Dark Embrace 83). Deck SIZE already matches his; the CARDS do not.

That table was produced ad hoc. This makes it a harness, because it answers the
question that decides whether floor gains can become wins: **is a policy that
scores more floors actually building a different deck, or just playing the same
deck further?** Elite-taking is capability-gated (`_route_bias_probe.py`:
forcing elites costs -1.93 floors because the deck cannot beat them cheaply),
and relics sit behind elites, so deck quality is upstream of the whole economy.
A policy that gains floors without changing what it drafts should be expected to
plateau before the win threshold.

Compares any number of checkpoints on identical seeds and reports card
frequencies, type mix, and the specific cards the human archive flags.

Run from slay-sim/:
    python -m lightspeed._deck_composition --checkpoints a.pt b.pt --runs 150
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

_STATE: dict = {}

# The exhaust/defensive engine the human archive is built around, against the
# raw attacks v31 favours. Named explicitly so the comparison cannot be
# retrofitted to whatever the run happens to produce.
HUMAN_STAPLES = ["ShrugItOff", "FeelNoPain", "DarkEmbrace", "TrueGrit",
                 "BurningPact", "Offering", "SecondWind", "Armaments"]
KNOWN_BAD = ["Clash", "PerfectedStrike", "TwinStrike", "Anger", "IronWave"]


def _worker_init(sims: int, ascension: int) -> None:
    import torch

    from .search_config import DEFAULT_SEARCH_CONFIG_PATH

    torch.set_num_threads(1)
    _STATE.update(sims=sims, ascension=ascension,
                  search_config=DEFAULT_SEARCH_CONFIG_PATH, policies={})


def _policy(path: str):
    import torch

    from .eval_whole_run_policy import load_policy

    if path not in _STATE["policies"]:
        _STATE["policies"][path] = load_policy(path, torch.device("cpu"))
    return _STATE["policies"][path]


def _play(job: tuple[str, int]) -> dict:
    import torch
    import slaythespire as sts

    from .whole_run_env import RunConfig, WholeRunEnv

    path, seed = job
    policy = _policy(path)
    env = WholeRunEnv(RunConfig(
        ascension=_STATE["ascension"], combat_sims=_STATE["sims"],
        deterministic_combat=True, search_config_path=_STATE["search_config"]))
    obs = env.reset(seed)
    with torch.inference_mode():
        while (env.gc.outcome.name == "UNDECIDED"
               and env.steps < env.config.max_decisions):
            action, _, _, _ = policy.act(obs, sample=False)
            obs, _, done, _ = env.step(action)
            if done:
                break

    representation = sts.getNNRepresentation(env.gc)
    cards = collections.Counter()
    for card_id in representation.deck.cards:
        try:
            cards[sts.CardId(int(card_id)).name] += 1
        except (ValueError, TypeError):
            cards[f"id{int(card_id)}"] += 1
    upgrades = sum(int(u) > 0 for u in representation.deck.upgrades)
    return {"checkpoint": os.path.basename(path), "seed": seed,
            "floor": int(env.gc.floor_num), "deck_size": len(representation.deck.cards),
            "upgrades": upgrades, "relics": len(representation.relics.relics),
            "cards": dict(cards)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--runs", type=int, default=150)
    parser.add_argument("--seed-base", type=int, default=1_003_000)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--top", type=int, default=14)
    parser.add_argument("--out", default="runs/deck_composition.jsonl")
    args = parser.parse_args()

    jobs = [(path, args.seed_base + offset)
            for path in args.checkpoints for offset in range(args.runs)]
    print(f"{len(args.checkpoints)} checkpoints x {args.runs} seeds "
          f"on {args.workers} workers", flush=True)

    rows = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                             initargs=(args.sims, args.ascension)) as pool:
        for done, row in enumerate(pool.map(_play, jobs, chunksize=4), start=1):
            rows.append(row)
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)

    with open(args.out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    by_checkpoint: dict[str, list[dict]] = {}
    for row in rows:
        by_checkpoint.setdefault(row["checkpoint"], []).append(row)

    print()
    for name, group in by_checkpoint.items():
        floors = sum(r["floor"] for r in group) / len(group)
        print(f"{name}: floor {floors:.2f}, deck {sum(r['deck_size'] for r in group) / len(group):.1f}, "
              f"upgrades {sum(r['upgrades'] for r in group) / len(group):.1f}, "
              f"relics {sum(r['relics'] for r in group) / len(group):.1f}")

    names = list(by_checkpoint)
    totals = {
        name: collections.Counter(
            {card: count / len(group)
             for card, count in _sum_cards(group).items()})
        for name, group in by_checkpoint.items()
    }
    union = set()
    for name in names:
        union |= {card for card, _ in totals[name].most_common(args.top)}
    union |= set(HUMAN_STAPLES) | set(KNOWN_BAD)

    header = f"{'card':>18}" + "".join(f"{n[:22]:>24}" for n in names)
    print(f"\ncards per run\n{header}")
    for card in sorted(union, key=lambda c: -max(totals[n].get(c, 0) for n in names)):
        row = f"{card:>18}"
        for name in names:
            row += f"{totals[name].get(card, 0.0):>24.2f}"
        flag = ""
        if card in HUMAN_STAPLES:
            flag = "  <- human staple"
        elif card in KNOWN_BAD:
            flag = "  <- v31 over-drafts"
        print(row + flag)

    print(f"\n{'group':>18}" + "".join(f"{n[:22]:>24}" for n in names))
    for label, cards in (("human staples", HUMAN_STAPLES),
                         ("v31 over-drafts", KNOWN_BAD)):
        row = f"{label:>18}"
        for name in names:
            row += f"{sum(totals[name].get(c, 0.0) for c in cards):>24.2f}"
        print(row)
    print(f"\nwrote {args.out}")


def _sum_cards(group: list[dict]) -> collections.Counter:
    total = collections.Counter()
    for row in group:
        total.update(row["cards"])
    return total


if __name__ == "__main__":
    main()
