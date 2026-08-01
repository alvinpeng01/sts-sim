"""Measure a deck's fighting strength by simulation, not by regression.

Why measured rather than learned from run outcomes.  Every attempt to learn a
value signal from full runs has hit the same wall: the true per-decision
advantage has sd 0.0096 against an episode-return sd of 0.084
(06-experiment-log.md), so run outcomes are almost pure noise at the level of a
single card pick.  A deck played against a FIXED encounter battery is the
opposite regime -- the score is nearly deterministic given the seeds, two decks
differing by one card share almost all of their variance, and common random
numbers across candidates make the *difference* quieter still.  It is the one
place in this project where pairing works well; run-level pairing removes only
39%.

What this is for, in order of how directly it attacks a measured defect:

1. **A drafting signal that can see the deck.**  The AIVAT control variate
   failed at corr(floor, correction) = 0.014 because every candidate at a card
   reward shares a floor and the critic is ~80% a floor lookup -- it cannot tell
   "take Feel No Pain" from "take Clash".  `strength_delta` is exactly that
   quantity.
2. **Potential-based reward shaping.**  With Phi(s) = deck strength, applying
   `gamma * Phi(s') - Phi(s)` is policy-invariant -- it provably cannot change
   the optimal policy -- while converting a delayed, noisy consequence into an
   immediate low-variance signal.  That is aimed at credit assignment, which is
   the measured problem, rather than at the objective, which is a separate one.
3. **A deck-sensitive critic input**, fixing the floor-lookup defect at its
   source.

The battery is weighted toward where runs actually die.  07-known-issues.md puts
47.5% of deaths at floors 16-17 and 33-34 -- act bosses -- and
03-combat-search.md records that the CMA-ES fitness set failed precisely by
omitting those and fighting at the wrong power level.  So elites AND bosses for
each act are included, and the deck is scored at the act it is actually in.

Every fight starts from the same supplied HP, so fights are independent and the
score is a property of the deck rather than of a particular run's HP trajectory.

Run from slay-sim/:
    python -m lightspeed.deck_strength --validate --workers 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

# Elites and bosses per act. Ordinary monster rooms are deliberately excluded:
# they are not what kills runs, and including them dilutes the signal with
# fights every deck wins.
BATTERY = {
    1: ["GREMLIN_NOB", "LAGAVULIN", "THREE_SENTRIES",
        "THE_GUARDIAN", "HEXAGHOST", "SLIME_BOSS"],
    2: ["GREMLIN_LEADER", "BOOK_OF_STABBING", "SLAVERS",
        "CHAMP", "COLLECTOR", "AUTOMATON"],
    3: ["GIANT_HEAD", "NEMESIS", "REPTOMANCER",
        "AWAKENED_ONE", "TIME_EATER", "DONU_AND_DECA"],
}

# A flat death penalty makes the measure jump ~1.5 units whenever a fight flips
# outcome, which swamps the ~0.1-unit HP differences that distinguish two decks
# and leaves the delta dominated by binary flips. Losses instead carry partial
# credit for how much of the encounter the deck removed -- the same shape the
# search's own loss evaluation uses (`lossProgressCreditWeight`).
LOSS_BASE = -1.0
LOSS_PROGRESS_CREDIT = 0.75

# Relics with a per-RUN charge, not a per-fight effect. The battery rebuilds a
# fresh context for every fight, so a charge that should apply to three combats
# in a real run applies to all 36 here. Neow's Lament ("enemies in your next
# three combats have 1 HP") turns the entire battery into free wins and reports
# a floor-14 deck at perfect strength -- measured, not hypothetical.
CHARGE_RELICS = {"NEOWS_LAMENT", "LIZARD_TAIL", "OMAMORI"}

_STATE: dict = {}


def _worker_init(sims: int, ascension: int) -> None:
    import slaythespire as sts

    from .search_config import ensure_search_config

    # Explicit: this module builds GameContexts directly rather than through
    # WholeRunEnv, so nothing else would apply the tuned search configuration.
    # A worker running compiled defaults would silently measure a different
    # combat -- the failure mode that contaminated pilot1 iteration 31.
    _STATE["config"] = ensure_search_config()
    _STATE["sims"] = sims
    _STATE["ascension"] = ascension
    _STATE["sts"] = sts


def battery_for(act: int, span: int = 2) -> list[tuple[int, str]]:
    """Encounters from this act and the next `span - 1`, since a deck drafted
    now has to survive what comes later, not only what is on screen."""
    fights = []
    for offset in range(span):
        target = act + offset
        if target in BATTERY:
            fights.extend((target, name) for name in BATTERY[target])
    return fights or [(3, name) for name in BATTERY[3]]


def _fight(job: dict) -> dict:
    from ._human_deck_combat import build_battle, play

    sts = _STATE["sts"]
    deck = [(int(c), int(u)) for c, u in job["deck"]]
    relics = [r for r in job["relics"]
              if sts.RelicId(int(r)).name not in CHARGE_RELICS]
    encounter = getattr(sts.MonsterEncounter, job["encounter"])
    battle, missing = build_battle(
        deck, relics, job["cur_hp"], job["max_hp"], encounter,
        _STATE["ascension"], act=job["act"])
    damage, outcome = play(battle, _STATE["sims"], job["seed"])
    survived = outcome == sts.BattleOutcome.PLAYER_VICTORY
    if survived:
        score = (job["cur_hp"] - damage) / max(1, job["max_hp"])
        removed = 1.0
    else:
        total = sum(max(0, m.max_hp) for m in battle.monsters) or 1
        left = sum(max(0, m.cur_hp) for m in battle.monsters)
        removed = 1.0 - left / total
        score = LOSS_BASE + LOSS_PROGRESS_CREDIT * removed
    return {"label": job["label"], "encounter": job["encounter"],
            "act": job["act"], "seed": job["seed"], "damage": damage,
            "survived": survived, "removed": removed, "score": score,
            "missing": missing}


def strength_jobs(label: str, deck, relics, cur_hp: int, max_hp: int, act: int,
                  seeds: list[int], span: int = 2) -> list[dict]:
    """One job per (encounter, seed). Seeds are shared across every deck being
    compared, which is what makes the DIFFERENCE between two decks quiet."""
    return [
        {"label": label, "deck": deck, "relics": list(relics),
         "cur_hp": int(cur_hp), "max_hp": int(max_hp), "act": fight_act,
         "encounter": name, "seed": seed}
        for fight_act, name in battery_for(act, span)
        for seed in seeds
    ]


def summarize(rows: list[dict]) -> dict[str, dict]:
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    out = {}
    for label, group in by_label.items():
        scores = [r["score"] for r in group]
        out[label] = {
            "strength": statistics.mean(scores),
            "sem": (statistics.stdev(scores) / len(scores) ** 0.5
                    if len(scores) > 1 else float("nan")),
            "deaths": sum(1 for r in group if not r["survived"]),
            "fights": len(group),
            "per_fight": {
                (r["encounter"], r["seed"]): r["score"] for r in group},
        }
    return out


def paired_delta(base: dict, variant: dict) -> tuple[float, float]:
    """Difference on identical (encounter, seed) fights -- the whole point of
    sharing seeds. Comparing means instead would drown the effect."""
    shared = sorted(set(base["per_fight"]) & set(variant["per_fight"]))
    diffs = [variant["per_fight"][k] - base["per_fight"][k] for k in shared]
    mean = statistics.mean(diffs)
    sem = (statistics.stdev(diffs) / len(diffs) ** 0.5
           if len(diffs) > 1 else float("nan"))
    return mean, sem


def run_jobs(jobs: list[dict], sims: int, ascension: int,
             workers: int) -> list[dict]:
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(sims, ascension)) as pool:
        for row in pool.map(_fight, jobs, chunksize=4):
            rows.append(row)
    return rows


def deck_from_context(gc) -> tuple[list[tuple[int, int]], list[int]]:
    import slaythespire as sts

    representation = sts.getNNRepresentation(gc)
    deck = [(int(c), int(u)) for c, u in
            zip(representation.deck.cards, representation.deck.upgrades)]
    relics = [int(r) for r in representation.relics.relics]
    return deck, relics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true",
                        help="acceptance test: does the measure reproduce facts "
                             "the project already knows?")
    parser.add_argument("--policy",
                        default="runs/whole_run_transformer_v37_critic.pt")
    parser.add_argument("--decks", type=int, default=4,
                        help="base decks sampled from real runs")
    parser.add_argument("--seed-base", type=int, default=1_003_000)
    parser.add_argument("--stop-floor", type=int, default=14)
    parser.add_argument("--battery-seeds", type=int, default=3)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default="runs/deck_strength_validate.json")
    args = parser.parse_args()

    if not args.validate:
        raise SystemExit("nothing to do; pass --validate")

    import torch
    import slaythespire as sts

    from .eval_whole_run_policy import load_policy
    from .whole_run_env import RunConfig, WholeRunEnv

    torch.set_num_threads(1)
    policy = load_policy(args.policy, torch.device("cpu"))

    # The cards the archive says we over-draft, the cards it says we under-draft,
    # and a neutral control. Named before measuring, so the test cannot be
    # retrofitted to whatever comes out.
    CANDIDATES = ["CLASH", "PERFECTED_STRIKE", "IRON_WAVE", "TWIN_STRIKE",
                  "FEEL_NO_PAIN", "SHRUG_IT_OFF", "DARK_EMBRACE", "OFFERING",
                  "SECOND_WIND", "BURNING_PACT", "INFLAME", "BODY_SLAM"]

    bases = []
    # Runs die before the target floor often enough that a fixed seed range
    # yields too few decks; keep drawing seeds until enough survive.
    for index in range(args.decks * 8):
        if len(bases) >= args.decks:
            break
        env = WholeRunEnv(RunConfig(ascension=args.ascension,
                                    combat_sims=args.sims,
                                    deterministic_combat=True))
        obs = env.reset(args.seed_base + index)
        with torch.inference_mode():
            while (env.gc.outcome.name == "UNDECIDED"
                   and int(env.gc.floor_num) < args.stop_floor
                   and env.steps < env.config.max_decisions):
                action, _, _, _ = policy.act(obs, sample=False)
                obs, _, done, _ = env.step(action)
                if done:
                    break
        if env.gc.outcome.name != "UNDECIDED":
            continue
        deck, relics = deck_from_context(env.gc)
        bases.append({"seed": args.seed_base + index, "deck": deck,
                      "relics": relics, "act": int(env.gc.act),
                      "max_hp": int(env.gc.max_hp),
                      "floor": int(env.gc.floor_num)})
        print(f"base deck {index}: floor {env.gc.floor_num}, "
              f"{len(deck)} cards, {len(relics)} relics", flush=True)

    if not bases:
        raise SystemExit("no surviving base decks; lower --stop-floor")
    seeds = [11 * i + 7 for i in range(args.battery_seeds)]
    jobs = []
    for base in bases:
        tag = f"seed{base['seed']}"
        jobs += strength_jobs(f"{tag}|base", base["deck"], base["relics"],
                              base["max_hp"], base["max_hp"], base["act"], seeds)
        for card in CANDIDATES:
            try:
                card_id = int(getattr(sts.CardId, card))
            except AttributeError:
                continue
            jobs += strength_jobs(f"{tag}|{card}", base["deck"] + [(card_id, 0)],
                                  base["relics"], base["max_hp"],
                                  base["max_hp"], base["act"], seeds)

    print(f"\n{len(jobs)} fights ({len(bases)} decks x "
          f"{1 + len(CANDIDATES)} variants x {len(battery_for(bases[0]['act']))} "
          f"encounters x {len(seeds)} seeds)", flush=True)
    started = time.perf_counter()
    rows = run_jobs(jobs, args.sims, args.ascension, args.workers)
    elapsed = time.perf_counter() - started
    print(f"{elapsed:.0f}s ({1000 * elapsed / len(rows):.0f} ms/fight)\n")

    summary = summarize(rows)
    deltas: dict[str, list[float]] = {}
    for base in bases:
        tag = f"seed{base['seed']}"
        base_key = f"{tag}|base"
        if base_key not in summary:
            continue
        for card in CANDIDATES:
            key = f"{tag}|{card}"
            if key in summary:
                mean, _ = paired_delta(summary[base_key], summary[key])
                deltas.setdefault(card, []).append(mean)

    print(f"{'card':>18}  {'mean delta strength':>20}  {'+/-':>7}  {'decks':>5}")
    for card, values in sorted(deltas.items(), key=lambda kv: -statistics.mean(kv[1])):
        mean = statistics.mean(values)
        sem = (statistics.stdev(values) / len(values) ** 0.5
               if len(values) > 1 else float("nan"))
        print(f"{card:>18}  {mean:>20.4f}  {sem:>7.4f}  {len(values):>5}")

    print(f"\n{'deck':>28}  {'strength':>9}  {'deaths':>7}")
    for label in sorted(summary):
        if label.endswith("|base"):
            entry = summary[label]
            print(f"{label:>28}  {entry['strength']:>9.4f}  "
                  f"{entry['deaths']:>3}/{entry['fights']:<3}")

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"deltas": {k: v for k, v in deltas.items()},
                   "bases": bases, "config": vars(args)}, handle, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
