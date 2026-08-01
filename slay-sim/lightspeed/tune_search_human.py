"""CMA-ES tune the native search against a HUMAN baseline on his own decks.

`tune_search_cma.py` scores synthetic fights: decks from `weighted_ironclad_deck`,
relics only if a generator is passed (it wasn't), fitness = win rate + an HP-
fraction bonus. Two defects were measured and written up in
`docs/03-combat-search.md`: the fitness set fights at the wrong power level
(Burning Blood only, against bosses a real run reaches holding 8-11 relics, worth
+0.406 win rate), and 87% of its compute buys almost no gradient because 13 of
its 15 encounters sit at ~1.00 win rate for every candidate.

This tuner fixes both, and fixes the second one differently than that doc
proposes. Rather than reweighting toward bosses -- which discards the easy fights
-- it changes what is measured. Every fight carries a real human's damage on the
same floor of the same run, so a fight we always win still says whether we won it
at 8 HP or 16. Measured on the full 560: 430 (77%) are wins that score a flat 1.0
under a win-rate objective and are all live under this one.

The power level needs no sampling approximation either. Each fight is rebuilt
from an actual A20 Heart run at that floor: his deck, his relics, his HP, his
encounter.

Objective: mean (human_damage - our_hp_paid) per fight, so 0 == human parity and
the shipped config scores about -10.6. A death pays all remaining HP.

Each candidate is scored on a 65% sample of the train split, redrawn every
generation, with all candidates in a generation sharing that sample and the
search RNG (common random numbers) so a score difference is the parameters and
nothing else. A fight costs ~0.024s at 100 sims, so ~250 fights evaluate in ~6s
and the sample can be this large -- `tune_search_cma.py` had to subsample far
harder, and its per-generation seed-set noise (sd ~0.06) swamped real
improvements (0.02-0.05).

Redrawing the fights matters more than it looks. A first version held all 388
fixed, which makes the confirmation round a test of search-RNG luck ONLY and
blind to fight-set overfitting: three generations of that already scored
-3.13 +/- 1.14 HP WORSE than the shipped config on the held-out split while
looking better on train. Validate with `_human_deck_eval.py` regardless -- 35
parameters will fit whatever they are shown.

100 sims, not 300: measured flat from 43 to 1500 sims on this same benchmark
(43->300 is -2.17 +/- 1.10 HP, 300->1500 is -0.97 +/- 1.23), so 100 buys the same
play for a third of the compute.

Adds `rollout_temperature` to the search space. It is 0.0 in the shipped config
and absent from `tune_search_cma.py`'s PARAM_NAMES entirely -- never tuned. At 0
the rollout is argmax, so every rollout from a node replays essentially the same
line and extra simulations mostly re-measure it; that is the mechanism for the
flat sim curve above (22 of 40 fights took identical damage at 5x budget).

    python -m lightspeed.tune_search_human --minutes 420 --workers 11
"""
from __future__ import annotations

import argparse
import json
import math
import os
import multiprocessing as mp
import random
import time
import zlib

import cma
import numpy as np

from .tune_search_cma import (ADDITIVE_BOUNDS, PARAM_NAMES as BASE_PARAM_NAMES,
                              _param_kind, raw_value)
from .paths import (HUMAN_BENCHMARK, LIGHTSPEED_DIR, RUNS_DIR,
                    native_build_path)

# MUST be a module-level constant, not a flag: workers are spawned (not forked)
# on Windows, so they re-import this module and never see anything main() sets.
BENCHMARK_PATH = str(HUMAN_BENCHMARK)
OUT_PATH = str(RUNS_DIR / "tuned_search_human.json")
LOG_PATH = str(LIGHTSPEED_DIR / "tune_search_human_progress.log")

SIMS = 100
SIGMA0 = 0.15
# Pinned in every worker, deliberately NOT in PARAM_NAMES. honest_draw_order is a
# REGIME, not a knob: if CMA-ES could move it, the cheapest way for a candidate to
# score well would be to turn draw-order clairvoyance back on, which is worth
# ~4.9 HP and would swamp every real parameter. Set by --honest.
def _honest_draw_order() -> float:
    """1.0 when tuning without draw-order clairvoyance.

    Read from the environment rather than a module constant because workers are
    SPAWNED on Windows: they re-import this module and never see anything main()
    assigns, but they do inherit os.environ. Same reason BENCHMARK_PATH is a
    module-level literal.
    """
    return 1.0 if os.environ.get("STS_TUNE_HONEST") == "1" else 0.0
# Search seeds averaged per fight inside the objective. One is not enough: an
# identical config scored against ITSELF reports +1.60 +/- 0.68 (t = 2.34) at
# k=1, per-fight sd 10.82 HP (docs/04-evaluation.md). A 42-parameter optimiser
# handed an objective that noisy fits the noise -- which tune_search_cma.py's own
# docstring records happening to it, with per-generation seed noise (sd ~0.06)
# swamping real improvements (0.02-0.05).
def _score_seeds() -> int:
    """Search seeds averaged per fight; see _honest_draw_order for why env."""
    return max(1, int(os.environ.get("STS_TUNE_SCORE_SEEDS", "2")))
# Fraction of the train split scored per evaluation -- see _evaluate_candidate.
# At 1730 train fights this is ~600 per evaluation, already far more than the
# comparison needs (all candidates share the subset, so the pairing does the
# variance reduction, not the sample size). Kept low deliberately: a smaller
# share means successive generations see MORE different fights, which is the
# whole defence against fitting the training decks.
FIGHT_SAMPLE_FRACTION = 0.35
TUNE_ASCENSION = 20

# Validation cadence. The whole val split is scored, split across workers, so a
# val pass costs about one generation. Every 10 generations with patience 12
# means "stop after 120 generations with no validation gain" -- the previous run
# made its last real progress around generation 200.
VAL_EVERY = 10
VAL_PATIENCE = 12
VAL_SHARDS = 11
# Ignore validation gains smaller than this: the val split is ~555 fights, whose
# paired standard error is well under 1 HP, but unpaired run-to-run wobble is not.
VAL_MIN_DELTA = 0.05

# rollout_temperature has never been in a search space -- see the module docstring.
# Bounds from nativeScoreAction's own comment: raw action scores span roughly 4-30,
# so 8.0 is well past "meaningfully peaked" and 0.0 keeps the current argmax
# reachable, letting CMA-ES turn it back off if it does not pay.
# Seven more that tune_search_cma.py never searched. All but the last two sit at
# 0.0 in the shipped config -- present in the engine, contributing nothing.
# win_hp_fraction_weight is the one to watch: the docs record our win reward's HP
# term as a much smaller PROPORTION of the total than Silver Automaton's, and
# HP-preservation on won fights is exactly where silverbot beats us (-6.58 vs our
# -11.79 on the same 528 fights, 2026-07-31).
NEWLY_TUNED = [
    "win_hp_fraction_weight",
    "self_damage_score_penalty",
    "attack_damage_score_weight",
    "block_weight",
    "win_bonus_weight",
    "early_act_easy_pool_hp_safety_weight",
]
# The rollout's potion branch is deliberately NOT here, and neither are mast_weight,
# seq_halving_candidates or backup_max_weight. All exist in the engine, all default to
# verified no-ops, and all measured null or worse in single-parameter A/Bs on 500 train
# fights (see docs/03-combat-search.md). This follows the precedent power_horizon_weight
# and boss_power_multiplier already set: adding parameters that measure null only widens
# the surface CMA-ES can overfit. Re-add one only alongside a reason to expect it to pay.
PARAM_NAMES = list(BASE_PARAM_NAMES) + ["rollout_temperature"] + NEWLY_TUNED
EXTRA_ADDITIVE_BOUNDS = {
    "rollout_temperature": (0.0, 8.0),
    # Scaled by 100 in the reward, so single digits already rival the absolute-HP
    # term; negative would mean preferring to end fights at lower HP.
    "win_hp_fraction_weight": (0.0, 10.0),
    # Ironclad pays HP for Offering, Hemokinesis, Bloodletting and Combust, and
    # nothing currently prices that. Penalty, so non-negative.
    "self_damage_score_penalty": (0.0, 10.0),
    # Both sit alongside nativeScoreAction's existing per-action terms
    # (attack_base/skill_base/power_score are ~4-10), so bound to that scale.
    "attack_damage_score_weight": (0.0, 5.0),
    "block_weight": (0.0, 5.0),
    # Already 1.0 in the engine and multiplicative in effect, but tuned additively
    # here because its compiled default is a real value rather than an off-state.
    "win_bonus_weight": (0.0, 5.0),
    "early_act_easy_pool_hp_safety_weight": (0.0, 5.0),
}

_ROLLOUT_T_IDX = PARAM_NAMES.index("rollout_temperature")

_worker_fights: list | None = None
_worker_val: list | None = None


def _score_fights(fights, sims: int, seed_base: int, ascension: int) -> float:
    """Mean (human_damage - our_hp_paid) over `fights`. Higher is better; 0 is
    human parity. Shared by the training objective and the validation pass so the
    two are never measured on subtly different rules."""
    import slaythespire as sts
    from lightspeed._human_deck_combat import build_battle, play

    total = 0.0
    counted = 0
    for rec in fights:
        try:
            bc, _ = build_battle(rec["deck"], rec["relics"], rec["cur_hp"],
                                 rec["max_hp"],
                                 getattr(sts.MonsterEncounter, rec["encounter"]),
                                 ascension, rec["act"], rec.get("potions", ()))
        except Exception:  # noqa: BLE001 - a fight we cannot rebuild scores nothing
            continue
        # crc32, not hash(): Python randomizes str/tuple hashing per process, so
        # hash() would give the same fight a different search seed in every
        # worker and destroy the pairing this whole comparison rests on.
        fight_key = zlib.crc32(f"{rec['run_id']}:{rec['floor']}".encode())
        per_seed = 0.0
        seeds = _score_seeds()
        for seed in range(seeds):
            if seed:
                bc, _ = build_battle(rec["deck"], rec["relics"], rec["cur_hp"],
                                     rec["max_hp"],
                                     getattr(sts.MonsterEncounter,
                                             rec["encounter"]),
                                     ascension, rec["act"],
                                     rec.get("potions", ()))
            damage, outcome = play(bc, sims,
                                   ((seed_base + seed * 7919) << 20) ^ fight_key)
            # A death pays every point of HP that was left, which is what it
            # costs a run. Without this a death would score as merely "the damage
            # before dying" and could look cheaper than a won fight.
            hp_paid = (rec["cur_hp"]
                       if outcome != sts.BattleOutcome.PLAYER_VICTORY
                       else damage)
            per_seed += rec["human_damage"] - hp_paid
        total += per_seed / seeds
        counted += 1
    return total / max(1, counted)


def _evaluate_validation(args) -> tuple[float, int]:
    """Score a parameter vector on the VALIDATION split -- a fixed seed and the
    whole split, so successive calls are comparable to each other. This is what
    decides which point along the trajectory to keep and when to stop; the
    training score cannot do either, because it is measured on the fights the
    optimizer is selecting against."""
    x, defaults, shard, sims = args
    import slaythespire as sts

    params = {name: _raw_for(name, x[i], defaults[name])
              for i, name in enumerate(PARAM_NAMES)}
    params["honest_draw_order"] = _honest_draw_order()
    sts.set_search_params(params)
    shard_fights = _worker_val[shard::VAL_SHARDS]
    return (_score_fights(shard_fights, sims, 0, TUNE_ASCENSION)
            * len(shard_fights), len(shard_fights))


def _log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _bounds_for(name: str):
    if name in EXTRA_ADDITIVE_BOUNDS:
        return EXTRA_ADDITIVE_BOUNDS[name]
    kind = _param_kind(name)
    if kind == "additive":
        return ADDITIVE_BOUNDS[name]
    if kind == "signed_mult":
        return (-3.0, 10.0)
    return (0.02, 10.0)


def _kind_for(name: str) -> str:
    return "additive" if name in EXTRA_ADDITIVE_BOUNDS else _param_kind(name)


def _raw_for(name: str, x_i: float, default_i: float) -> float:
    # raw_value dispatches on _param_kind, which does not know about the params
    # this module adds; an additive param's raw value is just x itself.
    if name in EXTRA_ADDITIVE_BOUNDS:
        return float(x_i)
    return raw_value(name, x_i, default_i)


def _fitness_config() -> dict:
    return {"objective": "human_hp_delta", "sims": SIMS,
            "ascension": TUNE_ASCENSION, "split": "train",
            "benchmark": "baalorlord_a20_heart_20runs"}


def _worker_init() -> None:
    global _worker_fights, _worker_val
    import sys
    sys.path.insert(0, native_build_path())
    import slaythespire as sts  # noqa: F401 - imported so the child has it loaded
    from lightspeed.search_config import (DEFAULT_SEARCH_CONFIG_PATH,
                                          ensure_search_config)

    # Apply the SHIPPED config before anything else. Without this a worker sits
    # at native compiled defaults, so every parameter outside PARAM_NAMES (20 of
    # the 55) takes a different value here than in `_human_deck_eval`, which
    # applies the shipped config first -- the tuner would then be optimising a
    # regime we never actually measure or ship.
    ensure_search_config(DEFAULT_SEARCH_CONFIG_PATH)

    with open(BENCHMARK_PATH, encoding="utf-8") as handle:
        records = json.load(handle)
    _worker_fights = [r for r in records if r["split"] == "train"]
    # Benchmarks written before the 3-way split have no "val" rows; the caller
    # checks this and falls back to wall-clock stopping rather than silently
    # selecting on an empty set.
    _worker_val = [r for r in records if r["split"] == "val"]


def _evaluate_candidate(args) -> float:
    """NEGATIVE mean (human_damage - our_hp_paid). CMA-ES minimizes, so lower is
    better and the shipped config sits near +10.6."""
    x, defaults, seed_base, sims = args
    import slaythespire as sts

    params = {name: _raw_for(name, x[i], defaults[name])
              for i, name in enumerate(PARAM_NAMES)}
    params["honest_draw_order"] = _honest_draw_order()
    sts.set_search_params(params)

    # Resample WHICH fights each generation, not just the search RNG. Scoring the
    # same 388 fights every time makes the confirmation round a test of RNG luck
    # only -- it cannot see fight-set overfitting, and measured on the held-out
    # split, three generations of that was already -3.13 +/- 1.14 HP WORSE than
    # the shipped config while looking better on train. Every candidate in a
    # generation still shares one subset (paired), and the confirmation round
    # draws a different one, so a candidate has to survive fights it was not
    # selected on.
    rng = random.Random(seed_base)
    fights = rng.sample(_worker_fights,
                        max(1, int(len(_worker_fights) * FIGHT_SAMPLE_FRACTION)))
    # Common random numbers: every candidate in a generation faces identical
    # fights AND identical search randomness, so a score difference is the
    # parameters and nothing else. seed_base varies per generation so the
    # optimizer cannot overfit one RNG draw or one fight subset.
    return -_score_fights(fights, sims, seed_base, TUNE_ASCENSION)


def main() -> None:
    global SIMS, OUT_PATH, LOG_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=420.0)
    parser.add_argument("--honest", action="store_true",
                        help="tune with draw-order clairvoyance REMOVED. The "
                             "shipped 42 parameters were all fitted with the "
                             "search knowing the draw order; nothing has ever "
                             "been tuned for honest play.")
    parser.add_argument("--score-seeds", type=int, default=2,
                        help="search seeds averaged per fight in the objective")
    parser.add_argument("--workers", type=int, default=11,
                        help="parallel processes; hardware, not algorithm")
    parser.add_argument("--popsize", type=int, default=0,
                        help="CMA-ES candidates per generation; 0 = 4+3*ln(n), "
                             "the library default for this dimensionality")
    parser.add_argument("--sims", type=int, default=SIMS)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--log", default=LOG_PATH)
    args = parser.parse_args()
    SIMS = args.sims
    OUT_PATH = args.out
    LOG_PATH = args.log

    import sys
    sys.path.insert(0, native_build_path())
    import slaythespire as sts

    # Native compiled defaults, captured BEFORE the shipped config is applied.
    # x-space is always relative to these (x=1.0 means "the compiled default"),
    # so they must be read first or the whole parameterisation shifts.
    defaults = sts.get_search_params()
    from lightspeed.search_config import (DEFAULT_SEARCH_CONFIG_PATH,
                                          ensure_search_config)
    shipped = ensure_search_config(DEFAULT_SEARCH_CONFIG_PATH)["params"]
    with open(BENCHMARK_PATH, encoding="utf-8") as handle:
        records = json.load(handle)
    n_train = sum(1 for r in records if r["split"] == "train")
    # popsize is an ALGORITHM choice and workers is a HARDWARE one; tying them
    # together (as tune_search_cma.py does) silently under-sizes the population.
    # At 35 dimensions the library default is 14, where 11 workers were giving
    # 11. It costs nothing here: 11 workers need two rounds for 12 tasks and two
    # rounds for 15, so the extra candidates ride along in slack that already
    # existed.
    if args.honest:
        os.environ["STS_TUNE_HONEST"] = "1"
    os.environ["STS_TUNE_SCORE_SEEDS"] = str(args.score_seeds)
    popsize = args.popsize or (4 + int(3 * math.log(len(PARAM_NAMES))))
    _log(f"=== human-baseline CMA-ES: {n_train} train fights, {SIMS} sims, "
         f"{len(PARAM_NAMES)} params (incl. rollout_temperature), "
         f"popsize {popsize} on {args.workers} workers, {args.minutes:.0f}m ===")
    _log("objective: mean(human_damage - our_hp_paid); 0 == human parity")

    # Warm-start from the SHIPPED config, not the compiled defaults. Generation 1
    # then measures the config we actually run, so every accepted candidate is an
    # improvement on it rather than on a weaker starting point the tuner has to
    # spend generations climbing back to. Measured live: starting cold, gen 10
    # validated at -11.214 against the shipped config's -8.784 -- 2.4 HP spent
    # rediscovering ground already held.
    x0 = []
    for name in PARAM_NAMES:
        raw = float(shipped.get(name, defaults[name]))
        if _kind_for(name) == "additive":
            x0.append(raw)
        else:
            # Multiplicative dimensions are ratios to the compiled default; a
            # default of 0 has no ratio, so those stay at 1.0.
            x0.append(raw / defaults[name] if defaults[name] else 1.0)
    lo = [b[0] for b in (_bounds_for(n) for n in PARAM_NAMES)]
    hi = [b[1] for b in (_bounds_for(n) for n in PARAM_NAMES)]
    clamped = [n for n, x, a, b in zip(PARAM_NAMES, x0, lo, hi) if not a <= x <= b]
    if clamped:
        _log(f"warm-start clamped to bounds: {clamped}")
        x0 = [min(max(x, a), b) for x, a, b in zip(x0, lo, hi)]

    bounds_pairs = [_bounds_for(name) for name in PARAM_NAMES]
    cma_stds = [((hi - lo) / 4.0) / SIGMA0 if _kind_for(name) == "additive" else 1.0
                for name, (lo, hi) in zip(PARAM_NAMES, bounds_pairs)]
    es = cma.CMAEvolutionStrategy(
        x0, SIGMA0,
        {"popsize": popsize,
         "bounds": [[lo for lo, _ in bounds_pairs], [hi for _, hi in bounds_pairs]],
         "CMA_stds": cma_stds, "verbose": -9},
    )

    with open(BENCHMARK_PATH, encoding="utf-8") as handle:
        has_val = any(r["split"] == "val" for r in json.load(handle))
    if not has_val:
        _log("WARNING: benchmark has no val split -- falling back to wall-clock "
             "stopping and saving the last accepted point, which the previous "
             "run showed is an arbitrary sample of the plateau")
    best_val = float("-inf")
    stalled = 0
    val_history: list[tuple[int, float]] = []

    incumbent_x = np.array(x0, dtype=float)
    best_params = {name: _raw_for(name, incumbent_x[i], defaults[name])
                   for i, name in enumerate(PARAM_NAMES)}
    start = time.time()
    gen = 0
    seed_base = 1
    saved_any = False

    with mp.Pool(args.workers, initializer=_worker_init) as pool:
        while time.time() - start < args.minutes * 60.0 and not es.stop():
            gen += 1
            candidates = es.ask()
            # Incumbent last so it shares this generation's seed_base exactly --
            # a candidate must beat it on the SAME fights and the SAME RNG.
            arg_list = [(np.array(c), defaults, seed_base, args.sims) for c in candidates]
            arg_list.append((incumbent_x, defaults, seed_base, args.sims))
            seed_base += 1
            values = pool.map(_evaluate_candidate, arg_list)
            fitnesses, incumbent_score = values[:-1], -values[-1]
            es.tell(candidates, fitnesses)

            best_idx = int(np.argmin(fitnesses))
            gen_best = -fitnesses[best_idx]
            improved = False
            confirm_note = ""
            if gen_best > incumbent_score:
                # The best of popsize noisy draws beats one draw of equal true
                # quality most of the time, so re-run the pair on a FRESH RNG
                # draw neither was selected on before accepting.
                confirm = pool.map(_evaluate_candidate, [
                    (np.array(candidates[best_idx], dtype=float), defaults, seed_base, args.sims),
                    (incumbent_x, defaults, seed_base, args.sims)])
                seed_base += 1
                challenger, incumbent_confirmed = -confirm[0], -confirm[1]
                improved = challenger > incumbent_confirmed
                confirm_note = (f" | confirm chal={challenger:.3f} "
                                f"inc={incumbent_confirmed:.3f}"
                                f"{'  <-- ACCEPTED' if improved else '  (rejected)'}")
                if improved:
                    incumbent_x = np.array(candidates[best_idx], dtype=float)
                    best_params = {name: _raw_for(name, incumbent_x[i], defaults[name])
                                   for i, name in enumerate(PARAM_NAMES)}
                    saved_any = True
            _log(f"gen {gen:4d} (t={(time.time()-start)/60:5.1f}m): "
                 f"mean={-float(np.mean(fitnesses)):7.3f} best={gen_best:7.3f} "
                 f"inc={incumbent_score:7.3f} "
                 # Index by NAME. This was incumbent_x[-1], which was correct only
                 # while rollout_temperature happened to be the last parameter --
                 # adding seven more silently repointed it at
                 # boss_silver_card_play_prior_weight and printed negative
                 # "temperatures" that its own (0, 8) bounds forbid.
                 f"rollout_T={_raw_for('rollout_temperature', incumbent_x[_ROLLOUT_T_IDX], 0.0):.2f}"
                 f"{confirm_note}")

            # Validation pass. The training score cannot say which point along
            # the trajectory to keep: the previous 727-generation run improved
            # train from -24.9 to -13.5 by generation 200 and then random-walked
            # for 500 more at a 48.4% acceptance rate, so its final artifact was
            # an arbitrary sample of a plateau rather than the best point found.
            # This scores the incumbent on runs the optimizer never selects
            # against, keeps the best one, and stops when it stalls.
            if has_val and gen % VAL_EVERY == 0:
                shards = pool.map(_evaluate_validation,
                                  [(incumbent_x, defaults, s, args.sims) for s in range(VAL_SHARDS)])
                val_score = sum(t for t, _ in shards) / max(1, sum(n for _, n in shards))
                val_history.append((gen, val_score))
                better = val_score > best_val + VAL_MIN_DELTA
                if better:
                    best_val = val_score
                    stalled = 0
                    with open(OUT_PATH, "w", encoding="utf-8") as handle:
                        json.dump({"score": val_score, "train_score": incumbent_score,
                                   "fitness_config": _fitness_config(),
                                   "generation": gen, "params": best_params}, handle,
                                  indent=2)
                    saved_any = True
                else:
                    stalled += 1
                _log(f"    val @ gen {gen}: {val_score:+7.3f} "
                     f"(best {best_val:+7.3f}, stalled {stalled}/{VAL_PATIENCE})"
                     f"{'  <-- SAVED' if better else ''}")
                if stalled >= VAL_PATIENCE:
                    _log(f"stopping: validation has not improved in "
                         f"{VAL_PATIENCE * VAL_EVERY} generations")
                    break

    _log(f"=== done: {gen} generations in {(time.time()-start)/60:.1f}m ===")
    if saved_any:
        _log(f"best saved to {OUT_PATH}")
        _log("NOT applied to the active config -- validate on the held-out test "
             "split first: python -m lightspeed._human_deck_eval --params " + OUT_PATH)
    else:
        _log("no candidate beat the shipped config on a confirmation round")


if __name__ == "__main__":
    main()
