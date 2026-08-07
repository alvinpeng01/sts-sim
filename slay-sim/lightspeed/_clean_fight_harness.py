"""PAIRED clean-fight harness: the low-variance replacement for Heart win rate.

Why this exists. Every combat experiment in this project has been scored on
Heart win rate at n=24, where a 2-win swing is inside sampling noise and a 79%
death rate makes mean HP meaningless (it averages survivors only, while the
human survived by construction). Today that cost us: a 13/24 "breakthrough"
evaporated to a 22/48 null, and a train-side sims peak failed to replicate.

The benchmark already carried a better signal. Every record has human_damage --
the HP the human paid on that exact fight from that exact state -- so the score
is a WITHIN-FIGHT paired difference, not a comparison of averages over different
situations. Restricted to encounters the model almost never dies in, there is no
survivorship bias either.

Two modes:
  baseline  -- run one config, dump per-fight results to JSON
  compare   -- run config B, load a baseline dump, and report the PAIRED
               per-fight difference with a paired t-test. Pairing removes
               fight-to-fight variance, which is most of the variance.

Validated: two null-controls (same config in a fresh process, and a no-op
override setting c_puct to its own shipped value) both return a paired delta of
EXACTLY 0.000 +/- 0.000, so any nonzero delta is a real effect of the change and
not run-to-run drift.

Set STS_SPLIT=val+test to confirm a train-side hit on held-out fights. Do that
before believing anything: a 6-point c_puct sweep showed a consistent -0.65 HP
plateau on train (t=-2.4) that collapsed to -0.16 (t=-0.7) held out.

Usage:
  python -m lightspeed._clean_fight_harness out.json '{"...":1}' [sims] [k]
  python -m lightspeed._clean_fight_harness new.json '{"...":1}' [sims] [k] --vs base.json
"""
from __future__ import annotations

import glob
import json
import os
import math
import statistics
import sys
import time

sys.path.insert(0, r"C:\Users\Alvin\grok\sts-project\slay-sim")

from lightspeed.paths import HUMAN_BENCHMARK, native_build_path  # noqa: E402

# STS_ENGINE_DIR selects a side-built engine (build/honest_test) when the main
# .pyd is link-locked by a running training process. Must win over the default.
_engine = os.environ.get("STS_ENGINE_DIR")
if _engine:
    os.add_dll_directory(_engine)
    sys.path.insert(0, _engine)
sys.path.insert(1 if _engine else 0, native_build_path())

import slaythespire as sts  # noqa: E402

from lightspeed.replay_human_runs import seed_to_long  # noqa: E402
from lightspeed.search_config import (  # noqa: E402
    DEFAULT_SEARCH_CONFIG_PATH, ensure_search_config)

# Encounters measured at <=7% model death rate on train (gap_by_encounter.py),
# so mean HP is an honest comparison rather than a survivor average.
CLEAN = ["LAGAVULIN", "NEMESIS", "THREE_SHAPES", "FOUR_SHAPES",
         "SPHERIC_GUARDIAN", "SHELLED_PARASITE_AND_FUNGI",
         "SPHERE_AND_TWO_SHAPES", "GREMLIN_NOB"]

# The other half of the deficit: encounters we DIE in, where the scoring metric
# has to be survival rather than HP (an HP mean over survivors is exactly the
# survivorship trap this harness exists to avoid). STS_ENCOUNTERS=boss selects
# these and switches the report to a paired McNemar test on death/survival.
BOSS = ["THE_HEART", "SHIELD_AND_SPEAR", "TIME_EATER", "AUTOMATON",
        "THE_GUARDIAN", "AWAKENED_ONE", "COLLECTOR", "DONU_AND_DECA",
        "HEXAGHOST", "CHAMP", "SLIME_BOSS", "REPTOMANCER", "BOOK_OF_STABBING",
        "GREMLIN_LEADER"]

OUT = sys.argv[1]
OVERRIDES = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
SIMS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
K = int(sys.argv[4]) if len(sys.argv) > 4 else 2
VS = sys.argv[sys.argv.index("--vs") + 1] if "--vs" in sys.argv else None

ARCHIVE = r"C:/Users/Alvin/Documents/Codex/2026-07-29/baalorlord-run-dataset"
SEEDS = {}
for f in glob.glob(ARCHIVE + "/**/*.jsonl", recursive=True):
    try:
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if isinstance(r, dict) and r.get("seed") and r.get("run_id") is not None:
                SEEDS.setdefault(str(r["run_id"]), r["seed"])
    except Exception:
        pass


def build(rec):
    rid = str(rec["run_id"])
    seed = seed_to_long(SEEDS[rid]) if rid in SEEDS else 1
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 20)
    for _ in range(len(gc.deck)):
        gc.remove_card(0)
    for cid, up in rec["deck"]:
        c = sts.Card(sts.CardId(cid))
        for _ in range(up):
            c.upgrade()
        gc.obtain_card(c)
    for x in rec["relics"]:
        try:
            gc.obtain_relic(sts.RelicId(x))
        except Exception:
            pass
    for p in rec.get("potions", ()):
        try:
            gc.obtain_potion(sts.Potion(p))
        except Exception:
            pass
    gc.act = int(rec["act"]); gc.max_hp = int(rec["max_hp"])
    gc.cur_hp = int(rec["cur_hp"]); gc.floor_num = int(rec["floor"])
    return sts.new_battle(gc, getattr(sts.MonsterEncounter, rec["encounter"]))


SPLITS = os.environ.get("STS_SPLIT", "train").split("+")
_sel = os.environ.get("STS_ENCOUNTERS", "clean")
ENCOUNTERS = BOSS if _sel == "boss" else (
    CLEAN if _sel == "clean" else _sel.split(","))
recs = [r for r in json.load(open(HUMAN_BENCHMARK, encoding="utf-8"))
        if r["split"] in SPLITS and r["encounter"] in ENCOUNTERS
        and r.get("human_damage") is not None]

# Optional bilinear student (docs/14). Loading it is inert unless a weight
# selects it, so an unset STS_STUDENT changes nothing.
_student = os.environ.get("STS_STUDENT")
if _student:
    sts.load_bilinear_student(json.load(open(_student, encoding="utf-8")))

ensure_search_config(DEFAULT_SEARCH_CONFIG_PATH)
# REGIME, not a knob. 0.0 = draw-order clairvoyance, which tune_search_human
# measures at ~4.9 HP -- enough to swamp every real parameter -- so an A/B is
# only meaningful within ONE regime, and a result found under clairvoyance does
# not automatically transfer to honest play. STS_HONEST=1 selects honest.
params = {"honest_draw_order":
          1.0 if os.environ.get("STS_HONEST") == "1" else 0.0}
params.update(OVERRIDES)
sts.set_search_params(params)
sts.set_leaf_eval_mode("rollout", 3)

print("CLEAN HARNESS: %d fights x k=%d, sims=%d" % (len(recs), K, SIMS), flush=True)
print("overrides: %s" % (OVERRIDES or "none (baseline)"), flush=True)

MAX_STEPS = int(os.environ.get("STS_MAX_STEPS", "400"))
results = {}          # "runid|floor|enc|seedidx" -> damage (or None if died)
deaths = 0
stalls = 0
t0 = time.time()
for i, rec in enumerate(recs):
    for s_ in range(K):
        key = "%s|%s|%s|%d" % (rec["run_id"], rec["floor"], rec["encounter"], s_)
        try:
            bc = build(rec)
        except Exception:
            continue
        hp0 = bc.player_hp
        # Hard step cap. A config that values survival with no time cost can
        # prefer stalling forever to dying, which hangs an uncapped loop --
        # horizon_value_mode=1 does exactly that on boss fights. Cap-hits are
        # counted separately from deaths: a stall is neither a win nor a loss,
        # and silently scoring it as either would hide the pathology.
        steps = 0
        while (bc.outcome == sts.BattleOutcome.UNDECIDED
               and bc.get_legal_actions() and steps < MAX_STEPS):
            a, _ = sts.run_mcts_search(bc, SIMS, None, i * 31 + s_)
            a.execute(bc)
            steps += 1
        if steps >= MAX_STEPS:
            stalls += 1
            results[key] = None
        elif bc.outcome == sts.BattleOutcome.PLAYER_VICTORY:
            results[key] = {"dmg": hp0 - bc.player_hp,
                            "human": float(rec["human_damage"]),
                            "enc": rec["encounter"]}
        else:
            deaths += 1
            results[key] = None

took = time.time() - t0
live = {k: v for k, v in results.items() if v}
gaps = [v["dmg"] - v["human"] for v in live.values()]
print("\nvs HUMAN: %+.2f HP/fight  (n=%d, SE %.2f)  deaths %d (%.1f%%)  [%.0fs]"
      % (statistics.mean(gaps), len(gaps),
         statistics.stdev(gaps) / math.sqrt(len(gaps)), deaths,
         100.0 * deaths / max(len(results), 1), took), flush=True)

json.dump({"overrides": OVERRIDES, "sims": SIMS, "k": K, "results": results},
          open(OUT, "w"), indent=0)
if stalls:
    print("  STALLS: %d fights hit the %d-step cap (neither win nor loss) -- a "
          "config that values survival with no time cost prefers stalling to "
          "dying" % (stalls, MAX_STEPS), flush=True)
print("wrote", OUT, flush=True)

if VS:
    base = json.load(open(VS))["results"]
    pairs = [(base[k]["dmg"], v["dmg"]) for k, v in live.items()
             if k in base and base[k]]
    if pairs:
        d = [b - a for a, b in pairs]       # positive = new config pays MORE
        mean = statistics.mean(d)
        se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0
        t = mean / se if se else 0.0
        print("\nPAIRED vs %s on %d shared fights:" % (VS, len(pairs)), flush=True)
        print("  HP delta %+.3f +/- %.3f   t=%.2f   %s"
              % (mean, se, t,
                 "BETTER (pays less)" if mean < 0 else "worse (pays more)"),
              flush=True)
        print("  detectable at 2 SE: %.2f HP" % (2 * se), flush=True)

    # Deaths end the run, so they are the primary outcome on the boss set and
    # are not commensurate with HP. Paired McNemar over every shared fight:
    # only the DISCORDANT pairs carry information about which config survives
    # more, and pairing removes the fight-to-fight difficulty variance.
    shared = [k for k in results if k in base]
    b = sum(1 for k in shared if base[k] is None and results[k] is not None)
    c = sum(1 for k in shared if base[k] is not None and results[k] is None)
    if b + c:
        chi = (abs(b - c) - 1) ** 2 / float(b + c)      # continuity-corrected
        base_d = sum(1 for k in shared if base[k] is None)
        new_d = sum(1 for k in shared if results[k] is None)
        print("\nSURVIVAL (McNemar, n=%d shared fights):" % len(shared),
              flush=True)
        print("  deaths %d -> %d  (%.1f%% -> %.1f%%)"
              % (base_d, new_d, 100.0 * base_d / len(shared),
                 100.0 * new_d / len(shared)), flush=True)
        print("  discordant: saved %d, lost %d   chi2=%.2f  %s"
              % (b, c, chi,
                 "SIGNIFICANT (p<0.05)" if chi > 3.84 else "not significant"),
              flush=True)
print("done", flush=True)
