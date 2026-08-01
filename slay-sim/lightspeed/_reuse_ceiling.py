"""How much work could tree reuse actually recycle?

Rerooting saves the subtree under the action actually taken. Two things bound
what that is worth:

  1. N_best / N_total -- the share of the search's simulations that sit under
     the chosen root action. This is the most work a reroot could possibly
     carry forward.
  2. Whether that action consumes RNG. After a chance action (END_TURN, and any
     card that shuffles) the real next state is drawn from the real fight's own
     RNG, which is not the RNG the search sampled its children with -- so the
     real state will generally match no cached child and the reroot must fall
     back to a fresh tree.

Reusable fraction per decision = (N_best / N_total) if deterministic else 0.
"""
from __future__ import annotations

import statistics
import sys

import slaythespire as sts
from lightspeed.search_config import ensure_search_config

ensure_search_config()

SIMS = 300
ENCOUNTERS = ["JAW_WORM", "GREMLIN_NOB", "THE_GUARDIAN", "AUTOMATON",
              "TIME_EATER", "THREE_SENTRIES"]

rows = []
for enc_name in ENCOUNTERS:
    enc = getattr(sts.MonsterEncounter, enc_name)
    for seed in range(4):
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 7000 + seed, 20)
        try:
            bc = sts.new_battle(gc, enc)
        except Exception as exc:
            print(f"  skip {enc_name}: {exc}")
            break
        decisions = 0
        while bc.outcome == sts.BattleOutcome.UNDECIDED and decisions < 60:
            action, visits = sts.run_mcts_search(bc, SIMS, None, 12345 + decisions)
            visits = list(visits)
            total = sum(visits)
            if total <= 0 or len(visits) < 2:
                action.execute(bc)
                decisions += 1
                continue
            best = max(visits)
            # Does the chosen action consume RNG? Probe on a copy, exactly the
            # way the search itself classifies actions.
            probe = bc.copy_self()
            before = probe.rng_counter_sum()
            action.execute(probe)
            deterministic = probe.rng_counter_sum() == before
            rows.append({
                "encounter": enc_name,
                "share": best / total,
                "deterministic": deterministic,
                "reusable": (best / total) if deterministic else 0.0,
                "n_actions": len(visits),
                "is_end_turn": action.action_type == sts.ActionType.END_TURN,
            })
            action.execute(bc)
            decisions += 1

if not rows:
    raise SystemExit("no decisions measured")

det = [r for r in rows if r["deterministic"]]
share_all = statistics.mean(r["share"] for r in rows)
reusable = statistics.mean(r["reusable"] for r in rows)
print()
print("=" * 72)
print(f"{len(rows)} real decisions across {len(ENCOUNTERS)} encounters at {SIMS} sims")
print("=" * 72)
print(f"  mean N_best/N_total (share of tree under chosen action) : {share_all:6.3f}")
print(f"  decisions whose chosen action is deterministic          : "
      f"{len(det)}/{len(rows)} ({100*len(det)/len(rows):.1f}%)")
print(f"  END_TURN decisions                                      : "
      f"{sum(r['is_end_turn'] for r in rows)}/{len(rows)}")
print()
print(f"  MEAN REUSABLE FRACTION                                  : {reusable:6.3f}")
print(f"  => effective budget multiplier if reuse were perfect    : "
      f"{1 + reusable:6.3f}x  ({SIMS} -> {int(SIMS * (1 + reusable))} sims)")
print()
print("  per encounter:")
for enc in ENCOUNTERS:
    sub = [r for r in rows if r["encounter"] == enc]
    if not sub:
        continue
    print(f"    {enc:18s} n={len(sub):3d}  share={statistics.mean(r['share'] for r in sub):.3f}"
          f"  det={100*sum(r['deterministic'] for r in sub)/len(sub):5.1f}%"
          f"  reusable={statistics.mean(r['reusable'] for r in sub):.3f}")
