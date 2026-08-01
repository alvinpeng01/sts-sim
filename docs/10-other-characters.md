# Other characters: what exists, and what starting would cost

Groundwork for a future Silent/Defect/Watcher effort. Nothing here is on the
Ironclad path and no number in the rest of these docs depends on it. Read out of
the tree on 2026-07-31; the harness is `lightspeed/_class_card_audit.py`.

## The engine is not the problem

Earlier revisions of [01](01-architecture.md), [03](03-combat-search.md) and
[07](07-known-issues.md) all said Defect and Watcher were "largely
unimplemented". That was wrong. Measured by cross-referencing `cardTypes` and
`cardColors` against the three type switches in `BattleContext.cpp`:

| Character | Cards | Implemented |
|---|---:|---:|
| Ironclad | 75 | 75 |
| Silent | 75 | 75 |
| Defect | 75 | 75 |
| Watcher | 75 | 75 |

(Reflex, Tactician and Deus Ex Machina have no case because they are unplayable
by design — they trigger on discard or draw.) Orbs, stances, per-character
starting decks and card pools are all wired: `BattleContext.cpp:66` gives the
Defect 3 orb slots, `GameContext.cpp:440-520` builds each character's opening
deck and relic.

Since the card-select enumeration fix (see [07](07-known-issues.md)), a native
MCTS playout runs clean for all four characters.

## The search is the problem, and it is not a tuning problem

The search's *state abstraction* has no representation of what Defect and
Watcher actually do. Verified by grep over `bindings/slaythespire.cpp`: **zero**
references to orbs, and the only matches for "stance" are the substring inside
`CardInstance`, the monster move `THE_CHAMP_DEFENSIVE_STANCE`, and the
`STANCE_POTION` enum binding.

| Mechanic | Character | In `NativeStateKey`? | Scored by the rollout heuristic? | Seen by leaf / terminal eval? |
|---|---|---|---|---|
| Poison | Silent | **yes** — monster status index 0 | partly: `isSilentPoisonApplier` exists, but `silentPoisonApplyBonus` is **0.0** in the shipped config | indirectly, via monster HP |
| Orbs, Focus | Defect | no | no | no |
| Stance, Mantra | Watcher | no | no | no |
| Ironclad powers | Ironclad | yes — 14 of the 19 tracked player statuses are Ironclad-specific | yes, tuned | yes |

`PLAYER_STATUS_IDS` (the 19 statuses the key carries) is
`ARTIFACT, BARRICADE, METALLICIZE, RITUAL, RAGE, RUPTURE, COMBUST, DEMON_FORM,
DARK_EMBRACE, EVOLVE, FEEL_NO_PAIN, FIRE_BREATHING, JUGGERNAUT, PANACHE,
ENVENOM, FLAME_BARRIER, BRUTALITY, REGEN, CORRUPTION`. `ENVENOM` is Silent's;
`ARTIFACT`/`RITUAL`/`REGEN`/`METALLICIZE` are shared; everything else is
Ironclad. There is no `FOCUS` and no `MANTRA`.

What that means concretely:

- For the **Defect**, a position holding three Dark orbs at 40 damage each
  evaluates identically to one with empty orb slots, and channelling Lightning
  versus Frost versus Dark scores the same, because both are just a card of some
  type. Evoke timing — the entire character — is invisible.
- For the **Watcher**, ending a turn in Wrath (double damage taken) evaluates
  the same as ending it in Calm. The search cannot see the one decision that
  decides Watcher fights.

**Tuning cannot fix this.** CMA-ES optimizes weights over features the search
computes; it cannot invent a feature for a mechanic the state does not carry. So
the usual "re-tune for the new character" plan is premature for two of the three.

## Work order, cheapest first

1. **Silent is close to tunable today.** Poison is already keyed, and the
   scoring term exists and is switched off. The missing piece is data:
   `cardPickRateWeight` was learned from ~5,500 Ironclad decisions and every
   other character falls back to a 0.05 smoothing floor, which makes the
   per-card prior — weight **21.8**, one of the strongest live terms — pure
   noise off-Ironclad. `silverPrior` has the same problem: it is Silver
   Automaton's ranking, and Silverbot is an Ironclad bot.
2. **Defect and Watcher need the state abstraction extended first**: orb
   count/type/focus and stance/mantra into `NativeStateKey`, and into
   `nativeLeafFeatures`. Only then are heuristic terms (channel value, evoke
   timing, stance-exit safety) meaningful, and only then is tuning worth compute.
3. **Then** per-character CMA-ES — which should not start before the relic-level
   defect in the fitness objective is fixed, since that is worth an order of
   magnitude more than anything else measured ([03](03-combat-search.md)).

Two details that make step 2 cheaper than it looks. `NativeStateKey` is inert in
production — every consumer is gated behind `g_useStateMerging`, which is off —
so extending it breaks nothing today while being a precondition for state
merging or tree reuse ever working. And `NATIVE_LEAF_FEATURE_DIM` is 10, feeding
only `nativePolicyNetScore`, whose `policyNetWeight` is 0.0; nothing consumes
those features today, so widening the array is free right now and will not be
later.

One hazard: `PLAYER_STATUS_IDS` is duplicated verbatim in two places in
`bindings/slaythespire.cpp` (the state-key builder and the feature builder).
They must stay in sync; nothing enforces it.

## Above the search

The whole-run side is Ironclad-only by construction:
`sts.CharacterClass.IRONCLAD` is hardcoded in five non-test files
(`whole_run_env.py:257`, `env.py:822`, `benchmark_full_runs.py:40`,
`compare_tier_combat.py:126`, `eval_heart1_hybrid.py:23`). The policy embeds
cards by raw `CardId` so it can *represent* other characters' cards, but it has
no training signal for them, and every baseline and eval `.jsonl` in `runs/` is
Ironclad. A new character starts a lineage with no comparison points.

## The standing reason to keep exercising them

Independent of ever targeting another character, running all four is a cheap
fuzzer for shared machinery, and it has already paid for itself. Every defect
found on 2026-07-31 — five empty card-select enumerations, two enumerations
emitting actions the validator rejects, four cards dispatched to a switch that
did not implement them — was found by exercising Silent, Defect and Watcher, and
Ironclad-only testing would never have reached any of them.

The bug *classes* are shared even when the instances are not: `default:`
silently returning an empty vector, enumeration drifting from validation, and
`assert()` being inert under `-DNDEBUG`. That first class did reach Ironclad
once, through Melange granting Scry.

`lightspeed/_class_card_audit.py` runs all four in seconds. Run it before any
change to `Action.cpp`, `BattleContext.cpp`'s type switches, or the bindings'
action enumeration.
