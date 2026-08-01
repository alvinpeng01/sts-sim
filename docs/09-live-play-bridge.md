# Live play: bridge, overlay, autobattle

An optional path that plays alongside the real game. It is independent of the
training pipeline and shares only the native engine and the tuned search config.

## The pieces

```
Slay the Spire (+ ModTheSpire, BaseMod, CommunicationMod)
   │
   ├── CommunicationMod ──► launches  slay-sim/run_bridge.py  as a subprocess
   │        stdin/stdout JSON protocol
   │
   ├── slay-sim/sts/bridge/communication_mod.py     the bridge process
   │        ├── native_recommend.py    native MCTS on a reconstruction of the live state
   │        ├── predict.py             damage-only fallback
   │        └── state_mapper.py        legacy CombatState reconstruction (not on the hot path)
   │
   └── stsmod/ (STSPredictor.jar)      BaseMod overlay: draws the panel, F9 toggle
```

`run_bridge.py` exists because CommunicationMod builds its subprocess command by
naive whitespace-splitting a config string, with no shell and no working
directory — confirmed by decompiling `CommunicationMod.class`. So the entry point
must live at a path without spaces and must fix up `sys.path` from `__file__`
rather than relying on cwd.

## Two prediction layers

Tried in order; each falls back to the next, and a bug in either is logged and
never crashes the bridge or blocks the game.

1. **Full recommendation** — `native_recommend.native_recommend()` runs the
   project's own native C++ MCTS with `lightspeed/tuned_search_params.json` on a
   reconstruction of the live JSON, and returns an actual card to play.
2. **Damage-only (v1)** — `predict.py` does net-incoming-damage arithmetic
   straight off the live JSON's telegraphed intents. No recommendation, no
   assumptions about monster AI. Always available.

`native_recommend` is deliberately **self-sufficient**: it builds its own
lightweight shadow combat rather than going through
`state_mapper.build_combat_state()`. Replaying this project's own captured live
data found that 257 of 300 sampled real states were a Taskmaster fight, which
`sts/enemies.py` has no class for — so requiring the Python engine's
reconstruction to succeed first was gating the native engine behind a *smaller*
monster roster than the native engine itself has.

It also needs the **outer** `game_state` payload, not just `combat_state`, for
ascension level, relics and potions. An earlier version hardcoded ascension 20
and passed no relics or potions regardless of the real game.

## Autobattle

Off by default at every game launch. Toggled in-game with **F9**
(`STSPredictorMod.java`, a `PostUpdateSubscriber`), which writes
`~/sts_autobattle_enabled.txt`. The bridge re-reads that file fresh on every
state push — never cached.

A command is only ever sent when **all** of these hold:

- autobattle is on;
- layer 1 produced a real recommendation — autobattle **never** acts on the v1
  damage-only fallback, which has no action to act on, only a number;
- the resolved command appears in that state's own `available_commands`, the same
  check the no-op `state` command always had to clear.

Otherwise the bridge writes `state`, exactly as it did when it was advisory-only.

### Index conventions

`_build_command` was verified against `CommunicationMod.class`'s
`executePlayCommand` bytecode, decompiled rather than assumed:

- **Card index is 1-based** against the live hand: `hand[i]` → `play i+1`.
- **Monster index is 0-based** against the JSON's own **unfiltered** `monsters`
  array.

The second one is the subtle half. Dead or gone monsters are filtered out when
reconstructing combat, so `combat.monsters` position drifts from the JSON's as
soon as anything dies. Each mapped monster therefore carries a `json_index` tag,
and the builder returns `None` — refusing to act — rather than guessing when a
chosen card is no longer in hand or a target has no verified index.
`tests/test_autobattle.py` covers both directions.

## Open: the reconstruction guesses monster intents

`native_recommend.py` never sets `move_name` on a `NativeMonsterSpec`, so
`build_battle_context` rolls a plausible move instead of using the one the game
is telegraphing — in every fight, not just under Runic Dome. CommunicationMod
supplies `intent`, `move_id`, `move_base_damage`, `move_adjusted_damage` and
`move_hits` on 100% of monster records in this project's own captures, and
`intent` is `UNKNOWN` only 1.5% of the time.

Measured against the capture, the rolled move matches the telegraphed one only
**12.5% of the time** (125/1000 single-monster states) — and the failure mode
includes predicting **zero** incoming damage against Snecko and Snake Plant,
which makes the search suppress defensive cards and attack into the hit.

This is the largest remaining defect on the live path. Detail, and the 21-entry
fingerprint table that would resolve it, in
[07-known-issues.md](07-known-issues.md).

## Fixed 2026-07-30: the reconstructed state let cards be replayed

`sts.build_battle_context()` used to produce a `BattleContext` in which a played
card was copied to the discard pile but **never removed from hand**, so the
search planned against an infinite hand — every live recommendation and every
autobattle decision was affected. Root cause was a `uniqueId` signedness
mismatch; detail and verification in [07-known-issues.md](07-known-issues.md).

Any recommendation logged before that fix should be treated as unreliable.
Training and evaluation were never affected — they do not use this constructor.

## The Windows rename race

The overlay file is written atomically (temp file + rename). On Windows that
rename can fail with `PermissionError` / "Access is denied" if the Java mod has
the file open for reading at that instant — a real race between the mod's
per-frame poll and the bridge's per-state write, and Windows-specific, since
POSIX allows renaming over an open destination.

Confirmed from a live session log: about 1.6% of writes were failing silently
this way for a whole session. The symptom was a stale-by-one-cycle overlay, not a
crash. Fixed with a short retry-with-backoff around the rename, giving up after a
handful of attempts rather than retrying forever.

## Installing

`stsmod/STSPredictor.jar` is prebuilt; no Java toolchain needed. See
`slay-sim/stsmod/README.md` for the mod half and the full user-facing behaviour
of autobattle. The mod itself never reads game state and never acts — it polls
one text file the bridge writes and writes one toggle file the bridge reads.

The mod half needs only `slay-sim/sts/` and numpy. The **recommendation** layer
needs the compiled `slaythespire` module; without it the bridge still runs, in
damage-only mode.

## Captured data

`slay-sim/sts_raw_states.log` (50 MB) and `sts_predictions.log` (37 MB) are real
CommunicationMod captures from a live session. They back
`tests/test_native_recommend.py` and are the reason several id-mapping bugs were
found rather than guessed at — for example Looter's real CommunicationMod id
being `Mugger`, and Taskmaster's being `SlaverBoss`.
