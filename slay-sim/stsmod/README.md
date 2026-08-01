# STS Predictor Overlay

A tiny BaseMod mod that renders the `sts/bridge/` recommendation as a panel
in the top-left corner of the game window, and (new) an F9 toggle for
**autobattle** -- letting the Python bridge actually play recommended
cards/end your turn for you, instead of only ever suggesting.

This mod itself does not read game state and never acts on your behalf --
it only polls one text file the separate Python bridge process
(`sts.bridge.communication_mod`) writes, and writes one text file of its
own (the autobattle toggle) that the bridge reads back. The actual
game-state reading, searching, and (when autobattle is on) command-sending
all happen in that separate Python process, launched by CommunicationMod.
Compiled against BaseMod, ModTheSpire, and the game's own `desktop-1.0.jar`
(introspected with `javap` for the real API, not written from memory), and
verified against a real live run this time, not just compiled cleanly and
untested.

## Install (no Java toolchain needed — `STSPredictor.jar` is prebuilt)

1. Copy `STSPredictor.jar` into your StS1 mods folder — the same place
   BaseMod/StSLib/etc. already live for ModTheSpire to find. On Windows
   that's typically wherever you subscribed workshop mods land, or
   ModTheSpire's own configured mods directory (check `Edit ModTheSpire.json`
   / the mods list in the ModTheSpire launcher for the exact folder it's
   scanning — same place your other installed mods showed up).
2. Launch via ModTheSpire, check the box for **STS Predictor Overlay**
   alongside BaseMod, StSLib, and CommunicationMod.
3. That's it — no build step. Java bytecode runs the same on any OS; the
   jar was compiled once here and just needs to be present.

If you already had a previous version of this mod installed, replacing the
jar takes effect on your **next launch** — a JVM that's already running has
the old class bytecode loaded in memory regardless of what's on disk.

## Autobattle (new)

Press **F9** any time (in or out of combat) to toggle autobattle on/off.
The overlay's top line always shows the current state — white/yellow
`Autobattle: OFF` or red `Autobattle: ON` — specifically so it's never
ambiguous whether the mod is just watching or actually playing your turns.

**Defaults OFF, every single time the game starts** — this mod deliberately
does not remember your last session's setting; you opt in fresh each time
you launch. If you never press F9, this behaves exactly like the
advisory-only version always did.

Even with it on, the Python side (`communication_mod.py`) refuses to act
unless *all* of these hold, every single state push:

- autobattle is genuinely on (it re-reads the toggle file fresh each time,
  not a cached value);
- a real recommendation was computed this state — it **never** acts on
  the damage-only (v1) fallback, since that layer has no actual
  card/target choice to send, only a damage number;
- the command it would send is currently listed as legal in that state's
  own `available_commands` (the same strict check this mod's `state`
  command always required — a wrong command sent to a live run is not a
  mistake to allow silently, and that bar didn't lower just because more
  commands are now in scope);
- for a card play, the chosen card is still actually found in your current
  hand (a last-instant guard against acting on anything stale).

Anything failing any of those falls back to sending `state` (the same
no-op as always) for that state push, and logs why. Every autobattle
action is also logged with an explicit `[autobattle]` line in
`sts_predictions.log`, separate from ordinary recommendation logging, so
what actually got played is always auditable after the fact.

**Scope**: autobattle only ever plays cards or ends your turn — it does not
touch potions, map choices, rewards, shops, or anything outside combat
(the underlying search doesn't reason about those at all, so there's
nothing correct to send).

## How the two sides connect

- `sts.bridge.communication_mod` (Python, launched by CommunicationMod)
  writes `~/sts_latest_recommendation.txt` every time a new game state
  arrives — overwritten, not appended, and written atomically (temp file +
  rename) so this mod never reads a half-written file mid-update. On
  Windows specifically, that rename can transiently fail if this mod
  happens to have the file open for reading at that exact instant (a real,
  if narrow, race — confirmed from a live run's own log, not theoretical);
  the Python side retries a few times with backoff before giving up and
  logging a warning, rather than crashing or wedging the bridge.
- This mod polls that file's last-modified timestamp once a frame (cheap)
  and only re-reads its content when that timestamp changes.
- This mod also writes `~/sts_autobattle_enabled.txt` (`"true"`/`"false"`)
  whenever you press F9; the Python side re-reads that fresh on every
  state push.
- `~` resolves to the same real path on both sides — Python's `Path.home()`
  and Java's `System.getProperty("user.home")` — so this works regardless
  of where either side's code physically lives on disk, including across
  the file transfer to a different machine.
- When you're not in combat, `communication_mod.py` writes an empty
  recommendation file, so the recommendation lines disappear on their own
  rather than showing stale advice (the autobattle toggle line itself
  still always shows, in or out of combat).

## Rebuilding from source (only needed if you change the .java)

Needs a JDK (not just a JRE) plus the same three jars on the classpath:
BaseMod.jar, ModTheSpire.jar, and the game's own desktop-1.0.jar (find
paths via wherever Steam/ModTheSpire put them on your system), targeting
Java 8 bytecode specifically -- the game bundles its own Java 8 JRE, and a
jar compiled with a newer `--release` will fail to load with
`UnsupportedClassVersionError` at mod-init time (hit this for real once
already; see the main project README's own note on it).

```
javac --release 8 -cp "BaseMod.jar;ModTheSpire.jar;desktop-1.0.jar" -d build src/stspredictor/STSPredictorMod.java
cp ModTheSpire.json build/
jar cf STSPredictor.jar -C build .
```
(On Linux/macOS, use `:` instead of `;` between classpath entries.)
