"""Wires predict.py / native_recommend.py's search to CommunicationMod's
stdin/stdout JSON protocol.

Run:  PYTHONPATH=. .venv/bin/python -m sts.bridge.communication_mod
(CommunicationMod itself launches this as a subprocess per its config's
`command=` line -- you won't normally run it by hand once that's set up.)

Two layers of prediction, tried in order, each falling back to the next:
  1. FULL RECOMMENDATION -- native_recommend.native_recommend() runs THIS
     PROJECT'S OWN native combat engine (sts_lightspeed's slaythespire C++
     MCTS + lightspeed/tuned_search_params.json, tuned and validated all
     last session at 83%+ win rate) on a sts.build_battle_context()
     reconstruction of the live JSON, to pick an actual card to play, not
     just a damage number. This is what "wiring the mod" means -- see
     native_recommend.py's own docstring for the monster/card/status id
     mapping tables (many confirmed against this project's own real
     CommunicationMod captures, sts_raw_states.log) and for why this
     module does NOT use state_mapper.py's older CombatState reconstruction
     (a real, data-backed finding: sts/enemies.py's smaller monster roster
     was silently gating the native engine behind coverage it doesn't
     actually need). Previously this layer used sts.mcts.mcts_choose_action()
     (a separate, pure-Python MCTS + an earlier learned value net, built on
     top of state_mapper's CombatState) -- kept importable via git history,
     not deleted, but no longer wired in here.
  2. DAMAGE-ONLY (v1) -- if native_recommend hits an UnmappedMonsterError (an
     id it doesn't recognize -- expected until the id tables are corrected
     against real captured data) or raises for any other reason, falls back
     to predict.py's v1 mode: no recommendation, just net-incoming-damage
     arithmetic straight off the live JSON's own telegraphed intents.
     Always available, zero unverified assumptions beyond predict.py's own
     DAMAGING_INTENTS/field-name notes.
A prediction bug at either layer is logged and never crashes the bridge or
blocks the game -- see the broad except in main()'s loop.

ADVISORY BY DEFAULT, AUTOBATTLE OPT-IN: the only thing ever written to
stdout used to unconditionally be the literal command "state" -- but this
now supports an autobattle mode (see AUTOBATTLE_PATH below) where, if
explicitly turned on (from the in-game overlay, defaults OFF), a computed
"play"/"end" command is sent instead of "state" when -- and only when --
every one of these holds:
  - autobattle is turned on (the toggle file says so);
  - layer 1 (the real recommendation) succeeded this state -- autobattle
    NEVER acts on a v1 damage-only fallback, since that layer has no
    CombatState/action to act on at all, only a damage number;
  - the resolved command is *actually* listed in this state's own
    `available_commands` (same strict check "state" always got -- sending
    the wrong command to a live run, confirming a card removal etc., is
    not a mistake to allow silently, and that bar doesn't lower just
    because more commands are now in scope);
  - for a "play", the target card is still found at the hand index we
    computed it from (a last-second sanity check against acting on stale
    data -- see _build_command).
Any of those failing falls back to "state", exactly like before. Every
prediction is routed to a separate log file instead of stdout, since
CommunicationMod owns stdout; autobattle actions get an explicit log line
of their own (see _log's "[autobattle]" prefix) so what actually got
played is always auditable after the fact, not just what was recommended.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

from .predict import summarize
from .native_recommend import native_recommend, UnmappedMonsterError

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = _PROJECT_ROOT / "sts_predictions.log"
RAW_LOG_PATH = _PROJECT_ROOT / "sts_raw_states.log"  # for verifying schema assumptions later

# The in-game overlay mod (stsmod/) runs inside the game's own JVM, a
# separate process from this one -- CommunicationMod only launches THIS
# script as its child, not the mod. They share state through plain files
# instead. Deliberately Path.home(), not project-relative: the mod doesn't
# know or care where slay-sim/ lives on disk, and `System.getProperty
# ("user.home")` on the Java side resolves to the same place on any OS, so
# this works identically whether slay-sim/ ends up in a different folder,
# a different drive, even a different machine account setup.
LATEST_PATH = Path.home() / "sts_latest_recommendation.txt"
# Toggled in-game by pressing the overlay mod's autobattle key (F9 by
# default -- see STSPredictorMod.java); contains the literal text "true"
# when on. Missing/anything-else means off, which is the safe default: a
# fresh install with this file absent behaves exactly like the
# advisory-only mod did before autobattle existed.
AUTOBATTLE_PATH = Path.home() / "sts_autobattle_enabled.txt"

_search_ready = False


def _init_search():
    """Loads the native engine's tuned search params lazily and once, so
    layer-2 fallback still works with zero extra dependencies if this ever
    fails (slaythespire not built, tuned_search_params.json missing, ...).

    This project's own C++ engine (sts_lightspeed/slaythespire), tuned and
    validated all last session at 83%+ win rate -- see native_recommend.py's
    own docstring. Replaces the OLD sts.mcts/value_net_weights_v2.npz path
    (kept importable via git history, not deleted, in case a direct
    A/B is ever wanted, but no longer wired in here)."""
    global _search_ready
    if _search_ready:
        return True
    try:
        from . import native_recommend as _nr
        _nr._ensure_params_loaded()
        _search_ready = True
    except Exception as e:
        _log(LOG_PATH, f"[warn] native search engine unavailable ({type(e).__name__}: {e}), "
                        f"recommendations will fall back to v1 damage-only mode.")
        _search_ready = False
    return _search_ready


def _log(path: Path, line: str) -> None:
    with path.open("a") as f:
        f.write(line + "\n")


# RAW_LOG_PATH logs every raw state line unconditionally (see below) --
# found at ~7GB/537K lines, almost entirely idle-poll noise (the same
# "in_game: false" line repeated every tick at the menu), since nothing
# ever capped or rotated it. This keeps only the most recent MAX_RAW_LOG_BYTES
# worth of lines instead of accumulating forever.
MAX_RAW_LOG_BYTES = 50 * 1024 * 1024  # 50MB -- plenty for schema verification, not unbounded


def _log_capped(path: Path, line: str, max_bytes: int) -> None:
    """Same as _log, but truncates the file down to roughly its last
    max_bytes once it exceeds that size -- a simple tail-keeping rotation.
    path.stat() is O(1) (cheap on every call); the actual truncate-rewrite
    only runs once the file has grown max_bytes past its last truncation,
    so the amortized cost per write stays low."""
    _log(path, line)
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return
    if size > max_bytes:
        with path.open("rb") as f:
            f.seek(-max_bytes, 2)
            tail = f.read()
        # Drop a possibly-truncated partial first line so every remaining
        # line is a complete, valid JSON record.
        first_newline = tail.find(b"\n")
        if first_newline != -1:
            tail = tail[first_newline + 1:]
        with path.open("wb") as f:
            f.write(tail)


def _write_latest(text: str) -> None:
    """Overwrites (not appends) the small file the in-game overlay mod
    polls -- see stsmod/'s README for the read side. Written atomically
    (write to a temp file, then rename) so the overlay never reads a
    half-written file mid-update; on the same filesystem a rename is a
    single directory-entry swap, not a byte-by-byte operation, so there's
    no window where the file exists but is partially the old/new content.

    Real bug found via a live run's own log (not theoretical): on Windows,
    a rename onto an existing destination (`MoveFileEx`) fails outright if
    another process has that destination open without FILE_SHARE_DELETE at
    that exact instant -- unlike POSIX, where a rename can't be blocked by
    a concurrent reader at all. The overlay mod (STSPredictorMod.java)
    polls this exact file's mtime every frame and opens it to re-read
    whenever that changes, so there's a real, if narrow, window on every
    single write where its read and this rename can collide -- a live
    session's log showed this happening on ~1.6% of writes
    (PermissionError, WinError 5 "Access is denied"), silently leaving
    that one update's overlay content stale until the next successful
    write. Retrying a few times with a short backoff is the standard fix
    for this exact category of Windows-only transient file-lock race;
    giving up after a handful of attempts (rather than retrying forever)
    keeps a truly stuck lock from hanging the whole bridge loop."""
    tmp = LATEST_PATH.with_suffix(".tmp")
    tmp.write_text(text)
    last_error = None
    for attempt in range(5):
        try:
            tmp.replace(LATEST_PATH)
            return
        except PermissionError as e:
            last_error = e
            time.sleep(0.01 * (attempt + 1))
    _log(LOG_PATH, f"[warn] _write_latest: rename failed after retries ({last_error}); "
                    f"overlay will show stale content until the next successful write.")


def _autobattle_enabled() -> bool:
    """Re-read on every state push rather than cached/mtime-polled (unlike
    the Java side's overlay-file read) -- state pushes are nowhere near
    frequent enough (per-combat-update, not per-frame) for a plain read of
    a few bytes to matter, and a stale cached "on" surviving after the
    player turns it back off in-game is a worse failure mode than the
    trivial extra I/O."""
    try:
        return AUTOBATTLE_PATH.read_text().strip().lower() == "true"
    except FileNotFoundError:
        return False


def _build_command(action, combat, available_commands) -> Optional[str]:
    """Turns a chosen (action, combat) into the literal string
    CommunicationMod expects, or None if it can't be built safely --
    never a guess. Verified against CommunicationMod.class's own
    executePlayCommand/executeEndCommand bytecode (decompiled, not
    assumed): card index is 1-based (hand[i] -> command index i+1) and
    monster index is 0-based, matching the JSON's own (unfiltered)
    `monsters` array position -- see state_mapper.py's `json_index` tag on
    each mapped Monster for why that's not simply combat.monsters.index(m)
    once anything in the fight has died.
    """
    if action[0] == "end":
        if "end" not in available_commands:
            return None
        return "end"

    _, card, target = action
    if "play" not in available_commands:
        return None
    try:
        hand_idx = combat.hand.index(card)
    except ValueError:
        return None  # the chosen card isn't (or isn't still) in hand -- don't guess
    command = f"play {hand_idx + 1}"
    if target is not None:
        json_idx = getattr(target, "json_index", None)
        if json_idx is None:
            return None  # no verified index to target with -- refuse rather than guess
        command += f" {json_idx}"
    return command


def _try_recommend(combat_state_json: dict, game_state: dict):
    """NATIVE ENGINE search recommendation. Raises on any failure -- callers
    fall back to v1, this never partially-degrades silently. Returns
    (description, action, combat) -- the description is for the log/overlay,
    (action, combat) is what autobattle needs to actually build a command
    later in handle_state, kept together so there's only one recommendation
    computed per state, not one for display and a second for acting.

    game_state (the OUTER payload["game_state"], not combat_state_json) is needed
    for ascension_level/relics/potions, which live one level up from combat_state_json
    -- see native_recommend()'s own docstring for why this matters (a real bug: an
    earlier version hardcoded ascension=20 and passed zero relics/potions regardless
    of the actual live game).

    Deliberately does NOT call state_mapper.build_combat_state() -- see
    native_recommend.py's own module docstring for why (a real, data-backed
    finding: the older engine's roster gaps were silently gating the native
    engine behind a smaller monster coverage than it actually has).
    native_recommend() builds its own lightweight state directly from the
    raw JSON."""
    return native_recommend(combat_state_json, game_state, log_fn=lambda msg: _log(LOG_PATH, msg))


def handle_state(payload: dict) -> Optional[str]:
    """Returns the command to send to CommunicationMod for this state, or
    None to mean "just send 'state'" -- see main()'s loop for how that's
    used. Only ever returns non-None when autobattle is on AND a real
    recommendation was computed AND _build_command's own checks all pass;
    see the module docstring for the full list."""
    game_state = payload.get("game_state") or {}
    combat_state = game_state.get("combat_state")
    if combat_state is None:
        _write_latest("")  # not in a fight -- clear the overlay rather than show stale info
        return None

    result = summarize(combat_state)
    ts = time.strftime("%H:%M:%S")
    base_line = (
        f"[{ts}] turn {result['turn']}  HP {result['player_hp']}  "
        f"block {result['player_block']}  net incoming {result['net_incoming_damage']}  "
        f"| " + ", ".join(f"{m['name']}:{m['damage']}" for m in result["per_monster"])
    )

    recommendation = None
    action = combat = None
    if _init_search():
        try:
            recommendation, action, combat = _try_recommend(combat_state, game_state)
        except UnmappedMonsterError as e:
            recommendation = f"(no recommendation: {e})"
        except Exception as e:
            recommendation = f"(no recommendation: {type(e).__name__}: {e})"

    autobattle_on = _autobattle_enabled()
    command = None
    if autobattle_on and action is not None:
        available = payload.get("available_commands") or []
        command = _build_command(action, combat, available)
        if command is not None:
            _log(LOG_PATH, f"[autobattle] sending: {command}")

    _log(LOG_PATH, base_line + (f"  || {recommendation}" if recommendation else ""))

    # Short, multi-line, screen-friendly form for the in-game overlay --
    # the log line above is for the human tailing a terminal; this one is
    # for stsmod/'s renderer, which just splits on "\n" and draws each line.
    overlay_lines = [f"Turn {result['turn']}  HP {result['player_hp']}  Block {result['player_block']}"]
    if result["net_incoming_damage"]:
        overlay_lines.append(f"Incoming: {result['net_incoming_damage']}")
    if recommendation:
        overlay_lines.append(recommendation)
    if autobattle_on:
        overlay_lines.append(f"[autobattle -> {command}]" if command else "[autobattle: waiting]")
    _write_latest("\n".join(overlay_lines))

    return command


def main() -> None:
    # CommunicationMod's startExternalProcess() blocks for up to
    # maxInitializationTimeout seconds waiting for ANY line on this
    # process's stdout right after spawning it -- before it ever sends the
    # first game state (confirmed by decompiling CommunicationMod.class;
    # the line is consumed directly by the handshake, never passed to its
    # command executor, so content doesn't matter). Without this, both
    # sides wait on each other forever: CommunicationMod logs "Timed out
    # while waiting for signal from external process."
    sys.stdout.write("ready\n")
    sys.stdout.flush()

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        _log_capped(RAW_LOG_PATH, raw_line, MAX_RAW_LOG_BYTES)  # keep every raw state for later schema verification, but capped

        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as e:
            _log(LOG_PATH, f"[error] malformed state line: {e}")
            continue

        command = None
        try:
            command = handle_state(payload)
        except Exception as e:  # a prediction bug should never crash the bridge or block the game
            _log(LOG_PATH, f"[error] prediction failed: {type(e).__name__}: {e}")

        if not payload.get("ready_for_command"):
            continue

        available = payload.get("available_commands") or []
        if "state" not in available:
            _log(LOG_PATH,
                 f"[fatal] 'state' not in available_commands ({available}) -- "
                 f"refusing to guess a command, exiting.")
            sys.stdout.write("state\n")  # best-effort last attempt, then stop
            sys.stdout.flush()
            return

        # command is only ever non-None when handle_state's own autobattle
        # checks all passed (see its docstring) -- otherwise this is
        # exactly the unconditional "state" this bridge always sent.
        sys.stdout.write((command or "state") + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
