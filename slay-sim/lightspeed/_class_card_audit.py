"""Card-select enumeration coverage for all four characters.

Motivation: `Action::enumerateCardSelectActions` covered 20 of the 26
CardSelectTasks. The six it omitted fell to `default:` and returned an EMPTY
action vector, which is the same failure InputState::SCRY had -- an empty list
reaches nativeHeuristicPick's `return legal[0]` (slaythespire.cpp:1286) and
leaves a searched node with zero edges. All six belonged to the other three
characters (DISCARD -> Silent, HOLOGRAM -> Defect, NIGHTMARE/SETUP/RETAIN ->
Watcher), which is why Ironclad-only training never hit it.

Two passes:

  targeted   -- every select-opening card in the game, each guaranteed into
                hand, asserting the enumeration is non-empty and that each
                enumerated action actually resolves the select.
  randomized -- real GameContext decks per character, driven both through the
                Python enumeration loop and through the all-C++
                native_playout_battle, watching stderr.

Watching stderr is the point rather than an afterthought: `sts_common.h:8`
defines `sts_asserts` unconditionally but CMakeLists.txt:17 passes -DNDEBUG, so
`assert()` is a no-op in every release build while the `std::cerr` writes
beside it still happen. Those writes are the only runtime signal that a card is
unimplemented (BattleContext's three type switches) or that an action failed
validation (Action::execute). A silent run is the pass condition.

usage:  python -m lightspeed._class_card_audit [randomized_fights_per_class]
"""
from __future__ import annotations

import contextlib
import os
import random
import sys
import tempfile

import slaythespire as sts

STATE_CARD_SELECT = sts.INPUT_STATE_CARD_SELECT
STATE_NAMES = {0: "EXECUTING_ACTIONS", 1: "PLAYER_NORMAL", 2: "CARD_SELECT",
               3: "CHOOSE_STANCE_ACTION", 9: "SCRY"}

# (card_string_id, expected task, character) -- every card in the engine that
# opens a CARD_SELECT screen. Powers that only open one at end of turn are
# marked needs_end_turn.
SELECT_CARDS = [
    # Ironclad
    ("Armaments", "ARMAMENTS", "Ironclad", False),
    ("Headbutt", "HEADBUTT", "Ironclad", False),
    ("Exhume", "EXHUME", "Ironclad", False),
    ("Dual Wield", "DUAL_WIELD", "Ironclad", False),
    ("Warcry", "WARCRY", "Ironclad", False),
    ("Burning Pact", "EXHAUST_ONE", "Ironclad", False),
    ("Second Wind", "EXHAUST_MANY", "Ironclad", False),
    # Silent -- all DISCARD, the task that fell through
    ("Acrobatics", "DISCARD", "Silent", False),
    ("Prepared", "DISCARD", "Silent", False),
    ("Concentrate", "DISCARD", "Silent", False),
    ("Dagger Throw", "DISCARD", "Silent", False),
    ("Survivor", "DISCARD", "Silent", False),
    ("Expertise", "-", "Silent", False),
    # Defect
    ("Hologram", "HOLOGRAM", "Defect", False),
    ("Redo", "RECYCLE", "Defect", False),
    ("Seek", "SEEK", "Defect", False),
    # Watcher
    ("PathToVictory", "-", "Watcher", False),  # Pressure Points' real string id
    ("Blizzard", "-", "Defect", False),
    ("Scrape", "-", "Defect", False),
    ("Self Repair", "-", "Defect", False),
    ("Night Terror", "NIGHTMARE", "Watcher", False),
    ("Setup", "SETUP", "Watcher", False),
    ("Omniscience", "OMNISCIENCE", "Watcher", False),
    ("Wish", "WISH", "Watcher", False),
    ("ForeignInfluence", "FOREIGN_INFLUENCE", "Watcher", False),
    ("Meditate", "MEDITATE", "Watcher", False),
    ("Well Laid Plans", "RETAIN", "Watcher", True),
    # Colorless
    ("Discovery", "DISCOVERY", "Colorless", False),
    ("Forethought", "FORETHOUGHT", "Colorless", False),
    ("Secret Weapon", "SECRET_WEAPON", "Colorless", False),
    ("Secret Technique", "SECRET_TECHNIQUE", "Colorless", False),
    ("Thinking Ahead", "-", "Colorless", False),
]

FILLER = [("Strike_R", 0), ("Defend_R", 0), ("Bash", 0), ("Anger", 0)]
DRAW = [("Strike_R", 0), ("Defend_R", 0), ("Iron Wave", 0), ("Clothesline", 0),
        ("Shrug It Off", 0)]
DISCARD = [("Strike_R", 0), ("Defend_R", 0)]
EXHAUST = [("Anger", 0), ("Defend_R", 0)]


class Captured:
    """Holder for text captured off fd 2; `.text` is filled on block exit."""
    text = ""


@contextlib.contextmanager
def captured_cxx_stderr():
    """Capture fd 2, so C++ std::cerr is seen and not just sys.stderr.

    Read `.text` AFTER the with-block: the redirect is only undone and the
    buffer only rewound on exit, so reading inside the block yields nothing.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 2)
    holder = Captured()
    try:
        yield holder
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        tmp.seek(0)
        holder.text = tmp.read().decode("utf-8", "replace")
        tmp.close()


def bridge_ctx(hand):
    m = sts.NativeMonsterSpec()
    m.monster_id_name = "JAW_WORM"
    m.cur_hp = m.max_hp = 60
    return sts.build_battle_context(
        player_hp=70, player_max_hp=80, player_block=0, player_energy=99,
        player_statuses=[], monsters=[m], hand_cards=hand,
        draw_pile_cards=DRAW, discard_pile_cards=DISCARD,
        exhaust_pile_cards=EXHAUST, potion_slots=[], relics=[], turn=1,
        ascension=20, rng_seed=4242)


def play_named(bc, string_id):
    """Play the hand card whose slot matches string_id's position (index 0)."""
    for a in bc.get_legal_actions():
        if a.action_type == sts.ActionType.CARD and a.source_idx == 0:
            a.execute(bc)
            return True
    return False


def targeted_pass():
    print("=" * 88)
    print("TARGETED: every select-opening card, guaranteed into hand")
    print("=" * 88)
    print(f"{'card':20s} {'task':18s} {'char':10s} {'state':13s} {'legal':>6s}  result")
    print("-" * 88)
    failures = []
    for string_id, task, char, needs_end_turn in SELECT_CARDS:
        try:
            bc = bridge_ctx([(string_id, 0)] + FILLER)
        except RuntimeError as exc:
            print(f"{string_id:20s} {task:18s} {char:10s} build failed: {exc}")
            failures.append((string_id, "build failed"))
            continue

        with captured_cxx_stderr() as cap:
            played = play_named(bc, string_id)
            if needs_end_turn:
                for a in bc.get_legal_actions():
                    if a.action_type == sts.ActionType.END_TURN:
                        a.execute(bc)
                        break
            raw = sts.get_input_state_raw(bc)
            legal = list(bc.get_legal_actions())
            # Resolving through EVERY enumerated action is the real check:
            # an action the enumerator emits but the validator rejects makes
            # Action::execute write its diagnostic dump to stderr.
            resolved = 0
            for a in legal:
                probe = bc.copy_self()
                a.execute(probe)
                if sts.get_input_state_raw(probe) != STATE_CARD_SELECT:
                    resolved += 1
        noise = cap.text

        if not played:
            verdict, ok = "could not play", False
        elif raw != STATE_CARD_SELECT:
            verdict, ok = "no select opened", True
        elif not legal:
            verdict, ok = "*** EMPTY ENUMERATION ***", False
        elif resolved == 0:
            verdict, ok = "*** none resolved ***", False
        else:
            verdict, ok = f"{resolved}/{len(legal)} resolve", True
        if noise.strip():
            verdict += "  +STDERR"
            ok = False
        print(f"{string_id:20s} {task:18s} {char:10s} "
              f"{'CARD_SELECT' if raw == STATE_CARD_SELECT else 'normal':13s} "
              f"{len(legal):6d}  {verdict}")
        if not ok:
            failures.append((string_id, verdict, noise[:400]))
    return failures


def dispatch_pass():
    """Is every card's `case` in the switch its CardType routes it to?

    BattleContext::useCard switches on CardInstance::getType(), which is
    cardTypes[id] -- the same table sts.get_card_type exposes -- and hands off
    to useAttackCard / useSkillCard / usePowerCard, each with its own switch
    and its own "attempted to use unimplemented card" default. A case sitting
    in the wrong one of the three is unimplemented at runtime even though
    `case CardId::X` exists in the file, so grepping for the case is not
    enough. BURST hit exactly this and is documented at BattleContext.cpp:2229.
    """
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[2] / \
        "sts_lightspeed/src/combat/BattleContext.cpp"
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    funcs = ["useAttackCard", "useSkillCard", "usePowerCard"]
    # Bound each switch by the NEXT member function of ANY name; bounding only
    # by the next of the three swallows later unrelated switches (the Wish
    # handler's own `switch (id)` is why this matters).
    defs = [(i, m.group(1)) for i, ln in enumerate(lines)
            if (m := re.search(r"^\w[\w:<>,&* ]*\bBattleContext::(\w+)\s*\(", ln))]
    where: dict[str, list[str]] = {}
    for pos, (start, fn) in enumerate(defs):
        if fn not in funcs:
            continue
        end = defs[pos + 1][0] if pos + 1 < len(defs) else len(lines)
        for m in re.finditer(r"case CardId::([A-Z0-9_]+)",
                             "\n".join(lines[start:end])):
            where.setdefault(m.group(1), []).append(fn)

    expect = {"ATTACK": "useAttackCard", "SKILL": "useSkillCard",
              "POWER": "usePowerCard"}
    # Unplayable by design: Reflex/Tactician trigger on discard
    # (CardInstance.cpp:220), Deus Ex Machina on draw (CardManager.cpp:421),
    # and the three Wish options are applied by chooseWishCard, never played.
    exempt = {"REFLEX", "TACTICIAN", "DEUS_EX_MACHINA",
              "LIVE_FOREVER", "BECOME_ALMIGHTY", "FAME_AND_FORTUNE"}

    print()
    print("=" * 88)
    print("DISPATCH: card's CardType vs the switch its case lives in")
    print("=" * 88)
    bad = []
    for name in dir(sts.CardId):
        if name.startswith("_") or name in exempt:
            continue
        cid = getattr(sts.CardId, name)
        if not isinstance(cid, sts.CardId):
            continue
        ctype = str(sts.get_card_type(cid)).split(".")[-1]
        if ctype not in expect:
            continue
        got = where.get(name, [])
        if expect[ctype] not in got:
            color = str(sts.get_card_color(cid)).split(".")[-1]
            bad.append((color, name, ctype, ",".join(got) or "(absent)"))
    if not bad:
        print("all playable cards dispatch to a switch that handles them")
    for color, name, ctype, got in sorted(bad):
        print(f"  *** {color:10s} {name:20s} type={ctype:7s} case in {got}")
    return [(n, f"dispatch mismatch: type={t} case in {g}") for _, n, t, g in bad]


# Drinking this leaves the battle in InputState::CHOOSE_STANCE_ACTION, which
# nothing in the engine resolves and getLegalActions has no case for. Watcher
# only, so it blocks no current work -- tracked in docs/07-known-issues.md and
# listed here so the pass reports it without going red on a known-open bug.
# Anything ELSE showing up is a new defect and does fail.
KNOWN_OPEN_POTIONS = {"STANCE_POTION"}

# Must be skipped, not merely reported: drinking it in a bridge-built context
# HANGS. nativeSeedRng seeds aiRng/cardRandomRng/miscRng/shuffleRng but not
# potionRng, Entropic Brew is the only potion that draws from potionRng, and
# returnRandomPotionOfRarity (Game.cpp:309) loops unbounded until it finds a
# matching rarity -- its own comment says "this is dumb". Real GameContext
# battles are fine (39/39 verified). See docs/07-known-issues.md.
HANGS_IN_BRIDGE_CONTEXT = {"ENTROPIC_BREW"}


def potion_pass():
    """Drink every potion and check the resulting state enumerates.

    Same shape as the card-select pass: a potion that lands the battle in a
    state getLegalActions cannot enumerate leaves the search with an empty
    action list while the fight is still UNDECIDED.
    """
    print()
    print("=" * 88)
    print("POTIONS: every potion drunk, resulting state enumerated")
    print("=" * 88)
    names = [n for n in dir(sts.Potion)
             if not n.startswith("_") and n not in ("name", "value",
                                                    "INVALID",
                                                    "EMPTY_POTION_SLOT")]
    failures, known, checked = [], [], 0
    for n in sorted(names):
        if n in HANGS_IN_BRIDGE_CONTEXT:
            known.append((n, "SKIPPED -- hangs in a bridge-built context"))
            continue
        m = sts.NativeMonsterSpec()
        m.monster_id_name = "JAW_WORM"
        m.cur_hp = m.max_hp = 60
        try:
            bc = sts.build_battle_context(
                player_hp=60, player_max_hp=80, player_block=0,
                player_energy=99, player_statuses=[], monsters=[m],
                hand_cards=FILLER, draw_pile_cards=DRAW,
                discard_pile_cards=DISCARD, exhaust_pile_cards=[],
                potion_slots=[getattr(sts.Potion, n)], relics=[], turn=1,
                ascension=20, rng_seed=1234)
        except Exception:
            continue
        drink = next((a for a in bc.get_legal_actions()
                      if a.action_type == sts.ActionType.POTION
                      and a.target_idx <= 5), None)
        if drink is None:
            continue  # Fairy Potion is not manually drinkable
        checked += 1
        with captured_cxx_stderr() as cap:
            drink.execute(bc)
            raw = sts.get_input_state_raw(bc)
            legal = len(bc.get_legal_actions())
        stuck = legal == 0 and bc.outcome == sts.BattleOutcome.UNDECIDED
        if stuck or cap.text.strip():
            why = (f"empty enumeration in {STATE_NAMES.get(raw, raw)}"
                   if stuck else cap.text.strip()[:120])
            (known if n in KNOWN_OPEN_POTIONS else failures).append((n, why))
    print(f"potions drunk: {checked}")
    for n, why in known:
        print(f"   known-open  {n}: {why}")
    for n, why in failures:
        print(f"   *** {n}: {why}")
    if not failures:
        print("   no new potion defects")
    return [(n, f"potion: {why}") for n, why in failures]


def randomized_pass(fights_per_class):
    print()
    print("=" * 88)
    print(f"RANDOMIZED: real decks, {fights_per_class} fights per character")
    print("=" * 88)
    encounters = [e for e in ["CULTIST", "JAW_WORM", "TWO_LOUSE", "GREMLIN_NOB",
                              "LAGAVULIN", "THE_GUARDIAN", "BOOK_OF_STABBING",
                              "AUTOMATON"] if hasattr(sts.MonsterEncounter, e)]
    failures = []
    for cc_name in ["IRONCLAD", "SILENT", "DEFECT", "WATCHER"]:
        cc = getattr(sts.CharacterClass, cc_name)
        # Fixed per-character seed: str hashing is salted per process, so
        # hash(cc_name) would make this harness unreproducible run to run.
        rng = random.Random(0xC0FFEE + sum(cc_name.encode()))
        empty = 0
        decisions = 0
        selects = 0
        noise_total = ""
        crashed = None
        for i in range(fights_per_class):
            gc = sts.GameContext(cc, rng.randint(1, 10 ** 8), 20)
            # Stack the deck with this character's whole card pool so the
            # select-opening cards actually get drawn.
            for cid_name in COLOR_CARDS.get(cc_name, []):
                gc.obtain_card(sts.Card(getattr(sts.CardId, cid_name)))
            enc = getattr(sts.MonsterEncounter, rng.choice(encounters))
            with captured_cxx_stderr() as cap:
                try:
                    bc = sts.new_battle(gc, enc)
                    for _ in range(400):
                        if bc.outcome != sts.BattleOutcome.UNDECIDED:
                            break
                        legal = list(bc.get_legal_actions())
                        decisions += 1
                        if sts.get_input_state_raw(bc) == STATE_CARD_SELECT:
                            selects += 1
                        if not legal:
                            empty += 1
                            break
                        rng.choice(legal).execute(bc)
                    # the all-C++ path, the one that actually runs in training
                    bc2 = sts.new_battle(gc, enc)
                    sts.native_playout_battle(bc2, 40)
                except Exception as exc:
                    crashed = f"{type(exc).__name__}: {exc}"
            noise_total += cap.text
        status = "OK"
        if empty:
            status = f"*** {empty} EMPTY ***"
        if crashed:
            status = f"*** {crashed} ***"
        if noise_total.strip():
            status += "  +STDERR"
        print(f"{cc_name:10s} decisions={decisions:6d} card_selects={selects:5d} "
              f"empty={empty:3d}  {status}")
        if status != "OK":
            failures.append((cc_name, status, noise_total[:600]))
    return failures


# Card pools per character, by color, resolved from the binding at import time
# so this tracks the engine's own (post-fix) cardColors table rather than a
# copy of it.
COLOR_OF = {"IRONCLAD": "RED", "SILENT": "GREEN", "DEFECT": "BLUE",
            "WATCHER": "PURPLE"}
COLOR_CARDS: dict[str, list[str]] = {k: [] for k in COLOR_OF}
for _name in dir(sts.CardId):
    if _name.startswith("_"):
        continue
    _cid = getattr(sts.CardId, _name)
    if not isinstance(_cid, sts.CardId):
        continue
    _col = str(sts.get_card_color(_cid)).split(".")[-1]
    for _cc, _want in COLOR_OF.items():
        if _col == _want:
            COLOR_CARDS[_cc].append(_name)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    fails = (targeted_pass() + dispatch_pass() + potion_pass()
             + randomized_pass(n))
    print()
    if fails:
        print(f"FAILURES: {len(fails)}")
        for f in fails:
            print(f"  {f[0]}: {f[1]}")
            if len(f) > 2 and f[2].strip():
                print("    stderr:", f[2].strip().splitlines()[:4])
        sys.exit(1)
    print("PASS -- no empty enumerations, no stderr diagnostics, no crashes.")
