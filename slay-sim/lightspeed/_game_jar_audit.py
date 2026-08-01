"""Validate the C++ engine against the real game's own bytecode.

Every other oracle available here is second-hand. `silverbot-reference` is a
fork of the same upstream engine, so a bug inherited from upstream sits in both
and is invisible to a diff; it also implements only 121 of our 345 cards. The
live capture (`sts_raw_states.log`) is genuine ground truth but covers 28 cards
from one session. The game's shipped `desktop-1.0.jar` is neither: it is the
actual source of truth, complete, and offline.

Two things make this tractable:

  * Only A0 and A20 matter for this project, and every ascension gate in the
    game sits at 2, 3, 4, 9, 17, 18 or 19. A0 is therefore below all of them and
    A20 above all of them, so the gate THRESHOLDS cannot matter -- only the base
    value and the top-tier value do. That is why a mislabelled constant like
    TheCollector's `A_2_BLOCK_AMT` (actually applied on an A9 gate) is harmless
    here.
  * Ascension does not scale cards at all, so a card is just (base, upgraded),
    read from `<init>`'s `putfield baseDamage` and `upgrade()`'s
    `upgradeDamage(delta)`.

Found by this harness on 2026-07-31, all confirmed against the game and all
recorded in docs/07-known-issues.md: Lagavulin retaining Metallicize on its
scheduled wake, Champ's Gloat strength tiers, Darkling's Chomp hit count, and
Writhing Mass's flail block.

usage:  python -m lightspeed._game_jar_audit [cards|monsters|all] [filter]
        "cards" covers both base damage and energy costs.
        STS_JAR=<path> overrides jar discovery.
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

JAR_CANDIDATES = [
    r"C:/Program Files (x86)/Steam/steamapps/common/SlayTheSpire/desktop-1.0.jar",
    r"C:/Program Files/Steam/steamapps/common/SlayTheSpire/desktop-1.0.jar",
]
HDR = pathlib.Path(__file__).resolve().parents[2] / \
    "sts_lightspeed/include/constants/Cards.h"
WORK = pathlib.Path(os.environ.get("TEMP", "/tmp")) / "sts_jar_audit"

PUSH = r"(?:bipush|sipush)\s+(-?\d+)|iconst_(\d)"
TIER = re.compile(r"^A_(\d+)_(.+)$")


def find_jar() -> pathlib.Path:
    if os.environ.get("STS_JAR"):
        return pathlib.Path(os.environ["STS_JAR"])
    for c in JAR_CANDIDATES:
        if pathlib.Path(c).exists():
            return pathlib.Path(c)
    raise SystemExit("desktop-1.0.jar not found; set STS_JAR")


def find_javap() -> str:
    exe = shutil.which("javap")
    if exe:
        return exe
    for base in pathlib.Path("C:/Program Files/Eclipse Adoptium").glob("jdk*/bin/javap.exe"):
        return str(base)
    raise SystemExit("javap not found; a JDK is required (a JRE is not enough)")


def consts_before(lines: list[str], idx: int) -> int | None:
    for j in range(idx - 1, max(-1, idx - 6), -1):
        m = re.search(PUSH, lines[j])
        if m:
            return int(m.group(1) if m.group(1) is not None else m.group(2))
    return None


def disassemble(javap: str, paths: list[str]) -> list[str]:
    """javap in batches; returns per-class chunks."""
    chunks: list[str] = []
    for i in range(0, len(paths), 40):
        dis = subprocess.run([javap, "-p", "-c", "-constants", *paths[i:i + 40]],
                             capture_output=True, text=True).stdout
        chunks.extend(re.split(r"\nCompiled from ", dis))
    return chunks


# Cards the game names differently from our CardId. Everything else matches on
# a case/punctuation-insensitive comparison; these six do not, and left
# unmapped they are silently never compared.
CARD_ALIASES = {
    "BULLSEYE": "LockOn",
    "CHARGE_BATTERY": "ConserveBattery",
    "CRIPPLING_CLOUD": "CripplingPoison",
    "DEFEND_PURPLE": "Defend_Watcher",
    "JUDGMENT": "Judgement",          # the game spells it the British way
    "VOID": "VoidCard",
}


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def audit_cards(jar: pathlib.Path, javap: str) -> int:
    z = zipfile.ZipFile(jar)
    classes = [n for n in z.namelist()
               if n.startswith("com/megacrit/cardcrawl/cards/")
               and n.endswith(".class") and "$" not in n
               and "/deprecated/" not in n and n.count("/") == 5]
    z.extractall(WORK, classes)
    game: dict[str, tuple[int, int]] = {}
    for chunk in disassemble(javap, [str(WORK / c) for c in classes]):
        m = re.search(r"class ([\w.]+) ", chunk)
        if not m:
            continue
        lines = chunk.splitlines()
        base = delta = None
        for k, ln in enumerate(lines):
            if "putfield" in ln and "baseDamage:I" in ln and base is None:
                base = consts_before(lines, k)
            if "invokevirtual" in ln and "upgradeDamage" in ln and delta is None:
                delta = consts_before(lines, k)
        if base is not None:
            game[m.group(1).split(".")[-1]] = (base, base + (delta or 0))

    text = HDR.read_text(errors="replace")
    names = [x.strip().strip('"') for x in
             re.search(r"cardEnumStrings\[\]\s*=\s*\{(.*?)\};", text, re.S)
             .group(1).split(",") if x.strip()]
    tbl_src = re.search(r"cardBaseDamage\[2\]\[371\]\s*\{(.*?)\n\s*\};", text, re.S)
    tbl = [[int(x) for x in row.replace("\n", " ").split(",") if x.strip()]
           for row in re.findall(r"\{([^{}]*)\}", tbl_src.group(1))]

    by_norm = {norm(k): v for k, v in game.items()}
    compared = clean = unmatched = 0
    bad = []
    for i, cid in enumerate(names):
        if cid == "INVALID":
            continue
        key = norm(CARD_ALIASES.get(cid, cid))
        hit = by_norm.get(key)
        if hit is None:
            # only a miss if the game has that card at all -- most of ours that
            # miss here simply have no baseDamage (skills, powers, curses)
            unmatched += 1
            continue
        ob, ou = tbl[0][i], tbl[1][i]
        if ob < 0 and ou < 0:
            continue                       # our table marks "no damage" negative
        compared += 1
        if (ob, ou) == hit:
            clean += 1
        else:
            bad.append((cid, hit, (ob, ou)))

    print(f"CARDS  cardBaseDamage vs the game: compared={compared} "
          f"identical={clean} DIFFERING={len(bad)} (no-damage/unmatched={unmatched})")
    for cid, g, o in bad:
        print(f"   *** {cid:24s} game={g} ours={o}")
    return len(bad)


# Differences from the game that are deliberate or inert, so the costs pass can
# report clean and a new difference actually stands out.
#
#  * The game marks unplayable cards -2; we use -3 (Cards.h's only such return).
#    Both are negative, every `energy >= cost` test fails identically, and
#    nothing in the engine compares against the literal. Revisit only if Blue
#    Candle or Medical Kit -- which make curses and status cards playable -- are
#    ever wired up.
#  * The three Wish options are marked unplayable in the game and hit our
#    `default: return 1`. They are applied by chooseWishCard and never enter a
#    deck.
#  * Blood for Blood's upgrade() has two upgradeBaseCost branches because its
#    cost falls as you lose HP; a naive read of the first one is wrong, and ours
#    matches the correct branch.
COST_EXEMPT = {"BECOME_ALMIGHTY", "FAME_AND_FORTUNE", "LIVE_FOREVER",
               "BLOOD_FOR_BLOOD"}


def audit_costs(jar: pathlib.Path, javap: str) -> int:
    """Card energy costs vs the game.

    getEnergyCost ends in `default: return 1`, so a card nobody added silently
    costs 1 -- a wrong entry is indistinguishable from an absent one. That is
    how Safety and Beta stayed wrong, and the switch already carries a comment
    about Silent cards having done the same thing.
    """
    z = zipfile.ZipFile(jar)
    classes = [n for n in z.namelist()
               if n.startswith("com/megacrit/cardcrawl/cards/")
               and n.endswith(".class") and "$" not in n
               and "/deprecated/" not in n and n.count("/") == 5]
    z.extractall(WORK, classes)
    game: dict[str, tuple[int, int]] = {}
    for chunk in disassemble(javap, [str(WORK / c) for c in classes]):
        m = re.search(r"class ([\w.]+) ", chunk)
        if not m:
            continue
        lines = chunk.splitlines()
        cost = upcost = None
        for k, ln in enumerate(lines):
            if 'AbstractCard."<init>"' in ln and cost is None:
                for j in range(k - 1, max(-1, k - 14), -1):
                    if (pm := re.search(PUSH, lines[j])):
                        cost = int(pm.group(1)) if pm.group(1) is not None else \
                            (-1 if pm.group(2) == "m1" else int(pm.group(2)))
                        break
            if "upgradeBaseCost" in ln and upcost is None:
                for j in range(k - 1, max(-1, k - 5), -1):
                    if (pm := re.search(PUSH, lines[j])):
                        upcost = int(pm.group(1)) if pm.group(1) is not None else \
                            (-1 if pm.group(2) == "m1" else int(pm.group(2)))
                        break
        if cost is not None:
            game[m.group(1).split(".")[-1]] = (cost, upcost if upcost is not None else cost)

    text = HDR.read_text(errors="replace")
    names = [x.strip().strip('"') for x in
             re.search(r"cardEnumStrings\[\]\s*=\s*\{(.*?)\};", text, re.S)
             .group(1).split(",") if x.strip()]
    fn = re.search(r"getEnergyCost\(CardId id, bool upgraded\)\s*\{(.*?)\n    \}",
                   text, re.S)
    body = re.sub(r"//[^\n]*", " ", fn.group(1))
    ours: dict[str, tuple[int, int]] = {}
    pending: list[str] = []
    for line in body.splitlines():
        if (cm := re.search(r"case CardId::([A-Z0-9_]+)\s*:", line)):
            pending.append(cm.group(1))
            continue
        if (rm := re.search(r"return\s+(.+?);", line)) and pending:
            expr = rm.group(1).strip()
            if (tern := re.fullmatch(r"upgraded\s*\?\s*(-?\d+)\s*:\s*(-?\d+)", expr)):
                pair = (int(tern.group(2)), int(tern.group(1)))
            elif re.fullmatch(r"-?\d+", expr):
                pair = (int(expr), int(expr))
            else:
                pending = []
                continue
            for n in pending:
                ours[n] = pair
            pending = []
    for n in names:
        ours.setdefault(n, (1, 1))          # `default: return 1`

    by_norm = {norm(k): v for k, v in game.items()}
    compared = clean = 0
    bad = []
    for cid, mine in sorted(ours.items()):
        hit = by_norm.get(norm(CARD_ALIASES.get(cid, cid)))
        if hit is None or cid in COST_EXEMPT:
            continue
        if hit[0] <= -2 and mine[0] <= -2:
            continue                        # both mark unplayable, different sentinel
        compared += 1
        if mine == hit:
            clean += 1
        else:
            bad.append((cid, hit, mine))
    print(f"COSTS  getEnergyCost vs the game: compared={compared} "
          f"identical={clean} DIFFERING={len(bad)}")
    for cid, g, o in bad:
        print(f"   *** {cid:24s} game={g} ours={o}")
    return len(bad)


def audit_monsters(jar: pathlib.Path, javap: str, filt: str) -> None:
    """Print each monster constant resolved at A0 and A20.

    Reported rather than auto-compared: our side writes these as inline
    `asc2 ? x : y` expressions and 3-element tier arrays scattered through
    MonsterSpecific.cpp, so matching them mechanically is far less reliable
    than reading the two columns against the code.
    """
    z = zipfile.ZipFile(jar)
    classes = [n for n in z.namelist()
               if n.startswith("com/megacrit/cardcrawl/monsters/")
               and n.endswith(".class") and "$" not in n and n.count("/") == 5]
    z.extractall(WORK, classes)
    print(f"{'monster':22s} {'constant':24s} {'A0':>6s} {'A20':>6s}  tiers")
    print("-" * 80)
    for c in sorted(classes):
        name = c.split("/")[-1][:-6]
        if filt and filt.upper() not in name.upper():
            continue
        dis = subprocess.run([javap, "-p", "-constants", str(WORK / c)],
                             capture_output=True, text=True).stdout
        fields = dict(re.findall(r"static final int ([A-Z0-9_]+)\s*=\s*(-?\d+);", dis))
        groups: dict[str, dict[int, int]] = {}
        for k, v in fields.items():
            m = TIER.match(k)
            asc, key = (int(m.group(1)), m.group(2)) if m else (0, k)
            groups.setdefault(key, {})[asc] = int(v)
        for key, tiers in sorted(groups.items()):
            lo = tiers.get(0, tiers[min(tiers)])
            hi = tiers[max(tiers)]
            mark = "" if lo == hi else "  <- A0 != A20"
            spread = ",".join(f"A{a}={v}" for a, v in sorted(tiers.items()))
            print(f"{name:22s} {key:24s} {lo:6d} {hi:6d}  {spread}{mark}")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    filt = sys.argv[2] if len(sys.argv) > 2 else ""
    jar, javap = find_jar(), find_javap()
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"jar   : {jar}")
    print(f"javap : {javap}\n")
    rc = 0
    if mode in ("cards", "all"):
        rc |= 1 if audit_cards(jar, javap) else 0
        rc |= 1 if audit_costs(jar, javap) else 0
    if mode in ("monsters", "all"):
        audit_monsters(jar, javap, filt)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
