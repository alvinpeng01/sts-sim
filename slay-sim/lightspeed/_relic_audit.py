"""Relics: constants against the real game, and relics that do nothing.

Relics were the largest unexamined surface in the engine, and not a cheap one
to leave -- the CMA-ES work measured relics as worth **+0.406 win rate** at the
encounter level, an order of magnitude more than any other combat lever tested
(docs/03-combat-search.md). Before 2026-07-31 not one relic value had been
checked against anything.

Two passes:

  constants -- the game's per-relic int constants against our source near every
               mention of that RelicId. A SCREEN, not an exact comparison: our
               relic values are inline in the effect code rather than tabulated,
               so a miss is a candidate to read, not a verdict. Expect flags
               from turn counters -- our `bc.turn` is 0-indexed while the game's
               constants are 1-based, so Captain's Wheel is `turn == 2` against
               the game's TURN_ACTIVATION=3, and both are right.

  inert     -- relics that sit in a live pool but that NO behaviour code ever
               reads. Those are obtainable, displayed and saved, and do nothing.
               This pass is what found Toy Ornithopter, a common relic in the
               shared pool whose "heal 5 HP whenever you drink a potion" was
               never implemented.

  hooks     -- how many distinct moments the game fires a relic against how
               many places we check it.

usage:  python -m lightspeed._relic_audit
        STS_JAR=<path> overrides jar discovery.
"""
from __future__ import annotations

import collections
import os
import pathlib
import re
import shutil
import subprocess
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2] / "sts_lightspeed"
WORK = pathlib.Path(os.environ.get("TEMP", "/tmp")) / "sts_relic_audit"
JAR_CANDIDATES = [
    r"C:/Program Files (x86)/Steam/steamapps/common/SlayTheSpire/desktop-1.0.jar",
    r"C:/Program Files/Steam/steamapps/common/SlayTheSpire/desktop-1.0.jar",
]
# Layout constants on AbstractRelic, never gameplay values.
SKIP_FIELDS = {"RAW_W", "RAW_H", "W", "H"}
# Declaration sites: listing a RelicId here is not implementing it.
DECL_FILES = {"RelicPools.h", "SaveFileMappings.h"}
DECL_LINE = re.compile(r"relicEnumNames|relicNames|relicStringIds|relicPool"
                       r"|^\s*[A-Z_]+,\s*$")
CONTEXT = 6


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
    for b in pathlib.Path("C:/Program Files/Eclipse Adoptium").glob("jdk*/bin/javap.exe"):
        return str(b)
    raise SystemExit("javap not found; a JDK is required")


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def game_constants(jar, javap):
    z = zipfile.ZipFile(jar)
    cls = [n for n in z.namelist()
           if n.startswith("com/megacrit/cardcrawl/relics/")
           and n.endswith(".class") and "$" not in n]
    z.extractall(WORK, cls)
    paths = [str(WORK / c) for c in cls]
    out = {}
    for i in range(0, len(paths), 40):
        dis = subprocess.run([javap, "-p", "-constants", *paths[i:i + 40]],
                             capture_output=True, text=True).stdout
        for chunk in re.split(r"\nCompiled from ", dis):
            m = re.search(r"class ([\w.]+) ", chunk)
            if not m or m.group(1).endswith("AbstractRelic"):
                continue
            f = [(k, int(v)) for k, v in
                 re.findall(r"static final int (\w+)\s*=\s*(-?\d+);", chunk)
                 if k not in SKIP_FIELDS]
            if f:
                out[m.group(1).split(".")[-1]] = f
    return out


HOOK = re.compile(r"^\s+public [\w.$<>\[\], ]+ ((?:on|at)[A-Z]\w*)\(", re.M)
NON_GAMEPLAY = {"onRightClick", "onPreviewObtain", "onTrigger", "onUnequip"}
DISPLAY = re.compile(r"getUpdatedDescription|makeCopy|setDescription|"
                     r"updateDescription|initializeTips|renderTip|getPrice")
FUNC = re.compile(r"^[A-Za-z_][\w:<>,&*\s]*\b(\w+)::(\w+)\s*\(")


def audit_hooks(jar, javap) -> None:
    """How many distinct moments the game fires a relic vs how many we check it.

    The inert pass finds relics that fire nowhere. This finds the subtler case:
    one our engine checks in a single place that the game fires in three.

    EXPECT FLAGS, AND READ THEM. The two engines are built differently: the game
    gives each relic its own counter and therefore needs explicit atTurnStart /
    onVictory hooks to reset it, while we read shared per-turn counters
    (`attacksPlayedThisTurn`, `skillsPlayedThisTurn`, `cardsPlayedThisTurn`)
    that reset centrally. Kunai, Shuriken, Ornamental Fan and Letter Opener all
    flag for exactly that reason and all four are correct. As of 2026-07-31 all
    19 flags were read and none was a defect. Note the enclosing-function
    attribution is coarse: Gambling Chip has two call sites that both resolve to
    one name here, so a low site count can also just be this tool's bookkeeping.
    """
    paths = sorted(str(p) for p in (WORK / "com/megacrit/cardcrawl/relics").glob("*.class"))
    game: dict[str, set[str]] = {}
    for i in range(0, len(paths), 40):
        dis = subprocess.run([javap, "-p", *paths[i:i + 40]],
                             capture_output=True, text=True).stdout
        for chunk in re.split(r"\nCompiled from ", dis):
            m = re.search(r"class com\.megacrit\.cardcrawl\.relics\.(\w+)", chunk)
            if not m or m.group(1) == "AbstractRelic":
                continue
            hooks = {h for h in HOOK.findall(chunk)
                     if h not in NON_GAMEPLAY and not DISPLAY.search(h)}
            if hooks:
                game[m.group(1)] = hooks

    sites: dict[str, set[str]] = collections.defaultdict(set)
    for p in list(ROOT.glob("src/**/*.cpp")) + list(ROOT.glob("include/**/*.h")):
        if p.name in DECL_FILES or p.name == "Relics.h":
            continue
        cur = f"<{p.name}>"
        for line in p.read_text(errors="replace").splitlines():
            if line.lstrip().startswith("//"):
                continue
            fm = FUNC.match(line)
            if fm:
                cur = f"{fm.group(1)}::{fm.group(2)}"
            for m in re.finditer(r"\bR(?:elicId)?::([A-Z_0-9]+)", line):
                sites[m.group(1)].add(cur)
    by_norm = {norm(k): v for k, v in sites.items()}

    thin = []
    for rn, hooks in sorted(game.items()):
        mine = by_norm.get(norm(rn), set())
        if mine and len(hooks) > len(mine):
            thin.append((rn, sorted(hooks), sorted(s.split("::")[-1] for s in mine)))
    print(f"\nHOOKS      game relics with gameplay hooks: {len(game)}")
    print(f"           fewer call sites than game hooks: {len(thin)} "
          f"(candidates to read, not defects -- see docstring)")
    for rn, hooks, mine in thin:
        print(f"   ?  {rn:20s} game={','.join(hooks)[:38]:39s} ours={','.join(mine)[:44]}")


def main() -> int:
    jar, javap = find_jar(), find_javap()
    WORK.mkdir(parents=True, exist_ok=True)

    srcs = [re.sub(r"//[^\n]*", " ", p.read_text(errors="replace")).splitlines()
            for p in sorted(ROOT.glob("src/**/*.cpp"))]
    mention = collections.defaultdict(list)
    for lines in srcs:
        for i, ln in enumerate(lines):
            for m in re.finditer(r"\bR(?:elicId)?::([A-Z_0-9]+)", ln):
                mention[norm(m.group(1))].append((lines, i))

    game = game_constants(jar, javap)
    matched, flagged = 0, []
    for rn, fields in sorted(game.items()):
        sites = mention.get(norm(rn))
        if not sites:
            continue
        nums = set()
        for lines, i in sites:
            for j in range(max(0, i - CONTEXT), min(len(lines), i + CONTEXT + 1)):
                nums |= {int(x) for x in re.findall(r"(?<![\w.])(\d+)(?![\w.])", lines[j])}
        absent = [(k, v) for k, v in fields if v not in nums and v not in (0, 1)]
        if absent:
            flagged.append((rn, absent))
        else:
            matched += 1
    print(f"CONSTANTS  matched={matched}  flagged={len(flagged)}")
    for rn, absent in flagged:
        print(f"   ?  {rn:22s} {', '.join(f'{k}={v}' for k, v in absent)}")

    pools_txt = (ROOT / "include/constants/RelicPools.h").read_text(errors="replace")
    pools = set()
    for line in pools_txt.splitlines():
        if not line.lstrip().startswith("//"):
            pools |= set(re.findall(r"RelicId::([A-Z_0-9]+)", line))
    impl = set()
    for p in list(ROOT.glob("src/**/*.cpp")) + list(ROOT.glob("include/**/*.h")):
        if p.name in DECL_FILES:
            continue
        for line in p.read_text(errors="replace").splitlines():
            if line.lstrip().startswith("//") or DECL_LINE.search(line):
                continue
            impl |= set(re.findall(r"R(?:elicId)?::([A-Z_0-9]+)", line))
    audit_hooks(jar, javap)

    inert = sorted(pools - impl)
    print(f"\nINERT      obtainable but never read by any code: {len(inert)}")
    for r in inert:
        print(f"   *** {r}")
    return 1 if inert else 0


if __name__ == "__main__":
    raise SystemExit(main())
