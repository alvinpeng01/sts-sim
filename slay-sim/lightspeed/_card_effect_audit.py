"""Card BEHAVIOUR vs the real game: which effects a card queues.

Every other check in this repo compares NUMBERS. This compares what a card
actually does, which is a different question -- the Armaments leak had every
constant correct and still cost 3.2 mean floors.

A StS card's use() queues actions (`addToBot(new DamageAction(...))`,
`addToBot(new ApplyPowerAction(..., new VulnerablePower(...)))`); our case
bodies queue a parallel sequence of `Actions::` helpers and
`Buff/Debuff<MS::X>` calls. Canonicalising both into one vocabulary makes them
comparable, which catches a missing effect, an extra effect, the wrong power,
or the wrong target scope -- none of which a constant check can see.

Three things the bytecode side must get right, each found by getting it wrong:

  * javap names a class twice per site (the `new` opcode and the matching
    `invokespecial`), so only `new` may be counted.
  * Many cards open use() with `if (Settings.isDebug) { ... }` -- Bash's debug
    branch calls DamageAllEnemiesAction. Dead code in a real game; excluded by
    following the `ifeq` target.
  * Cosmetic actions (VFX/SFX/Wait) are noise and are dropped.

RESULTS, 2026-07-31, after mapping all five colours:

    red 66/75   green 62/72   blue 54/73   purple 62/72   colorless 29/39

EVERY residual was then read individually against both implementations -- all
58, not a sample. No behavioural defect was found in any of them. Every difference examined resolved as one of:

  * naming -- our `XAction` wrapper against the game's differently-named action
    for the same effect (Reaper, Sword Boomerang, Warcry, Second Wind, ...);
  * a deliberate implementation choice -- Berserk increments
    `player.energyPerTurn` rather than modelling a BerserkPower (stacks
    identically); Consume uses `IncreaseOrbSlots(-1)` where the game has a
    separate DecreaseMaxOrbAction; Expertise draws TO a target rather than a
    flat count;
  * THE TOOL'S CEILING -- our engine expresses many effects as inline lambdas
    or direct field writes, which a regex over `Actions::` calls simply cannot
    see. Chaos channels its orbs inside a lambda, All for One filters the
    discard pile inside one, Genetic Algorithm writes specialData directly. All
    three read as "ours is missing an effect" and none of them is.

That ceiling is why the residual counts do not shrink to zero and should not be
chased with more aliases. Going further means executing both engines and
comparing resulting state -- real differential testing -- not more parsing.

A wrong alias silently hides a real difference, so add entries one at a time
with the card open. Two were caught doing exactly that during this work:
"SECONDWIND": "EXHAUSTMANY" made Second Wind look like it opened a card-select
when it does not, and "WATCHER": "PWR:MANTRA" masked an extractor bug where the
game's `powers/watcher/` subpackage was being captured instead of the class
name, collapsing every Watcher power to one symbol.

usage: python -m lightspeed._card_effect_audit [red|green|blue|purple|colorless]
"""
from __future__ import annotations

import collections
import os
import pathlib
import re
import subprocess
import sys
import zipfile

JAR = os.environ.get("STS_JAR",
    r"C:/Program Files (x86)/Steam/steamapps/common/SlayTheSpire/desktop-1.0.jar")
JAVAP = r"C:/Program Files/Eclipse Adoptium/jdk-21.0.7.6-hotspot/bin/javap"
OUT = pathlib.Path("cardcls")
CPP = pathlib.Path(__file__).resolve().parents[2] / "sts_lightspeed/src/combat/BattleContext.cpp"

COSMETIC = {"VFXAction", "SFXAction", "WaitAction", "AbstractGameAction",
            "ApplyPowerAction", "DamageInfo",
            "PressEndTurnButtonAction", "NotStanceCheckAction",
            "ShakeScreenAction"}
GAME_MAP = {"DamageAction": "DMG", "DamageAllEnemiesAction": "DMG_ALL",
            "VampireDamageAllEnemiesAction": "DMG_ALL",
            "GainBlockAction": "BLK", "DrawCardAction": "DRAW",
            "GainEnergyAction": "NRG", "LoseHPAction": "LOSEHP",
            "MakeTempCardInHandAction": "CARD_HAND",
            "MakeTempCardInDrawPileAction": "CARD_DRAW",
            "MakeTempCardInDiscardAction": "CARD_DISCARD",
            "ExhaustAction": "EXHAUST_SELECT", "HealAction": "HEAL"}
OUR_MAP = {"AttackEnemy": "DMG", "DamageAllEnemy": "DMG_ALL",
           "AttackAllEnemy": "DMG_ALL", "GainBlock": "BLK",
           "DrawCards": "DRAW", "GainEnergy": "NRG", "PlayerLoseHp": "LOSEHP",
           "MakeTempCardInHand": "CARD_HAND",
           "MakeTempCardInDrawPile": "CARD_DRAW",
           "ShuffleTempCardIntoDrawPile": "CARD_DRAW",
           "MakeTempCardInDiscard": "CARD_DISCARD",
           "ChooseExhaustOne": "EXHAUST_SELECT", "HealPlayer": "HEAL"}

ALIAS = {
    "ENTRENCH": "DOUBLEYOURBLOCK",       # ours wraps it; the game names the effect
    "HEADBUTT": "DISCARDPILETOTOPOFDECK",
    "INFERNALBLADE": "CARD_HAND",        # ours wraps MakeTempCardInHand
    "SECONDWIND": "BLOCKPERNONATTACK",
}


# One canonical name per effect, applied to BOTH sides after extraction, so the
# two engines' differing helper names collapse together without either side
# being privileged. Every entry below was adjudicated by reading the card in
# both implementations -- three looked like real differences and were not:
#   Expertise      draws TO a target in ours (with an empirically-verified +1
#                  for the in-flight copy), not a flat DrawCards.
#   Consume        our IncreaseOrbSlots(-1) is the signed equivalent of the
#                  game's DecreaseMaxOrbAction -- not a sign error.
#   Sneaky Strike  the "if you discarded" condition is in C++ rather than in a
#                  GainEnergyIfDiscardAction.
# Add new entries one at a time with the card open; a wrong alias silently
# hides a real difference, which is how Second Wind briefly looked like it
# opened a card-select when it does not.
CANON = {
    # orbs (Defect)
    "CHANNELORB": "CHANNEL", "TEMPEST": "CHANNEL",
    "INCREASEORBSLOTS": "ORBSLOTS", "DECREASEMAXORB": "ORBSLOTS",
    "MULTICAST": "EVOKEORB",
    # discard family (Silent)
    "CHOOSEDISCARDCARDS": "DISCARD", "DISCARDRANDOMCARDINHAND": "DISCARD",
    "DISCARDNONATTACKCARDSINHAND": "UNLOAD",
    "GAINENERGYIFDISCARD": "NRG", "AGGREGATEENERGY": "NRG",
    "BOUNCINGFLASK": "POISONRANDOMENEMY",
    # cards created into hand / draw pile
    "BLADEFURY": "CARD_HAND", "DISTRACTION": "CARD_HAND",
    "WHITENOISE": "CARD_HAND",
    "PUTRANDOMCARDSINDRAWPILE": "CARD_DRAW",
    # damage variants that are just a named action in the game
    "BANE": "DMG", "BARRAGE": "DMG", "SUNDER": "DMG", "FLECHETTE": "DMG",
    "DAMAGEPERATTACKPLAYED": "DMG",
    # block / draw variants
    "REINFORCEDBODY": "BLK", "EXPERTISE": "DRAW", "COMPILEDRIVER": "DRAW",
    # powers named differently on the two sides
    "PWR:INTANGIBLEPLAYER": "PWR:INTANGIBLE", "PWR:CHOKE": "PWR:CHOKED",
    "PWR:RETAINCARD": "PWR:WELLLAIDPLANS",
    "FORTHEEYES": "PWR:WEAK",
    # Defect: orbs and power naming
    "INCREASEMAXORB": "ORBSLOTS", "DOUBLEENERGY": "NRG",
    "PWR:HEATSINK": "PWR:HEATSINKS", "PWR:HELLO": "PWR:HELLOWORLD",
    "PWR:DRAW": "PWR:MACHINELEARNING", "PWR:ECHO": "PWR:ECHOFORM",
    "PWR:REPAIR": "PWR:SELFREPAIR", "PWR:AMPLIFY": "PWR:DUPLICATION",
    "COLLECT": "PWR:COLLECT",
    # Watcher: powers the game names for their effect, we name for their card
    "PWR:BLOCKRETURN": "PWR:TALKTOTHEHAND",
    "PWR:FREEATTACK": "PWR:FREEATTACKPOWER",
    "PWR:VIGOR": "PWR:WREATHOFFLAME",
    "PWR:ENDTURNDEATH": "PWR:BLASPHEMER",
    "PWR:ENERGYDOWN": "PWR:FASTING",
    "MEDITATE": "BETTERDISCARDPILETOHAND",
    # named actions that are just damage / block / draw / energy / a debuff
    "WALLOP": "DMG", "LESSONLEARNED": "DMG", "HALT": "BLK",
    "SANCTITY": "DRAW", "FOLLOWUP": "NRG",
    "HEADSTOMP": "PWR:WEAK", "CRUSHJOINTS": "PWR:VULNERABLE",
    # misc 1:1 renames
    "OBTAINPOTION": "ALCHEMIZE", "APPLYBULLETTIME": "BULLETTIME",
    "REDO": "RECURSION",
}


def canonical(eff):
    return [CANON.get(e, e) for e in eff]


INSTR = re.compile(r"^\s*(\d+):\s+(\S+)(.*)$")
NEWCLS = re.compile(r"// class com/megacrit/cardcrawl/(?:actions|powers)(?:/[a-z]+)*/(\w+)")


def canon_power(sym: str) -> str:
    return "PWR:" + re.sub(r"[^A-Z0-9]", "", sym[:-5].upper())


def game_effects(colour: str) -> dict[str, list[str]]:
    z = zipfile.ZipFile(JAR)
    classes = [n for n in z.namelist()
               if n.startswith(f"com/megacrit/cardcrawl/cards/{colour}/")
               and n.endswith(".class") and "$" not in n]
    z.extractall(OUT, classes)
    out: dict[str, list[str]] = {}
    for c in classes:
        name = c.split("/")[-1][:-6]
        dis = subprocess.run([JAVAP, "-p", "-c", str(OUT / c)],
                             capture_output=True, text=True).stdout
        m = re.search(r"public void use\(.*?\n(.*?)(?=\n  [a-zA-Z]|\Z)", dis, re.S)
        if not m:
            continue
        skip_until = -1
        eff: list[str] = []
        prev_debug = False
        for line in m.group(1).splitlines():
            im = INSTR.match(line)
            if not im:
                continue
            off, op, rest = int(im.group(1)), im.group(2), im.group(3)
            if off >= skip_until:
                skip_until = -1
            if "Settings.isDebug" in rest:
                prev_debug = True
                continue
            if prev_debug and op.startswith("if"):
                tgt = re.search(r"(\d+)", rest)
                if tgt:
                    skip_until = int(tgt.group(1))
                prev_debug = False
                continue
            prev_debug = False
            if skip_until > 0 and off < skip_until:
                continue                     # inside the debug-only branch
            if op != "new":
                continue
            cm = NEWCLS.search(rest)
            if not cm:
                continue
            sym = cm.group(1)
            if sym in COSMETIC:
                continue
            if sym.endswith("Power"):
                if sym == "AbstractPower":
                    continue
                eff.append(canon_power(sym))
            else:
                eff.append(GAME_MAP.get(sym, sym.replace("Action", "").upper()))
        out[name] = canonical(eff)
    return out


def our_effects() -> dict[str, list[str]]:
    text = re.sub(r"//[^\n]*", " ",
                  re.sub(r"/\*.*?\*/", " ", CPP.read_text(errors="replace"), flags=re.S))
    lines = text.splitlines()
    defs = [(i, mm.group(1)) for i, ln in enumerate(lines)
            if (mm := re.search(r"^\w[\w:<>,&* ]*\bBattleContext::(\w+)\s*\(", ln))]
    bodies: dict[str, str] = {}
    for pos, (start, fn) in enumerate(defs):
        if fn not in {"useAttackCard", "useSkillCard", "usePowerCard"}:
            continue
        end = defs[pos + 1][0] if pos + 1 < len(defs) else len(lines)
        parts = re.split(r"(case\s+CardId::[A-Z0-9_]+\s*:)", "\n".join(lines[start:end]))
        pending: list[str] = []
        for chunk in parts:
            mm = re.fullmatch(r"case\s+CardId::([A-Z0-9_]+)\s*:", chunk.strip())
            if mm:
                pending.append(mm.group(1))
                continue
            if pending and chunk.strip():
                for n in pending:
                    bodies[n] = chunk
                pending = []
    out: dict[str, list[str]] = {}
    tok = re.compile(
        r"(?:Actions::)?(?:Buff|Debuff)(?:Player|Enemy|AllEnemy)<(?:MS|PS)::(\w+)>"
        r"|\b(?:buff|debuff|setHasStatus)<(?:MS|PS)::(\w+)>"
        r"|Actions::(\w+)")
    for cid, body in bodies.items():
        eff: list[str] = []
        for tm in tok.finditer(body):
            p = tm.group(1) or tm.group(2)
            if p:
                eff.append("PWR:" + re.sub(r"[^A-Z0-9]", "", p.upper()))
                continue
            sym = tm.group(3)
            if sym in ("SetState", "RollMove", "NoOpRollMove"):
                continue
            sym = OUR_MAP.get(sym, re.sub(r"ACTION$", "", sym.upper()))
            eff.append(ALIAS.get(sym, sym))
        out[cid] = canonical(eff)
    return out


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


colour = sys.argv[1] if len(sys.argv) > 1 else "red"
game, ours = game_effects(colour), our_effects()
by_norm = {norm(k): (k, v) for k, v in game.items()}
same, rows = 0, []
for cid, oe in sorted(ours.items()):
    hit = by_norm.get(norm(cid))
    if hit is None:
        continue
    if collections.Counter(hit[1]) == collections.Counter(oe):
        same += 1
    else:
        rows.append((cid, hit[1], oe))
print(f"{colour}: compared={same + len(rows)}  identical={same}  DIFFER={len(rows)}\n")
for cid, ge, oe in rows:
    print(f"{cid}\n    game: {ge}\n    ours: {oe}")
