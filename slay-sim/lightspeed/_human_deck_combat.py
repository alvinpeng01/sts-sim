"""Fight a human's actual decks with our search, to separate piloting from deckbuilding.

Baalorlord's A20 Heart runs take 13.9-16.4 damage on Act 1 elites; our search on
`env.py`'s calibrated Act 1 elite state takes 32-36. That 2.2x gap conflates two
things: he may simply have a better deck by the time he fights an elite, or our
combat may be worse at piloting the same cards.

This isolates them. For each fight in the dataset it replays the run up to that
floor with the importer's own reconstruction, rebuilds that exact deck, relic
set, HP and encounter in our engine, and plays the fight with the native search.
His damage on that floor is recorded in the archive text, so every fight carries
its own paired human result.

A gap that survives here is piloting. A gap that disappears is deckbuilding.

Measured 2026-07-31 on 40 Act 1 elites: with his exact deck we still die in 9 of
40 fights he won all of, and take 32.4 damage against his 15.9 on the ones we
win. Raising the search from 300 to 1500 sims moved that by -0.97 +/- 1.23 HP and
left the same 9 fights lost, so the binding constraint is the leaf evaluation,
not search depth.

This is also the fastest combat benchmark available: ~570 paired fights against a
ground-truth human, scoring HP directly, where the alternative is reading combat
quality out of full-run floor counts at +/-0.5 floors of noise.

    python -m lightspeed._human_deck_combat --dataset <archive.jsonl> --sims 300
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics

import slaythespire as sts

from .import_baalorlord_runs import ReconstructedRun
from .search_config import DEFAULT_SEARCH_CONFIG_PATH, ensure_search_config


# Leading count words: the archive writes "3 Sentries" where the enum says
# THREE_SENTRIES.
COUNT_WORDS = {"2": "TWO", "3": "THREE", "4": "FOUR"}

# Names the normaliser cannot reach, almost all event fights whose archive text
# is a sentence rather than an encounter name.
ENCOUNTER_ALIASES = {
    "SPHERE_AND_2_SHAPES": "SPHERE_AND_TWO_SHAPES",
    "FOUGHT_BANDITS": "MASKED_BANDITS_EVENT",
    "FOUGHT_THE_MUSHROOM_LAIR": "MUSHROOMS_EVENT",
    "FOUGHT_THE_MYSTERIOUS_SPHERE": "MYSTERIOUS_SPHERE_EVENT",
    "FOUGHT_THE_COLOSSEUM_SLAVERS": "COLOSSEUM_EVENT_SLAVERS",
    "FOUGHT_THE_COLOSSEUM_NOBS": "COLOSSEUM_EVENT_NOBS",
}

_ENUM_NAMES = {n for n in dir(sts.MonsterEncounter) if n.isupper()}


def resolve_encounter(display_name: str):
    """Map an archive display name onto a MonsterEncounter, or None."""
    tokens = re.sub(r"[^A-Za-z0-9 ]", " ", display_name).split()
    # Archive prose prefixes: "- Fought Bandits", "Fought 2 Orb Walkers".
    while tokens and tokens[0].upper() in {"FOUGHT", "THE"} and len(tokens) > 1:
        tokens = tokens[1:]
    if tokens and tokens[0] in COUNT_WORDS:
        tokens[0] = COUNT_WORDS[tokens[0]]
    candidate = "_".join(token.upper() for token in tokens)
    candidate = ENCOUNTER_ALIASES.get(candidate, candidate)
    for variant in (candidate, f"THE_{candidate}", candidate.replace("THE_", ""),
                    f"{candidate}S", candidate.rstrip("S")):
        if variant in _ENUM_NAMES:
            return getattr(sts.MonsterEncounter, variant), variant
    return None, candidate


def parse_fight(row: dict) -> tuple[str, int, int] | None:
    """Pull (enemy, damage, turns) out of the archive's rendered detail text."""
    lines = row.get("detail_lines") or []
    enemy = damage = turns = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        match = re.match(r"^(\d+) damage$", stripped)
        if match:
            damage = int(match.group(1))
            if index:
                enemy = lines[index - 1].strip()
        match = re.match(r"^(\d+) turns?$", stripped)
        if match:
            turns = int(match.group(1))
    if enemy is None or damage is None:
        return None
    return enemy, damage, turns or 0


def build_battle(deck, relics, cur_hp, max_hp, encounter, ascension, act=1,
                 potions=()):
    """A GameContext carrying the human's exact deck/relics/HP, then a battle."""
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 1, ascension)
    # `gc.deck` returns a *copy* of the underlying vector, so clearing or
    # appending to it silently does nothing -- an early version of this harness
    # did exactly that and measured 40 fights with the default starter deck.
    # obtain_card/remove_card are the real mutators, and they preserve upgrades.
    for _ in range(len(gc.deck)):
        gc.remove_card(0)
    for card_id, upgrades in deck:
        card = sts.Card(sts.CardId(card_id))
        for _ in range(upgrades):
            card.upgrade()
        gc.obtain_card(card)
    if len(gc.deck) != len(deck):
        raise RuntimeError(
            f"deck rebuild mismatch: wanted {len(deck)}, got {len(gc.deck)}")
    missing = []
    for relic_id in relics:
        try:
            gc.obtain_relic(sts.RelicId(relic_id))
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            missing.append(f"{relic_id}:{type(error).__name__}")
    for potion_id in potions:
        try:
            gc.obtain_potion(sts.Potion(potion_id))
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            missing.append(f"potion {potion_id}:{type(error).__name__}")
    gc.act = int(act)
    gc.max_hp = int(max_hp)
    gc.cur_hp = int(cur_hp)
    return sts.new_battle(gc, encounter), missing


def play(bc, sims: int, seed: int):
    """Play the fight out with the native search; return (damage, outcome)."""
    start = bc.player_hp
    for step in range(600):
        if bc.outcome != sts.BattleOutcome.UNDECIDED:
            break
        if not bc.get_legal_actions():
            break
        action, _ = sts.run_mcts_search(bc, sims, None, (seed << 20) ^ step)
        action.execute(bc)
    return start - bc.player_hp, bc.outcome


def collect(dataset: str, acts, rooms):
    """Every fight we can pair: reconstructed pre-fight state + his damage."""
    rows = [json.loads(line) for line in open(dataset, encoding="utf-8")
            if line.strip()]
    runs = collections.defaultdict(list)
    for row in rows:
        runs[row["run_id"]].append(row)

    records, unresolved = [], collections.Counter()
    for run_id, floors in runs.items():
        floors.sort(key=lambda r: int(r["floor"]))
        state = ReconstructedRun()
        previous_hp = None
        for row in floors:
            fight = parse_fight(row)
            act = int(row.get("act") or 0)
            room = row.get("map_node") or ""
            wanted = (not acts or act in acts) and (not rooms or room in rooms)
            if fight and wanted and row.get("hp_current") is not None:
                encounter, name = resolve_encounter(fight[0])
                if encounter is None:
                    unresolved[fight[0]] += 1
                elif previous_hp is not None and previous_hp > 0:
                    # Pre-fight HP is the previous floor's recorded HP. That is
                    # exact and needs no assumption about which healing relic he
                    # was carrying, unlike back-computing from damage taken.
                    records.append({
                        "run_id": run_id, "floor": int(row["floor"]),
                        "act": act, "room": room,
                        "enemy": fight[0], "encounter": name,
                        "human_damage": fight[1], "human_turns": fight[2],
                        "deck": list(state.deck), "relics": list(state.relics),
                        "potions": list(state.potions),
                        "cur_hp": int(previous_hp), "max_hp": int(row["hp_max"]),
                    })
            # Rebuild *before* applying this floor: rewards on a fight floor are
            # obtained after the fight, so applying first would hand our engine
            # the relic it just won.
            state.apply_floor(row)
            if row.get("hp_current") is not None:
                previous_hp = int(row["hp_current"])
    return records, unresolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sims", type=int, default=300)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--acts", default="", help="e.g. 1,2")
    parser.add_argument("--rooms", default="",
                        help="e.g. fight_elite,fight_normal,boss_node")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    ensure_search_config(DEFAULT_SEARCH_CONFIG_PATH)

    acts = {int(a) for a in args.acts.split(",") if a.strip()}
    rooms = {r.strip() for r in args.rooms.split(",") if r.strip()}
    records, unresolved = collect(args.dataset, acts, rooms)
    if args.limit:
        records = records[: args.limit]
    print(f"paired fights to replay: {len(records)}")
    if unresolved:
        print(f"unresolved encounters: {dict(unresolved)}")

    skipped = 0
    for index, rec in enumerate(records):
        encounter = getattr(sts.MonsterEncounter, rec["encounter"])
        try:
            bc, missing = build_battle(rec["deck"], rec["relics"], rec["cur_hp"],
                                       rec["max_hp"], encounter, args.ascension,
                                       rec["act"], rec.get("potions", ()))
        except Exception as error:  # noqa: BLE001 - counted and reported
            skipped += 1
            rec["error"] = f"{type(error).__name__}: {error}"
            continue
        damage, outcome = play(bc, args.sims, index)
        rec["died"] = outcome != sts.BattleOutcome.PLAYER_VICTORY
        # A death is not a damage number; keep it separate rather than encoding
        # it as max_hp, which silently contaminates every mean.
        rec["our_damage"] = None if rec["died"] else damage
        rec["missing_relics"] = missing

    report(records, skipped)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            for rec in records:
                rec.pop("deck", None)
                handle.write(json.dumps(rec) + "\n")


def report(records, skipped: int) -> None:
    played = [r for r in records if "died" in r]
    if not played:
        print("nothing played")
        return

    def block(title, key):
        groups = collections.defaultdict(list)
        for rec in played:
            groups[key(rec)].append(rec)
        print(f"\n{title:22s}{'n':>4s}{'died':>6s}{'human':>8s}"
              f"{'ours':>8s}{'ratio':>7s}")
        for name in sorted(groups, key=str):
            sub = groups[name]
            won = [r for r in sub if not r["died"]]
            deaths = len(sub) - len(won)
            if not won:
                print(f"{str(name):22s}{len(sub):4d}{deaths:6d}"
                      f"{'-':>8s}{'-':>8s}{'-':>7s}")
                continue
            human = statistics.mean(r["human_damage"] for r in won)
            ours = statistics.mean(r["our_damage"] for r in won)
            print(f"{str(name):22s}{len(sub):4d}{deaths:6d}{human:8.1f}"
                  f"{ours:8.1f}{ours / max(1e-9, human):7.2f}")

    block("by act", lambda r: f"act {r['act']}")
    block("by room", lambda r: r["room"])

    won = [r for r in played if not r["died"]]
    deaths = len(played) - len(won)
    human = statistics.mean(r["human_damage"] for r in won)
    ours = statistics.mean(r["our_damage"] for r in won)
    print(f"\n{'TOTAL':22s}{len(played):4d}{deaths:6d}{human:8.1f}"
          f"{ours:8.1f}{ours / max(1e-9, human):7.2f}")
    print(f"\ndeaths                {deaths}/{len(played)} "
          f"({100 * deaths / len(played):.0f}%) -- he survived all of them")
    print(f"we matched or beat him {sum(1 for r in won if r['our_damage'] <= r['human_damage'])}"
          f"/{len(won)} of the fights we won")
    print(f"mean deck size        {statistics.mean(len(r['deck']) for r in records if 'deck' in r):.1f}"
          if any("deck" in r for r in records) else "")
    if skipped:
        print(f"skipped (build error) {skipped}")
    bad = collections.Counter(m for r in played for m in r.get("missing_relics", []))
    if bad:
        print(f"relics that failed to apply: {dict(bad)}")


if __name__ == "__main__":
    main()
