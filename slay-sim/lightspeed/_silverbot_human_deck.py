"""Play the human-deck benchmark with Silver Automaton instead of our search.

Our combat pays 1.29x a top human's HP across 2841 reconstructed fights
(`_human_deck_combat.py`). That number alone cannot say whether it is good: the
human is an upper reference, not an achievable one, and 1.29x might be near the
ceiling for any bot or might be leaving a lot on the table. Silver Automaton is
the strongest available reference on the other side, so running it on the SAME
fights turns one number into a bracket.

Must run in its own process. Both engines' Python modules are named
`slaythespire`, so only one can be imported per interpreter -- this file puts
`silverbot-reference/build` on the path FIRST and must never be imported from a
process that has already loaded ours.

The benchmark file is portable between the two engines without translation:
their CardId, RelicId, Potion and MonsterEncounter enums are integer-identical
to ours across all 371/181/44/64 members (verified 2026-07-31), which is
unsurprising since both are forks of the same upstream.

    python lightspeed/_silverbot_human_deck.py --split test --sims 100
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys

SILVERBOT_BUILD = r"C:\Users\Alvin\grok\sts-project\silverbot-reference\build"
BENCHMARK_PATH = r"C:\Users\Alvin\grok\sts-project\slay-sim\runs\human_fight_benchmark_100.json"

sys.path.insert(0, SILVERBOT_BUILD)
import slaythespire as sts  # noqa: E402 - must follow the path insert

if "silverbot" not in (getattr(sts, "__file__", "") or "").lower():
    raise RuntimeError(
        f"imported the wrong engine: {getattr(sts, '__file__', '?')} -- this "
        "script must run in a process that has not loaded sts_lightspeed")


def off_class_cards(rec: dict) -> list[str]:
    """Cards in this deck that silverbot has no implementation for.

    Precomputed into the benchmark by our engine, because silverbot's module
    exposes `CardColor` but no `get_card_color` to classify with.
    """
    if "off_class_cards" not in rec:
        raise KeyError(
            "benchmark lacks 'off_class_cards'; regenerate it with the "
            "annotation step in _human_deck_combat, or silverbot will abort "
            "the process on the first Prismatic Shard deck")
    return rec["off_class_cards"]


def play_fight(rec: dict, sims: int, ascension: int, seed: int):
    """Rebuild the human's state in silverbot's engine and let its agent play.

    Returns (hp_paid, died). A death pays every point of HP that was left, the
    same convention `_human_deck_combat.py` uses, so the two are comparable.
    """
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension)
    gc.clear_deck()
    for card_id, upgrades in rec["deck"]:
        gc.obtain_card(sts.Card(sts.CardId(card_id), int(upgrades)))
    for relic_id in rec["relics"]:
        try:
            gc.obtain_relic(sts.RelicId(relic_id))
        except Exception:  # noqa: BLE001 - counted by the caller via deck size
            pass
    for potion_id in rec.get("potions", ()):
        try:
            gc.obtain_potion(sts.Potion(potion_id))
        except Exception:  # noqa: BLE001
            pass
    gc.act = int(rec["act"])
    gc.max_hp = int(rec["max_hp"])
    gc.cur_hp = int(rec["cur_hp"])

    agent = sts.Agent()
    # Their agent narrates every action to stdout by default, which at 2841
    # fights is both unreadable and slow.
    agent.verbosity_level = 0
    agent.log_battle_outcomes = False
    agent.record_actions = False
    agent.simulation_count_base = sims
    # Their own comparison harness sets this; leaving it at 1 would give their
    # boss fights a smaller budget than their agent is designed around.
    agent.boss_simulation_multiplier = 2
    try:
        agent.playout_battle(gc, getattr(sts.MonsterEncounter, rec["encounter"]))
    except RuntimeError:
        # "regain control lambda was null" -- a post-battle reward callback that
        # only the full run-loop wires up. It fires ONLY on a win, and cur_hp is
        # already updated before it does, so the outcome is still readable. This
        # is documented in their own comparison_tests/silverbot_encounter_test.py.
        pass
    died = gc.cur_hp <= 0
    return (int(rec["cur_hp"]) if died else int(rec["cur_hp"]) - gc.cur_hp), died


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--write-playable", default=None,
                        help="dump the (run_id, floor) list actually played, so "
                             "our side can be scored on the identical subset")
    args = parser.parse_args()

    with open(BENCHMARK_PATH, encoding="utf-8") as handle:
        records = json.load(handle)
    fights = [r for r in records
              if args.split == "all" or r["split"] == args.split]
    # Silverbot ABORTS the process (assert, not exception) on an off-class card:
    # "attempted to use unimplemented card: Prepared". Those decks are correct --
    # 11 of the 100 runs carry Prismatic Shard, which legitimately offers Silent,
    # Defect and Watcher cards to an Ironclad, and our engine implements all four
    # characters so it plays them. Silverbot cannot, so they have to be filtered
    # BEFORE the call rather than caught after it.
    playable = [r for r in fights if not off_class_cards(r)]
    skipped = len(fights) - len(playable)
    fights = playable
    if args.limit:
        fights = fights[: args.limit]
    print(f"silverbot on {len(fights)} {args.split} fights at {args.sims} sims"
          + (f"  ({skipped} skipped: off-class cards from Prismatic Shard)"
             if skipped else ""))
    if args.write_playable:
        with open(args.write_playable, "w", encoding="utf-8") as handle:
            json.dump([[r["run_id"], r["floor"]] for r in fights], handle)

    results = []
    for index, rec in enumerate(fights):
        try:
            hp_paid, died = play_fight(rec, args.sims, args.ascension, index)
        except Exception as error:  # noqa: BLE001 - reported, not hidden
            print(f"  skip {rec['run_id']}@{rec['floor']} "
                  f"{rec['encounter']}: {type(error).__name__}: {error}")
            continue
        results.append({**{k: v for k, v in rec.items() if k != "deck"},
                        "hp_paid": hp_paid, "died": died})

    by_room = collections.defaultdict(list)
    for r in results:
        by_room[r["room"]].append(r)
    print(f"\n{'':16s}{'n':>5s}{'died':>7s}{'human':>8s}{'silver':>8s}{'ratio':>7s}")
    for room in sorted(by_room):
        sub = by_room[room]
        won = [r for r in sub if not r["died"]]
        if not won:
            continue
        human = statistics.mean(r["human_damage"] for r in won)
        silver = statistics.mean(r["hp_paid"] for r in won)
        print(f"{room:16s}{len(sub):5d}{len(sub)-len(won):7d}{human:8.1f}"
              f"{silver:8.1f}{silver/max(1e-9,human):7.2f}")
    won = [r for r in results if not r["died"]]
    human = statistics.mean(r["human_damage"] for r in won)
    silver = statistics.mean(r["hp_paid"] for r in won)
    print(f"{'TOTAL':16s}{len(results):5d}{len(results)-len(won):7d}"
          f"{human:8.1f}{silver:8.1f}{silver/human:7.2f}")
    objective = statistics.mean(r["human_damage"] - r["hp_paid"] for r in results)
    print(f"\nobjective (human_damage - hp_paid): {objective:+.3f}"
          "   <- directly comparable to ours")
    print(f"deaths: {len(results)-len(won)}/{len(results)} "
          f"({(len(results)-len(won))/len(results):.0%})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            for r in results:
                handle.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
