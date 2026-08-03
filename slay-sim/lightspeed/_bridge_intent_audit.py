"""How often does the live bridge guess the monster's intent correctly?

`sts/bridge/native_recommend.py` never sets `NativeMonsterSpec.move_name`, so
`sts.build_battle_context()` rolls a plausible move from the engine's own monster
AI instead of using the one CommunicationMod is reporting. The queued move drives
predicted incoming damage, which is what decides block-vs-attack — so if the
guess is wrong, the recommendation is planned against a fight that is not
happening.

This replays this project's own capture, rebuilds each state exactly the way the
bridge does (no `move_name`), and compares the rolled move's damage against the
damage the real game was telegraphing in that same state.

Run from slay-sim/:
    PYTHONPATH='../sts_lightspeed/build;.' python -m lightspeed._bridge_intent_audit

Measured 2026-07-30 on `sts_raw_states.log`: 125/1000 = 12.5% match, i.e. about
what coincidence gives. Worst cases predict *zero* incoming damage (Snecko,
Snake Plant), which additionally makes `blockSufficient` trivially true and fires
`defensiveCardSuppressionPenalty` against every defensive card in hand.

When the intent lookup is implemented this becomes its regression test: the match
rate on states with a known intent should go to ~100%.
"""
from __future__ import annotations

import collections
import itertools
import json
import sys

import slaythespire as sts

from sts.bridge.native_recommend import UnmappedMonsterError, _map_monster_id
from sts.bridge.intent_map import lookup_move_name


CAPTURE = "sts_raw_states.log"
# Single-monster states only: with several monsters alive the per-monster
# comparison is still valid but the reported mismatch attribution gets muddier,
# and single-monster states are already plentiful in the capture.
MAX_LINES = 400_000


def main() -> None:
    capture = sys.argv[1] if len(sys.argv) > 1 else CAPTURE
    match = miss = skipped = 0
    mismatches: collections.Counter = collections.Counter()
    unknown_intent = 0

    with open(capture, encoding="utf-8", errors="replace") as handle:
        for line in itertools.islice(handle, MAX_LINES):
            brace = line.find("{")
            if brace < 0:
                continue
            try:
                payload = json.loads(line[brace:])
            except ValueError:
                continue
            state = payload.get("game_state") or {}
            combat = state.get("combat_state")
            if not combat:
                continue
            alive = [m for m in combat.get("monsters", [])
                     if not m.get("is_gone") and m.get("current_hp", 0) > 0]
            if len(alive) != 1:
                continue
            monster = alive[0]
            if monster.get("intent") in (None, "UNKNOWN", "DEBUG"):
                unknown_intent += 1
                continue
            true_damage = monster.get("move_adjusted_damage", -1)
            true_hits = monster.get("move_hits", 1)
            if true_damage is None or true_damage < 0:
                continue
            try:
                name = _map_monster_id(monster.get("id", ""))
            except UnmappedMonsterError:
                skipped += 1
                continue

            spec = sts.NativeMonsterSpec()
            spec.monster_id_name = name
            spec.cur_hp = monster.get("current_hp", 1)
            spec.max_hp = monster.get("max_hp", 1)
            spec.block = monster.get("block", 0)
            spec.statuses = []
            # Same lookup the live bridge now applies. With the mapping in
            # place this file is the regression test its docstring promised:
            # the pre-mapping baseline was 12.5%.
            mapped = lookup_move_name(name, monster.get("move_id"))
            if mapped is not None:
                spec.move_name = mapped
            player = combat.get("player", {})
            try:
                bc = sts.build_battle_context(
                    player_hp=player.get("current_hp", 1),
                    player_max_hp=player.get("max_hp", 1),
                    player_block=player.get("block", 0),
                    player_energy=player.get("energy", 3),
                    player_statuses=[], monsters=[spec],
                    hand_cards=[], draw_pile_cards=[], discard_pile_cards=[],
                    exhaust_pile_cards=[], potion_slots=[], relics=[],
                    turn=combat.get("turn", 1),
                    ascension=state.get("ascension_level", 20),
                    rng_seed=12345)
                rolled_damage, rolled_hits = bc.get_monster_move_damage(0)
            except Exception:
                skipped += 1
                continue

            if rolled_damage == true_damage and rolled_hits == true_hits:
                match += 1
            else:
                miss += 1
                mismatches[(name, f"{true_damage}x{true_hits}",
                            f"{rolled_damage}x{rolled_hits}")] += 1

    total = match + miss
    print()
    print("=" * 76)
    print(f"single-monster states compared: {total}"
          f"   (unmapped/failed: {skipped}, UNKNOWN intent: {unknown_intent})")
    print("=" * 76)
    if not total:
        print("  nothing comparable found")
        return
    print(f"  guess MATCHED the telegraphed move : {match}/{total} = "
          f"{100 * match / total:.1f}%")
    print(f"  guess was WRONG                    : {miss}/{total} = "
          f"{100 * miss / total:.1f}%")
    print()
    print("  most common mismatches (true dmg x hits -> guessed):")
    for (name, true_desc, rolled_desc), count in mismatches.most_common(10):
        print(f"    {name:18s} true {true_desc:<8} -> guessed {rolled_desc:<8} "
              f"({count} states)")


if __name__ == "__main__":
    main()
