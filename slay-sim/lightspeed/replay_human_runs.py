"""Replay Baalorlord's actual runs in our engine, recovering his ROUTING decisions.

The importer (`import_baalorlord_runs.py`) emits only card rewards and boss relics
-- 526 demonstrations -- because the archive is not simulator-complete. But the
archive carries a base-35 StS **seed** per run, and our engine generates the same
map from it, so his recorded per-floor room sequence can be walked back into the
map choices that produced it.

That matters because the 2026-07-31 layer swap put the ENTIRE 15.71-floor gap in
the overworld policy (see docs/03-combat-search.md), and routing is the decision
class v31 is worst at. Routing is also the one class the archive cannot express
directly: it records the room he ARRIVED in, never the node he clicked.

Two things make naive matching fail, both learned by doing it wrong first:

- **`?` nodes.** A map EVENT symbol resolves at entry into a shop, treasure,
  monster fight or event, and the archive records the RESOLUTION. A run showing
  "shop" on floor 2 did not find a shop node on floor 2 -- shops do not generate
  that early -- it took a `?`. So EVENT must be accepted for several archive
  rooms.
- **Ambiguous rows.** When two reachable nodes share a room type, picking the
  first one silently takes the wrong branch and every later floor mismatches.
  Runs that "diverge at floor 3" are usually mis-stepped at floor 2. This does a
  DFS over whole-sequence-consistent paths instead; the map is <=7 wide and 15
  rows per act, so exhaustive search is free.

Emits one row per routing decision: the reconstructed pre-decision state, the
options offered, and which one he took.

    python -m lightspeed.replay_human_runs --out runs/human_routing.jsonl
"""
from __future__ import annotations

import os

import argparse
import collections
import json

import slaythespire as sts

from .import_baalorlord_runs import ReconstructedRun, resolve_card
from .whole_run_env import WholeRunEnv

# The run archive is external to this repository, so unlike every other path
# here it cannot be derived from __file__. Set STS_BAALOR_ARCHIVE or pass
# --archive. Empty rather than a fallback: a stale default would either read
# someone else's file or fail with a path that means nothing to whoever is
# running it.
ARCHIVE = os.environ.get("STS_BAALOR_ARCHIVE", "")

SEED_CHARS = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"

# Archive room name -> map symbols that can produce it. A `?` (EVENT) node
# resolves into shop/treasure/monster/event at entry, so those archive rooms
# accept EVENT as well as their own symbol.
ROOM_SYMBOLS = {
    "fight_normal": {"MONSTER", "EVENT"},
    "fight_elite": {"ELITE"},
    "event": {"EVENT"},
    "event_fight": {"EVENT"},
    "rest": {"REST"},
    "shop": {"SHOP", "EVENT"},
    "treasure_chest": {"TREASURE", "EVENT"},
    "boss_node": {"BOSS"},
}


def seed_to_long(text: str) -> int:
    """Base-35 StS seed string -> the unsigned 64-bit value GameContext takes."""
    n = 0
    for ch in text.strip().upper().replace("O", "0"):
        n = n * 35 + SEED_CHARS.index(ch)
    return n & 0xFFFFFFFFFFFFFFFF


def room_name(value: int) -> str:
    return str(sts.Room(int(value))).split(".")[-1]


def map_rows(gc) -> dict[int, dict[int, str]]:
    """{y: {x: ROOM_NAME}} for the whole act's map."""
    nn = sts.getNNRepresentation(gc)
    rows: dict[int, dict[int, str]] = collections.defaultdict(dict)
    for x, y, room in zip(nn.map.xs, nn.map.ys, nn.map.room_types):
        rows[int(y)][int(x)] = room_name(room)
    return rows


def map_graph(gc):
    """({y: {x: ROOM}}, {(y, x): [next_x, ...]}) for the current act's map.

    `path_xs` is per node: three slots holding the destination x on the next row,
    -1 where there is no edge. A node at row y is floor y+1.
    """
    nn = sts.getNNRepresentation(gc)
    rooms: dict[int, dict[int, str]] = collections.defaultdict(dict)
    edges: dict[tuple[int, int], list[int]] = {}
    for x, y, room, paths in zip(nn.map.xs, nn.map.ys, nn.map.room_types,
                                 nn.map.path_xs):
        x, y = int(x), int(y)
        rooms[y][x] = room_name(room)
        edges[(y, x)] = sorted({int(v) for v in paths if int(v) >= 0})
    return rooms, edges


def solve_path(rooms, edges, start_row, start_options, wanted):
    """Longest room-type-consistent path from `start_options` on `start_row`.

    Returns the list of x's, one per row. Greedy matching is not enough: when two
    reachable nodes share a room type, committing to the first silently takes the
    wrong branch and every later floor mismatches -- runs that appear to diverge
    at floor 3 are usually mis-stepped at floor 2. The map is <=7 wide over 15
    rows, so exhaustive DFS costs nothing and removes the whole failure mode.
    """
    best: list[int] = []

    def walk(row: int, x: int, path: list[int]):
        nonlocal best
        if len(path) > len(best):
            best = list(path)
        depth = row - start_row + 1
        if depth >= len(wanted):
            return
        allowed = ROOM_SYMBOLS.get(wanted[depth], set())
        for nxt in edges.get((row, x), []):
            if rooms.get(row + 1, {}).get(nxt) in allowed:
                path.append(nxt)
                walk(row + 1, nxt, path)
                path.pop()

    if not wanted:
        return best
    first_allowed = ROOM_SYMBOLS.get(wanted[0], set())
    for x in start_options:
        if rooms.get(start_row, {}).get(x) in first_allowed:
            walk(start_row, x, [x])
    return best


def states_by_floor(floors):
    """His deck/relics/potions/HP as of ENTERING each floor.

    Without this the replay follows his map choices but takes `actions[0]` for
    every card reward, campfire and shop, so it accumulates an arbitrary deck,
    loses A20 fights his deck would have won, and the run ends. Measured: only 23
    of 100 replays reached act 2 and the extraction was act-1 heavy purely
    because of this.
    """
    state = ReconstructedRun()
    out = {}
    previous_hp = None
    for row in floors:
        floor = int(row["floor"])
        out[floor] = (list(state.deck), list(state.relics), list(state.potions),
                      previous_hp, int(row["hp_max"]))
        state.apply_floor(row)
        if row.get("hp_current") is not None:
            previous_hp = int(row["hp_current"])
    return out


def sync_to_human(gc, snapshot):
    """Overwrite our GameContext's deck/relics/potions/HP with his."""
    deck, relics, potions, hp, max_hp = snapshot
    if not deck:
        return
    for _ in range(len(gc.deck)):
        gc.remove_card(0)
    for card_id, upgrades in deck:
        card = sts.Card(sts.CardId(card_id))
        for _ in range(upgrades):
            card.upgrade()
        gc.obtain_card(card)
    for relic_id in relics:
        try:
            gc.obtain_relic(sts.RelicId(relic_id))
        except Exception:  # noqa: BLE001 - already held, or not applicable
            pass
    gc.max_hp = int(max_hp)
    if hp:
        gc.cur_hp = max(1, min(int(max_hp), int(hp)))


def encode_choice(encoder, gc, chosen, decision_type):
    """One-hot training row for `chosen`, or None if it is not a live option.

    Indexes against env.legal_actions(), NOT the raw action list: that method
    drops provably-immediately-losing actions, so the orderings differ and a raw
    index would silently mislabel.
    """
    if encoder is None:
        return None
    encoder.gc = gc
    safe = encoder.legal_actions()
    bits = [a.bits for a in safe]
    if chosen.bits not in bits or len(safe) < 2:
        return None
    observation = encoder.observation()
    target = [0.0] * len(safe)
    target[bits.index(chosen.bits)] = 1.0
    return {"observation": {k: v for k, v in observation.items()
                            if k != "action_text"},
            "target_probabilities": target,
            "decision_type": decision_type}


def upgradeable_indices(gc):
    """Deck positions offered by the smith card-select, in the engine's order."""
    out = []
    for i, card in enumerate(gc.deck):
        if card.upgraded:
            continue
        kind = sts.get_card_type(sts.CardId(int(card.id)))
        if kind in (sts.CardType.CURSE, sts.CardType.STATUS):
            continue
        out.append((i, int(card.id)))
    return out


def replay(run_rows, sims: int, ascension: int, max_steps: int = 2000,
           encoder: WholeRunEnv | None = None):
    """Walk one archived run, emitting a row per map decision.

    His deck, relics and HP are pinned at every floor, so combat is fought with
    the cards he actually held and the recorded state is HIS rather than an
    artefact of our arbitrary reward picks. A combat loss is also un-done rather
    than ending the walk: the object is to recover which map node he clicked, and
    our combat quality is irrelevant to that.
    """
    floors = sorted(run_rows, key=lambda r: int(r["floor"]))
    by_floor = {int(r["floor"]): r for r in floors}
    snapshots = states_by_floor(floors)
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD,
                         seed_to_long(floors[0]["seed"]), ascension)
    synced_floor = -1

    out, mismatches = [], 0
    pending_smith = None
    for _ in range(max_steps):
        if gc.outcome != sts.GameOutcome.UNDECIDED:
            break
        if gc.screen_state == sts.ScreenState.BATTLE:
            # Heal to full BEFORE the fight, not after. He survived every one of
            # these; our combat is not what is being recovered, and a loss here
            # is unrecoverable -- clearing `outcome` leaves the screen stuck on a
            # dead player and the walk spins until max_steps. That single bug
            # froze most replays at the act-1 boss.
            gc.cur_hp = gc.max_hp
            sts.native_playout_current_battle(gc, sims)
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                break
            recorded = by_floor.get(gc.floor_num)
            if recorded and recorded.get("hp_current") is not None:
                # Back to HIS HP, so the next decision is recorded from his
                # trajectory rather than from a full-healed artefact.
                gc.cur_hp = max(1, min(gc.max_hp, int(recorded["hp_current"])))
            continue

        if gc.floor_num != synced_floor and gc.floor_num in snapshots:
            sync_to_human(gc, snapshots[gc.floor_num])
            synced_floor = gc.floor_num

        actions = sts.GameAction.getAllActionsInState(gc)
        if not actions:
            break

        if gc.screen_state == sts.ScreenState.REWARDS:
            # Drafting is where the two agents diverge most: v31's top picks are
            # Perfected Strike / Twin Strike / Clash, nine of its top fifteen
            # absent from his top twenty-five, while he drafts an exhaust engine
            # (Feel No Pain 97, Dark Embrace 83, Second Wind, Burning Pact).
            # Deck SIZE matches at floor 10 (16.6 vs 16.2) -- the difference is
            # which cards, not how many. Unlike routing, card choice is largely
            # capability-independent, so it does not carry the failure that made
            # the routing clone collapse by 15.8 floors.
            card_actions = [a for a in actions
                            if a.rewards_action_type == sts.RewardsActionType.CARD]
            if not card_actions:
                actions[0].execute(gc)
                continue
            offered = [int(c.id) for c in gc.get_card_reward()]
            record = by_floor.get(gc.floor_num) or {}
            wanted = [resolve_card(n) for n in (record.get("card_picked") or [])]
            wanted_ids = {r[0] for r in wanted if r}
            picked_action = next(
                (a for a in card_actions
                 if a.idx2 < len(offered) and offered[a.idx2] in wanted_ids), None)
            if picked_action is None and any(
                    "singing bowl" in str(n).lower()
                    for n in (record.get("card_picked") or [])):
                # The archive renders Singing Bowl's +2 max-HP choice as a
                # "picked card"; the engine encodes it as idx2 == 5.
                picked_action = next((a for a in actions if a.idx2 == 5), None)
            if picked_action is None:
                # Card rewards are RNG-dependent and our replay's RNG has already
                # diverged from his (we fight different fights), so the cards on
                # offer here are usually NOT the ones he chose from. Only label a
                # SKIP when he genuinely declined -- i.e. he recorded no pick at
                # all. Falling through to SKIP whenever his pick is merely absent
                # teaches the model to skip everything: it labelled 71 of ~110
                # rewards as SKIP before this check existed.
                declined = not (record.get("card_picked") or [])
                if declined and (record.get("skipped") or []):
                    picked_action = next(
                        (a for a in actions
                         if a.rewards_action_type == sts.RewardsActionType.SKIP),
                        None)
            if picked_action is None:
                # His choice is not representable here; take a card so the deck
                # keeps growing, but emit no training row.
                card_actions[0].execute(gc)
                continue
            encoded = encode_choice(encoder, gc, picked_action, "rewards")
            if encoded:
                took = (sts.CardId(offered[picked_action.idx2]).name
                        if picked_action.rewards_action_type == sts.RewardsActionType.CARD
                        and picked_action.idx2 < len(offered) else "SKIP")
                out.append({"encoded": encoded, "run_id": floors[0]["run_id"],
                            "floor": gc.floor_num, "act": gc.act,
                            "hp": gc.cur_hp, "max_hp": gc.max_hp,
                            "kind": "card_reward", "chosen": took,
                            "n_options": len(actions),
                            "options": [sts.CardId(c).name for c in offered] + ["SKIP"],
                            "chosen_idx": actions.index(picked_action)})
            picked_action.execute(gc)
            continue

        if gc.screen_state == sts.ScreenState.REST_ROOM:
            # docs/07: v31 resolves every campfire to REST (argmax collapse on a
            # 0.41/0.35 marginal) while he smiths 83% of 784 decisions. This is
            # the single most inverted behaviour measured, so capture it.
            recorded = (by_floor.get(gc.floor_num) or {}).get("campfire_action") or {}
            wanted_idx = {"rest": 0, "smith": 1}.get(recorded.get("action"))
            if wanted_idx is None and "key" in str(recorded.get("target", "")).lower():
                wanted_idx = 2
            picked = next((a for a in actions if a.idx1 == wanted_idx), None)
            if picked is None:
                actions[0].execute(gc)
                continue
            encoded = encode_choice(encoder, gc, picked, "rest")
            if encoded:
                out.append({"encoded": encoded, "run_id": floors[0]["run_id"],
                            "floor": gc.floor_num, "act": gc.act,
                            "hp": gc.cur_hp, "max_hp": gc.max_hp,
                            "kind": "campfire",
                            "chosen": recorded.get("action"),
                            "n_options": len(actions),
                            "options": [str(a.idx1) for a in actions],
                            "chosen_idx": [a.idx1 for a in actions].index(wanted_idx)})
            pending_smith = recorded.get("target") if wanted_idx == 1 else None
            picked.execute(gc)
            continue

        if gc.screen_state == sts.ScreenState.CARD_SELECT and pending_smith:
            resolved = resolve_card(pending_smith)
            options = upgradeable_indices(gc)
            target_id = resolved[0] if resolved else None
            slot = next((k for k, (_, cid) in enumerate(options)
                         if cid == target_id), None)
            picked = next((a for a in actions if a.idx1 == slot), None) if slot is not None else None
            if picked is None:
                pending_smith = None
                actions[0].execute(gc)
                continue
            encoded = encode_choice(encoder, gc, picked, "card_select")
            if encoded:
                out.append({"encoded": encoded, "run_id": floors[0]["run_id"],
                            "floor": gc.floor_num, "act": gc.act,
                            "hp": gc.cur_hp, "max_hp": gc.max_hp,
                            "kind": "smith_target", "chosen": pending_smith,
                            "n_options": len(actions),
                            "options": [str(a.idx1) for a in actions],
                            "chosen_idx": slot})
            pending_smith = None
            picked.execute(gc)
            continue

        if gc.screen_state == sts.ScreenState.MAP_SCREEN:
            target_floor = gc.floor_num + 1
            if target_floor not in by_floor:
                break
            rooms, edges = map_graph(gc)
            # Row within THIS act's map, not the absolute floor: every act
            # regenerates a map starting at row 0 while floors keep counting, so
            # `target_floor - 1` is only correct in act 1. cur_map_node_y is -1
            # before the first step of an act, which makes this 0 there.
            row = gc.cur_map_node_y + 1
            # His remaining rooms for this act, so the search can look past
            # ambiguous rows instead of committing to the first match.
            wanted, f, r = [], target_floor, row
            while f in by_floor and r in rooms:
                wanted.append(by_floor[f]["map_node"] or "")
                f += 1
                r += 1
            options = [(a, rooms.get(row, {}).get(a.idx1, "MISSING")) for a in actions]
            path = solve_path(rooms, edges, row, [a.idx1 for a in actions], wanted)
            if not path:
                # The boss row is absent from the NN map representation, so the
                # last pre-boss screen has nothing to match against. That is not
                # a decision (there is one way onward) and must not end the
                # replay -- doing so capped every run inside act 1.
                if len(actions) == 1:
                    actions[0].execute(gc)
                    continue
                mismatches += 1
                break
            chosen = next(a for a in actions if a.idx1 == path[0])
            encoded = None
            if encoder is not None:
                # The trainer consumes `observation` + `target_probabilities`, and
                # its policy loss is cross-entropy against a soft target -- a
                # one-hot human choice is a valid special case. Index against
                # env.legal_actions(), NOT the raw action list: that method drops
                # provably-immediately-losing actions, so the two orderings differ
                # and a raw index would silently mislabel.
                encoder.gc = gc
                safe = encoder.legal_actions()
                bits = [a.bits for a in safe]
                if chosen.bits in bits:
                    observation = encoder.observation()
                    target = [0.0] * len(safe)
                    target[bits.index(chosen.bits)] = 1.0
                    encoded = {"observation": {k: v for k, v in observation.items()
                                               if k != "action_text"},
                               "target_probabilities": target,
                               "decision_type": "map"}
            out.append({
                "encoded": encoded,
                "run_id": floors[0]["run_id"],
                "floor": target_floor,
                "act": gc.act,
                "hp": gc.cur_hp,
                "max_hp": gc.max_hp,
                "options": [name for _, name in options],
                "chosen": rooms.get(row, {}).get(path[0], "MISSING"),
                "chosen_idx": [a.idx1 for a in actions].index(path[0]),
                "n_options": len(options),
                "archive_room": by_floor[target_floor]["map_node"],
                "lookahead_matched": len(path),
            })
            chosen.execute(gc)
            continue

        actions[0].execute(gc)
    return out, mismatches, gc.floor_num


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default=ARCHIVE,
                        help="run archive .jsonl; defaults to $STS_BAALOR_ARCHIVE")
    parser.add_argument("--runs", type=int, default=0, help="0 = all")
    parser.add_argument("--start", type=int, default=0,
                        help="skip this many runs. The engine SEGFAULTS on at "
                             "least one archived run -- a hard crash, not "
                             "catchable -- so a slice lets that run be isolated "
                             "and stepped over instead of losing the extraction")
    parser.add_argument("--sims", type=int, default=30,
                        help="combat budget during replay; low is fine, HP is pinned")
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-out", default=None,
                        help="also write a torch payload in the format "
                             "train_whole_run_v27.py loads")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.archive, encoding="utf-8")
            if line.strip()]
    runs = collections.defaultdict(list)
    for row in rows:
        runs[row["run_id"]].append(row)
    encoder = WholeRunEnv() if args.train_out else None
    ordered = sorted(runs)[args.start:]
    run_ids = ordered[: args.runs] if args.runs else ordered
    print(f"replaying {len(run_ids)} runs at A{args.ascension} "
          f"(from index {args.start})", flush=True)

    # Write per run and flush. Buffering to the end meant one segfaulting run
    # destroyed the whole extraction silently: the process died, the shell
    # pipeline still reported exit 0 (that was `tail`'s status, not python's),
    # and no output file appeared at all.
    all_rows, depths, stalled = [], [], 0
    with open(args.out, "w", encoding="utf-8") as handle:
        for index, run_id in enumerate(run_ids):
            decisions, mismatches, _ = replay(runs[run_id], args.sims,
                                              args.ascension, encoder=encoder)
            for row in decisions:
                # `encoded` holds the v27 observation, which contains numpy
                # arrays and is not JSON-serialisable; it travels in the torch
                # payload instead.
                handle.write(json.dumps({k: v for k, v in row.items()
                                         if k != "encoded"}) + "\n")
            handle.flush()
            all_rows.extend(decisions)
            depths.append(len(decisions))
            stalled += mismatches
            if (index + 1) % 10 == 0:
                print(f"  {index+1}/{len(run_ids)}  {len(all_rows)} decisions "
                      f"(last {run_id})", flush=True)

    multi = [r for r in all_rows if r["n_options"] > 1]
    print(f"\n{len(all_rows)} routing decisions, {len(multi)} with >1 option")
    print(f"mean per run: {sum(depths)/max(1,len(depths)):.1f}   "
          f"median depth: {sorted(depths)[len(depths)//2] if depths else 0}")
    print(f"runs stopped by a room mismatch: {stalled}/{len(run_ids)}")
    picked = collections.Counter(r["chosen"] for r in multi)
    print(f"chosen room types (multi-option only): {dict(picked)}")
    print(f"wrote {args.out}")
    if args.train_out:
        import torch
        trainable = [r["encoded"] | {"act": r["act"], "floor": r["floor"]}
                     for r in all_rows if r.get("encoded")]
        torch.save({"rows": trainable,
                    "metadata": {"source": "baalorlord human replay",
                                 "ascension": args.ascension,
                                 "runs": len(run_ids),
                                 "decision_types": ["map"]}}, args.train_out)
        print(f"wrote {len(trainable)} trainable rows -> {args.train_out}")


if __name__ == "__main__":
    main()
