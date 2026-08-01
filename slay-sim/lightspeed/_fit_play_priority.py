"""Fit a per-card play-priority ranking from our own search, to replace a borrowed one.

`silverCardPlayRank` (`bindings/slaythespire.cpp:722`) is Silver Automaton's
hand-curated 133-card play ordering, re-encoded as a 372-entry lookup. It is
load-bearing -- removing it measures **-1.20 +/- 0.49 HP (t = -2.45)** on 500
paired train fights -- so it cannot simply be deleted. This fits the same slot
from this project's own data instead.

The slot is narrow and stays unchanged: `nativeScoreAction` reads
`silverCardPlayRank[cardId]`, converts a rank in 1..133 to
`(134 - rank) / 133`, and scales it by `silver_card_play_prior_weight`. Rank 0
means "no opinion" and contributes nothing. This script emits a table with
exactly those semantics, so nothing in the engine changes but the numbers.

**The metric is a conditional logit over the cards actually available**, not a
raw pick rate. A raw rate was tried first and is wrong: it measures how
RESTRICTIVE a card's legality is rather than how good it is. Clash came out 4th of
133 under it, because Clash is only legal when the hand is all attacks and is
usually the right play once that holds; every card landed between 0.20 and 0.30,
including Strike at 0.240 and Defend at 0.189, which is just the reciprocal of how
many options were in hand. Conditioning each observation on the set the card beat
removes that.

    utility[card], P(choose k) = softmax over the cards available at that decision

with the most common card pinned at zero for identifiability. Same shape as
`_routing_audit.conditional_logit`, which fits room utilities the same way.

**Collected with the silver prior switched OFF.** Otherwise the search's picks
are partly downstream of the very ranking being replaced, and the fit would
launder their table back into ours rather than measuring our own preferences.

Availability is counted once per DECISION per card, not once per legal action: a
targeted attack appears once per living monster, and counting each would make
every attack look less preferred in multi-monster fights purely by arithmetic.

    python -m lightspeed._fit_play_priority --out runs/play_priority.json
    python -m lightspeed._fit_play_priority --data runs/play_priority.json --emit-cpp
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from collections import defaultdict

import slaythespire as sts

from ._human_deck_combat import build_battle
from .search_config import DEFAULT_SEARCH_CONFIG_PATH, ensure_search_config

BENCHMARK_PATH = r"C:\Users\Alvin\grok\sts-project\slay-sim\runs\human_fight_benchmark_100.json"

# The engine's formula is (134 - rank) / 133, so ranks outside 1..133 would give
# a negative or >1 prior. Keeping the same width keeps this a drop-in.
TABLE_RANKS = 133
CARD_ID_COUNT = 372
# Below this many observations a pick rate is noise. Silver's table is a curated
# 133 out of ~370 cards, so leaving the long tail at "no opinion" matches both the
# formula's width and the spirit of the original.
MIN_OBSERVATIONS = 25


def collect(split: str, sims: int, ascension: int, limit: int) -> dict:
    ensure_search_config(DEFAULT_SEARCH_CONFIG_PATH)
    # The two that must move together -- a negative boss weight means "inherit the
    # general one", so zeroing only the general weight would leave bosses using
    # the borrowed ranking.
    sts.set_search_params({
        "silver_card_play_prior_weight": 0.0,
        "boss_silver_card_play_prior_weight": -1.0,
        "policy_net_weight": 0.0,
    })

    with open(BENCHMARK_PATH, encoding="utf-8") as handle:
        records = json.load(handle)
    fights = [r for r in records if split == "all" or r["split"] == split]
    if limit:
        fights = fights[:limit]
    print(f"collecting from {len(fights)} {split} fights at {sims} sims, "
          f"silver prior DISABLED")

    available: dict[int, int] = defaultdict(int)
    chosen: dict[int, int] = defaultdict(int)
    groups: list[dict] = []
    start = time.time()
    for index, record in enumerate(fights):
        try:
            battle, _ = build_battle(
                record["deck"], record["relics"], record["cur_hp"],
                record["max_hp"],
                getattr(sts.MonsterEncounter, record["encounter"]),
                ascension, record["act"], record.get("potions", ()))
        except Exception as error:  # noqa: BLE001 - reported, not hidden
            print(f"  skip {record['run_id']}@{record['floor']}: {error}")
            continue
        fight_key = zlib.crc32(f"{record['run_id']}:{record['floor']}".encode())
        for step in range(600):
            if battle.outcome != sts.BattleOutcome.UNDECIDED:
                break
            actions = battle.get_legal_actions()
            if not actions:
                break
            action, _ = sts.run_mcts_search(battle, sims, None,
                                            (fight_key << 20) ^ step)
            playable = sorted({int(battle.hand[a.source_idx].id)
                               for a in actions
                               if a.action_type == sts.ActionType.CARD})
            # Only decisions where a card was actually played teach anything about
            # relative card priority; ending the turn is a different decision.
            if len(playable) >= 2 and action.action_type == sts.ActionType.CARD:
                picked = int(battle.hand[action.source_idx].id)
                groups.append({"cards": playable, "chosen": playable.index(picked)})
                for card_id in playable:
                    available[card_id] += 1
                chosen[picked] += 1
            action.execute(battle)
        if (index + 1) % 400 == 0:
            print(f"  {index+1}/{len(fights)} fights, {len(groups)} decisions, "
                  f"{time.time()-start:.0f}s")

    return {
        "available": {str(k): v for k, v in available.items()},
        "chosen": {str(k): v for k, v in chosen.items()},
        "groups": groups,
        "meta": {"split": split, "sims": sims, "fights": len(fights)},
    }


def rank_table(payload) -> list[int]:
    """Conditional-logit utilities per card, converted to ranks 1..133.

    Each decision contributes one softmax over exactly the cards that were
    playable at that moment, so a card is scored against what it actually beat.
    The most-seen card is pinned at utility 0 for identifiability -- the scale is
    arbitrary anyway, since only the ordering reaches the engine.
    """
    import torch

    available = {int(k): v for k, v in payload["available"].items()}
    groups = payload["groups"]
    eligible = {c for c, n in available.items() if n >= MIN_OBSERVATIONS}
    reference = max(eligible, key=lambda c: available[c])
    free = sorted(eligible - {reference})
    slot = {c: i for i, c in enumerate(free)}
    print(f"{len(eligible)} cards cleared {MIN_OBSERVATIONS} observations; "
          f"{len(groups)} decisions; reference card id {reference}")

    # Rows whose whole comparison set is below the observation floor teach nothing
    # about the ranked cards and would only add noise through the reference.
    rows = [g for g in groups if any(c in eligible for c in g["cards"])]
    flat, offsets, picks, total = [], [], [], 0
    for g in rows:
        cards = g["cards"]
        offsets.append(total)
        flat.extend(slot.get(c, -1) for c in cards)
        picks.append(total + g["chosen"])
        total += len(cards)

    idx = torch.tensor(flat, dtype=torch.long)
    is_ref = idx < 0
    idx = idx.clamp_min(0)
    group_of = torch.repeat_interleave(
        torch.arange(len(rows)), torch.tensor([len(g["cards"]) for g in rows]))
    chosen_flat = torch.tensor(picks, dtype=torch.long)

    utility = torch.zeros(len(free), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([utility], max_iter=300,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        scores = torch.where(is_ref, torch.zeros(1, dtype=torch.float64),
                             utility[idx])
        top = torch.full((len(rows),), -1e30, dtype=torch.float64).scatter_reduce(
            0, group_of, scores, reduce="amax")
        shifted = (scores - top[group_of]).exp()
        denominator = torch.zeros(len(rows), dtype=torch.float64).index_add(
            0, group_of, shifted)
        # Ridge keeps rarely-seen cards from running to +/-inf when they happen to
        # win or lose every comparison they appear in.
        loss = ((top + denominator.log()) - scores[chosen_flat]).mean()             + 1e-3 * utility.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        scored = [(float(utility[slot[c]]), c) for c in free] + [(0.0, reference)]
    scored.sort(reverse=True)

    table = [0] * CARD_ID_COUNT
    for position, (_, card_id) in enumerate(scored[:TABLE_RANKS]):
        table[card_id] = position + 1
    return table


def emit_cpp(table: list[int]) -> str:
    body = []
    for start in range(0, CARD_ID_COUNT, 16):
        body.append("        " + " ".join(
            f"{v}," for v in table[start:start + 16]))
    return (
        "    // Per-card play priority, rank 1 = played most readily. Fitted from this\n"
        "    // project's own search by lightspeed/_fit_play_priority.py -- the rate at which\n"
        "    // the search chooses to play a card when it is available -- with the silver\n"
        "    // prior disabled during collection so the fit could not launder the ranking it\n"
        "    // replaces back into itself. Rank 0 means no opinion and contributes nothing.\n"
        "    // Consumed as (134 - rank) / 133, so the width is fixed at 133.\n"
        f"    constexpr std::array<unsigned char, {CARD_ID_COUNT}> cardPlayRank = {{\n"
        + "\n".join(body) + "\n    };\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="collect and write counts here")
    parser.add_argument("--data", help="load counts from here instead of collecting")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--emit-cpp", action="store_true")
    args = parser.parse_args()

    if args.data:
        with open(args.data, encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = collect(args.split, args.sims, args.ascension, args.limit)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            print(f"wrote {args.out}")

    table = rank_table(payload)
    if args.emit_cpp:
        print(emit_cpp(table))
    else:
        named = sorted(((v, k) for k, v in enumerate(table) if v))[:15]
        print("\ntop 15 by fitted play priority:")
        for rank, card_id in named:
            print(f"  {rank:3d}  {sts.get_card_name(sts.CardId(card_id))}"
                  if hasattr(sts, "get_card_name") else f"  {rank:3d}  CardId {card_id}")


if __name__ == "__main__":
    main()
