"""Does action IDENTITY carry the signal the 8 action features cannot?

Distillation into the rollout policy was refuted on 2026-07-31 -- worse at
matched wall clock AND at matched simulations -- and `docs/03-combat-search.md`
names the suspect rather than the cause. `nativeActionFeatures` is
`[is_attack, is_skill, is_power, is_other, target_hp_missing, target_block,
is_aoe_multi, card_pick_rate_weight]`: it never says WHICH card is being played.
Everything expressible in those 8 numbers is already hardcoded in
`nativeScoreAction`, which is why top-1 accuracy saturated immediately at
0.334 / 0.340 / 0.340 for hidden 4 / 8 / 16 against a 0.202 random baseline --
capacity was never the limit.

There is a second hole the docs do not name, found while writing this: every
non-CARD action returns the SAME vector, `{0,0,0,1,0,0,0,0}`. END_TURN, all 33
playable potions and every card-select option are one indistinguishable symbol
to the net. `potionScoreWeight` is tuned to 29.7, so the heuristic cares about
potions a great deal and the net cannot see them at all.

Closing either hole is an engine change (a wider feature vector, a loader schema
change for the embedding table, and a re-tune of everything downstream). This
script is the kill-gate that decides whether to start: `Action.source_idx`,
`bc.hand[i].id`, `.upgraded`, `.cost_for_turn` and `bc.potions` are all bound
already, so identity can be recovered in PYTHON at collection time and the
accuracy question answered offline, for the price of one re-collection.

**Pre-registered gate, fixed before the first run.** Ship the engine change only
if `+embed` beats `baseline` on val top-1 by more than the seed-to-seed spread
of both arms combined. Sweeping arms and reporting the best is exactly the
failure that manufactured a ~2 HP "Power horizon improvement" out of noise
(`docs/03-combat-search.md`), so every arm is trained at three seeds and the
spread is printed next to the mean. Accuracy is a GATE, not a deliverable: a net
that clears it still has to win at matched wall clock in
`_eval_rollout_policy_net.py`, which is the only measurement that ships anything.

Four arms, so a win can be attributed rather than just observed:

    baseline  the current 18 features -- reproduces the 0.334 on record
    +dense    18 + action-type discrimination and energy cost, no identity.
              Isolates how much is just "tell potions from END_TURN"
    +scalar   +dense plus ONE learned scalar per identity, added to the score.
              The cheapest possible engine change: a table lookup and an add,
              no MLP cost at all. A generalisation of cardPickRateWeight, which
              is already exactly this shape
    +embed    +dense plus a learned E-dim embedding concatenated into the MLP
              input. The only arm that can express card x state interaction,
              and the one the engine change is actually for

Cost, since the docs are emphatic that a net has to pay for itself: the
embedding is a TABLE LOOKUP, not a one-hot. Input width grows by E (4), not by
743, so first-layer work goes 18x4 -> 27x4 per action. A one-hot would be
389x4 and would lose on the clock before it started.

    python -m lightspeed._probe_card_identity --collect runs/card_identity_probe.pt
    python -m lightspeed._probe_card_identity --data runs/card_identity_probe.pt
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import zlib

import numpy as np
import torch
import torch.nn as nn

import slaythespire as sts

from ._human_deck_combat import build_battle
from .search_config import DEFAULT_SEARCH_CONFIG_PATH, ensure_search_config
from .paths import HUMAN_BENCHMARK

BENCHMARK_PATH = str(HUMAN_BENCHMARK)

# Identity encoding. Row 0 and row 1 are deliberately distinct: END_TURN is a
# real recurring decision with its own semantics, while CARD_SELECT is an
# UNKNOWN -- `Action::getSourceIdx` indexes a different pile per CardSelectTask
# (hand for ARMAMENTS/DUAL_WIELD, cardSelectInfo.cards for DISCOVERY/CODEX/WISH,
# draw/discard/exhaust for the rest) and `bc.cardSelectInfo` is not bound, so
# the pile cannot be resolved from Python. Guessing hand[] would silently
# mislabel every non-hand task, which is worse than one honest unknown symbol.
# Binding `card_select_task` would unlock these; noted, not needed for the gate.
IDENT_END_TURN = 0
IDENT_CARD_SELECT = 1
IDENT_CARD_BASE = 2

N_CARD_IDS = len(sts.CardId.__members__)
N_POTIONS = len(sts.Potion.__members__)
IDENT_POTION_BASE = IDENT_CARD_BASE + N_CARD_IDS * 2
N_IDENT = IDENT_POTION_BASE + N_POTIONS

# 10 state (leaf_features) + 8 action (action_features) = the exact 18 the engine
# feeds nativePolicyNetScore today. Kept as a contiguous prefix so the baseline
# arm is bit-identical to what was measured on 2026-07-31.
DIM_ENGINE = 18


def action_identity(bc, action) -> int:
    """Which concrete thing this action does, as an embedding row."""
    kind = action.action_type
    if kind == sts.ActionType.CARD:
        card = bc.hand[action.source_idx]
        return IDENT_CARD_BASE + int(card.id) * 2 + (1 if card.upgraded else 0)
    if kind == sts.ActionType.POTION:
        # Slot-indexed against bc.potions by construction -- see that property's
        # own comment on why empty slots are kept in the list.
        return IDENT_POTION_BASE + int(bc.potions[action.source_idx])
    if kind == sts.ActionType.END_TURN:
        return IDENT_END_TURN
    return IDENT_CARD_SELECT


def dense_extra(bc, action) -> list[float]:
    """The five features that need no embedding table to be worth having.

    Three of them exist only because every non-CARD action currently collapses
    into one `is_other` flag. The two energy terms are the part an embedding
    CANNOT carry: cost is per-INSTANCE, not per-card -- Setup, Streamline,
    Corruption and Blue Candle all move `costForTurn` away from the card's
    printed cost, and an X-cost card reports -1.
    """
    kind = action.action_type
    is_potion = 1.0 if kind == sts.ActionType.POTION else 0.0
    is_end_turn = 1.0 if kind == sts.ActionType.END_TURN else 0.0
    is_select = 1.0 if kind in (sts.ActionType.SINGLE_CARD_SELECT,
                                sts.ActionType.MULTI_CARD_SELECT) else 0.0
    cost, energy_left = 0.0, bc.player_energy / 3.0
    if kind == sts.ActionType.CARD:
        raw = bc.hand[action.source_idx].cost_for_turn
        # X-cost reports -1; it spends everything, so that is what it costs.
        spend = bc.player_energy if raw < 0 else raw
        cost = spend / 3.0
        energy_left = max(0, bc.player_energy - spend) / 3.0
    return [is_potion, is_end_turn, is_select, cost, energy_left]


def collect_fight(rec: dict, sims: int, ascension: int, max_steps: int = 600):
    """Play one fight with the search, recording what it chose and from what.

    Mirrors `collect_rollout_policy_data.collect_fight` exactly -- same skip of
    single-action decisions, same search seed derivation -- and adds the identity
    column plus the five extra dense features. Same states, so the baseline arm
    here is comparable to the numbers already on record.
    """
    bc, _ = build_battle(rec["deck"], rec["relics"], rec["cur_hp"], rec["max_hp"],
                         getattr(sts.MonsterEncounter, rec["encounter"]),
                         ascension, rec["act"], rec.get("potions", ()))
    fight_key = zlib.crc32(f"{rec['run_id']}:{rec['floor']}".encode())

    features, idents, group_sizes, chosen = [], [], [], []
    for step in range(max_steps):
        if bc.outcome != sts.BattleOutcome.UNDECIDED:
            break
        actions = bc.get_legal_actions()
        if not actions:
            break
        action, _ = sts.run_mcts_search(bc, sims, None, (fight_key << 20) ^ step)
        if len(actions) > 1:
            state = list(bc.leaf_features())
            rows, ids = [], []
            for a in actions:
                rows.append(state + list(bc.action_features(a)) + dense_extra(bc, a))
                ids.append(action_identity(bc, a))
            keys = [str(a) for a in actions]
            try:
                pick = keys.index(str(action))
            except ValueError:
                action.execute(bc)
                continue
            features.extend(rows)
            idents.extend(ids)
            group_sizes.append(len(actions))
            chosen.append(pick)
        action.execute(bc)
    return features, idents, group_sizes, chosen


def collect(out_path: str, split: str, sims: int, ascension: int, limit: int) -> None:
    ensure_search_config(DEFAULT_SEARCH_CONFIG_PATH)
    # A loaded net would make the search imitate itself. Off, as in collection.
    sts.set_search_params({"policy_net_weight": 0.0})

    with open(BENCHMARK_PATH, encoding="utf-8") as handle:
        records = json.load(handle)
    fights = [r for r in records if split == "all" or r["split"] == split]
    if limit:
        fights = fights[:limit]
    print(f"collecting from {len(fights)} {split} fights at {sims} sims")

    all_features, all_idents, all_groups, all_chosen, fight_of = [], [], [], [], []
    start = time.time()
    for index, rec in enumerate(fights):
        try:
            features, idents, groups, chosen = collect_fight(rec, sims, ascension)
        except Exception as error:  # noqa: BLE001 - reported, not hidden
            print(f"  skip {rec['run_id']}@{rec['floor']}: "
                  f"{type(error).__name__}: {error}")
            continue
        all_features.extend(features)
        all_idents.extend(idents)
        all_groups.extend(groups)
        all_chosen.extend(chosen)
        # Held out BY FIGHT below, not by decision: consecutive decisions inside
        # one fight share a deck, a monster and most of a state, so a
        # decision-level split leaks the answer across the boundary.
        fight_of.extend([index] * len(groups))
        if (index + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  {index+1}/{len(fights)} fights, {len(all_groups)} decisions, "
                  f"{elapsed:.0f}s ({elapsed/(index+1)*1000:.0f} ms/fight)")

    payload = {
        "x": torch.tensor(np.asarray(all_features, dtype=np.float32)),
        "ident": torch.tensor(np.asarray(all_idents, dtype=np.int64)),
        "groups": torch.tensor(np.asarray(all_groups, dtype=np.int64)),
        "chosen": torch.tensor(np.asarray(all_chosen, dtype=np.int64)),
        "fight": torch.tensor(np.asarray(fight_of, dtype=np.int64)),
        "meta": {"split": split, "sims": sims, "ascension": ascension,
                 "fights": len(fights), "n_ident": N_IDENT,
                 "dim_engine": DIM_ENGINE},
    }
    torch.save(payload, out_path)
    print(f"\n{len(all_groups)} decisions, {len(all_features)} scored actions")
    print(f"wrote {out_path} ({time.time()-start:.0f}s)")


class RankNet(nn.Module):
    """Dense stack ending in a scalar score per action, optionally identity-aware.

    `embed_dim > 0` concatenates a learned per-identity vector into the MLP input
    -- this is the arm the engine change would implement, as a table lookup, so
    the width the first layer sees grows by embed_dim and nothing else.
    `scalar` instead adds a learned per-identity bias straight to the output,
    which costs the engine literally one lookup and one add.
    """

    def __init__(self, dense_dim: int, hidden: list[int], n_ident: int = 0,
                 embed_dim: int = 0, scalar: bool = False):
        super().__init__()
        self.embed = nn.Embedding(n_ident, embed_dim) if embed_dim else None
        if self.embed is not None:
            nn.init.normal_(self.embed.weight, std=0.1)
        self.bias = nn.Embedding(n_ident, 1) if scalar else None
        if self.bias is not None:
            nn.init.zeros_(self.bias.weight)
        layers: list[nn.Module] = []
        prev = dense_dim + embed_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.Tanh()]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, ident):
        if self.embed is not None:
            x = torch.cat([x, self.embed(ident)], dim=-1)
        score = self.net(x).squeeze(-1)
        if self.bias is not None:
            score = score + self.bias(ident).squeeze(-1)
        return score


def grouped_cross_entropy(scores, groups, chosen):
    """Softmax cross-entropy within each decision's own action set.

    Lifted verbatim from `train_rollout_policy_net.py` so the two are measuring
    the same quantity; see that file for why it is scattered rather than looped.
    """
    n_groups = len(groups)
    group_of = torch.repeat_interleave(torch.arange(n_groups, device=scores.device),
                                       groups)
    offsets = torch.zeros(n_groups + 1, dtype=torch.long, device=scores.device)
    offsets[1:] = groups.cumsum(0)

    group_max = scores.new_full((n_groups,), float("-inf"))
    group_max = group_max.scatter_reduce(0, group_of, scores, reduce="amax")
    shifted = (scores - group_max[group_of]).exp()
    group_sum = scores.new_zeros(n_groups).index_add(0, group_of, shifted)
    logsumexp = group_max + group_sum.log()

    chosen_flat = offsets[:-1] + chosen
    loss = (logsumexp - scores[chosen_flat]).mean()

    best = scores.new_full((n_groups,), float("-inf")).scatter_reduce(
        0, group_of, scores, reduce="amax")
    is_best = scores == best[group_of]
    first_best = torch.zeros(n_groups, dtype=torch.long, device=scores.device)
    first_best.scatter_reduce_(0, group_of[is_best],
                               (torch.arange(len(scores), device=scores.device)
                                - offsets[:-1][group_of])[is_best],
                               reduce="amin", include_self=False)
    accuracy = (first_best == chosen).float().mean().item()
    return loss, accuracy


def train_arm(data, hidden: list[int], dense_dim: int, embed_dim: int,
              scalar: bool, seed: int, epochs: int, lr: float,
              embed_decay: float) -> float:
    """One arm at one seed. Returns best val top-1."""
    torch.manual_seed(seed)
    xtr, xva = data["xtr"][:, :dense_dim], data["xva"][:, :dense_dim]
    model = RankNet(dense_dim, hidden, N_IDENT, embed_dim, scalar)
    # Weight decay on the identity tables only. Most of the 743 card rows are
    # never seen (off-class cards cannot appear in an Ironclad benchmark) and the
    # rare ones are seen a handful of times, which is exactly where an embedding
    # memorises. The dense trunk is small enough not to need it.
    identity_params = [p for m in (model.embed, model.bias) if m is not None
                       for p in m.parameters()]
    trunk = list(model.net.parameters())
    opt = torch.optim.Adam(
        [{"params": trunk, "weight_decay": 0.0},
         {"params": identity_params, "weight_decay": embed_decay}], lr=lr)

    best = -1.0
    for _ in range(epochs):
        model.train()
        loss, _ = grouped_cross_entropy(model(xtr, data["itr"]),
                                        data["gtr"], data["ctr"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            _, vacc = grouped_cross_entropy(model(xva, data["iva"]),
                                            data["gva"], data["cva"])
        best = max(best, vacc)
    return best


def prepare(payload, val_frac: float) -> dict:
    """Split by fight, normalise the dense block on train only."""
    x, ident = payload["x"], payload["ident"]
    groups, chosen, fight = payload["groups"], payload["chosen"], payload["fight"]

    fights = int(fight.max()) + 1
    cutoff = fights - max(1, int(fights * val_frac))
    n_train = int((fight < cutoff).sum())
    train_actions = int(groups[:n_train].sum())

    xtr, xva = x[:train_actions], x[train_actions:]
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
    return {
        "xtr": (xtr - mu) / sd, "xva": (xva - mu) / sd,
        "itr": ident[:train_actions], "iva": ident[train_actions:],
        "gtr": groups[:n_train], "gva": groups[n_train:],
        "ctr": chosen[:n_train], "cva": chosen[n_train:],
        "n_train_fights": cutoff, "n_val_fights": fights - cutoff,
    }


def report_coverage(data) -> None:
    """How much of the identity table is actually exercised, and how thinly.

    An embedding row seen twice in training is a memorised constant, and the
    engine will hit it at inference on a deck this benchmark never contained.
    This is the arm's main honest risk, so it gets printed rather than assumed.
    """
    counts = torch.bincount(data["itr"], minlength=N_IDENT)
    seen = int((counts > 0).sum())
    val_counts = counts[data["iva"]]
    print(f"identity coverage: {seen}/{N_IDENT} rows seen in train")
    print(f"  val actions whose identity was seen >= 20x in train: "
          f"{float((val_counts >= 20).float().mean()):.3f}")
    print(f"  val actions with an UNSEEN identity: "
          f"{float((val_counts == 0).float().mean()):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", metavar="OUT",
                        help="collect a dataset to this path and exit")
    parser.add_argument("--data", help="train the four arms on this dataset")
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--hidden", default="8")
    parser.add_argument("--embed-dim", type=int, default=4)
    parser.add_argument("--embed-decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    if args.collect:
        collect(args.collect, args.split, args.sims, args.ascension, args.limit)
        return
    if not args.data:
        parser.error("pass --collect OUT to build a dataset or --data IN to train")

    payload = torch.load(args.data, weights_only=False)
    print(f"{len(payload['groups'])} decisions, {payload['x'].shape[0]} actions, "
          f"dim {payload['x'].shape[1]}")
    print(f"meta: {payload['meta']}\n")
    data = prepare(payload, args.val_frac)
    print(f"split by fight: {data['n_train_fights']} train / "
          f"{data['n_val_fights']} val, "
          f"{len(data['gtr'])} / {len(data['gva'])} decisions")
    report_coverage(data)

    baseline = float((1.0 / data["gva"].float()).mean())
    print(f"\nrandom-pick accuracy on val: {baseline:.3f}")

    hidden = [int(h) for h in args.hidden.split(",") if h.strip()]
    dim_all = int(payload["x"].shape[1])
    arms = [
        ("baseline (engine's 18)", DIM_ENGINE, 0, False),
        ("+dense (type, energy)", dim_all, 0, False),
        ("+scalar per identity", dim_all, 0, True),
        (f"+embed dim {args.embed_dim}", dim_all, args.embed_dim, False),
    ]

    print(f"\nhidden {hidden}, {args.seeds} seeds per arm, {args.epochs} epochs")
    print(f"{'arm':26s}{'val top-1':>12s}{'spread':>10s}{'vs random':>11s}")
    results = {}
    for label, dense_dim, embed_dim, scalar in arms:
        scores = [train_arm(data, hidden, dense_dim, embed_dim, scalar, seed,
                            args.epochs, args.lr, args.embed_decay)
                  for seed in range(args.seeds)]
        mean = statistics.mean(scores)
        spread = max(scores) - min(scores)
        results[label] = (mean, spread)
        print(f"{label:26s}{mean:12.3f}{spread:10.3f}{mean/baseline:10.2f}x")

    # The pre-registered gate, evaluated here rather than eyeballed from the
    # table -- a margin inside the combined seed spread is not a result.
    base_mean, base_spread = results["baseline (engine's 18)"]
    embed_mean, embed_spread = results[f"+embed dim {args.embed_dim}"]
    margin = embed_mean - base_mean
    noise = base_spread + embed_spread
    print(f"\n+embed - baseline = {margin:+.3f}, combined seed spread {noise:.3f}")
    if margin > noise:
        print("GATE PASSED: identity carries signal the 18 features do not. The "
              "engine change is worth starting -- then it must still win at "
              "matched wall clock in _eval_rollout_policy_net.py.")
    else:
        print("GATE FAILED: identity does not move top-1 beyond seed noise. The "
              "feature vector is NOT the limit; do not widen it. Record this in "
              "docs/03-combat-search.md's dead-ends table.")


if __name__ == "__main__":
    main()
