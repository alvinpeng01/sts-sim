"""Do drafting interventions buy floors? Skip bonus and human pick-rate prior.

Drafting is the largest thing the network does -- randomizing it costs
**-5.85 floors** (`_decision_ablation.py`), and reward screens are 40.7 of the
~81 decisions in a run. It is also where the clearest gap against agents that
actually win sits, and the gap is not subtle:

    cards per run      v37 24.5   pilot120 25.5   heart1 17.0   human 16.2
    SKIP chosen        v37 1/272 (0.4%)   pilot120 0/240 (0%)
    SKIP was legal     100% of those decisions

The policy sits at a corner: an action available every single time, taken
never. Dominion -- where deck dilution is settled theory rather than a
hypothesis -- treats buying nothing as a frequently-correct move, and its bots
must model trashing to compete. heart1 declines more than half its offers.

Two interventions, both pure inference-time logit biases with nothing trained:

* `--skip-bonus` added to SKIP actions on reward screens;
* `--pickrate-weight` scaling `(pick_rate[card] - 0.5)` on card actions, using
  the 72-card human list in `data/ironclad_pick_rates.json` (the same decisions
  `cardPickRateWeight` was learned from).

Scored against **paired floors**, which is the only metric that has ever
settled anything here. Note the precedent for caution: the equivalent
intervention on elites -- forcing the policy toward what winning runs do --
**cost 1.93 floors**, because elite-taking is capability-gated. Skipping should
not be, since a thinner deck helps consistency regardless of combat strength,
but that is a prediction and this harness is how it gets tested.

Run from slay-sim/:
    python -m lightspeed._draft_bias_probe --runs 240 --workers 6

RESULTS. Both interventions are refuted, and so is the follow-up that was
supposed to rescue them.

ROUND 1 (seeds 1_003_000+, n = 240, runs/draft_bias_probe.jsonl)

           arm    floor   paired vs base       t   skips/run   deck
      baseline    27.18         --           --       0.0      27.3
      skip+0.5    27.18    +0.00 +/-0.00      --      0.0      27.3
        skip+1    27.18    +0.00 +/-0.00      --      0.0      27.3
        skip+2    27.12    -0.06 +/-0.08    -0.74     0.1      27.2
        skip+4    24.10    -3.08 +/-0.51    -6.00     3.7      21.5
    pickrate+2    26.87    -0.31 +/-0.54    -0.57     0.0      26.9
    pickrate+5    25.49    -1.69 +/-0.58    -2.94     0.1      26.2

skip+0.5 and skip+1 change LITERALLY nothing -- not a small effect, zero runs
altered -- so the take-vs-skip logit gap exceeds 1. This is not a marginal
preference that a nudge can tip.

ROUND 2 (seeds 4_000_000+, n = 600, runs/draft_bias_probe_r2.jsonl) asked
whether the flat bias failed because it was INDISCRIMINATE. A blanket bonus
discards a Corruption and a Wild Strike alike; humans skip weak offers. The
weak<T arms fire only when every card on offer is below T, with the dose fixed
at the 4.0 known to flip the decision, so they differ from skip+N in WHICH
offers they decline, not how forcefully.

           arm    floor   paired vs base       t   declined   deck
      baseline    27.09         --           --      0.0%     27.2
     weak<0.15    26.53    -0.56 +/-0.21    -2.72     7.4%     26.0
     weak<0.25    26.32    -0.77 +/-0.24    -3.18    14.3%     25.0
        skip+3    26.28    -0.81 +/-0.21    -3.87    14.6%     25.0
     weak<0.35    26.03    -1.06 +/-0.27    -3.90    19.2%     24.2

THE CONTROLLED COMPARISON is skip+3 against weak<0.25. They decline the same
share of offers -- 14.6% versus 14.3% -- and differ only in which ones, one
choosing by nothing and the other by human pick rate. Paired on the same seeds:

    weak<0.25 minus skip+3:  +0.04 +/-0.24  (t = +0.18), 200/600 runs differ

Targeting buys nothing. The runs genuinely diverge, so this is a null with the
mechanism firing, not a null implementation.

What the dose curve shows is that the loss tracks SKIP VOLUME and is indifferent
to targeting:

     0.0% declined    0.00 floors
     7.4%            -0.56
    14.3%            -0.77
    14.6%            -0.81
    19.2%            -1.06
    35.0%            -3.08   (round 1 skip+4)

and deck size tracks floors monotonically across the whole tested range: 27.2 ->
27.09, 26.0 -> 26.53, 25.0 -> 26.3, 24.2 -> 26.03. For this agent MORE CARDS IS
MORE FLOORS, which is the reverse of the deck-dilution theory the docstring above
opens with. The prediction that "skipping should not be capability-gated, since a
thinner deck helps consistency regardless of combat strength" is wrong as stated.

CAVEAT, and it is not a small one. Every arm here biases a policy at inference
time that was TRAINED taking essentially every card. Its downstream play -- and
the combat search's rollout priors -- have only ever seen decks built that way,
so an off-distribution deck can lose floors for reasons that have nothing to do
with whether the draft was good. What is established is that you cannot bolt
thinning onto this policy after the fact. Whether an agent TRAINED to thin would
do better is a different experiment and is not answered here. The same caveat
applies to `_route_bias_probe.py`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

DEFAULT_CHECKPOINT = "runs/ppo/pilot1/policy_iter120.pt"
PICK_RATES = "lightspeed/data/ironclad_pick_rates.json"

_STATE: dict = {}


def _worker_init(checkpoint: str, sims: int, ascension: int) -> None:
    import torch
    import slaythespire as sts

    from .eval_whole_run_policy import load_policy
    from .search_config import DEFAULT_SEARCH_CONFIG_PATH

    torch.set_num_threads(1)
    _STATE["torch"] = torch
    _STATE["sts"] = sts
    _STATE["policy"] = load_policy(checkpoint, torch.device("cpu"))
    _STATE["sims"] = sims
    _STATE["ascension"] = ascension
    _STATE["search_config"] = DEFAULT_SEARCH_CONFIG_PATH
    with open(PICK_RATES, encoding="utf-8") as handle:
        rates = json.load(handle)
    # Our OWN depth-residualized card values (_card_value_audit.py), in floors.
    # The human prior lost 1.69 floors; three separate attempts to import a
    # better agent's revealed preferences have now failed the same way. This is
    # the same shape of prior built from OUR outcomes instead, which is the one
    # source not confounded by someone else's capability.
    own = {}
    own_path = "runs/card_value_audit.json"
    if os.path.exists(own_path):
        with open(own_path, encoding="utf-8") as handle:
            for row in json.load(handle)["rows"]:
                if hasattr(sts.CardId, row["card"]):
                    own[int(getattr(sts.CardId, row["card"]))] = row["residual"]
    _STATE["own_by_id"] = own
    # content id for a card action is 1 + CardId, set in whole_run_env.observation
    _STATE["rate_by_id"] = {
        int(getattr(sts.CardId, name)): value
        for name, value in rates.items() if hasattr(sts.CardId, name)
    }


def _offer_best_rate(obs, actions, sts, rate_by_id) -> float | None:
    """Highest human pick rate among the cards on offer, or None if unknown.

    Unknown cards return None and the caller declines to fire, rather than
    treating them as weak. The list covers 72 Ironclad cards; colourless and
    rarer additions are missing, and scoring an unknown card as 0 would make the
    rule skip exactly the offers it knows least about.
    """
    best = None
    for index, action in enumerate(actions):
        if action.rewards_action_type != sts.RewardsActionType.CARD:
            continue
        rate = rate_by_id.get(int(obs["action_content_ids"][index]) - 1)
        if rate is None:
            return None
        best = rate if best is None else max(best, rate)
    return best


def _play(job: tuple[str, float, float, float, float, float, int]) -> dict:
    torch = _STATE["torch"]
    sts = _STATE["sts"]
    policy = _STATE["policy"]

    from .whole_run_env import RunConfig, WholeRunEnv

    label, skip_bonus, pickrate_weight, weak_threshold, weak_bonus, own_weight, seed = job
    env = WholeRunEnv(RunConfig(
        ascension=_STATE["ascension"], combat_sims=_STATE["sims"],
        deterministic_combat=True, search_config_path=_STATE["search_config"]))
    obs = env.reset(seed)
    offers = taken = skipped = 0
    # How often the conditional rule could evaluate the offer and chose to fire.
    # Without these, a null result cannot be told apart from a rule that never
    # triggered -- which is the failure mode that made the routing probe add
    # rests_entered after the fact.
    weak_fired = weak_unknown = 0
    with torch.inference_mode():
        while (env.gc.outcome.name == "UNDECIDED"
               and env.steps < env.config.max_decisions):
            logits, _ = policy(obs)
            actions = env.legal_actions()
            is_card_screen = (
                env.gc.screen_state == sts.ScreenState.REWARDS
                and any(a.rewards_action_type == sts.RewardsActionType.CARD
                        for a in actions))
            if is_card_screen and (skip_bonus or pickrate_weight
                                   or weak_bonus or own_weight):
                bias = torch.zeros_like(logits)
                # Conditional skip: humans do not skip MORE, they skip WEAK
                # offers. A flat bonus cannot express that -- it discards a
                # Corruption and a Wild Strike at the same rate -- so this fires
                # only when every card on offer is below threshold.
                weak_extra = 0.0
                if weak_bonus:
                    best = _offer_best_rate(obs, actions, sts,
                                            _STATE["rate_by_id"])
                    if best is None:
                        weak_unknown += 1
                    elif best < weak_threshold:
                        weak_extra = weak_bonus
                        weak_fired += 1
                for index, action in enumerate(actions):
                    kind = action.rewards_action_type
                    if kind == sts.RewardsActionType.SKIP:
                        bias[index] += skip_bonus + weak_extra
                    elif kind == sts.RewardsActionType.CARD:
                        content = int(obs["action_content_ids"][index]) - 1
                        if pickrate_weight:
                            rate = _STATE["rate_by_id"].get(content)
                            if rate is not None:
                                # Centred so the prior reorders cards without a
                                # blanket shift for or against taking one at
                                # all; that is what --skip-bonus is for.
                                bias[index] += pickrate_weight * (rate - 0.5)
                        if own_weight:
                            value = _STATE["own_by_id"].get(content)
                            if value is not None:
                                # Already centred: a residual is a deviation
                                # from the depth-matched expectation.
                                bias[index] += own_weight * value
                logits = logits + bias
            index = int(torch.argmax(logits))
            if is_card_screen:
                offers += 1
                kind = actions[index].rewards_action_type
                taken += int(kind == sts.RewardsActionType.CARD)
                skipped += int(kind == sts.RewardsActionType.SKIP)
            obs, _, done, _ = env.step(index)
            if done:
                break

    representation = sts.getNNRepresentation(env.gc)
    return {"arm": label, "seed": seed, "floor": int(env.gc.floor_num),
            "act": int(env.gc.act), "outcome": env.gc.outcome.name,
            "deck": len(representation.deck.cards),
            "relics": len(representation.relics.relics),
            "offers": offers, "taken": taken, "skipped": skipped,
            "weak_fired": weak_fired, "weak_unknown": weak_unknown}


def summarize(rows: list[dict]) -> None:
    by_arm: dict[str, dict[int, dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], {})[row["seed"]] = row
    base = by_arm["baseline"]
    print(f"{'arm':>22}  {'floor':>6}  {'paired vs base':>17}  {'t':>6}  "
          f"{'deck':>5}  {'skip%':>6}  {'fired':>5}  {'unk':>5}")
    order = ["baseline"] + [a for a in by_arm if a != "baseline"]
    for arm in order:
        seeds = by_arm[arm]
        floors = [r["floor"] for r in seeds.values()]
        deck = statistics.mean(r["deck"] for r in seeds.values())
        offers = sum(r["offers"] for r in seeds.values())
        skips = sum(r["skipped"] for r in seeds.values())
        if arm == "baseline":
            delta = tstat = "--"
        else:
            shared = sorted(set(seeds) & set(base))
            diffs = [seeds[s]["floor"] - base[s]["floor"] for s in shared]
            mean = statistics.mean(diffs)
            sem = (statistics.stdev(diffs) / math.sqrt(len(diffs))
                   if len(diffs) > 1 else 0.0)
            delta = f"{mean:+.2f} +/-{sem:.2f}"
            # An arm that changes nothing gives sem 0; report it as such
            # rather than dividing by zero.
            tstat = f"{mean / sem:+.2f}" if sem > 0 else ("0.00" if mean == 0 else "inf")
        fired = sum(r.get("weak_fired", 0) for r in seeds.values())
        unknown = sum(r.get("weak_unknown", 0) for r in seeds.values())
        print(f"{arm:>22}  {statistics.mean(floors):>6.2f}  {delta:>17}  "
              f"{tstat:>6}  {deck:>5.1f}  "
              f"{100 * skips / max(1, offers):>5.1f}%  "
              f"{fired / len(seeds):>5.2f}  {unknown / len(seeds):>5.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--runs", type=int, default=240)
    parser.add_argument("--seed-base", type=int, default=1_003_000)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--arms", default=None,
                        help="comma-separated subset of arm labels")
    parser.add_argument("--out", default="runs/draft_bias_probe.jsonl")
    args = parser.parse_args()

    # (label, skip_bonus, pickrate_weight, weak_threshold, weak_bonus, own_weight)
    arms = [
        ("baseline", 0.0, 0.0, 0.0, 0.0, 0.0),
        ("skip+0.5", 0.5, 0.0, 0.0, 0.0, 0.0),
        ("skip+1", 1.0, 0.0, 0.0, 0.0, 0.0),
        ("skip+2", 2.0, 0.0, 0.0, 0.0, 0.0),
        # Fills the unexplored dose gap: +2 moved 8 runs and +4 moved 123, so
        # nothing is known between "no effect" and "-3.08 floors".
        ("skip+3", 3.0, 0.0, 0.0, 0.0, 0.0),
        ("skip+4", 4.0, 0.0, 0.0, 0.0, 0.0),
        ("pickrate+2", 0.0, 2.0, 0.0, 0.0, 0.0),
        ("pickrate+5", 0.0, 5.0, 0.0, 0.0, 0.0),
        ("skip+1,pickrate+2", 1.0, 2.0, 0.0, 0.0, 0.0),
        # Conditional skip. Bonus fixed at 4.0 -- the dose known to actually
        # flip the decision -- with the threshold carrying the whole variation,
        # so these differ from skip+4 in WHICH offers they decline and not in
        # how forcefully. Thresholds sit at roughly the 6%, 17% and 27% points
        # of the best-of-three offer distribution (marginal median 0.21).
        ("weak<0.15", 0.0, 0.0, 0.15, 4.0, 0.0),
        ("weak<0.25", 0.0, 0.0, 0.25, 4.0, 0.0),
        ("weak<0.35", 0.0, 0.0, 0.35, 4.0, 0.0),
        # A prior built from OUR OWN depth-residualized card values
        # (_card_value_audit.py) rather than another agent's revealed
        # preferences. Three imports of the latter have now lost floors; this
        # is the same shape of intervention sourced from our own outcomes,
        # which is the one signal not confounded by someone else's capability.
        ("own+0.25", 0.0, 0.0, 0.0, 0.0, 0.25),
        ("own+0.5", 0.0, 0.0, 0.0, 0.0, 0.5),
        ("own+1", 0.0, 0.0, 0.0, 0.0, 1.0),
        ("own-0.5", 0.0, 0.0, 0.0, 0.0, -0.5),
    ]
    if args.arms:
        wanted = {name.strip() for name in args.arms.split(",")} | {"baseline"}
        arms = [entry for entry in arms if entry[0] in wanted]
    jobs = [(label, skip, rate, weak_t, weak_b, own, args.seed_base + offset)
            for label, skip, rate, weak_t, weak_b, own in arms
            for offset in range(args.runs)]
    print(f"{len(arms)} arms x {args.runs} seeds = {len(jobs)} runs "
          f"at {args.sims} sims", flush=True)

    rows = []
    started = time.perf_counter()
    with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_worker_init,
            initargs=(args.checkpoint, args.sims, args.ascension)) as pool:
        for done, row in enumerate(pool.map(_play, jobs, chunksize=2), start=1):
            rows.append(row)
            if done % 300 == 0:
                print(f"  {done}/{len(jobs)} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"\nwrote {args.out} ({time.perf_counter() - started:.0f}s)\n")
    summarize(rows)


if __name__ == "__main__":
    main()
