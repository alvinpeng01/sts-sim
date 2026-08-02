"""Score inference-time routing biases directly against paired mean floor.

The routing audit (`_routing_audit.py`) fits a conditional logit over path
decisions and finds two coefficients with the WRONG SIGN relative to both
Silverbot and the human archive:

    room          silverbot     v31      Baalorlord
    ELITE            +0.22     -2.55        +1.93
    hp_frac x REST   -1.72     +1.19        -1.93

Cloning those preferences from the human archive is refuted -- it cost 15.80
floors, because his elite-taking is rational only given his deck, and the
extraction pins that deck into every observation (07-known-issues.md).  This
probe asks the cheaper question that imitation cannot: applied as a plain
additive logit bias on map screens, held to two or three scalars, does moving
these coefficients toward the reference values BUY FLOORS on our own state
distribution?

That distinguishes the two live hypotheses.  If elite avoidance is a valuation
error inherited from labels whose rollouts play elites badly, a positive bias
gains floors.  If it is a rational response to combat that is more expensive
than the human's, every positive bias loses floors -- and only a better policy
or better combat can fix it, not a coefficient.

Nothing is trained and no checkpoint is written: the bias is added to the
policy's logits at MAP_SCREEN only, so the arms differ in exactly one place and
share combat, seeds and search seeds.  Baseline is the (0, 0, 0) arm, evaluated
in the same process on the same seeds.

Run from slay-sim/:
    python -m lightspeed._route_bias_probe --runs 120 --workers 6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

MAP_SCREEN = 5
ROOM_REST = 1
ROOM_ELITE = 3

DEFAULT_CHECKPOINT = "runs/whole_run_transformer_yield10x_a20_v31.pt"

# (label, elite_bias, rest_bias, rest_when_hurt_bias).  The audit's fitted gaps
# are ELITE +2.8 (to silverbot) to +4.5 (to the human) and about -3 on the
# hp_frac x REST interaction, which is +3 on (1 - hp_frac) x REST.  The sweep
# brackets that range on both sides of zero, since the conditional logit's
# scale is not guaranteed to equal the net's own logit scale.
DEFAULT_SWEEP = [
    ("baseline", 0.0, 0.0, 0.0),
    ("elite+1", 1.0, 0.0, 0.0),
    ("elite+2", 2.0, 0.0, 0.0),
    ("elite+3", 3.0, 0.0, 0.0),
    ("elite+4.5", 4.5, 0.0, 0.0),
    ("elite-1", -1.0, 0.0, 0.0),
    ("hurtrest+2", 0.0, 0.0, 2.0),
    ("hurtrest+3", 0.0, 0.0, 3.0),
    ("rest+1.5", 0.0, 1.5, 0.0),
    ("elite+3,hurtrest+3", 3.0, 0.0, 3.0),
    # Second round: the first sweep put every rest arm at about +0.5 floors with
    # t ~ 1.1-1.5, which is suggestive and not decided, and put the three arms
    # within 0.02 floors of each other -- so it cannot yet distinguish "rest
    # more" from "rest more WHEN HURT".  These separate the two and extend the
    # range, on a fresh seed set.
    ("rest+3", 0.0, 3.0, 0.0),
    ("rest+5", 0.0, 5.0, 0.0),
    ("hurtrest+5", 0.0, 0.0, 5.0),
    ("rest+1.5,hurtrest+3", 0.0, 1.5, 3.0),
]

# ROUND 1 (seeds 1_003_000+, n = 120/arm, in runs/route_bias_probe.jsonl)
#
#              arm    floor   paired vs base      t     W/T/L      elites
#         baseline    22.18         --           --       --         0.85
#          elite+1    20.23    -1.95 +/-0.72   -2.69   27/46/47      2.22
#          elite+2    20.25    -1.93 +/-0.72   -2.67   27/46/47      2.23
#          elite+3    20.25    -1.93 +/-0.72   -2.67   27/46/47      2.23
#        elite+4.5    20.25    -1.93 +/-0.72   -2.67   27/46/47      2.23
#          elite-1    21.81    -0.38 +/-0.23   -1.62    3/109/8      0.81
#       hurtrest+2    22.70    +0.52 +/-0.35   +1.48    13/101/6     0.91
#       hurtrest+3    22.69    +0.51 +/-0.46   +1.11    14/97/9      0.90
#         rest+1.5    22.68    +0.50 +/-0.44   +1.15    18/88/14     0.97
# elite+3,hurtrest+3  20.59    -1.59 +/-0.78   -2.04   31/43/46      2.38
#
# ROUND 2 (seeds 2_000_000+, n = 350/arm, in runs/route_bias_probe_r2.jsonl)
#
#              arm    floor   paired vs base      t      W/T/L      elites
#         baseline    22.12         --           --        --         0.83
#          elite+2    21.88    -0.24 +/-0.43   -0.56  100/129/121     2.55
#       hurtrest+2    22.21    +0.08 +/-0.17   +0.49    27/302/21     0.89
#         rest+1.5    22.54    +0.41 +/-0.25   +1.64    53/262/35     0.99
#           rest+3    22.54    +0.41 +/-0.25   +1.64    53/262/35     0.99
#           rest+5    22.54    +0.41 +/-0.25   +1.64    53/262/35     0.99
#       hurtrest+5    22.33    +0.20 +/-0.19   +1.05    34/293/23     0.93
#
# Combined, inverse-variance across the two DISJOINT seed sets:
#
#          elite+2   -0.69 +/-0.37  (t = -1.85)   per round  -1.93  -0.24
#       hurtrest+2   +0.17 +/-0.15  (t = +1.09)   per round  +0.52  +0.08
#         rest+1.5   +0.44 +/-0.22  (t = +1.99)   per round  +0.50  +0.41
#
# READ ROUND 1 AND ROUND 2 TOGETHER, NEVER ROUND 1 ALONE. On round 1 the elite
# result looked decided at -1.93 +/-0.72 (t = -2.67). It did not replicate: the
# same arm on fresh seeds is -0.24 +/-0.43. The behaviour change was comparable
# in both rounds -- about 63% of runs altered, elite capture 0.83 -> 2.55 -- so
# what moved was not the policy but the estimate. Per-run paired sd on the elite
# arms is 7.94 floors, which at n = 120 is a standard error of 0.72, and a
# single round of that width will happily report a floor and a half that is not
# there. The honest combined statement is that a positive elite bias is
# DIRECTIONALLY negative and not established: -0.69 +/-0.37.
#
# Round 2 also reversed round 1 on WHICH rest rule is better. Round 1 mildly
# favoured the hp-conditioned one; round 2 puts unconditional rest at +0.41 and
# hp-conditioned at +0.08. Unconditional rest saturates at +1.5 -- the +1.5,
# +3 and +5 arms are identical run for run -- so there is no dose to tune, only
# a rule to accept or reject.
#
# rest+1.5 at t = 1.99 combined is the only arm still alive, and 1.99 is not a
# result. Round 3 (seeds 3_000_000+, n = 800) is the pre-registered test.
#
# ROUND 3 (seeds 3_000_000+, n = 800/arm, in runs/route_bias_probe_r3.jsonl)
# Pre-registered gate: ship rest+1.5 only if round 3 alone is positive with
# t >= 2 AND the three-round combined reaches t >= 3.
#
#              arm    floor   paired vs base      t      W/T/L    elites  rests
#         baseline    22.79         --           --        --       0.86   3.79
#       hurtrest+2    22.80    +0.00 +/-0.11   +0.03   57/699/44    0.92   3.99
#         rest+1.5    22.72    -0.08 +/-0.16   -0.50   77/637/86    0.97   4.15
#
# Combined, inverse-variance across all three disjoint seed sets:
#
#       hurtrest+2   +0.06 +/-0.09  (t = +0.68)   +0.52  +0.08  +0.00
#         rest+1.5   +0.10 +/-0.13  (t = +0.75)   +0.50  +0.41  -0.08
#
# BOTH SCALARS ARE REFUTED. The gate is failed on every clause: round 3 alone is
# negative, and combined t is 0.75 against a required 3.
#
# The mechanism is not in doubt, which is what makes this a refutation and not a
# null implementation. rest+1.5 moved rests per run from 3.79 to 4.15 and altered
# 163 of 800 runs; hurtrest+2 moved them to 3.99. The bias did exactly what it
# was designed to do and bought nothing.
#
# Read the sequence, because it is the most useful thing here:
#
#         n = 120   +0.50 +/-0.44
#         n = 350   +0.41 +/-0.25
#         n = 800   -0.08 +/-0.16
#
# Monotone decay toward zero as n grows, with each estimate inside the previous
# round's error bar. Nothing was ever inconsistent -- the early rounds were
# underpowered and the effect was never there. A project that stopped at round 1
# or 2 would have shipped this. The per-run paired sd is 4.4 floors for rest arms
# and 7.9 for elite arms, so at n = 120 a standard error is 0.4 to 0.7 floors and
# any single round of that width is a coin toss dressed as a measurement.
#
# Baseline mean floor also moved across rounds (22.18, 22.12, 22.79) purely from
# the seed set. That is why arms are only ever paired against a baseline run on
# the SAME seeds in the SAME round.
#
# Standing implication for the routing axis: this is now the fourth independent
# attempt on it -- map representation, human-archive imitation (-15.80),
# survival-weighted planning (-3.68) and inference-time biases (null) -- and none
# has bought a floor. The +15.71 floor gap is probably not in routing.

_STATE: dict = {}


def _worker_init(checkpoint: str, sims: int, ascension: int, search_config: str | None) -> None:
    import torch

    from .eval_whole_run_policy import load_policy
    from .search_config import DEFAULT_SEARCH_CONFIG_PATH

    torch.set_num_threads(1)
    _STATE["torch"] = torch
    _STATE["policy"] = load_policy(checkpoint, torch.device("cpu"))
    _STATE["sims"] = sims
    _STATE["ascension"] = ascension
    _STATE["search_config"] = (
        search_config if search_config is not None else DEFAULT_SEARCH_CONFIG_PATH)


def _biased_action(policy, obs, torch, elite: float, rest: float, rest_hurt: float,
                   hp_frac: float) -> tuple[int, bool]:
    """Argmax over logits plus the routing bias.  Returns (index, bias_applied)."""
    logits, _ = policy(obs)
    applied = False
    if int(obs["screen"]) == MAP_SCREEN and (elite or rest or rest_hurt):
        rooms = obs["action_target_rooms"]
        bias = torch.zeros_like(logits)
        rooms_t = torch.as_tensor(rooms, dtype=torch.long, device=logits.device)
        if elite:
            bias = bias + elite * (rooms_t == ROOM_ELITE).to(logits.dtype)
        rest_total = rest + rest_hurt * (1.0 - hp_frac)
        if rest_total:
            bias = bias + rest_total * (rooms_t == ROOM_REST).to(logits.dtype)
        if bool(torch.any(bias != 0)):
            applied = True
        logits = logits + bias
    return int(torch.argmax(logits)), applied


def _play(job: tuple[str, float, float, float, int]) -> dict:
    label, elite, rest, rest_hurt, seed = job
    torch = _STATE["torch"]
    policy = _STATE["policy"]

    from .whole_run_env import RunConfig, WholeRunEnv

    env = WholeRunEnv(RunConfig(
        ascension=_STATE["ascension"], combat_sims=_STATE["sims"],
        deterministic_combat=True, search_config_path=_STATE["search_config"]))
    obs = env.reset(seed)
    started = time.perf_counter()
    biased_decisions = 0
    map_decisions = 0
    elites_entered = 0
    # Tracked for the same reason as elites: the arms differ only in a logit
    # bias, so the only way to check that a REST bias actually changed REST
    # behaviour -- rather than moving floors through some third path -- is to
    # count the rooms chosen. Round 2 needed this and did not have it.
    rests_entered = 0
    prev_floor = int(env.gc.floor_num)
    with torch.inference_mode():
        while (env.gc.outcome.name == "UNDECIDED"
               and env.steps < env.config.max_decisions):
            if int(obs["screen"]) == MAP_SCREEN:
                map_decisions += 1
            hp_frac = float(env.gc.cur_hp) / max(1.0, float(env.gc.max_hp))
            action, applied = _biased_action(
                policy, obs, torch, elite, rest, rest_hurt, hp_frac)
            biased_decisions += int(applied)
            # The battle result carries `is_boss` but no elite flag, so elite
            # capture is read off the routing decision itself: choosing a node
            # whose room is ELITE is what fighting one means.
            if int(obs["screen"]) == MAP_SCREEN:
                room = int(obs["action_target_rooms"][action])
                if room == ROOM_ELITE:
                    elites_entered += 1
                elif room == ROOM_REST:
                    rests_entered += 1
            obs, _, done, _ = env.step(action)
            prev_floor = int(env.gc.floor_num)
            if done:
                break
    return {
        "arm": label, "elite_bias": elite, "rest_bias": rest,
        "rest_hurt_bias": rest_hurt, "seed": seed,
        "floor": int(env.gc.floor_num), "act": int(env.gc.act),
        "hp": int(env.gc.cur_hp), "outcome": str(env.gc.outcome),
        "battles": int(env.battles), "map_decisions": map_decisions,
        "biased_decisions": biased_decisions, "elites_entered": elites_entered,
        "rests_entered": rests_entered,
        "seconds": round(time.perf_counter() - started, 3),
        "final_floor": prev_floor,
    }


def summarize(rows: list[dict], baseline: str) -> str:
    by_arm: dict[str, dict[int, dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], {})[row["seed"]] = row
    base = by_arm.get(baseline, {})
    lines = [
        f"{'arm':>20}  {'n':>4}  {'floor':>6}  {'paired vs base':>16}  "
        f"{'t':>6}  {'W/T/L':>12}  {'elites':>7}  {'rests':>6}  {'wins':>5}"
    ]
    for arm, seeds in by_arm.items():
        floors = [r["floor"] for r in seeds.values()]
        elites = statistics.mean(r["elites_entered"] for r in seeds.values())
        rests = statistics.mean(r.get("rests_entered", 0) for r in seeds.values())
        wins = sum(1 for r in seeds.values() if "VICTORY" in r["outcome"])
        shared = sorted(set(seeds) & set(base))
        if arm == baseline or not shared:
            delta = "--"
            tstat = "--"
            wtl = "--"
        else:
            diffs = [seeds[s]["floor"] - base[s]["floor"] for s in shared]
            mean = statistics.mean(diffs)
            sem = (statistics.stdev(diffs) / math.sqrt(len(diffs))
                   if len(diffs) > 1 else float("nan"))
            delta = f"{mean:+.2f} +/-{sem:.2f}"
            tstat = f"{mean / sem:+.2f}" if sem else "inf"
            wtl = (f"{sum(1 for d in diffs if d > 0)}/"
                   f"{sum(1 for d in diffs if d == 0)}/"
                   f"{sum(1 for d in diffs if d < 0)}")
        lines.append(
            f"{arm:>20}  {len(floors):>4}  {statistics.mean(floors):>6.2f}  "
            f"{delta:>16}  {tstat:>6}  {wtl:>12}  {elites:>7.2f}  "
            f"{rests:>6.2f}  {wins:>5}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--runs", type=int, default=120)
    parser.add_argument("--seed-base", type=int, default=1_003_000)
    parser.add_argument("--sims", type=int, default=300)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--search-config", default=None)
    parser.add_argument("--arms", default=None,
                        help="comma-separated subset of sweep labels")
    parser.add_argument("--out", default="runs/route_bias_probe.jsonl")
    args = parser.parse_args()

    sweep = DEFAULT_SWEEP
    if args.arms:
        wanted = {name.strip() for name in args.arms.split(",")} | {"baseline"}
        sweep = [entry for entry in sweep if entry[0] in wanted]

    jobs = [(label, elite, rest, hurt, args.seed_base + offset)
            for label, elite, rest, hurt in sweep
            for offset in range(args.runs)]
    print(f"{len(sweep)} arms x {args.runs} seeds = {len(jobs)} runs "
          f"at {args.sims} sims on {args.workers} workers", flush=True)

    rows: list[dict] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_worker_init,
            initargs=(args.checkpoint, args.sims, args.ascension,
                      args.search_config)) as pool:
        for done, row in enumerate(pool.map(_play, jobs, chunksize=1), start=1):
            rows.append(row)
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"\nwrote {args.out} ({len(rows)} rows, "
          f"{time.perf_counter() - started:.0f}s)\n", flush=True)
    print(summarize(rows, "baseline"), flush=True)


if __name__ == "__main__":
    main()
