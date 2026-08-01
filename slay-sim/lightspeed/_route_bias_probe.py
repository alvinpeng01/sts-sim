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
    ("hurtrest+5", 0.0, 0.0, 5.0),
    ("rest+1.5,hurtrest+3", 0.0, 1.5, 3.0),
]

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
                if int(obs["action_target_rooms"][action]) == ROOM_ELITE:
                    elites_entered += 1
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
        f"{'t':>6}  {'W/T/L':>12}  {'elites':>7}  {'wins':>5}"
    ]
    for arm, seeds in by_arm.items():
        floors = [r["floor"] for r in seeds.values()]
        elites = statistics.mean(r["elites_entered"] for r in seeds.values())
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
            f"{delta:>16}  {tstat:>6}  {wtl:>12}  {elites:>7.2f}  {wins:>5}")
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
