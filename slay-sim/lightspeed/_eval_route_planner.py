"""Paired floor eval: the route planner at map decisions, v31 everywhere else.

The layer swap isolated the run policy as worth +15.71 floors; this isolates the
ROUTING half of it, the same way `_routing_audit.py --randomize-paths` does but
substituting a planner instead of noise (randomizing costs 4.87 floors, so the
harness is known to be sensitive enough to see a routing change).

Read the floor delta and nothing else. The routing clone that died at floor 7.08
had validation NLL falling monotonically for 18 epochs and human agreement rising
65% -> 69% the entire time; every intermediate metric pointed the right way. This
eval costs ~90 seconds and is the only one that caught it.

Elite capture is reported alongside because it is the mechanism under test, not a
success criterion. If capture jumps and floors fall, that is the capability
mismatch recurring -- the planner's survival estimate would be the thing to
distrust, not the idea.

    python -m lightspeed._eval_route_planner --runs 120 --elite 3 --rest 4
"""
from __future__ import annotations

import argparse
import math
import statistics

import torch

import slaythespire as sts

from ._route_planner import RouteWeights, choose
from .eval_whole_run_policy import load_policy
from .search_config import DEFAULT_SEARCH_CONFIG_PATH
from .whole_run_env import RunConfig, WholeRunEnv

CKPT = "runs/whole_run_transformer_yield10x_a20_v31.pt"


def play(policy, seed: int, sims: int, ascension: int,
         weights: RouteWeights | None):
    """One run. `weights=None` is stock v31; otherwise the planner owns map picks."""
    env = WholeRunEnv(RunConfig(
        ascension=ascension, combat_sims=sims, deterministic_combat=True,
        search_config_path=DEFAULT_SEARCH_CONFIG_PATH))
    obs = env.reset(seed)
    elites_taken = elites_offered = planner_overrides = map_decisions = 0
    with torch.inference_mode():
        while (env.gc.outcome.name == "UNDECIDED"
               and env.steps < env.config.max_decisions):
            is_map = env.gc.screen_state == sts.ScreenState.MAP_SCREEN
            logits, _, auxiliary, _, _ = policy.forward_detailed(obs)
            action = int(torch.argmax(logits))
            if is_map:
                rooms = [int(r) for r in obs["action_target_rooms"]]
                if len(rooms) >= 2:
                    map_decisions += 1
                    elites_offered += sum(
                        1 for r in rooms if r == int(sts.Room.ELITE))
                    if weights is not None:
                        rep = sts.getNNRepresentation(env.gc)
                        planned = choose(env.gc, rep.map, int(rep.mapY),
                                         env.legal_actions(), auxiliary, weights)
                        if planned != action:
                            planner_overrides += 1
                        action = planned
                    if 0 <= action < len(rooms) and rooms[action] == int(sts.Room.ELITE):
                        elites_taken += 1
            obs, _, done, _ = env.step(action)
            if done:
                break
    return {
        "floor": int(env.gc.floor_num),
        "won": env.gc.outcome.name == "PLAYER_VICTORY",
        "elites_taken": elites_taken,
        "elites_offered": elites_offered,
        "overrides": planner_overrides,
        "map_decisions": map_decisions,
    }


def summarise(label: str, rows) -> float:
    floors = [r["floor"] for r in rows]
    taken = sum(r["elites_taken"] for r in rows)
    offered = sum(r["elites_offered"] for r in rows)
    capture = taken / offered if offered else 0.0
    print(f"  {label:22s} floor {statistics.mean(floors):6.2f}   "
          f"wins {sum(r['won'] for r in rows):2d}   "
          f"elites {taken:3d}/{offered:3d} ({capture:.1%})")
    return statistics.mean(floors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=120)
    parser.add_argument("--seed-base", type=int, default=71_000_000)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--elite", type=float, default=3.0)
    parser.add_argument("--rest", type=float, default=4.0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    policy = load_policy(CKPT, torch.device("cpu"))
    policy.eval()
    weights = RouteWeights(elite=args.elite, rest=args.rest)

    print(f"{args.runs} paired A20 seeds at {args.sims} sims, "
          f"elite={args.elite} rest={args.rest}")
    base = [play(policy, args.seed_base + i, args.sims, args.ascension, None)
            for i in range(args.runs)]
    summarise("v31 (stock)", base)
    planned = [play(policy, args.seed_base + i, args.sims, args.ascension, weights)
               for i in range(args.runs)]
    summarise("v31 + planner", planned)

    deltas = [p["floor"] - b["floor"] for p, b in zip(planned, base)]
    mean = statistics.mean(deltas)
    stderr = statistics.stdev(deltas) / math.sqrt(len(deltas))
    changed = sum(1 for d in deltas if d)
    overrides = sum(p["overrides"] for p in planned)
    decisions = sum(p["map_decisions"] for p in planned)
    print(f"\n  paired delta {mean:+.2f} +/- {stderr:.2f} floors "
          f"(t={mean/stderr if stderr else 0:+.2f}), {changed}/{len(deltas)} runs differ")
    print(f"  planner overrode v31 on {overrides}/{decisions} map decisions "
          f"({overrides/max(1,decisions):.1%})")


if __name__ == "__main__":
    main()
