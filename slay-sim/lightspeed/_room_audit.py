"""Count what rooms a policy actually enters, per act.

The map features already expose elite counts and distances to the policy
(`whole_run_env.py:95-159`), but no evaluation records how many elites a run
actually fights. Elites are the main relic source, so under-fighting them
produces a deck that cannot clear act bosses — which is where 47.5% of deaths
land. This makes the routing behaviour visible.
"""
from __future__ import annotations

import argparse
import collections
import json

import torch

import slaythespire as sts
from .eval_whole_run_policy import load_policy
from .whole_run_env import RunConfig, WholeRunEnv
from .search_config import DEFAULT_SEARCH_CONFIG_PATH


def max_elites_available(map_rep, from_y: int) -> int:
    """Most elites reachable on any path from row `from_y` to the act boss.

    This is the denominator the elite count needs. Without it, "0.7 elites in
    act 1" cannot distinguish a policy that dodges elites from maps that never
    offered them cheaply. Mirrors the max_elites recurrence in
    `whole_run_env.map_route_features`.
    """
    xs = [int(v) for v in map_rep.xs]
    ys = [int(v) for v in map_rep.ys]
    rooms = [int(v) for v in map_rep.room_types]
    import numpy as np
    paths = np.asarray(map_rep.path_xs, dtype=np.int16).reshape((-1, 3))
    index = {(x, y): i for i, (x, y) in enumerate(zip(xs, ys))}
    successors: list[list[int]] = [[] for _ in xs]
    for i, (y, edges) in enumerate(zip(ys, paths)):
        for edge_x in edges:
            child = index.get((int(edge_x), y + 1))
            if int(edge_x) >= 0 and child is not None:
                successors[i].append(child)
    elite = int(sts.Room.ELITE)
    best = [0] * len(xs)
    for i in sorted(range(len(xs)), key=lambda k: ys[k], reverse=True):
        children = successors[i]
        best[i] = int(rooms[i] == elite) + (
            max(best[j] for j in children) if children else 0)
    # At act entry mapY is -1 (no node chosen yet), so start from the first row.
    start_y = from_y if from_y >= 0 else (min(ys) if ys else 0)
    reachable = [best[i] for i in range(len(xs)) if ys[i] == start_y]
    return max(reachable) if reachable else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--seed-base", type=int, default=18_900_000)
    parser.add_argument("--sims", type=int, default=300)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cpu")

    elite = int(sts.Room.ELITE)
    records = []
    for path in args.checkpoints:
        policy = load_policy(path, device)
        per_run = []
        for offset in range(args.runs):
            env = WholeRunEnv(RunConfig(
                ascension=args.ascension, combat_sims=args.sims,
                deterministic_combat=True,
                search_config_path=DEFAULT_SEARCH_CONFIG_PATH))
            obs = env.reset(args.seed_base + offset)
            rooms = collections.Counter()
            elites_by_act = collections.Counter()
            available_by_act = {}
            last = None
            with torch.inference_mode():
                while (env.gc.outcome.name == "UNDECIDED"
                       and env.steps < env.config.max_decisions):
                    node = (int(env.gc.cur_map_node_x), int(env.gc.cur_map_node_y),
                            int(env.gc.act))
                    room = int(env.gc.cur_room)
                    # Count each map node once; a node is revisited across the
                    # several decisions a single room can present.
                    if node != last:
                        rooms[sts.Room(room).name] += 1
                        if room == elite:
                            elites_by_act[int(env.gc.act)] += 1
                        last = node
                    act_now = int(env.gc.act)
                    if act_now not in available_by_act:
                        try:
                            rep = sts.getNNRepresentation(env.gc)
                            available_by_act[act_now] = max_elites_available(
                                rep.map, int(rep.mapY))
                        except Exception:
                            available_by_act[act_now] = -1
                    action, _, _, _ = policy.act(obs, sample=False)
                    obs, _, done, _ = env.step(action)
                    if done:
                        break
            per_run.append({
                "checkpoint": path.split("/")[-1], "seed": args.seed_base + offset,
                "floor": int(env.gc.floor_num), "act": int(env.gc.act),
                "elites_total": sum(elites_by_act.values()),
                "elites_act1": elites_by_act.get(1, 0),
                "elites_act2": elites_by_act.get(2, 0),
                "elites_act3": elites_by_act.get(3, 0),
                "available_act1": available_by_act.get(1, -1),
                "available_act2": available_by_act.get(2, -1),
                "rooms": dict(rooms),
            })
        records.extend(per_run)
        n = len(per_run)
        reached2 = [r for r in per_run if r["act"] >= 2]
        print(f"\n{path.split('/')[-1]}  ({n} runs)")
        print(f"  elites/run           {sum(r['elites_total'] for r in per_run)/n:.2f}")
        print(f"  elites in act 1      {sum(r['elites_act1'] for r in per_run)/n:.2f}")
        if reached2:
            print(f"  elites in act 2      "
                  f"{sum(r['elites_act2'] for r in reached2)/len(reached2):.2f}"
                  f"   (over {len(reached2)} runs that reached act 2)")
        avail = [r for r in per_run if r["available_act1"] >= 0]
        if avail:
            took = sum(r["elites_act1"] for r in avail)
            offered = sum(r["available_act1"] for r in avail)
            print(f"  act-1 elites TAKEN {took} of {offered} AVAILABLE "
                  f"({100*took/max(1,offered):.0f}% capture, over {len(avail)} runs)")
            adist = collections.Counter(r["available_act1"] for r in avail)
            print("  act-1 elites available distribution: "
                  + " ".join(f"{k}:{adist[k]}" for k in sorted(adist)))
        dist = collections.Counter(r["elites_act1"] for r in per_run)
        print("  act-1 elite count distribution: "
              + " ".join(f"{k}:{dist[k]}" for k in sorted(dist)))
        agg = collections.Counter()
        for r in per_run:
            agg.update(r["rooms"])
        print("  rooms/run: " + " ".join(
            f"{k}={v/n:.1f}" for k, v in sorted(agg.items(), key=lambda kv: -kv[1])))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
