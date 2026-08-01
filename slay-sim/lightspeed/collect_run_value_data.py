"""Collect uncensored critic data from our own full runs, and fit the value head.

Why not the harvest rows.  `--harvest-rate` writes a free
`(observation, action, return)` row for every decision inside a counterfactual
continuation, and 02-training-pipeline.md recommends them for the value and
auxiliary heads.  Measured 2026-07-31 on `runs/v37_trunc/` (82,861 rows), that
recommendation does not hold for a **truncated** generation:

    observed returns      n=30794  mean -0.096  sd 0.375   (train)
    bootstrapped returns  n=37646  mean +1.798  sd 1.267

54.4% of the returns were estimated by the previous model's own
`terminal_floor` head, so training on them re-learns the old critic.  The
remaining 46% are not a random sample -- a branch has a true terminal return
precisely when it ENDED inside the 20-decision window, and over 99% of those
sit at `-0.4 + 0.1 x floors_gained`, i.e. they are deaths.  The censoring is
perfectly correlated with the target, and there is no uncensored held-out
subset left to validate against.  Truncated harvest cannot certify a critic.

What this does instead.  Play complete runs under the current policy and label
every overworld state with the return actually realized from it, which is
uncensored by construction and on-policy by construction.  The target is the
undiscounted sum of `WholeRunEnv`'s own step rewards from that state onward
(gamma = 1, episodic), so a critic fitted here predicts the quantity PPO's
advantage estimator needs rather than a proxy:

    return(s) = 0.01 * (final_floor - floor_s)
              + 0.002 * (final_hp - hp_s)
              + (+1 victory | -1 loss)

Features are the frozen trunk's `state` tensor -- the exact input `self.value`
consumes -- computed in the worker so runs stream as 96 floats rather than
observation dicts.  Train and validation use disjoint seed ranges.

This is also the collection half of the PPO loop: same env, same policy
snapshot, same per-worker structure, one extra tensor per step.

Run from slay-sim/:
    python -m lightspeed.collect_run_value_data --runs 3000 --workers 6
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch import nn

from .train_value_from_harvest import fit_head, r2, ridge_ceiling

DEFAULT_CHECKPOINT = "runs/whole_run_transformer_postfix-trunc_a20_v37.pt"

_STATE: dict = {}


def _worker_init(checkpoint: str, sims: int, ascension: int) -> None:
    from .eval_whole_run_policy import load_policy
    from .search_config import DEFAULT_SEARCH_CONFIG_PATH

    torch.set_num_threads(1)
    _STATE["policy"] = load_policy(checkpoint, torch.device("cpu"))
    _STATE["sims"] = sims
    _STATE["ascension"] = ascension
    _STATE["search_config"] = DEFAULT_SEARCH_CONFIG_PATH


def _play(seed: int) -> dict:
    from .train_value_from_harvest import _state_features
    from .whole_run_env import RunConfig, WholeRunEnv

    policy = _STATE["policy"]
    env = WholeRunEnv(RunConfig(
        ascension=_STATE["ascension"], combat_sims=_STATE["sims"],
        deterministic_combat=True, search_config_path=_STATE["search_config"]))
    obs = env.reset(seed)
    features, floors, hps, acts, screens = [], [], [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        while (env.gc.outcome.name == "UNDECIDED"
               and env.steps < env.config.max_decisions):
            features.append(_state_features(policy, obs))
            floors.append(int(env.gc.floor_num))
            hps.append(int(env.gc.cur_hp))
            acts.append(int(env.gc.act))
            screens.append(int(obs["screen"]))
            action, _, _, _ = policy.act(obs, sample=False)
            obs, _, done, _ = env.step(action)
            if done:
                break

    final_floor, final_hp = int(env.gc.floor_num), int(env.gc.cur_hp)
    outcome = env.gc.outcome.name
    terminal = (1.0 if outcome == "PLAYER_VICTORY"
                else -1.0 if outcome == "PLAYER_LOSS" else 0.0)
    floors_a = np.asarray(floors, dtype=np.float32)
    hps_a = np.asarray(hps, dtype=np.float32)
    returns = (0.01 * (final_floor - floors_a)
               + 0.002 * (final_hp - hps_a) + terminal).astype(np.float32)
    return {
        "seed": seed,
        "features": (np.stack(features).astype(np.float32) if features
                     else np.zeros((0, 96), np.float32)),
        "returns": returns,
        "floors": np.asarray(floors, dtype=np.int16),
        "acts": np.asarray(acts, dtype=np.int16),
        "screens": np.asarray(screens, dtype=np.int16),
        "final_floor": final_floor,
        "outcome": outcome,
        "seconds": time.perf_counter() - started,
    }


def collect(checkpoint: str, seeds: list[int], sims: int, ascension: int,
            workers: int) -> list[dict]:
    print(f"collecting {len(seeds)} runs at {sims} sims on {workers} workers",
          flush=True)
    started = time.perf_counter()
    episodes = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(checkpoint, sims, ascension)) as pool:
        for done, episode in enumerate(pool.map(_play, seeds, chunksize=4),
                                       start=1):
            episodes.append(episode)
            if done % 200 == 0:
                rows = sum(len(e["returns"]) for e in episodes)
                floor = np.mean([e["final_floor"] for e in episodes])
                print(f"  {done}/{len(seeds)} runs, {rows} rows, "
                      f"mean floor {floor:.2f} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
    return episodes


def stack(episodes: list[dict]) -> dict:
    return {
        "features": np.concatenate([e["features"] for e in episodes]),
        "returns": np.concatenate([e["returns"] for e in episodes]),
        "floors": np.concatenate([e["floors"] for e in episodes]),
        "acts": np.concatenate([e["acts"] for e in episodes]),
        "screens": np.concatenate([e["screens"] for e in episodes]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--runs", type=int, default=3000)
    parser.add_argument("--seed-base", type=int, default=7_000_000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--sims", type=int, default=100,
                        help="floors are flat from 100 to 1500 sims, so 100 is "
                             "the cheapest budget that does not change the task")
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cache", default="runs/run_value_data.pt")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    torch.manual_seed(0)

    if args.reuse_cache and os.path.exists(args.cache):
        payload = torch.load(args.cache, weights_only=False, map_location="cpu")
        episodes = payload["episodes"]
        print(f"reusing {len(episodes)} cached episodes from {args.cache}")
    else:
        seeds = [args.seed_base + i for i in range(args.runs)]
        episodes = collect(args.checkpoint, seeds, args.sims,
                           args.ascension, args.workers)
        torch.save({"checkpoint": args.checkpoint, "episodes": episodes},
                   args.cache)
        print(f"cached episodes -> {args.cache}", flush=True)

    # Split by SEED, not by row: states inside one run share a terminal
    # outcome, so a row-level split leaks the answer across the boundary.
    cut = int(len(episodes) * (1.0 - args.validation_fraction))
    train, validation = stack(episodes[:cut]), stack(episodes[cut:])
    torch.set_num_threads(max(1, args.workers))

    floors = [e["final_floor"] for e in episodes]
    wins = sum(1 for e in episodes if e["outcome"] == "PLAYER_VICTORY")
    print(f"\n{len(episodes)} runs: mean floor {np.mean(floors):.2f}, "
          f"{wins} victories, "
          f"{np.mean([e['seconds'] for e in episodes]):.2f}s/run")
    print(f"rows: {len(train['returns'])} train / "
          f"{len(validation['returns'])} validation "
          f"({len(train['returns']) / cut:.1f} per run)")
    print(f"target: train mean {train['returns'].mean():+.3f} "
          f"sd {train['returns'].std():.3f} | "
          f"val mean {validation['returns'].mean():+.3f} "
          f"sd {validation['returns'].std():.3f}\n")

    val_x, val_y = validation["features"], validation["returns"]
    # A ridge probe is a *ceiling* only for a linear head, which is what the
    # auxiliary heads are. `value` is a 2-layer MLP, so this is a linear
    # BASELINE it is expected to beat -- and does, by a wide margin, because
    # remaining progress is not a linear function of these features.
    linear = ridge_ceiling(train["features"], train["returns"], val_x, val_y)
    print(f"linear-probe baseline: val R2 {linear:+.4f}")

    dim = train["features"].shape[1]
    head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
    head, best = fit_head(head, train["features"], train["returns"],
                          val_x, val_y, args.epochs, args.batch, args.lr,
                          "value")

    baseline = r2(np.full_like(val_y, train["returns"].mean()), val_y)
    print(f"\nvalue head : val R2 {best:+.4f}")
    print(f"linear probe: val R2 {linear:+.4f}")
    print(f"predict-mean: val R2 {baseline:+.4f}")

    with torch.inference_mode():
        prediction = head(torch.from_numpy(val_x)).squeeze(-1).numpy()
    print(f"\n{'act':>4}  {'n':>6}  {'R2':>8}")
    for act in sorted(set(validation["acts"].tolist())):
        mask = validation["acts"] == act
        if mask.sum() > 50:
            print(f"{act:>4}  {int(mask.sum()):>6}  {r2(prediction[mask], val_y[mask]):>+8.4f}")

    if args.out:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        for source, destination in (("0", "value.0"), ("2", "value.2")):
            state[f"{destination}.weight"] = head.state_dict()[f"{source}.weight"]
            state[f"{destination}.bias"] = head.state_dict()[f"{source}.bias"]
        torch.save(state, args.out)
        print(f"\nwrote {args.out}: {os.path.basename(args.checkpoint)} with a "
              f"value head fitted to on-policy returns (val R2 {best:+.4f})")


if __name__ == "__main__":
    main()
