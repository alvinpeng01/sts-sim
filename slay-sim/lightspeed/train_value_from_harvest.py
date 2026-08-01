"""Fit the transformer's value head on the free harvest rows.

`train_whole_run_v27.py` has four loss terms and **none of them touches
`self.value`** -- the critic has only ever been trained by
`pretrain_whole_run_value.py`, which pays for fresh episodes.  Meanwhile
`--harvest-rate 0.02` has been writing `(observation, action, return)` rows for
every decision *inside* a counterfactual continuation, already simulated and
therefore free: **82,861 rows in `runs/v37_trunc/`**, 20x the 4,008 the policy
head trains on.  They are correlated and slightly off-policy, which makes them
wrong for a policy head and right for a critic.

Method.  `value` reads only `state` -- the encoder's CLS token plus v27's two
adapters -- so with the trunk frozen every row's features can be computed once
and cached, after which fitting the head is seconds rather than hours, and a
ridge regression on the same features gives the head's achievable ceiling.  This
is the measurement 02-training-pipeline.md already applies to the auxiliary
heads, reused here.

Two hazards this script is built around:

1. **54.4% of harvest returns are bootstrapped** -- `--truncate-after` ended the
   branch early and the tail was estimated by the v28 model's own
   `terminal_floor` head.  Training a critic on those partly re-learns the old
   critic.  The default arm therefore fits observed terminal returns only, and
   `--include-bootstrapped` measures what the extra 45k rows are worth.
   Validation R2 is always reported on observed rows alone.
2. **Train/validation must not share episodes.**  Harvest rows carry no seed, so
   the split is taken from the generator's own dataset boundary: rows from
   `*_validation_*.harvest.pt` were produced by separate validation runs with
   their own seeds.

The target is `rollout_score` in its own units (mean 0.907, sd 1.344), the same
scale the labels use.  A PPO critic must predict the discounted return of
whatever reward its env emits, which is NOT this; what transfers is the
representation and a head that is already monotone in run outcome, with the
final scale re-fit against the RL reward.

Run from slay-sim/:
    python -m lightspeed.train_value_from_harvest --workers 6
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch import nn

DEFAULT_CHECKPOINT = "runs/whole_run_transformer_postfix-trunc_a20_v37.pt"
DEFAULT_GLOB = "runs/v37_trunc/*.harvest.pt"
FEATURE_CACHE = "runs/harvest_value_features.pt"

_STATE: dict = {}


def _worker_init(checkpoint: str) -> None:
    from .eval_whole_run_policy import load_policy

    torch.set_num_threads(1)
    _STATE["policy"] = load_policy(checkpoint, torch.device("cpu"))


def _state_features(policy, obs) -> np.ndarray:
    """The exact tensor `self.value` consumes, with the trunk frozen."""
    from .v27_features import augment_v27_observation

    device = next(policy.parameters()).device
    augmented = augment_v27_observation(obs)
    encoded = policy.encoder(policy._state_tokens(augmented, device))[0]
    state = encoded[0]
    state = state + policy.deck_summary_adapter(
        policy._tensor(augmented["v27_deck_summary"], torch.float32, device))
    state = state + policy.strategic_context_adapter(
        policy._tensor(augmented["v27_strategic_context"], torch.float32, device))
    return state.detach().numpy()


def _extract_shard(path: str) -> dict:
    policy = _STATE["policy"]
    payload = torch.load(path, weights_only=False, map_location="cpu")
    features, returns, bootstrapped, acts, types = [], [], [], [], []
    with torch.inference_mode():
        for row in payload["rows"]:
            features.append(_state_features(policy, row["observation"]))
            returns.append(float(row["return"]))
            bootstrapped.append(bool(row["bootstrapped"]))
            acts.append(int(row["act"]))
            types.append(row["decision_type"])
    return {
        "path": path,
        "split": "validation" if "_validation_" in os.path.basename(path) else "train",
        "features": np.stack(features).astype(np.float32),
        "returns": np.asarray(returns, dtype=np.float32),
        "bootstrapped": np.asarray(bootstrapped, dtype=bool),
        "acts": np.asarray(acts, dtype=np.int16),
        "types": types,
    }


def build_cache(checkpoint: str, pattern: str, workers: int, cache_path: str) -> list[dict]:
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise SystemExit(f"no harvest shards matched {pattern}")
    print(f"extracting frozen-trunk features from {len(shards)} shards "
          f"on {workers} workers", flush=True)
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(checkpoint,)) as pool:
        for done, shard in enumerate(pool.map(_extract_shard, shards), start=1):
            results.append(shard)
            print(f"  {done}/{len(shards)}  {os.path.basename(shard['path'])}  "
                  f"{len(shard['returns'])} rows "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)
    torch.save({"checkpoint": checkpoint, "shards": results}, cache_path)
    print(f"cached features -> {cache_path}", flush=True)
    return results


def assemble(shards: list[dict]) -> dict:
    out = {}
    for split in ("train", "validation"):
        chosen = [s for s in shards if s["split"] == split]
        out[split] = {
            "features": np.concatenate([s["features"] for s in chosen]),
            "returns": np.concatenate([s["returns"] for s in chosen]),
            "bootstrapped": np.concatenate([s["bootstrapped"] for s in chosen]),
            "acts": np.concatenate([s["acts"] for s in chosen]),
        }
    return out


def r2(prediction: np.ndarray, target: np.ndarray) -> float:
    residual = float(((prediction - target) ** 2).sum())
    total = float(((target - target.mean()) ** 2).sum())
    return 1.0 - residual / total if total > 0 else float("nan")


def ridge_ceiling(train_x, train_y, val_x, val_y, alpha: float = 1.0) -> float:
    """Closed-form linear probe: what these frozen features can support."""
    design = np.concatenate([train_x, np.ones((len(train_x), 1), np.float32)], axis=1)
    gram = design.T @ design + alpha * np.eye(design.shape[1], dtype=np.float32)
    weights = np.linalg.solve(gram, design.T @ train_y)
    val_design = np.concatenate([val_x, np.ones((len(val_x), 1), np.float32)], axis=1)
    return r2(val_design @ weights, val_y)


def fit_head(head: nn.Module, train_x, train_y, val_x, val_y, epochs: int,
             batch: int, lr: float, label: str) -> tuple[nn.Module, float]:
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    train_x_t = torch.from_numpy(train_x)
    train_y_t = torch.from_numpy(train_y)
    val_x_t = torch.from_numpy(val_x)
    best_state, best = None, -float("inf")
    for epoch in range(epochs):
        head.train()
        order = torch.randperm(len(train_x_t))
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            loss = nn.functional.mse_loss(
                head(train_x_t[index]).squeeze(-1), train_y_t[index])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            score = r2(head(val_x_t).squeeze(-1).numpy(), val_y)
        marker = ""
        if score > best:
            best, best_state = score, {k: v.clone() for k, v in head.state_dict().items()}
            marker = "  *"
        print(f"    {label} epoch {epoch + 1:>2}  val R2 {score:+.4f}{marker}", flush=True)
    if best_state is not None:
        head.load_state_dict(best_state)
    return head, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--harvest-glob", default=DEFAULT_GLOB)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cache", default=FEATURE_CACHE)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--include-bootstrapped", action="store_true",
                        help="also train on truncation-bootstrapped returns "
                             "(measured as a second arm by default)")
    parser.add_argument("--out", default=None,
                        help="write a copy of the checkpoint with the fitted "
                             "value head; omit to measure only")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.workers))
    torch.manual_seed(0)

    if os.path.exists(args.cache) and not args.rebuild_cache:
        payload = torch.load(args.cache, weights_only=False, map_location="cpu")
        if payload.get("checkpoint") != args.checkpoint:
            raise SystemExit(
                f"cache was built from {payload.get('checkpoint')}; "
                f"pass --rebuild-cache to redo it for {args.checkpoint}")
        shards = payload["shards"]
        print(f"loaded cached features from {args.cache}", flush=True)
    else:
        shards = build_cache(args.checkpoint, args.harvest_glob,
                             args.workers, args.cache)

    data = assemble(shards)
    train, validation = data["train"], data["validation"]
    val_observed = ~validation["bootstrapped"]
    val_x = validation["features"][val_observed]
    val_y = validation["returns"][val_observed]

    print(f"\ntrain rows {len(train['returns'])} "
          f"({int((~train['bootstrapped']).sum())} observed, "
          f"{int(train['bootstrapped'].sum())} bootstrapped)")
    print(f"validation rows {len(validation['returns'])} "
          f"-- scoring on {len(val_y)} observed only")
    print(f"target: mean {val_y.mean():+.3f} sd {val_y.std():.3f}\n")

    arms = [("observed-only", ~train["bootstrapped"])]
    if args.include_bootstrapped or True:
        arms.append(("all-rows", np.ones(len(train["returns"]), dtype=bool)))

    dim = train["features"].shape[1]
    results = {}
    for label, mask in arms:
        train_x = train["features"][mask]
        train_y = train["returns"][mask]
        ceiling = ridge_ceiling(train_x, train_y, val_x, val_y)
        print(f"  {label}: {len(train_y)} rows, "
              f"linear-probe ceiling val R2 {ceiling:+.4f}")
        head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        head, best = fit_head(head, train_x, train_y, val_x, val_y,
                              args.epochs, args.batch, args.lr, label)
        results[label] = (best, ceiling, head)
        print(f"  {label}: best val R2 {best:+.4f} "
              f"(ceiling {ceiling:+.4f})\n", flush=True)

    print(f"{'arm':>16}  {'val R2':>8}  {'ceiling':>8}  {'train mean':>10}")
    for label, (best, ceiling, _) in results.items():
        mask = dict(arms)[label]
        baseline = r2(np.full_like(val_y, train["returns"][mask].mean()), val_y)
        print(f"{label:>16}  {best:>+8.4f}  {ceiling:>+8.4f}  "
              f"{baseline:>+10.4f}")

    # The scoreboard above is not a fair comparison and must not be read as one.
    # Validation is observed-only because bootstrapped targets are the old
    # critic's output, and observed rows are >99% deaths -- so the `all-rows`
    # arm is trained on a population (mean +1.80) that the validation subset
    # (mean -0.13) does not represent, and its R2 measures that shift rather
    # than any fit. See the module docstring: truncated harvest cannot certify
    # a critic in either direction. Use `collect_run_value_data.py`.
    print("\nNOTE: observed rows are >99% deaths (censoring is correlated with"
          "\nthe target), so neither arm above is a usable critic. This script"
          "\nexists to document that; collect_run_value_data.py replaces it.")

    if args.out:
        best_label = max(results, key=lambda k: results[k][0])
        head = results[best_label][2]
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        for source, destination in (("0", "value.0"), ("2", "value.2")):
            state[f"{destination}.weight"] = head.state_dict()[f"{source}.weight"]
            state[f"{destination}.bias"] = head.state_dict()[f"{source}.bias"]
        torch.save(state, args.out)
        print(f"\nwrote {args.out} with the {best_label} value head "
              f"(val R2 {results[best_label][0]:+.4f}); every other tensor is "
              f"{os.path.basename(args.checkpoint)} unchanged")


if __name__ == "__main__":
    main()
