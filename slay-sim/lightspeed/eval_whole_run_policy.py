"""Matched native-MCTS evaluation for self-trained whole-run policies."""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .whole_run_env import RunConfig, WholeRunEnv
from .search_config import DEFAULT_SEARCH_CONFIG_PATH
from .whole_run_transformer import WholeRunTransformerPolicy
from .whole_run_transformer_v27 import WholeRunTransformerPolicyV27


# Attention head count leaves no trace in the state dict — nn.MultiheadAttention
# packs qkv as (3*dim, dim) whatever the head count is — so it cannot be inferred
# and a wrong value loads silently while computing different attention. These are
# the head counts every checkpoint in the lineage was trained with, keyed by dim.
HEADS_BY_DIM = {96: 4, 192: 6}


def checkpoint_architecture(state) -> tuple[int, int, int]:
    """Recover (dim, layers, heads) from a checkpoint's own tensor shapes.

    Each checkpoint must be evaluated at the architecture it was trained at.
    Taking these from a caller-supplied CLI value means one wrong flag silently
    drops a baseline from a paired comparison, which is exactly how the v28
    baseline vanished from the v28-vs-v30 evaluation.
    """
    dim = state["card.weight"].shape[1]
    layers = 1 + max(
        int(key.split(".")[2]) for key in state if key.startswith("encoder.layers."))
    if dim not in HEADS_BY_DIM:
        raise RuntimeError(
            f"unknown dim={dim}; add its head count to HEADS_BY_DIM")
    return dim, layers, HEADS_BY_DIM[dim]


def load_policy(path: str, device):
    state = torch.load(path, map_location=device, weights_only=True)
    policy_class = (
        WholeRunTransformerPolicyV27
        if any(key.startswith("decision_experts.") for key in state)
        else WholeRunTransformerPolicy
    )
    dim, layers, heads = checkpoint_architecture(state)
    policy = policy_class(dim=dim, layers=layers, heads=heads).to(device)
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"{path}: compatible load new={missing} unused={unexpected}", flush=True)
    print(
        f"{path}: architecture={policy_class.__name__} "
        f"dim={dim} layers={layers} heads={heads}", flush=True)
    return policy.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="one or more policy checkpoints")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=1_003_000)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument(
        "--search-config", default=None,
        help="optional isolated native-search configuration artifact")
    parser.add_argument(
        "--torch-threads", type=int, default=1,
        help="CPU intra-op threads; one is fastest for this small policy")
    parser.add_argument(
        "--stochastic-combat", action="store_true",
        help="use random MCTS search seeds; deterministic paired combat is the default")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--sample", action="store_true",
        help="sample actions instead of taking the argmax. The net has only "
             "learned marginals on some screens, where argmax collapses a "
             "41/35 split into 100/0 behaviour")
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="sampling temperature; <1 sharpens toward argmax")
    parser.add_argument(
        "--policy-seed", type=int, default=0,
        help="seed for action sampling so --sample runs stay reproducible")
    args = parser.parse_args()
    if args.sample:
        torch.manual_seed(args.policy_seed)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []
    for path in args.checkpoints:
        policy = load_policy(path, device)
        rows = []
        for offset in range(args.runs):
            env = WholeRunEnv(RunConfig(
                ascension=args.ascension, combat_sims=args.sims,
                deterministic_combat=not args.stochastic_combat,
                search_config_path=(args.search_config
                                    if args.search_config is not None
                                    else DEFAULT_SEARCH_CONFIG_PATH)))
            obs = env.reset(args.seed_base + offset)
            started = time.perf_counter()
            with torch.inference_mode():
                while (env.gc.outcome.name == "UNDECIDED"
                       and env.steps < env.config.max_decisions):
                    action, _, _, _ = policy.act(
                        obs, sample=args.sample, temperature=args.temperature)
                    obs, _, done, _ = env.step(action)
                    if done:
                        break
            row = {"checkpoint": os.path.basename(path), "seed": args.seed_base + offset,
                   "floor": env.gc.floor_num, "act": env.gc.act, "hp": env.gc.cur_hp,
                   "outcome": str(env.gc.outcome), "seconds": round(time.perf_counter() - started, 3),
                   **env.combat_audit}
            rows.append(row); all_rows.append(row)
            print(json.dumps(row), flush=True)
        summary = {"checkpoint": os.path.basename(path), "runs": len(rows),
                   "mean_floor": sum(row["floor"] for row in rows) / len(rows),
                   "median_floor": sorted(row["floor"] for row in rows)[len(rows) // 2],
                   "act2_plus": sum(row["act"] >= 2 for row in rows),
                   "act3_plus": sum(row["act"] >= 3 for row in rows),
                   "act4_plus": sum(row["act"] >= 4 for row in rows),
                   "early_floor_10_or_less": sum(row["floor"] <= 10 for row in rows),
                   "victories": sum("VICTORY" in row["outcome"] for row in rows),
                   "mean_seconds": sum(row["seconds"] for row in rows) / len(rows),
                   "search_simulations_total": sum(
                       row["search_simulations_total"] for row in rows),
                   "stall_fallback_decisions": sum(
                       row["stall_fallback_decisions"] for row in rows),
                   "stall_progress_override_decisions": sum(
                       row.get("stall_progress_override_decisions", 0) for row in rows),
                   "soft_tempo_override_decisions": sum(
                       row.get("soft_tempo_override_decisions", 0) for row in rows),
                   "stall_recovery_search_decisions": sum(
                       row.get("stall_recovery_search_decisions", 0) for row in rows),
                   "turn_limit_battles": sum(
                       row["turn_limit_battles"] for row in rows),
                   "safety_filter_events": sum(
                       row["safety_filter_events"] for row in rows),
                   "immediate_loss_actions_filtered": sum(
                       row["immediate_loss_actions_filtered"] for row in rows)}
        print(json.dumps({"summary": summary}), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            for row in all_rows:
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
