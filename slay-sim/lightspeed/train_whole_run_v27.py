"""Train isolated v27 experts, structured adapters, and auxiliary heads."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import math
import os
import random

import torch
from torch.nn import functional as F

from .whole_run_transformer_v27 import WholeRunTransformerPolicyV27


SURVIVAL_TARGETS = {
    "next_combat_survival", "next_rest_reach", "act_boss_survival"}


SCRATCH = "scratch"


def load_model(path: str, device, dim: int = 96, layers: int = 2, heads: int = 4):
    """Build the model, warm-starting from `path` unless it is the SCRATCH sentinel.

    A dim mismatch is fatal. Swallowing it produced the v30 run: the weights
    cold-started, and because the anchor is a deepcopy of whatever this returns,
    the KL anchor term spent every epoch pulling the model toward a *random*
    reference — inverting the regularizer it exists to provide. Pass
    `--load scratch --anchor-weight 0` to train from scratch on purpose.
    """
    model = WholeRunTransformerPolicyV27(dim=dim, layers=layers, heads=heads).to(device)
    if path == SCRATCH:
        print(f"v27 fresh init dim={dim} layers={layers} heads={heads}", flush=True)
        return model
    try:
        missing, unexpected = model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True), strict=False)
    except RuntimeError as error:
        raise RuntimeError(
            f"warm-start from {path} is incompatible with "
            f"dim={dim} layers={layers} heads={heads}: {error}. "
            f"Project a compatible checkpoint, match the checkpoint's "
            f"architecture, or pass --load {SCRATCH} --anchor-weight 0 to "
            f"train from scratch deliberately.") from error
    if missing or unexpected:
        print(f"compatible v27 load new={missing} unused={unexpected}", flush=True)
    return model


def normalized_observation(row):
    obs = row["observation"]
    if "act" not in obs or "floor" not in obs:
        obs = dict(obs)
        obs.setdefault("act", int(row.get("act", 1)))
        obs.setdefault("floor", int(row.get("floor", 0)))
    return obs


def bootstrap_member(row, member: int) -> bool:
    identity = (
        f"{row.get('seed', 0)}:{row.get('floor', 0)}:"
        f"{row.get('decision_type', '')}:{member}").encode()
    return hashlib.blake2b(identity, digest_size=1).digest()[0] < 192


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--validation-dataset", nargs="+", required=True)
    parser.add_argument("--load", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--batch", type=int, default=32,
        help="rows accumulated per optimizer step; 1 reproduces the pre-v31 "
             "behaviour, which could not fit its training data")
    parser.add_argument(
        "--warmup-fraction", type=float, default=0.05,
        help="share of total steps spent linearly warming the learning rate")
    parser.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="also write <out>_ep<N>.pt every N epochs. Validation NLL has "
             "picked the wrong epoch three times running (v30, v32, v33), so "
             "the only reliable selector is a paired floor evaluation across "
             "epochs — which needs the intermediate weights kept")
    parser.add_argument("--anchor-weight", type=float, default=0.15)
    parser.add_argument("--ensemble-weight", type=float, default=0.25)
    parser.add_argument("--auxiliary-weight", type=float, default=0.20)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--scope",
        choices=("experts", "experts-structure", "human-adapter", "all-v27",
                 "full"),
        default="experts-structure")
    parser.add_argument("--seed", type=int, default=7_100_000)
    args = parser.parse_args()
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def rows_from(paths):
        rows = []
        for path in paths:
            rows.extend(torch.load(
                path, map_location="cpu", weights_only=False)["rows"])
        return rows

    rows = rows_from(args.dataset)
    validation = rows_from(args.validation_dataset)
    if not rows or not validation:
        raise RuntimeError("v27 requires non-empty train and validation rows")
    if args.load == SCRATCH and args.anchor_weight != 0.0:
        raise RuntimeError(
            "a from-scratch model has no reference to anchor against; "
            "pass --anchor-weight 0 with --load scratch")
    counts = Counter(row["decision_type"] for row in rows)
    model = load_model(args.load, device, dim=args.dim, layers=args.layers, heads=args.heads)
    anchor = copy.deepcopy(model).eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    prefixes = ["decision_experts.", "uncertainty_heads."]
    if args.scope in ("experts-structure", "all-v27"):
        prefixes.extend((
            "deck_summary_adapter.", "strategic_context_adapter.",
            "action_card_structure_adapter."))
    if args.scope == "all-v27":
        prefixes.append("auxiliary_heads.")
    elif args.scope == "human-adapter":
        # Human demonstrations currently cover reward and boss-relic screens.
        # Isolate their zero-initialized residual so imitation training cannot
        # overwrite V28's search-trained experts or shared representation.
        prefixes = ["human_score."]
    for name, parameter in model.named_parameters():
        # `full` trains the embeddings, transformer trunk, and score/value heads
        # alongside the v27 residuals. Every other scope freezes them — which is
        # correct when the trunk arrives fitted from the previous version, and
        # ruinous when it does not: v30 cold-started a random trunk and then
        # froze 76% of the model, leaving the residuals to learn on top of random
        # features. Use this scope whenever the trunk is not inherited.
        parameter.requires_grad_(
            args.scope == "full"
            or any(name.startswith(prefix) for prefix in prefixes))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"scope={args.scope} trainable={trainable:,}/{total:,} "
        f"({100 * trainable / total:.1f}%)", flush=True)
    if args.batch < 1:
        raise ValueError("--batch must be positive")
    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(rows) / args.batch)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_fraction))

    def learning_rate_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    print(
        f"batch={args.batch} steps/epoch={steps_per_epoch} "
        f"total_steps={total_steps} warmup={warmup_steps} lr={args.lr}",
        flush=True)
    best_nll = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        random.shuffle(rows)
        model.train()
        totals = Counter()
        optimizer.zero_grad()
        for index, row in enumerate(rows):
            obs = normalized_observation(row)
            target = torch.as_tensor(
                row["target_probabilities"], dtype=torch.float32, device=device)
            logits, _, auxiliary, _, ensemble = model.forward_detailed(obs)
            with torch.no_grad():
                anchor_logits, _ = anchor(obs)
            policy_loss = -(target * F.log_softmax(logits, dim=-1)).sum()
            anchor_loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(anchor_logits, dim=-1), reduction="sum")
            ensemble_loss = torch.zeros((), device=device)
            used_members = 0
            for member in range(len(ensemble)):
                if bootstrap_member(row, member):
                    ensemble_loss = ensemble_loss - (
                        target * F.log_softmax(
                            ensemble[member], dim=-1)).sum()
                    used_members += 1
            ensemble_loss = ensemble_loss / max(1, used_members)
            auxiliary_loss = torch.zeros((), device=device)
            auxiliary_count = 0
            for name, value in row.get("auxiliary_targets", {}).items():
                if name not in auxiliary:
                    continue
                expected = torch.as_tensor(
                    value, dtype=torch.float32, device=device)
                if name in SURVIVAL_TARGETS:
                    auxiliary_loss = auxiliary_loss + F.binary_cross_entropy_with_logits(
                        auxiliary[name], expected)
                else:
                    auxiliary_loss = auxiliary_loss + F.mse_loss(
                        auxiliary[name], expected)
                auxiliary_count += 1
            auxiliary_loss = auxiliary_loss / max(1, auxiliary_count)
            type_weight = len(rows) / (
                len(counts) * counts[row["decision_type"]])
            loss = type_weight * (
                policy_loss + args.anchor_weight * anchor_loss
                + args.ensemble_weight * ensemble_loss
                + args.auxiliary_weight * auxiliary_loss)
            # Accumulate across `--batch` rows before stepping. At batch 1 the
            # per-row type_weight scales the entire step, and AdamW's m/sqrt(v)
            # cancels a constant gradient scale — so the 76x spread between
            # boss_relic and rewards did essentially nothing. Weights only bite
            # when they differ *within* an accumulation window.
            (loss / args.batch).backward()
            if (index + 1) % args.batch == 0 or index + 1 == len(rows):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            totals["loss"] += float(loss)
            totals["policy"] += float(policy_loss)
            totals["anchor"] += float(anchor_loss)
            totals["correct"] += int(
                torch.argmax(logits) == torch.argmax(target))

        model.eval()
        validation_nll = 0.0
        validation_correct = 0
        mean_uncertainty = 0.0
        with torch.no_grad():
            for row in validation:
                target = torch.as_tensor(
                    row["target_probabilities"],
                    dtype=torch.float32, device=device)
                logits, _, _, uncertainty, _ = model.forward_detailed(
                    normalized_observation(row))
                validation_nll += float(
                    -(target * F.log_softmax(logits, dim=-1)).sum())
                validation_correct += int(
                    torch.argmax(logits) == torch.argmax(target))
                mean_uncertainty += float(uncertainty.mean())
        validation_nll /= len(validation)
        mean_uncertainty /= len(validation)
        if validation_nll < best_nll:
            best_nll = validation_nll
            best_state = copy.deepcopy(model.state_dict())
        if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
            stamp = f"{os.path.splitext(args.out)[0]}_ep{epoch + 1}.pt"
            os.makedirs(os.path.dirname(stamp) or ".", exist_ok=True)
            torch.save(model.state_dict(), stamp)
            print(f"checkpoint={stamp}", flush=True)
        print(
            f"epoch={epoch} loss={totals['loss']/len(rows):.4f} "
            f"policy={totals['policy']/len(rows):.4f} "
            f"anchor={totals['anchor']/len(rows):.4f} "
            f"argmax={totals['correct']}/{len(rows)} "
            f"val_nll={validation_nll:.4f} "
            f"val_argmax={validation_correct}/{len(validation)} "
            f"val_uncertainty={mean_uncertainty:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(best_state, args.out)
    # Also keep the final-epoch weights. Selection is by validation NLL, which
    # this project has repeatedly measured as a poor proxy for mean floor — v29
    # improved NLL and lost 0.983 floors; v31 barely moved NLL and gained 2.60.
    # v32 was early-stopped at epoch 7 of 30 on that metric, so the weights that
    # actually did its fitting were discarded and could not be evaluated. Saving
    # both lets a paired floor comparison decide which checkpoint to promote.
    final_path = f"{os.path.splitext(args.out)[0]}_final.pt"
    torch.save(model.state_dict(), final_path)
    print(
        f"saved={args.out} best_val_nll={best_nll:.4f} "
        f"train={len(rows)} validation={len(validation)}", flush=True)
    print(f"saved_final={final_path} (last epoch, not val-selected)", flush=True)


if __name__ == "__main__":
    main()
