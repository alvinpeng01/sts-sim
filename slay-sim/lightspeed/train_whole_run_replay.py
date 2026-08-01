"""Train a whole-run policy from stored soft counterfactual rollout labels."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import os
import random

import torch
from torch.nn import functional as F

from .whole_run_transformer import WholeRunTransformerPolicy


def load_model(path: str, device):
    model = WholeRunTransformerPolicy().to(device)
    missing, unexpected = model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True), strict=False)
    if missing or unexpected:
        print(f"compatible model load new={missing} unused={unexpected}", flush=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, nargs="+",
                        help="one or more rollout datasets to combine")
    parser.add_argument("--load", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--anchor-weight", type=float, default=0.10)
    parser.add_argument("--train-scope",
                        choices=("all", "act-adapter", "phase-adapter",
                                 "human-adapter", "route-adapter",
                                 "consequence-adapter"),
                        default="all",
                        help="optionally update only the per-act policy adapter")
    parser.add_argument("--validation-dataset", nargs="+", default=None,
                        help="optional held-out datasets used to select the saved epoch")
    parser.add_argument(
        "--types", default=None,
        help="optional comma-separated decision types to train and validate on")
    parser.add_argument("--seed", type=int, default=1_021_000)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for dataset_path in args.dataset:
        payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
        rows.extend(payload["rows"])
    requested_types = None
    if args.types:
        requested_types = {
            item.strip() for item in args.types.split(",") if item.strip()}
        rows = [row for row in rows if row["decision_type"] in requested_types]
    if not rows:
        raise RuntimeError("rollout dataset contains no labels")
    counts = Counter(row["decision_type"] for row in rows)
    model = load_model(args.load, device)
    anchor = copy.deepcopy(model).eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    if args.train_scope == "act-adapter":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("act_"))
    elif args.train_scope == "phase-adapter":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("phase_score."))
    elif args.train_scope == "human-adapter":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("human_score."))
    elif args.train_scope == "route-adapter":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("route_"))
    elif args.train_scope == "consequence-adapter":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("action_consequence."))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    validation_rows = []
    for dataset_path in args.validation_dataset or ():
        validation_rows.extend(torch.load(
            dataset_path, map_location="cpu", weights_only=False)["rows"])
    if requested_types is not None:
        validation_rows = [
            row for row in validation_rows
            if row["decision_type"] in requested_types]
    best_validation = float("inf")
    best_state = None

    def normalized_observation(row):
        obs = row["observation"]
        if "act" not in obs:
            obs = dict(obs)
            obs["act"] = int(row.get("act", 1))
        if "floor" not in obs:
            if obs is row["observation"]:
                obs = dict(obs)
            obs["floor"] = int(row.get("floor", 0))
        return obs

    for epoch in range(args.epochs):
        random.shuffle(rows)
        total_loss = total_soft = total_anchor = 0.0
        correct = 0
        for row in rows:
            obs = normalized_observation(row)
            target = torch.as_tensor(
                row["target_probabilities"], dtype=torch.float32, device=device)
            logits, _ = model(obs)
            with torch.no_grad():
                anchor_logits, _ = anchor(obs)
            soft_loss = -(target * F.log_softmax(logits, dim=-1)).sum()
            anchor_probability = F.softmax(anchor_logits, dim=-1)
            anchor_loss = F.kl_div(
                F.log_softmax(logits, dim=-1), anchor_probability,
                reduction="sum")
            # Equalize decision types even if rare categories did not fill
            # their requested quota.
            type_weight = len(rows) / (len(counts) * counts[row["decision_type"]])
            loss = type_weight * (
                soft_loss + args.anchor_weight * anchor_loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss)
            total_soft += float(soft_loss)
            total_anchor += float(anchor_loss)
            correct += int(torch.argmax(logits) == torch.argmax(target))
        validation_text = ""
        if validation_rows:
            model.eval()
            validation_nll = 0.0
            validation_correct = 0
            with torch.no_grad():
                for row in validation_rows:
                    target = torch.as_tensor(
                        row["target_probabilities"], dtype=torch.float32, device=device)
                    logits, _ = model(normalized_observation(row))
                    validation_nll += float(
                        -(target * F.log_softmax(logits, dim=-1)).sum())
                    validation_correct += int(
                        torch.argmax(logits) == torch.argmax(target))
            validation_nll /= len(validation_rows)
            validation_text = (
                f" val_nll={validation_nll:.4f} "
                f"val_argmax={validation_correct}/{len(validation_rows)}")
            if validation_nll < best_validation:
                best_validation = validation_nll
                best_state = copy.deepcopy(model.state_dict())
            model.train()
        print(
            f"epoch={epoch} loss={total_loss / len(rows):.4f} "
            f"soft={total_soft / len(rows):.4f} anchor={total_anchor / len(rows):.4f} "
            f"argmax={correct}/{len(rows)}{validation_text}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(best_state if best_state is not None else model.state_dict(), args.out)
    print(f"saved={args.out} rows={len(rows)} types={dict(counts)}", flush=True)


if __name__ == "__main__":
    main()
