"""Can the model fit at all? Deliberately overfit a small subset.

If train NLL drives to the target-entropy floor on 200 rows, the flat curves are
an optimization problem (batch size 1, low lr, anchor pinning) and batching plus
a higher learning rate will fix them. If it cannot, the trainable subspace lacks
the capacity or the observation encoding discards what the targets need.

Runs four configs on the same 200 rows so the comparison is clean.
"""
from __future__ import annotations

import random
import sys
import time

import torch
from torch.nn import functional as F

from lightspeed.train_whole_run_v27 import load_model, normalized_observation

PREFIXES = (
    "decision_experts.", "uncertainty_heads.", "deck_summary_adapter.",
    "strategic_context_adapter.", "action_card_structure_adapter.",
    "auxiliary_heads.")


def entropy_floor(rows) -> float:
    total = 0.0
    for row in rows:
        p = torch.tensor(row["target_probabilities"], dtype=torch.float64)
        p = p / p.sum()
        total += float(-(p * torch.log(p.clamp_min(1e-12))).sum())
    return total / len(rows)


def run(rows, *, scope, lr, epochs, batch, anchor_weight, device, floor):
    model = load_model(
        "runs/whole_run_transformer_outcome_a20_v28.pt", device)
    anchor = None
    if anchor_weight:
        import copy
        anchor = copy.deepcopy(model).eval()
        for p in anchor.parameters():
            p.requires_grad_(False)
    for name, p in model.named_parameters():
        p.requires_grad_(scope == "full" or name.startswith(PREFIXES))
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    started = time.perf_counter()
    first = last = None
    for epoch in range(epochs):
        random.shuffle(rows)
        total = 0.0
        optimizer.zero_grad()
        for index, row in enumerate(rows):
            target = torch.as_tensor(
                row["target_probabilities"], dtype=torch.float32, device=device)
            logits, _, _, _, _ = model.forward_detailed(
                normalized_observation(row))
            loss = -(target * F.log_softmax(logits, dim=-1)).sum()
            total += float(loss)
            if anchor is not None:
                with torch.no_grad():
                    anchor_logits, _ = anchor(normalized_observation(row))
                loss = loss + anchor_weight * F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    F.softmax(anchor_logits, dim=-1), reduction="sum")
            (loss / batch).backward()
            if (index + 1) % batch == 0 or index + 1 == len(rows):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
        mean = total / len(rows)
        if first is None:
            first = mean
        last = mean
    seconds = time.perf_counter() - started
    print(
        f"  scope={scope:8s} batch={batch:3d} lr={lr:<8g} anchor={anchor_weight:<5g} "
        f"params={n_train:>9,} | NLL {first:.4f} -> {last:.4f} "
        f"| gap above floor {last - floor:+.4f} | {seconds:.0f}s", flush=True)


def main() -> None:
    device = torch.device("cpu")
    torch.set_num_threads(1)
    paths = [
        "runs/v31_yield10x/whole_run_v31_act1_train_800.pt",
        "runs/v31_yield10x/whole_run_v31_act2_train_1000.pt",
    ]
    rows = []
    for path in paths:
        rows += torch.load(path, map_location="cpu", weights_only=False)["rows"]
    random.seed(0)
    subset = random.sample(rows, 200)
    floor = entropy_floor(subset)
    print(f"200 rows, irreducible floor = {floor:.4f} nats\n")

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    # Baseline reproduces production settings; each later row relaxes one thing.
    run(subset, scope="all-v27", lr=3e-5, epochs=epochs, batch=1,
        anchor_weight=0.25, device=device, floor=floor)
    run(subset, scope="all-v27", lr=3e-5, epochs=epochs, batch=1,
        anchor_weight=0.0, device=device, floor=floor)
    run(subset, scope="all-v27", lr=1e-3, epochs=epochs, batch=32,
        anchor_weight=0.0, device=device, floor=floor)
    run(subset, scope="full", lr=1e-3, epochs=epochs, batch=32,
        anchor_weight=0.0, device=device, floor=floor)


if __name__ == "__main__":
    main()
