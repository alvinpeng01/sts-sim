"""Does the fit plateau, or was the short probe just step-starved?

The 40-epoch probe gave batch-32 configs only ~280 optimizer steps against the
batch-1 configs' 8,000, so their better loss came with 28x less optimization.
This runs the two batched configs far longer to find where they actually settle
relative to the target-entropy floor.
"""
from __future__ import annotations

import random
import sys
import time

import torch
from torch.nn import functional as F

from lightspeed.train_whole_run_v27 import load_model, normalized_observation
from lightspeed._overfit_probe import PREFIXES, entropy_floor


def run(rows, *, scope, lr, epochs, batch, device, floor, report_every):
    model = load_model("runs/whole_run_transformer_outcome_a20_v28.pt", device)
    for name, p in model.named_parameters():
        p.requires_grad_(scope == "full" or name.startswith(PREFIXES))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    steps = 0
    started = time.perf_counter()
    print(f"--- scope={scope} batch={batch} lr={lr} params={sum(p.numel() for p in trainable):,}", flush=True)
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
            (loss / batch).backward()
            if (index + 1) % batch == 0 or index + 1 == len(rows):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                steps += 1
        if (epoch + 1) % report_every == 0 or epoch == 0:
            mean = total / len(rows)
            print(
                f"  epoch={epoch+1:4d} steps={steps:6d} NLL={mean:.4f} "
                f"gap={mean - floor:+.4f} ({time.perf_counter()-started:.0f}s)",
                flush=True)


def main() -> None:
    device = torch.device("cpu")
    torch.set_num_threads(1)
    rows = []
    for path in ("runs/v31_yield10x/whole_run_v31_act1_train_800.pt",
                 "runs/v31_yield10x/whole_run_v31_act2_train_1000.pt"):
        rows += torch.load(path, map_location="cpu", weights_only=False)["rows"]
    random.seed(0)
    subset = random.sample(rows, 200)
    floor = entropy_floor(subset)
    print(f"200 rows, irreducible floor = {floor:.4f} nats\n", flush=True)
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    run(subset, scope="all-v27", lr=1e-3, epochs=epochs, batch=32,
        device=device, floor=floor, report_every=max(1, epochs // 8))
    run(subset, scope="full", lr=1e-3, epochs=epochs, batch=32,
        device=device, floor=floor, report_every=max(1, epochs // 8))


if __name__ == "__main__":
    main()
