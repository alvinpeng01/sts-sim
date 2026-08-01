"""Measure policy agreement with an offline human-demonstration dataset."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math

import torch

from .whole_run_transformer_v27 import WholeRunTransformerPolicyV27


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.dataset, map_location="cpu", weights_only=False)
    rows = payload["rows"]
    # V27 is backward-compatible with older checkpoints (its residuals are
    # zero-initialized) and preserves V28's expert/ensemble parameters.
    policy = WholeRunTransformerPolicyV27().to(device)
    missing, unexpected = policy.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True),
        strict=False,
    )
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint parameters: {unexpected}")
    policy.eval()

    groups = defaultdict(lambda: {
        "count": 0, "correct": 0, "nll": 0.0, "margin": 0.0,
    })
    with torch.no_grad():
        for row in rows:
            target = torch.as_tensor(
                row["target_probabilities"], dtype=torch.float32, device=device)
            chosen = int(torch.argmax(target))
            logits, _ = policy(row["observation"])
            log_probs = torch.log_softmax(logits, dim=0)
            predicted = int(torch.argmax(logits))
            sorted_logits = torch.sort(logits, descending=True).values
            margin = float(sorted_logits[0] - sorted_logits[1]) if len(logits) > 1 else 0.0
            act = int(row["observation"].get("act", 0))
            for key in ("all", row["decision_type"], f"act_{act}"):
                group = groups[key]
                group["count"] += 1
                group["correct"] += int(predicted == chosen)
                group["nll"] += float(-log_probs[chosen])
                group["margin"] += margin

    result = {
        key: {
            **values,
            "accuracy": values["correct"] / values["count"],
            "mean_nll": values["nll"] / values["count"],
            "mean_top_margin": values["margin"] / values["count"],
            "perplexity": math.exp(min(20.0, values["nll"] / values["count"])),
        }
        for key, values in sorted(groups.items())
    }
    print(json.dumps({
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "compatible_missing_parameters": missing,
        "groups": result,
    }, indent=2))


if __name__ == "__main__":
    main()
