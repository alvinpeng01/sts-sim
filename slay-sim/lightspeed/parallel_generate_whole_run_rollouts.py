"""Resumable process-isolated wrapper for whole-run counterfactual labels."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(torch.load(path, map_location="cpu", weights_only=False)["rows"])
    except (EOFError, OSError, RuntimeError):
        return 0


def _logged_label_count(path: Path) -> int:
    """Count a live worker's labels from its append-only log.

    Progress reporting must never deserialize a shard's `.partial` file. That
    file is rewritten every `--save-every` labels by another process, so a torn
    read is routine rather than exceptional, and `_row_count`'s except clause
    does not cover `pickle.UnpicklingError` — which is precisely what killed the
    v31 Act 3 stage after four stages had already succeeded. A text log read can
    only ever come up one line short.
    """
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.startswith("label="))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run independent native-search label shards and merge them.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-labels", type=int, default=1_000)
    parser.add_argument("--max-episodes", type=int, default=2_000)
    parser.add_argument("--per-type", type=int, default=125)
    parser.add_argument("--seed", type=int, default=2_500_000)
    parser.add_argument(
        "--fresh", action="store_true",
        help="rerun completed shards instead of resuming them")
    args, forwarded = parser.parse_known_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if "--policy" not in forwarded:
        raise ValueError("pass --policy CHECKPOINT for the underlying generator")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels_per_worker = math.ceil(args.max_labels / args.workers)
    episodes_per_worker = math.ceil(args.max_episodes / args.workers)
    types_per_worker = math.ceil(args.per_type / args.workers)
    processes = []
    logs = []
    shard_paths = []
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    for worker in range(args.workers):
        shard = output.with_suffix(output.suffix + f".shard{worker}.pt")
        shard_paths.append(shard)
        if shard.exists() and not args.fresh and _row_count(shard):
            print(f"worker={worker} resume rows={_row_count(shard)}", flush=True)
            continue
        log_path = output.with_suffix(output.suffix + f".shard{worker}.log")
        log_handle = open(log_path, "w", encoding="utf-8")
        logs.append(log_handle)
        command = [
            sys.executable, "-m", "lightspeed.generate_whole_run_rollouts",
            *forwarded,
            "--out", str(shard),
            "--max-labels", str(labels_per_worker),
            "--max-episodes", str(episodes_per_worker),
            "--per-type", str(types_per_worker),
            "--seed", str(args.seed + worker * 1_000_003),
        ]
        if Path(str(shard) + ".partial").exists() and not args.fresh:
            command.append("--resume")
        process = subprocess.Popen(
            command, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
        processes.append((worker, process, shard, log_path))
        print(f"worker={worker} pid={process.pid} log={log_path}", flush=True)

    spawned = [entry[1] for entry in processes]
    try:
        while processes:
            active = []
            for worker, process, shard, log_path in processes:
                code = process.poll()
                if code is None:
                    active.append((worker, process, shard, log_path))
                elif code:
                    raise RuntimeError(
                        f"worker {worker} failed with exit code {code}; "
                        f"see {log_path}")
                else:
                    print(
                        f"worker={worker} complete rows={_row_count(shard)}",
                        flush=True)
            processes = active
            if processes:
                progress = ", ".join(
                    f"{worker}:{max(_row_count(shard), _logged_label_count(log_path))}"
                    for worker, _, shard, log_path in processes)
                print(f"active={len(processes)} rows=[{progress}]", flush=True)
                time.sleep(5)
    finally:
        # Never leave workers behind. Each holds a core and keeps rewriting its
        # own shard, so survivors of a failed coordinator interleave with the
        # next attempt's workers on the same paths and corrupt its output. That
        # is what happened when the v31 Act 3 progress line raised: six orphans
        # outlived the coordinator and had to be killed by hand before Act 3
        # could be regenerated. A failing worker also used to orphan its five
        # siblings, since the raise above skipped straight past cleanup.
        for process in spawned:
            if process.poll() is None:
                process.terminate()
        for process in spawned:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        for handle in logs:
            if not handle.closed:
                handle.close()
    rows = []
    shard_metadata = []
    for shard in shard_paths:
        if not shard.exists():
            continue
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        rows.extend(payload["rows"])
        shard_metadata.append(payload.get("metadata", {}))
    rows = rows[:args.max_labels]
    torch.save({
        "rows": rows,
        "metadata": {
            "workers": args.workers,
            "requested_labels": args.max_labels,
            "shards": [str(path) for path in shard_paths],
            "shard_metadata": shard_metadata,
            "forwarded_args": forwarded,
        },
    }, output)
    print(f"merged={output} rows={len(rows)} shards={len(shard_metadata)}", flush=True)


if __name__ == "__main__":
    main()
