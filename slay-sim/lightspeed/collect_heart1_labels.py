"""Query Silverbot's heart1 overworld policy at states OUR policy visits.

The layer swap (03-combat-search.md) established that heart1's overworld policy
is worth +15.71 +/- 3.13 floors on our own engine and our own combat.  This
collects it as supervision.

Why this is not the refuted human-imitation experiment.  Cloning Baalorlord
cost 15.80 floors for two reasons, both identified in 07-known-issues.md:

  1. capability mismatch -- his elite-taking is correct given HIS deck;
  2. distribution mismatch -- the extraction pins his deck into every
     observation, so every labelled state is one our policy never occupies.

heart1 has neither problem.  It is a policy whose actions are already measured
good WITH OUR COMBAT UNDERNEATH, and unlike a human archive it can be queried
at any state, so labels are collected ON our policy's own distribution
(DAgger).  It also sidesteps the label-noise wall entirely: the target is a
policy's action distribution, not a 0.803-SNR Monte Carlo estimate over
counterfactual rollouts.

The action correspondence is exact rather than heuristic.  `construct_choice`
builds heart1's choice list FROM the `sts.GameAction` list we hand it and keeps
the correspondence, so `path_to_action_and_desc` returns one of our own action
objects; matching on `.bits` is therefore identity, not a guess.

Run from slay-sim/ (their package must be importable; it binds `slaythespire`
to whichever build is on PYTHONPATH, which is ours):
    PYTHONPATH='../sts_lightspeed/build;.;../silverbot-reference'
    python -m lightspeed.collect_heart1_labels --runs 20 --out runs/heart1_labels.pt
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import time

import numpy as np
import torch

import slaythespire as sts

from .eval_whole_run_policy import load_policy
from .search_config import DEFAULT_SEARCH_CONFIG_PATH
from .whole_run_env import RunConfig, WholeRunEnv

DEFAULT_STUDENT = "runs/whole_run_transformer_yield10x_a20_v31.pt"
DEFAULT_TEACHER = "../silverbot-reference/runs/heart1.pt"


def teacher_distribution(service, gc, actions, our_bits):
    """heart1's probability over OUR candidate list, or None if unmappable.

    Returns (probs, coverage) where coverage is the share of heart1's mass that
    landed on a candidate we actually offered.  Mass can be lost when their
    choice space encodes an option our action list splits differently; a screen
    whose coverage is materially below 1.0 must not be trained on.
    """
    from silverbot.playouts import construct_choice, path_to_action_and_desc
    from silverbot.network import choice_space

    try:
        choice = construct_choice(gc, sts.getNNRepresentation(gc), actions)
    except (ValueError, AssertionError, IndexError):
        # Boss-relic screens use a RELIC/SKIP encoding lightspeed does not
        # share; eval_heart1_hybrid.py falls back to a legal action there.
        return None, 0.0
    if choice is None:
        return None, 0.0

    batch_tensors, output = service.get_logits(choice)
    logits = output[0] if isinstance(output, tuple) else output
    probs = torch.softmax(torch.as_tensor(np.asarray(logits), dtype=torch.float32), dim=0)

    index_of = {bits: i for i, bits in enumerate(our_bits)}
    mapped = torch.zeros(len(our_bits), dtype=torch.float32)
    for ix in range(probs.shape[0]):
        weight = float(probs[ix])
        if weight <= 1e-6:
            continue
        try:
            path = choice_space.ix_to_path(batch_tensors["choices"], ix)
            action, _ = path_to_action_and_desc(choice, path, gc)
        except (IndexError, ValueError, AssertionError, KeyError):
            continue
        slot = index_of.get(int(action.bits))
        if slot is not None:
            mapped[slot] += weight
    coverage = float(mapped.sum())
    if coverage <= 1e-3:
        return None, coverage
    return mapped / coverage, coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default=DEFAULT_STUDENT,
                        help="policy that DRIVES the run; labels are collected "
                             "on the states it visits")
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=2_100_000)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.0,
                        help="probability of following the TEACHER's action "
                             "instead of the student's (DAgger mixing); 0 is "
                             "pure on-student collection")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", default="runs/heart1_labels.pt")
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = torch.device("cpu")
    student = load_policy(args.student, device)

    from silverbot.network import NN, ModelHP, load_network_backward_compatible
    from silverbot.playouts import NNService, take_free_rewards

    # Same construction as eval_heart1_hybrid.py: the checkpoint's architecture
    # is not recoverable from its state dict, so heart1's is spelled out.
    network = NN(ModelHP(use_value_head=True, dim=256, n_layers=4)).to(device)
    network = load_network_backward_compatible(
        network, torch.load(args.teacher, map_location=device, weights_only=True))
    network.eval()
    service = NNService(network, batch_size=8, batch_size_factor=1,
                        torch_compile_mode="no")

    rows: list[dict] = []
    coverage_by_screen: dict[str, list[float]] = collections.defaultdict(list)
    agree = total = 0
    started = time.perf_counter()

    for offset in range(args.runs):
        seed = args.seed_base + offset
        rng = random.Random(seed)
        env = WholeRunEnv(RunConfig(
            ascension=args.ascension, combat_sims=args.sims,
            deterministic_combat=True,
            search_config_path=DEFAULT_SEARCH_CONFIG_PATH))
        obs = env.reset(seed)
        with torch.inference_mode():
            while (env.gc.outcome.name == "UNDECIDED"
                   and env.steps < env.config.max_decisions):
                actions = env.legal_actions()
                take_free_rewards(env.gc)
                if env.gc.outcome.name != "UNDECIDED":
                    break
                if len(actions) != len(env.legal_actions()):
                    obs = env.observation()
                    actions = env.legal_actions()
                our_bits = [int(a.bits) for a in actions]

                student_index, _, _, _ = student.act(obs, sample=False)
                screen = str(env.gc.screen_state).split(".")[-1]

                probs, coverage = (None, 0.0)
                if len(actions) > 1:
                    probs, coverage = teacher_distribution(
                        service, env.gc, actions, our_bits)
                    coverage_by_screen[screen].append(coverage)

                if probs is not None:
                    teacher_index = int(torch.argmax(probs))
                    agree += int(teacher_index == student_index)
                    total += 1
                    rows.append({
                        "observation": {k: v for k, v in obs.items()
                                        if k != "action_text"},
                        "target_probabilities": probs.numpy(),
                        "screen": screen,
                        "seed": seed,
                        "floor": int(env.gc.floor_num),
                        "act": int(env.gc.act),
                        "coverage": coverage,
                    })
                    if args.beta > 0 and rng.random() < args.beta:
                        student_index = teacher_index

                obs, _, done, _ = env.step(student_index)
                if done:
                    break
        print(f"seed {seed}: floor {env.gc.floor_num} "
              f"rows {len(rows)} ({time.perf_counter() - started:.0f}s)",
              flush=True)

    service.stop()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(rows, args.out)

    print(f"\nwrote {args.out}: {len(rows)} rows from {args.runs} runs")
    print(f"teacher/student argmax agreement: {agree}/{total} "
          f"({100.0 * agree / max(1, total):.1f}%)")
    print(f"\n{'screen':>22}  {'n':>5}  {'mean coverage':>13}  {'unmappable':>10}")
    for screen, values in sorted(coverage_by_screen.items()):
        unmappable = sum(1 for v in values if v <= 1e-3)
        print(f"{screen:>22}  {len(values):>5}  {np.mean(values):>13.3f}  "
              f"{unmappable:>10}")


if __name__ == "__main__":
    main()
