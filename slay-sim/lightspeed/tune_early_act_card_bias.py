"""Tune a compact, early-Act rollout card-bias vector with paired A20 fights.

This is deliberately not a neural-policy training loop. It optimizes eight
interpretable rollout corrections against the remaining Act 1 HP-chip gap,
while preserving the native expectimax tree and its speed. Every candidate in
a generation sees the same game and search seeds; any provisional winner must
also beat the incumbent on a fresh confirmation seed set.

Run from ``slay-sim`` after building sts_lightspeed::slaythespire:

    PYTHONPATH=".;../sts_lightspeed/build" python -m lightspeed.tune_early_act_card_bias
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import cma
import numpy as np


SIMS = 100
EPISODES_PER_ENCOUNTER = 8
ENCOUNTERS = (
    "JAW_WORM", "TWO_LOUSE", "GREMLIN_GANG", "EXORDIUM_THUGS",
    "GREMLIN_NOB", "THREE_SENTRIES",
)
# Chosen from the matched Silverbot trace, not guessed card quality: these are
# exactly the cards whose early-Act use frequency materially diverged. Keeping
# the vector small makes the held-out gate meaningful rather than searching a
# 372-dimensional lookup table against a handful of fights.
BIAS_CARD_NAMES = (
    "DEFEND_RED", "BASH", "SHRUG_IT_OFF", "CORRUPTION",
    "OFFERING", "INTIMIDATE", "IMPERVIOUS", "REAPER",
)
BIAS_BOUND = 5.0
CHECKPOINT_PATH = Path(__file__).with_name("tuned_search_params.json")

_envs = None
_card_ids = None


def _load_active_config() -> dict:
    with CHECKPOINT_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _worker_init() -> None:
    global _envs, _card_ids
    import slaythespire as sts
    from lightspeed.cards import weighted_ironclad_deck
    from lightspeed.env import IroncladFightEnv, build_full_encounter_resources
    from lightspeed.search_config import apply_search_config

    apply_search_config(_load_active_config())
    # These are experimental controls with a compiled off-state; make that
    # explicit so a stale interpreter cannot leak a previous sweep setting.
    sts.set_search_params({
        "attack_damage_score_weight": 0.0,
        "direct_block_score_weight": 0.0,
    })
    _card_ids = [int(getattr(sts.CardId, name)) for name in BIAS_CARD_NAMES]
    resources = build_full_encounter_resources()
    _envs = [
        IroncladFightEnv(
            encounter=getattr(sts.MonsterEncounter, name),
            encounter_resources=resources, deck_generator=weighted_ironclad_deck,
            ascension=20,
        )
        for name in ENCOUNTERS
    ]


def _score(candidate: np.ndarray, seed_base: int) -> float:
    """Mean ``win + 2 * HP fraction`` on paired Act 1 episodes."""
    import slaythespire as sts

    biases = {card_id: float(value) for card_id, value in zip(_card_ids, candidate)}
    sts.set_early_act_card_biases(biases)
    score = 0.0
    count = 0
    for encounter_index, env in enumerate(_envs):
        for episode in range(EPISODES_PER_ENCOUNTER):
            seed = seed_base + episode
            env.reset(seed=seed)
            done = False
            step = 0
            info = None
            while not done and step < 150:
                search_seed = (seed << 32) ^ (encounter_index << 16) ^ step
                action, _ = sts.run_mcts_search(env.bc, SIMS, None, search_seed)
                _, _, done, info = env.step(action)
                step += 1
            if info["outcome"] == sts.BattleOutcome.PLAYER_VICTORY:
                score += 1.0 + 2.0 * info["player_hp"] / env.bc.player_max_hp
            count += 1
    return score / count


def _evaluate(args: tuple[list[float], int]) -> float:
    candidate, seed_base = args
    return -_score(np.asarray(candidate, dtype=float), seed_base)


def _format(candidate: np.ndarray) -> dict[str, float]:
    return {name: round(float(value), 3) for name, value in zip(BIAS_CARD_NAMES, candidate)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # Start from no early corrections. The active global checkpoint is already
    # loaded by every worker; this search only discovers its sparse adjustment.
    incumbent = np.zeros(len(BIAS_CARD_NAMES), dtype=float)
    es = cma.CMAEvolutionStrategy(
        incumbent, 1.0,
        {"popsize": args.workers, "bounds": [-BIAS_BOUND, BIAS_BOUND], "verbose": -9},
    )
    seed_base = 40_000
    with mp.Pool(args.workers, initializer=_worker_init) as pool:
        for generation in range(1, args.generations + 1):
            candidates = es.ask()
            paired_args = [(list(candidate), seed_base) for candidate in candidates]
            incumbent_score = -pool.map(_evaluate, [(list(incumbent), seed_base)])[0]
            scores = [-value for value in pool.map(_evaluate, paired_args)]
            es.tell(candidates, [-score for score in scores])
            best_index = int(np.argmax(scores))
            challenger = np.asarray(candidates[best_index], dtype=float)
            challenger_score = scores[best_index]
            accepted = False
            confirmation = None
            if challenger_score > incumbent_score:
                # Fresh paired confirmation prevents the per-generation
                # maximum from winning purely through selection noise.
                confirmation_seed = seed_base + EPISODES_PER_ENCOUNTER
                challenger_confirm, incumbent_confirm = [
                    -value for value in pool.map(_evaluate, [
                        (list(challenger), confirmation_seed),
                        (list(incumbent), confirmation_seed),
                    ])
                ]
                confirmation = (challenger_confirm, incumbent_confirm)
                if challenger_confirm > incumbent_confirm:
                    incumbent = challenger
                    accepted = True
            print(
                f"gen={generation:02d} incumbent={incumbent_score:.4f} "
                f"challenger={challenger_score:.4f} accepted={accepted} "
                f"biases={_format(incumbent)}"
                + ("" if confirmation is None else
                   f" confirm={confirmation[0]:.4f}/{confirmation[1]:.4f}"),
                flush=True,
            )
            seed_base += 2 * EPISODES_PER_ENCOUNTER

    print(json.dumps({"best_biases": _format(incumbent)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
