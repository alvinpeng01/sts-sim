# slay-sim/ AGENTS.md — Python policy, training, and bridge

The top-level `../AGENTS.md` covers the project as a whole and the hazards that
span both halves. This file is the Python side only.

## Overview

Two unrelated things live here and are easy to confuse:

- **`lightspeed/`** — the real pipeline. The whole-run policy, label generation,
  training, evaluation, and search tuning. All of it drives the **C++** engine
  through `import slaythespire`.
- **`sts/`** — an independent pure-Python STS engine. Used by the tests and the
  live-game bridge. **Never used for training.**

`lightspeed/` also still contains the older combat-only PPO stack (`env.py`,
`ppo.py`, `policy.py`, `train_*.py`, `checkpoint_*.pt`). It is not on the
whole-run path.

## Structure

```
sts/              engine: combat.py (CombatState), cards.py (Ironclad/Colorless/Curse),
                  enemies.py (Acts 1-3), relics.py, potions.py, powers.py, orbs.py,
                  value_net.py (POWER_VOCAB at :47 — append-only)
sts/bridge/       live bridge: communication_mod.py (stdio + autobattle),
                  native_recommend.py (native MCTS on live state), predict.py, state_mapper.py
lightspeed/
  whole_run_env.py            WholeRunEnv over the native GameContext
  whole_run_transformer.py    base policy; whole_run_transformer_v27.py adds experts/
                              uncertainty/auxiliary heads/adapters; v27_features.py their inputs
  generate_whole_run_rollouts.py / parallel_...py   label generation
  train_whole_run_v27.py      the training loop
  eval_whole_run_policy.py    paired evaluation
  search_config.py            loads/applies/verifies tuned_search_params.json
  tune_search_cma.py          CMA-ES over the NATIVE search params
  expectimax_search.py        Python MCTS — distillation/tuning only, not the whole-run path
  az_search.py                deprecated PUCT; az_search_debug.py is a debug artifact
  run_label_quality_v31.py    current launcher; run_long_training_v26/v28/v30.py superseded
  _run_audit.py, _room_audit.py, _overfit_probe*.py    diagnostics
  env.py, ppo.py, policy.py   older combat-only PPO stack
stsmod/           Java BaseMod overlay, prebuilt STSPredictor.jar, F9 autobattle toggle
tests/            19 files, 157 tests. No conftest, no pytest.ini.
runs/             ~830 files: checkpoints, .pt datasets, .jsonl evals, manifests, logs
```

## Where to look

| Task | Location |
|---|---|
| Run the current experiment | `lightspeed/run_label_quality_v31.py` |
| Train | `lightspeed/train_whole_run_v27.py` |
| Generate labels | `lightspeed/parallel_generate_whole_run_rollouts.py` |
| Evaluate two checkpoints | `lightspeed/eval_whole_run_policy.py` |
| Audit run economy / routing | `lightspeed/_run_audit.py`, `_room_audit.py` |
| Tune search params | `lightspeed/tune_search_cma.py` |
| Import human data | `lightspeed/import_baalorlord_runs.py` |
| Run the bridge live | `run_bridge.py` → `sts/bridge/communication_mod.py` |
| Demo the pure engine | `demo.py` |
| Add a card to the Python engine | `sts/cards.py` — dataclass + `play()`, register in `CARDS` |

## Commands

```bash
python -m pytest -q                    # 157 tests
python -m tests.test_combat            # single test file standalone

# PowerShell, from slay-sim/
$env:PYTHONPATH='..\sts_lightspeed\build;.'
python -m lightspeed.run_label_quality_v31 --arm yield --label-scale 10
python -m lightspeed.eval_whole_run_policy <a.pt> <b.pt> --runs 200 --seed-base 18_900_000 --sims 300
```

## Conventions

- `from __future__ import annotations` in every file.
- Search configuration only ever through `search_config.ensure_search_config()`,
  which resets, applies, and then verifies against the live runtime. Never call
  `sts.set_search_params()` directly outside the tuner.
- No `__init__.py` in `tests/`; pytest discovers by naming convention.
- Training scripts are single-use per experiment; copy rather than parameterize.

## Anti-patterns

- Never let autobattle act on the v1 damage-only fallback.
- Never reorder or insert into `POWER_VOCAB` (`sts/value_net.py:47`).
- Do not modify `env.py`'s reward without treating every `checkpoint_*.pt` value
  head as stale.
- `expectimax_search.py`, not `az_search.py`, is the Python-side combat search —
  but neither is on the whole-run path, which uses the C++ engine directly.
- Do not promote a checkpoint on validation NLL. Use a paired floor comparison.
- Do not re-run a generation arm into a directory that already holds data;
  existing datasets and shards are silently reused. Pass `--data-dir` and `--tag`.

See `../docs/` for the reasoning behind all of the above, and `../AGENTS.md` for
the cross-cutting hazards.
