# AGENTS.md

`README.md` says what this project is. This file says how to work on it.
`docs/` holds the reasoning; this is the operational summary.

## Layout

```
slay-sim/            Python: policy, training pipeline, evaluation, pure-Python engine, bridge
  sts/               pure-Python STS engine — tests and live bridge only, NOT training
  lightspeed/        everything RL: env, policy, generation, training, evaluation, tuning
  stsmod/            Java BaseMod overlay (prebuilt jar)
  tests/             19 files, 157 tests
  runs/              checkpoints, datasets, manifests, eval .jsonl, logs
sts_lightspeed/      C++17 engine + native MCTS + pybind11 bindings — the training runtime
silverbot-reference/ Daniel Ziegler's fork. Reference only. Do not develop here.
FULL_RUN_RL_DESIGN.md  design doc for a hierarchical RunPolicy + CombatSolver. Not implemented.
```

No `pyproject.toml`. Everything runs from `slay-sim/` with `PYTHONPATH` set to
`../sts_lightspeed/build` and `.`. C++ builds with CMake + MinGW GCC; targets
`slaythespire` (the `.pyd`), `test`, `main`, `small-test`.

## Where to look

| Task | Location |
|---|---|
| Run the current experiment | `lightspeed/run_label_quality_v31.py` (+ `run_v31.cmd`) |
| Train | `lightspeed/train_whole_run_v27.py` |
| Generate labels | `lightspeed/parallel_generate_whole_run_rollouts.py` → `generate_whole_run_rollouts.py` |
| Evaluate | `lightspeed/eval_whole_run_policy.py` |
| RL environment | `lightspeed/whole_run_env.py` |
| Model | `lightspeed/whole_run_transformer_v27.py`, `whole_run_transformer.py`, `v27_features.py` |
| Combat search (authoritative) | `sts_lightspeed/bindings/slaythespire.cpp` |
| Search config | `lightspeed/search_config.py`, `tuned_search_params.json` |
| Search tuning | `lightspeed/tune_search_cma.py` |
| Combat engine (C++) | `sts_lightspeed/src/combat/`, `src/game/GameContext.cpp` |
| Combat engine (Python) | `slay-sim/sts/combat.py`, `cards.py`, `enemies.py` |
| Live bridge | `slay-sim/run_bridge.py` → `sts/bridge/communication_mod.py` |
| Behaviour audits | `lightspeed/_run_audit.py` (deck/relics/gold), `_room_audit.py` (elite capture vs available), `_routing_audit.py` (conditional logit + `--intervention`), `_overfit_probe*.py` |
| Label quality | `lightspeed/_label_snr.py` — paired SNR per unit compute; the metric to judge any generation change on |

## Commands

```bash
# Build the native engine (from sts_lightspeed/)
mkdir -p build && cd build && cmake .. && cmake --build . --target slaythespire -j8

# Tests (from slay-sim/)
python -m pytest -q

# Train (PowerShell, from slay-sim/)
$env:PYTHONPATH='..\sts_lightspeed\build;.'
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1
python -m lightspeed.run_label_quality_v31 --arm yield --label-scale 10

# Paired evaluation (from slay-sim/)
python -m lightspeed.eval_whole_run_policy <a.pt> <b.pt> `
  --runs 200 --seed-base 18_900_000 --sims 300 --ascension 20 --torch-threads 1 --out runs/x.jsonl

# Pure-Python demo, no C++ needed
python demo.py
```

## Conventions

- `from __future__ import annotations` in every Python file. Non-negotiable.
- Heavy type annotations; narrative docstrings (not Google/NumPy/Sphinx style).
  Comments explain *why*, and frequently cite the measurement that motivated the
  code. Match that register.
- Cards in the Python engine: `@dataclass` with name/cost/type/rarity, a
  `play(state, target)` method, registered in the `CARDS` dict.
- Training and launcher scripts are single-use per experiment. Copy and modify;
  do not parameterize an old one. `run_v*.cmd` files are provenance records.
- C++: PascalCase types, camelCase methods, UPPER_SNAKE_CASE enums, `fixed_list.h`
  on hot paths, `FOREACH_ACTIONTYPE` macro codegen, flattened namespaces for
  pybind11 (`Neow::Bonus` → `NeowBonus`).
- `DRIFT WARNING` comments in `slaythespire.cpp` mark sections that must stay in
  sync with a named Python original.

## Anti-patterns

- Do **not** make backward-compatible changes. Refactor clean; no shims.
- Do **not** handle unexpected states — throw or assert, do not swallow. The v30
  disaster was one swallowed `RuntimeError`.
- Never swallow an error to make a test pass.
- Do not assume warnings are harmless.
- No defensive programming.
- Do not reorder `POWER_VOCAB` in `sts/value_net.py:47` — it is the encoding
  order for a trained net and is append-only. (This applies to that net only, not
  to the whole-run transformers, which key card embeddings off raw `CardId`.)
- Do not change the reward function in `env.py` without treating every
  `checkpoint_*.pt` value head as stale.
- `az_search.py` is deprecated for combat; `az_search_debug.py` is a debug
  artifact, not production.
- Autobattle must never act on the v1 damage-only fallback
  (`sts/bridge/communication_mod.py`).

## Hazards that have already cost time

1. **Two terminal rewards.** `nativeTerminalReward` (env.py / PPO) is a flat
   `−400 + turn` on a loss. `nativeExpectimaxTerminalReward` (the **search**) adds
   `lossProgressCreditWeight·(1 − monsterHpRatio)`, tuned to 566.8, giving ~550
   points of gradient. Do not "fix" the flat loss again — see
   `docs/03-combat-search.md`.
2. **Scope silently decides what trains.** Every scope except `full` freezes the
   trunk. Correct when the trunk is inherited; ruinous when it is not. Read the
   `scope=… trainable=…/…` line the trainer prints.
3. **Validation NLL is a bad selector.** It has picked the wrong epoch three runs
   running. Decide promotions on a paired floor comparison. `--checkpoint-every`
   exists so the epoch curve can be swept on floors.
4. **Head count is unrecoverable from a checkpoint.** `nn.MultiheadAttention`
   packs qkv identically at any head count, so `HEADS_BY_DIM` in
   `eval_whole_run_policy.py` is a hardcoded map. Add new dims there.
5. **Existing datasets are silently reused.** The launcher prints `HAVE <path>`
   and skips generation; shard files resume. Use `--data-dir` and `--tag` when
   re-running an arm, or two experiments become the same run under two names —
   which is exactly what happened to v32 and v33.
6. **`set_search_params` is unlocked global state.** Never have two
   configurations in flight in one process; CMA-ES uses one process per candidate.
7. **Cross-engine comparisons are invalid.** The Armaments upgrade leak was fixed
   on 2026-07-30; every number measured before it came from a different game.
   Check the `.pyd` mtime against the eval log's.
8. **Two engines named "the engine".** `slay-sim/sts/` and `sts_lightspeed/` are
   independent and not in parity. Training only ever uses the C++ one.
9. **Card uniqueIds are load-bearing.** `BattleContext::useCard` removes the
   played card via `removeFromHandById(c.uniqueId)`. Any construction path that
   leaves cards at `CardInstance`'s -1 default lets them be replayed forever —
   this happened in `build_battle_context` and went unnoticed for a while.
   Fixed 2026-07-30; see `docs/07-known-issues.md`.

## Current state, in one paragraph

v31 (`runs/whole_run_transformer_yield10x_a20_v31.pt`) is the best checkpoint:
**+1.41 ± 0.50 paired floors (t = 2.82) over v28 at A20**, re-baselined
2026-08-01 on the post-rebuild engine (`runs/rebaseline_v28_v31_200seeds_20260801.jsonl`,
200 paired seeds at 300 sims); mean floors 23.05 vs 21.64, 0 victories at A20,
13/100 at A0. Capacity (v30) and search budget (300 vs 900 sims) were both tested
and rejected, and fitting the data better (v32, v36) produced worse policies. The
binding constraint is the **overworld policy**, established by layer swap
(+15.71 floors from swapping only that layer); imitation of the human archive is
refuted and on-policy RL is the remaining path. See `docs/03-combat-search.md`
and `docs/06-experiment-log.md`.
