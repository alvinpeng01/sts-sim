# Slay the Spire AI

A Slay the Spire agent for the Ironclad. Combat is played by an expectimax MCTS
running in a C++17 engine; overworld decisions — card picks, paths, campfires,
shops, events, Neow — are made by a transformer policy trained on soft targets
from counterfactual rollouts.

Two directories, one pybind11 boundary:

- **`sts_lightspeed/`** — a C++17 fork of gamerpuppy's RNG-accurate STS engine,
  extended with the native MCTS search, Silent card implementations the upstream
  engine never had, and pybind11 bindings exposing it to Python as
  `slaythespire`. This is the training and evaluation runtime.
- **`slay-sim/`** — the Python side: the policy, the label-generation and
  training pipeline, the evaluation harness, a second independent pure-Python STS
  engine used for tests and the live-game bridge, and a BaseMod overlay.

Full documentation is in [`docs/`](docs/README.md). Design intent that has **not**
been implemented is in [`FULL_RUN_RL_DESIGN.md`](FULL_RUN_RL_DESIGN.md).

## Where it stands

| | |
|---|---|
| Best checkpoint | `slay-sim/runs/whole_run_transformer_yield10x_a20_v31.pt` |
| Ascension 20 | mean floor **23.57** of 56 over 200 seeds; **0 victories** |
| Ascension 0 | mean floor 39.21 over 100 seeds; **13 victories** |
| vs. the previous best (v28) | **+1.83 ± 0.55** paired floors at A20 |
| Tests | 157 passing in 19 files |

A20 numbers are post-Armaments-fix; the A0 numbers are pre-fix and not directly
comparable. See [docs/05-model-lineage.md](docs/05-model-lineage.md).

The measured bottleneck is **training label volume**. Model capacity (4× params)
and combat search budget (3× sims) were both tested and both bought less than
raising labels-per-episode did. See
[docs/06-experiment-log.md](docs/06-experiment-log.md).

## Setup

Requires CMake 3.19+, a C++17 compiler, and Python dev headers matching the
Python you will run `slay-sim/` with — pybind11 links against a specific version
at build time. The checked-in build is `cp313` on MinGW-w64 / Windows.

### 1. Build the native engine

```bash
cd sts_lightspeed
mkdir -p build && cd build
cmake ..
cmake --build . --target slaythespire -j8
```

This produces `build/slaythespire.cp<version>-<platform>.pyd` (or `.so`). Other
targets: `test` (benchmarks + MCTS), `main` (interactive), `small-test`.

### 2. Python environment

```bash
cd slay-sim
python -m venv .venv
.venv/Scripts/pip install numpy torch pytest
```

There is no `pyproject.toml`. Make the compiled module importable by putting
`sts_lightspeed/build` on `PYTHONPATH`.

### 3. Verify

```bash
cd slay-sim
python -c "import slaythespire; print('native engine ok')"
python -m pytest -q          # 157 tests
```

The pytest suite covers the pure-Python `sts/` engine and the pipeline glue; the
import check above is the real test that the native build worked.

### 4. Run something

```bash
# Pure Python, no C++ needed
python demo.py

# Evaluate two checkpoints head-to-head on shared seeds  (PowerShell)
$env:PYTHONPATH='..\sts_lightspeed\build;.'
python -m lightspeed.eval_whole_run_policy `
  runs/whole_run_transformer_outcome_a20_v28.pt `
  runs/whole_run_transformer_yield10x_a20_v31.pt `
  --runs 200 --seed-base 18_900_000 --sims 300 --ascension 20 --torch-threads 1

# Train the current experiment arm
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1
python -m lightspeed.run_label_quality_v31 --arm yield --label-scale 10
```

The optional in-game overlay and live bridge are covered in
[docs/09-live-play-bridge.md](docs/09-live-play-bridge.md) and
`slay-sim/stsmod/README.md`; that half needs only `slay-sim/sts/` and numpy.

## Not included in transfers

`slay-sim/.venv/` (~1 GB, reinstall instead) and `sts_lightspeed/build/`
artifacts (rebuild instead).

Also deliberately absent: `sts_lightspeed/.git/`. Its origin remote had a GitHub
Personal Access Token embedded in the URL
(`https://github_pat_...@github.com/gamerpuppy/sts_lightspeed.git`), found during
this project and never used beyond confirming it existed. **That token should be
treated as compromised and rotated on GitHub.** To restore history, re-clone
upstream fresh and re-apply these changes, or push this copy to your own remote
with your own credentials. Do not reuse that URL.

## Reading order

1. [docs/01-architecture.md](docs/01-architecture.md) — what the pieces are and
   which one is authoritative.
2. [docs/02-training-pipeline.md](docs/02-training-pipeline.md) — how a label is
   made and how the model is trained.
3. [docs/04-evaluation.md](docs/04-evaluation.md) — before you compare two
   numbers.
4. [docs/06-experiment-log.md](docs/06-experiment-log.md) — why the project is
   where it is.
5. [AGENTS.md](AGENTS.md) — conventions and hazards, if you are going to change
   code.
