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
| Ascension 20 | mean floor **23.05** of 56 over 200 seeds; **0 victories** |
| Ascension 0 | mean floor 39.21 over 100 seeds; **13 victories** |
| vs. the previous best (v28, 21.64) | **+1.41 ± 0.50** paired floors at A20 (t = 2.82) |
| Tests | 185 passing in 21 files |

A20 numbers were re-baselined 2026-08-01 on the post-rebuild engine; the A0
numbers predate several engine fixes and are not directly comparable. See
[docs/05-model-lineage.md](docs/05-model-lineage.md) and the re-baseline table in
[docs/README.md](docs/README.md).

**The binding constraint is the overworld policy, not combat.** Established by
layer swap: Silverbot's `heart1.pt` overworld policy driving OUR engine and OUR
combat moved the mean floor 21.29 → 37.00 (**+15.71 ± 3.13**) and produced this
stack's first A20 victories, with combat byte-identical between the arms. Combat
search budget, search configuration, and four separate algorithmic changes to the
search have all since measured flat or negative against floors. Training label
volume was the *previous* diagnosis and remains the bottleneck for the supervised
path specifically. See [docs/03-combat-search.md](docs/03-combat-search.md) for
the layer swap and [docs/06-experiment-log.md](docs/06-experiment-log.md) for the
label work.

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
python -m pytest -q          # 185 tests
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

## Version control

This tree is a git repository as of 2026-08-01; before that it had none.
`.gitignore` carries the reasoning per entry, but in short it tracks source,
tests and build inputs (1,857 files, 1.8 MB) and excludes `silverbot-reference/`
(348 MB vendored fork), `slay-sim/runs/` (1.3 GB of checkpoints, datasets and
eval output), the five `sts_lightspeed/build*/` trees, and `*.pt` / `*.pyd` /
`*.jsonl` / `*.log` by extension — the extension rules are needed because ~25
checkpoints sit inside `slay-sim/lightspeed/` intermixed with source.

`docs/` is currently **untracked rather than ignored**, pending a decision on
whether to version it.

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
