# Slay the Spire AI

A Slay the Spire agent for the Ironclad. Combat is played by an expectimax MCTS
running in a C++17 engine; overworld decisions — card picks, paths, campfires,
shops, events, Neow — are made by a transformer policy trained on soft targets
from counterfactual rollouts.

Two directories, one pybind11 boundary:

- **`sts_lightspeed/`** — a C++17 fork of gamerpuppy's RNG-accurate STS engine,
  extended with the native MCTS search and pybind11 bindings exposing it to
  Python as `slaythespire`. This is the training and evaluation runtime, and the
  authoritative engine.
- **`slay-sim/`** — the Python side: the policy, the label-generation and
  training pipeline, the evaluation harnesses, a second independent pure-Python
  STS engine used for tests and the live-game bridge, and a BaseMod overlay.

Full documentation is in [`docs/`](docs/README.md). Design intent that has **not**
been implemented is in [`FULL_RUN_RL_DESIGN.md`](FULL_RUN_RL_DESIGN.md).
Conventions and hazards for anyone changing code are in [`AGENTS.md`](AGENTS.md).

---

## Where it stands

| | |
|---|---|
| Best checkpoint | `slay-sim/runs/whole_run_transformer_yield10x_a20_v31.pt` |
| Ascension 20 | mean floor **23.05** of 56 over 200 paired seeds; **0 victories** |
| Ascension 0 | mean floor 39.21 over 100 seeds; **13 victories** |
| vs. previous best (v28, 21.64) | **+1.41 ± 0.50** paired floors, t = 2.82 |
| Combat vs. a top human | **−5.78** mean HP against his own decks (0 = parity) |
| Tests | **185 passing** in 21 files |
| Engine parameters | 68 runtime-tunable, 42 overridden by the shipped config |

A20 figures re-baselined 2026-08-01 on the post-rebuild engine
(`runs/rebaseline_v28_v31_200seeds_20260801.jsonl`). **The A0 figures predate
several engine fixes and are not comparable to anything measured after them.**

### The one thing to know

**The binding constraint is the overworld policy, not combat.** Established by
layer swap on 2026-07-31: Silverbot's `heart1.pt` overworld policy driving *our*
engine and *our* combat moved the mean floor 21.29 → **37.00** (+15.71 ± 3.13,
t = 5.02) and produced this stack's first A20 victories — with combat
byte-identical between the arms. `heart1` is the same architecture class as v31
(a trained net, one forward pass per decision, no search), so this is a
**training-quality gap, not a design gap**.

Everything downstream of that follows: combat work has repeatedly measured flat
or negative against floors, and the run policy has ~15 floors and the difference
between 0% and ~20% win rate sitting in it.

---

## Setup

Requires CMake 3.19+, a C++17 compiler, and Python dev headers matching the
Python you will run `slay-sim/` with — pybind11 links against a specific version
at build time. The checked-in build is `cp313` on MinGW-w64 / Windows.

### 1. Build the native engine

```bash
cd sts_lightspeed && mkdir -p build && cd build && cmake .. && cmake --build . --target slaythespire -j8
```

Produces `build/slaythespire.cp<version>-<platform>.pyd` (or `.so`). Other
targets: `test`, `main`, `small-test`.

**`build/CMakeCache.txt` carries `STS_PGO=use`**, so a plain rebuild in that
directory *is* the PGO build (`-fprofile-use` against `pgo/*.gcda`). Do not
"fix" this by reconfiguring without `STS_PGO`. On a fresh platform, build with
`-DSTS_PGO=` — profile data is compiler- and platform-specific and will not
transfer.

If the link fails with `cannot open output file ... Permission denied`, a running
Python process is holding the `.pyd`. The compile succeeded; only the link
didn't. Wait for the job rather than killing it.

### 2. Python environment

```bash
cd slay-sim && python -m venv .venv && .venv/Scripts/pip install numpy torch pytest
```

There is no `pyproject.toml`. Put `sts_lightspeed/build` on `PYTHONPATH`.

### 3. Verify

```bash
cd slay-sim && python -c "import slaythespire; print('native engine ok')" && python -m pytest -q
```

185 tests. The import check is the real test that the native build worked.

### 4. Run something

```bash
python demo.py    # pure Python, no C++ needed
```

```bash
python -m lightspeed.eval_whole_run_policy runs/whole_run_transformer_outcome_a20_v28.pt runs/whole_run_transformer_yield10x_a20_v31.pt --runs 200 --seed-base 18900000 --sims 300 --ascension 20 --torch-threads 1
```

---

## Repository and version control

Under git as of 2026-08-01; before that it had none. **1,857 files, 1.8 MB
tracked.** `.gitignore` carries per-entry reasoning; in short:

| excluded | why |
|---|---|
| `silverbot-reference/` | 348 MB vendored read-only fork — never developed here, but see the dependency note below |
| `slay-sim/runs/` | 1.3 GB of checkpoints, datasets, eval output — regenerable |
| `sts_lightspeed/build*/` | five CMake trees including the `.pyd` |
| `*.pt` `*.pyd` `*.jsonl` `*.log` | by extension, because ~25 checkpoints sit inside `slay-sim/lightspeed/` intermixed with source |

`docs/` is currently **untracked rather than ignored**, pending a decision on
whether to version it — `git add docs` is all it takes.

### `silverbot-reference/` is excluded but not optional

It is Daniel Ziegler's Silver Automaton fork, kept as a reference and never
developed in. Excluding it from git is deliberate, but **a fresh clone will not
be able to run two harnesses**, and one of them matters a great deal:

| harness | dependency | why it matters |
|---|---|---|
| `eval_heart1_hybrid.py` | imports `silverbot.network` / `silverbot.playouts`, loads `../silverbot-reference/runs/heart1.pt` | **This is the layer-swap instrument** — the measurement that established the run policy as the binding constraint, and the tool for judging any future run-policy work |
| `_silverbot_human_deck.py` | runs silverbot in a separate process against `silverbot-reference/build` (its own compiled module, since both engines' Python modules are named `slaythespire`) | The second arm on the combat benchmark — what turns our HP number into a bracket rather than a bare figure |

Everything else — `compare_tier_combat.py`'s silver arm aside — references the
fork only in comments. **The test suite does not depend on it**; all 185 tests
pass without it present.

To restore: obtain the fork, place it at `sts_lightspeed/../silverbot-reference/`,
and build its native module separately. `_silverbot_human_deck.py` also carries a
hardcoded absolute Windows path to that build directory.

**Security note, still outstanding.** The original `sts_lightspeed/.git/` had a
GitHub Personal Access Token embedded in its origin URL
(`https://github_pat_...@github.com/gamerpuppy/sts_lightspeed.git`). **That token
should be treated as compromised and rotated.** The repository created today has
no remote configured; that does not change the exposure.

---

## What has been established

The value of this project is largely in what has been *ruled out*. Each row is
backed by a harness under `slay-sim/lightspeed/` — see
[docs/README.md](docs/README.md) for the full script table.

### Confirmed

| finding | measurement |
|---|---|
| The overworld policy is the binding constraint | +15.71 ± 3.13 floors from swapping only that layer (`eval_heart1_hybrid.py`) |
| The combat search is **draw-order clairvoyant** | −3.78 ± 0.84 HP/fight when blinded; our apparent lead over Silverbot was this cheat |
| Relic power level dominates combat tuning | +0.406 win rate at the encounter level, z = 8.10 — an order of magnitude above any other combat lever |
| v31's routing preference is **inverted**, not miscalibrated | ELITE logit −2.55 against the human's +1.93 and Silverbot's +0.22 |
| v31 does not rest when hurt | hp_frac × REST +1.19 against −1.93 (human) and −1.72 (Silverbot) |
| v31's aux heads predict combat outcomes | `next_combat_survival` AUC **0.817**, `next_combat_hp` R² **0.302**, on 4,617 on-policy decisions |
| The engine matches the real game on constants | All 120 attack cards, 66 monster HP values, 64/76 relics verified against `desktop-1.0.jar` |

### Refuted

| hypothesis | result |
|---|---|
| More combat search budget buys floors | +0.115 ± 0.507 floors at 15× sims |
| Better combat config buys floors | Config worth +6.05 benchmark HP measured **−0.23 ± 0.29 floors** |
| Model capacity is the limit | v30 (4× params) was worse |
| The map representation is the routing problem | The features Silverbot credits already exist in `map_route_features` |
| Imitating the human archive | Routing **−15.80 ± 0.74** floors, drafting −5.42 ± 0.81 — distribution mismatch |
| Distilling the search into the rollout policy | −2.12 ± 0.84 HP at matched wall clock |
| Widening `nativeActionFeatures` with card identity | +2.3pp top-1 against a net costing 4.97× search speed |
| A cheaper leaf evaluation | −19.14 ± 1.16 HP at matched sims |
| Max-Monte-Carlo backup (MaxUCT) | **−1.76 ± 0.60** HP at the published setting |
| Gumbel-Top-k root candidate selection | **−2.08 ± 0.55** HP at m = 4 |
| MAST (online per-card table) | Null at every weight tested |
| Rollout potion scoring | Null to negative at every setting tested |
| Honest draws would un-flatten the sims curve | Clairvoyant gains **+3.47 ± 0.74** from 100→900 sims; honest gains **+0.75 ± 0.79** — the reverse |
| Survival-weighted route planning (elite weight 3.0) | **−3.68 ± 1.10** floors, elite capture 3.3% → 89.2% |

### Measurement traps this project has actually fallen into

Read [docs/04-evaluation.md](docs/04-evaluation.md) before comparing two numbers.
The short list:

- **Validation NLL picks the wrong epoch.** It has done so three runs running.
  Decide promotions on paired floor comparisons.
- **Imitation metrics lie.** The routing clone had validation NLL falling
  monotonically for 18 epochs and human agreement rising 65% → 69%, while dying
  at floor 7.08.
- **Sweeping then reporting manufactures wins.** With ~100 boss fights per split,
  best-of-seven reliably produces a ~2 HP "improvement" from noise. Sweep on
  train or val; keep test for a single pre-registered setting.
- **A val spot-check is not a result.** One arm read +1.0 HP on 120 val fights
  and −0.58 ± 0.56 on 500 train fights.
- **Engine rebuilds invalidate comparisons.** Check the `.pyd` mtime against the
  eval log's.

---

## Recent updates

### 2026-08-01

**Version control established** — the tree had none. Two commits: baseline
snapshot, then this README's correction.

**Re-baselined v28/v31 on the rebuilt engine.** The prediction in the docs held
exactly: ranking survives, absolute numbers move down. v31 23.57 → **23.05**,
v28 21.76 → **21.64**, paired margin +1.80 → **+1.41 ± 0.50**.

**Five new engine parameters, all defaulting to verified no-ops** (250 val fights
byte-identical across the rebuild). Every one measured null or negative and is
deliberately **not** in `tune_search_human.py`'s search space, following the
precedent `power_horizon_weight` set:

| parameter | what it does |
|---|---|
| `mast_weight`, `mast_min_visits` | MAST — online per-(card, upgraded) average-return table biasing the rollout |
| `seq_halving_candidates` | Gumbel-Top-k root candidate selection before sequential halving |
| `backup_max_weight` | blends Monte-Carlo backup toward Max-Monte-Carlo (MaxUCT) |
| `honest_draw_order` | removes draw-order clairvoyance via in-tree order resampling |
| `rollout_potion_*` (5) | potion scoring in the rollout, which previously scored every non-CARD action a flat 5.0 |

**`honest_draw_order` is the one worth knowing about.** The mechanism was more
specific than "the root copy carries the order": drawing from an ordered pile
consumes no RNG, so the `nativeRngCounterSum` probe classified every mid-turn
draw card as deterministic and cached a single order. Only END_TURN was ever a
chance node. It costs −4.78 ± 0.71 HP, and the case for enabling it is honest
measurement and a trustworthy live bridge — **not floors**.

**Auxiliary heads verified as trained and useful** — `next_combat_survival` at
AUC 0.817 and `next_combat_hp` at R² 0.302 are a working combat-outcome model,
which is what a run-level planner needs to price a route without simulating
fights. `terminal_floor` is not usable (R² 0.025, badly under-dispersed); use
`run_critic.py`'s value head instead.

**Batched forward for the whole-run transformer** (`lightspeed/batched_policy.py`,
**3.66×** at batch 32). Nothing in this project had ever run the model on a
batch — `train_whole_run_v27.py`'s `--batch` is gradient accumulation over a
per-row loop. Verified against the single-observation path to 1e-4.

**A stale documentation claim corrected.** `attackDamageScoreWeight`,
`selfDamageScorePenalty`, `blockWeight` and `winHpFractionWeight` were documented
as "still off at 0.0" and are all four tuned on in the shipped config. Found by a
test that had been failing since the config shipped.

### 2026-07-31

Layer swap isolating the run policy (+15.71 floors). Human-deck combat benchmark
built, with Silver Automaton as a second arm on identical fights. Draw-order
clairvoyance found and costed. Eight scoring terms turned on (+4.99 ± 0.69 HP).
Card-select enumeration fixed for the other three characters. Engine validated
against the game's own bytecode — four monster behaviours, seven inert relics,
two card costs corrected. Human archive imitation refuted.

### 2026-07-30

Armaments upgrade leak fixed — combat upgrades were writing through to the master
deck permanently, worth ~3 floors. `cardColors[]` corrected in 8 entries. SCRY
segfault fixed. Live-bridge card-identity bug fixed (played cards were never
removed from hand).

---

## Open problems

Roughly in order of expected value.

1. **On-policy RL for the run policy.** Imitation is refuted and representation is
   closed, leaving this as the path to the +15.71 floors the layer swap proved
   available. Episodes are cheap (0.54 s at 100 combat sims); the update is the
   bottleneck at ~22 ms/transition batch-1, which `batched_policy.py` improves
   3.66× and process-parallel gradient accumulation improves further.
2. **Route planning, second attempt.** The framing is untested — only one
   parameterisation is refuted. The planner applies a single survival probability
   to every fight, so an elite carries the same modelled risk as an ordinary
   monster while being worth more, and it takes 89% of them. Give ELITE nodes a
   harsher survival factor, and sweep the elite weight down.
3. **Live-bridge monster intents.** The bridge discards the telegraphed intent and
   rolls its own guess, **wrong 87.5% of the time**, with failure modes that zero
   `dangerFraction` and suppress every Defend in hand. The mapping is 21
   `(monster, move_id)` pairs. Mechanical, and the largest real-world combat gain
   on the books.
4. **Live-path robustness.** Entropic Brew hangs the bridge (`potionRng` unseeded
   in `nativeSeedRng`); Stance Potion deadlocks the battle (Watcher-only).
5. **Differential testing.** Constants are verified against the game; *behaviour*
   is not — modifier ordering, monster move-selection AI, helper-action values.

Not worth doing, with reasons in the docs: tree reuse (1.34× ceiling), state
merging, fixing Runic Dome clairvoyance (3% of runs, makes them harder).

---

## Reading order

1. [docs/01-architecture.md](docs/01-architecture.md) — what the pieces are and
   which one is authoritative.
2. [docs/02-training-pipeline.md](docs/02-training-pipeline.md) — how a label is
   made and how the model is trained.
3. [docs/04-evaluation.md](docs/04-evaluation.md) — before you compare two
   numbers.
4. [docs/03-combat-search.md](docs/03-combat-search.md) — the search, its tuning,
   and its long dead-ends table.
5. [docs/07-known-issues.md](docs/07-known-issues.md) — open defects and
   behavioural weaknesses.
6. [AGENTS.md](AGENTS.md) — conventions and hazards, if you are going to change
   code.

A claim in the docs without either a file:line reference or a harness in
[docs/README.md](docs/README.md)'s script table should be treated as unverified.
