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

Design intent that has **not** been implemented is in
[`FULL_RUN_RL_DESIGN.md`](FULL_RUN_RL_DESIGN.md). Conventions and hazards for
anyone changing code are in [`AGENTS.md`](AGENTS.md).

> **On `docs/`.** This README cites a `docs/` tree — twelve files recording every
> measurement, the harness behind each, and the reasoning connecting them. **It is
> not published in this repository.** References to `docs/…` below are to material
> kept outside it, and are left in place because they say precisely where a claim
> came from. Everything asserted here is also backed by a harness under
> `slay-sim/lightspeed/`, which *is* published.

---

## Where it stands

| | |
|---|---|
| Best checkpoint | `slay-sim/runs/whole_run_transformer_yield10x_a20_v31.pt` |
| Ascension 20 | mean floor **23.05** of 56 over 200 paired seeds; **0 victories** |
| Ascension 0 | mean floor 39.21 over 100 seeds; **13 victories** |
| vs. previous best (v28, 21.64) | **+1.41 ± 0.50** paired floors, t = 2.82 |
| Combat vs. a top human | **−5.78** mean HP against his own decks (0 = parity) |
| Tests | **412 passing** in 22 files (1 xfail: a known open defect) |
| Engine parameters | 68 runtime-tunable, 42 overridden by the shipped config |

> **In flux as of 2026-08-01.** The borrowed play-priority table is removed in
> source but not yet in the binary — the rebuild is blocked by a running job. On
> landing, the parameter counts become **66 / 40**. See the update log.

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

412 tests. The import check is the real test that the native build worked.

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
tracked**, MIT licensed. `.gitignore` carries per-entry reasoning; in short:

| excluded | why |
|---|---|
| `silverbot-reference/` | 348 MB vendored read-only fork — reference material, and nothing tracked here depends on it |
| `slay-sim/runs/` | 1.3 GB of checkpoints, datasets, eval output — regenerable |
| `sts_lightspeed/build*/` | five CMake trees including the `.pyd` |
| `*.pt` `*.pyd` `*.jsonl` `*.log` | by extension, because ~25 checkpoints sit inside `slay-sim/lightspeed/` intermixed with source |
| `docs/` | the documentation tree, deliberately unpublished |

`docs/` is **ignored** — see the note at the top of this file. It exists in the
working tree and is not published.

### No harness depends on an external agent

The three scripts that required Daniel Ziegler's fork to be present —
`eval_heart1_hybrid.py`, `collect_heart1_labels.py` and
`_silverbot_human_deck.py` — were removed on 2026-08-01, so everything tracked
here runs against this repository alone. The whole suite passes with no external
agent installed.

**The findings those harnesses produced remain valid and are recorded in
`docs/README.md`** — the layer swap (+15.71 ± 3.13 floors), the
combat bracket against a second agent, and the routing-coefficient comparison.
**They are no longer reproducible in-tree.** The layer swap in particular is the
measurement behind this project's stated top priority, so anyone revisiting that
conclusion will need to reconstruct the harness. What it did was mechanically
simple: drive our `GameContext` with an external overworld policy while
`native_playout_current_battle` owns every combat decision, which holds combat
byte-identical and isolates the run layer.

**Security note.** The original `sts_lightspeed/.git/` carried a GitHub Personal
Access Token embedded in its origin URL, of the form
`https://github_pat_...@github.com/gamerpuppy/sts_lightspeed.git`.

What is verified: that history was discarded, and searching this machine found no
surviving copy — no git config under the project tree, no global credential
helper, no credential store, nothing in the source archive the project arrived
in. Nothing in this repository contains it, and it has no remote.

**It is not this account's token.** Checked against both the classic and
fine-grained token pages: nothing there. Combined with the fact that it
authenticated against *gamerpuppy's* repository from a `.git` directory inherited
rather than created here, the token belongs to whoever assembled this project
copy.

That bounds the exposure but does not close it. Removing local copies does not
invalidate a credential, and this account cannot revoke one it does not own — so
the remaining remedy is notifying whoever put the original tree together. Until
then it should be assumed live, on someone else's account.

Nothing in this repository contains it, and nothing here can leak it further.

---

## What has been established

The value of this project is largely in what has been *ruled out*. Most rows are
backed by a harness under `slay-sim/lightspeed/` — see
`docs/README.md` for the full script table. Rows marked † were
produced by a harness that has since been removed and are no longer reproducible
in-tree.

### Confirmed

| finding | measurement |
|---|---|
| The overworld policy is the binding constraint † | +15.71 ± 3.13 floors from swapping only that layer |
| The combat search is **draw-order clairvoyant** | −3.78 ± 0.84 HP/fight when blinded; our apparent lead over Silverbot was this cheat |
| The borrowed play-priority table was load-bearing | removing it costs **−1.20 ± 0.49 HP** (t = −2.45) on 500 paired train fights |
| Relic power level dominates combat tuning | +0.406 win rate at the encounter level, z = 8.10 — an order of magnitude above any other combat lever |
| v31's routing preference is **inverted**, not miscalibrated † | ELITE logit −2.55 against the human's +1.93 and Silverbot's +0.22 |
| v31 does not rest when hurt † | hp_frac × REST +1.19 against −1.93 (human) and −1.72 (Silverbot) |
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
| Progressive widening explains that flat honest response | **Refuted on val.** Looked like +4.27 ± 1.21 (t = 3.52) on a disjoint *train* slice; on val neither the grid best nor the hand-picked setting beats shipped (−0.69 ± 1.01 and −1.60 ± 0.93). Confirming on train is not confirming |
| Survival-weighted route planning (elite weight 3.0) | **−3.68 ± 1.10** floors, elite capture 3.3% → 89.2% |
| A canonical (unordered) draw pile beats a per-sample ordered one | **+0.48 ± 0.58** (t = 0.84) on 150 train fights at k = 3 — inside the noise floor. See below |

### Measurement traps this project has actually fallen into

Read `docs/04-evaluation.md` before comparing two numbers.
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
- **One search seed per fight is not a measurement.** With the config held
  identical and only the seed set changed, 250 paired fights reported
  **+1.60 ± 0.68, t = 2.34** — a significant result from changing nothing, with
  151 of 250 fights differing. Pairing controls the fight, not the search:
  `rollout_temperature` is 2.489, so the rollout samples, and any change that
  alters one action re-rolls the rest of that fight. Averaging 3 seeds cuts the
  per-fight sd 13.73 → 7.82 and the spurious t to 0.86. `_param_ab.py` defaults
  to `--seeds 3`; **effects of 1–2 HP measured at one seed are not
  distinguishable from noise**, which covers most of the combat results here.

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

**The belief-search headroom was already banked, and a canonical pile adds
nothing on top.** Silverbot reports a large win for averaging over draw orders
inside the tree versus committing to one sampled order per decision. That gap is
not available here: `honest_draw_order = 1` already seeds its permutation from a
per-sample index, so every DPW chance sample and every rollout gets its own
order — the averaging shape, not the committing one. The only residual leak was
narrower: a sample's order is inherited by its whole subtree, so a deep decision
can plan around draws it should not know. `honest_draw_order = 2` closes that by
re-permuting the remaining pile after every action that drew, anywhere in the
tree or the rollout, which is what an unordered pile with lazy draws would give
without restructuring `CardManager` (shared with the real-game path). It measures
**+0.48 ± 0.58 (t = 0.84)** — a null. The modes are genuinely distinct (116/150
fights differ), so this is a refutation and not a no-op. Keep `1`.

**A real bug fell out of it.** Card-select and scry actions are *positions* into
`drawPile`, and validity is what sits at that position — `SECRET_WEAPON` is legal
only if `drawPile[idx].getType() == ATTACK`. The permutation ran on the child copy
*after* the action had been enumerated against the parent's order, so the index
could land on a card of the wrong type; the engine then dumped the entire
`BattleContext` to stderr and applied the action anyway. It fired once in 900
plays, found by attributing a stray 6 KB stderr dump. Fixed by skipping the
pre-execute permutation for exactly those action types, which loses no honesty —
a card-select screen shows the player the pile, so the choice is over identities
and the order carries no hidden information to protect. Pinned by
`tests/test_honest_draw_pile.py`, which fails on the unguarded engine in both
regimes and passes on the fixed one. Worth **+0.82 ± 0.61** HP, i.e. not
measurably worth anything: skipping a shuffle changes how much `rng` is consumed,
so 96 of 150 fights simply re-roll. It is a correctness fix, not a gain.

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

**Route planning tried and refuted at one parameterisation.** Survival-weighted
planning over the act-map DAG, using the aux heads above to price routes, took
elite capture 3.3% → 89.2% and cost **−3.68 ± 1.10 floors**. Notably it did *not*
collapse the way the imitation clone did (floor 20.43 against that clone's 7.08),
so conditioning elite appetite on our own survival estimate does real work — just
not enough. The likely defect is applying one survival probability to every fight,
which makes an elite carry the same modelled risk as an ordinary monster.
`_route_planner.py` and `_eval_route_planner.py` remain; the eval takes any
map-decision rule.

**Three harnesses removed** — `eval_heart1_hybrid.py`, `collect_heart1_labels.py`
and `_silverbot_human_deck.py` all required an external agent installed. Nothing
tracked here now depends on one.

**A borrowed data table removed, and a bug found removing it.** The rollout's
per-card play-priority prior was Silver Automaton's hand-curated 133-card
ordering, re-encoded — verified as 132 of 133 entries matching their list exactly
(one transcription slip: `GHOSTLY_ARMOR` ranked 83rd where it should be 8th).
Removed to avoid carrying their data, at a measured cost of −1.20 ± 0.49 HP.
`_fit_play_priority.py` refills the slot from our own search via a conditional
logit over the cards available at each decision, which ranks Corruption, Fiend
Fire, Immolate and Uppercut at the top; it is **not wired in**, because its data
was collected on the engine that still has the status-card bug.

**Unplayable cards are playable** — see Open problems. Found while chasing an
anomaly in that fit, and the most consequential thing on this list.

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

## Licence

MIT — see [`LICENSE`](LICENSE).

`sts_lightspeed/` is a fork of gamerpuppy's engine and retains its own upstream
notice at `sts_lightspeed/LICENSE.md` (MIT, Copyright © 2021 gamerpuppy), as MIT
requires. The vendored `json/` (nlohmann) and `pybind11/` trees keep theirs; eight
licence files are tracked in total.

No code or data from Silver Automaton remains in this repository — the one item
that was theirs rather than an idea, a curated card play-priority table, was
removed on 2026-08-01. See Acknowledgements.

## Acknowledgements

The C++ engine is a fork of **gamerpuppy's** RNG-accurate `sts_lightspeed`, which
is what makes any of this possible — an engine that reproduces the real game's
random number generation exactly, so a simulated run and a real one from the same
seed agree.

**Daniel Ziegler's Silver Automaton** was a significant source of inspiration.
It is an independent, considerably more mature Slay the Spire agent, and several
ideas here came from studying it. Named where they appear in the code, the
notable ones:

- **The "we have enough block" gate.** `HeuristicContext::blockSufficient` is a
  port of their `SimpleAgent`'s block heuristic, found while investigating why
  our rollouts finished fights with less HP than theirs at matched simulation
  counts. It is live in the shipped config.
- **The card play-priority prior** — *removed 2026-08-01, and the one case where
  this project carried their data rather than their idea.* `silverCardPlayRank`
  was their hand-curated 133-card ordering re-encoded as a 372-entry lookup: 132
  of 133 entries matched their list exactly, with one transcription slip. It was
  load-bearing (−1.20 ± 0.49 HP to remove) and removed anyway, since a curated
  ranking is their work and not an algorithm. `_fit_play_priority.py` refills the
  slot from this project's own search.
- **The allocation-free rollout.** `nativeHeuristicPickFast` scans the hand
  without materialising a legal-action vector, the same idea as their
  `SimpleAgent::chooseBattleCardPlay`. That path was 71.7% of simulation cost.
- **Draw-order clairvoyance.** They found this defect in their own engine first
  and measured it at roughly +34pp before removing it. We would not have gone
  looking otherwise.
- **The routing diagnostics.** `_routing_audit.py`'s conditional logit and
  `--randomize-paths` intervention are both borrowed from their map program, and
  their published coefficients are the reference our policy is measured against.
- **Validating against the real game.** Their "source audit + live bridge"
  programme is what prompted checking this engine against the game's own
  bytecode, which found four monster behaviours, seven inert relics and two card
  costs.

Their fork is not required to build or test anything here, and no tracked code
depends on it.

The human benchmark replays 100 Ascension 20 Heart runs by **Baalorlord**, used
as a ground-truth reference for combat quality.

## Reading order

The documentation tree is not published (see the note at the top), so for anyone
working from this repository alone:

1. [`AGENTS.md`](AGENTS.md) — conventions, layout, and the hazards that have
   already cost time. Read this first if you are going to change code.
2. `sts_lightspeed/bindings/slaythespire.cpp` — the search. Long, and heavily
   commented with the measurement that motivated each parameter.
3. `slay-sim/lightspeed/whole_run_env.py` and `whole_run_transformer*.py` — the
   overworld environment and policy.
4. `slay-sim/lightspeed/_*.py` — the audit and measurement harnesses. Every claim
   in this README traces to one of these; their docstrings carry the reasoning.

A claim without either a file:line reference or a harness behind it should be
treated as unverified.

