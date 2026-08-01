# Architecture

Everything below was read out of the working tree on 2026-07-30. Line references
and numbers come from the code, not from earlier documentation.

## What this project is

A Slay the Spire agent for the Ironclad, built from two halves that meet at one
pybind11 boundary:

- **`sts_lightspeed/`** — a C++17 fork of gamerpuppy's RNG-accurate STS engine,
  extended with an expectimax MCTS combat search and Python bindings. This is
  the training and evaluation runtime.
- **`slay-sim/`** — the Python side: a transformer policy for overworld
  decisions, the label-generation and training pipeline, the evaluation harness,
  a second independent pure-Python STS engine, and a live-game bridge.

**Combat is played by search, not by a network.** The policy only ranks legal
overworld actions — card picks, paths, campfires, shops, events, Neow. When
`WholeRunEnv` reaches a battle it calls
`sts.native_playout_current_battle_result()` and the whole fight is resolved in
C++ before the policy is consulted again (`whole_run_env.py:266-283`). The
network is never asked about a card play.

## Quick reference

| Question | Answer | Source |
|---|---|---|
| Character | Ironclad only | `sts.CharacterClass.IRONCLAD` hardcoded in 5 non-test files |
| Combat AI | Native expectimax MCTS, **sequential halving at the root** under the shipped config | `slaythespire.cpp:2245,2346`; `tuned_search_params.json` |
| Overworld AI | `WholeRunTransformerPolicyV27`, dim=96 / layers=2 / heads=4, **1,617,935 params** | measured by instantiating the class |
| Training | Outcome-supervised on soft targets from counterfactual rollouts. No PPO on the whole-run path. | `train_whole_run_v27.py` |
| Current best | `runs/whole_run_transformer_yield10x_a20_v31.pt` | [05-model-lineage.md](05-model-lineage.md) |
| Search params | 55 tunable doubles, all exposed to Python; 29 overridden by the shipped config | `slaythespire.cpp:120-332` |
| Tests | 157 tests in 19 files, all passing | `python -m pytest -q` |
| Ascension | 20 by default (`RunConfig.ascension = 20`) | `whole_run_env.py:170` |

## Two engines, and why

There are two complete, independent STS implementations here. Both get called
"the engine" informally, so:

| | `slay-sim/sts/` | `sts_lightspeed/src/` |
|---|---|---|
| Language | Python, ~11.3k lines | C++17 |
| Used for | Tests, the live-game bridge, standalone demos | All training, all evaluation, all search |
| Card coverage | Every Ironclad, Colorless and Curse card (`cards.py`, 4,600 lines) | **All four characters.** 75/75 Ironclad, 75/75 Silent, 75/75 Defect, 75/75 Watcher — see [07-known-issues.md](07-known-issues.md) |
| Monster coverage | Acts 1–3 incl. every Act 1 boss (`enemies.py`, 1,910 lines) | Full |
| Player choices | No choice-resolution mechanism — "choose a card" effects fall back to a random pick | Full `CardSelectTask` machinery (DISCARD, RETAIN, SETUP, NIGHTMARE, …) |

They are not kept in parity and must not be assumed to agree. The Python engine
exists because it is far easier to test and extend, and because the live bridge
needs to reconstruct state from JSON without a C++ build.

## The pybind11 boundary

```
slay-sim/lightspeed/*.py
        │  import slaythespire              (sts_lightspeed/build/ on PYTHONPATH)
        ▼
sts_lightspeed/bindings/slaythespire.cpp    5,310 lines — the authoritative runtime
        │
        ├── GameContext                     seeded overworld state machine
        ├── GameAction::getAllActionsInState(gc)
        ├── native_playout_current_battle_result()   plays an entire fight
        │      └── nativePlayoutBattle()  → nativeRunMctsSearch()  per decision
        ├── nativeHeuristicPickFast()       rollout policy
        ├── nativeExpectimaxTerminalReward()  leaf/terminal value for the search
        ├── nativeTerminalReward()          the *other* terminal reward, used by env.py/PPO
        └── get/set_search_params()         TunableParams — unlocked global state
```

`getNNRepresentation(gc)` is what the policy actually observes: fixed scalars,
deck ids + upgrade counts, relic ids + counters, potions, and the full map
(node coordinates, room types, outgoing edge x-coordinates).

### Which search is authoritative

Three search implementations exist and the naming has caused real confusion:

| Path | Status |
|---|---|
| `bindings/slaythespire.cpp` native MCTS | **Authoritative.** The entire whole-run pipeline runs here. |
| `lightspeed/expectimax_search.py` | The authoritative *Python* search. Used by `distillation.py`, `train_distillation_expectimax.py`, `train_policy_net.py`. Not on the whole-run path. |
| `lightspeed/az_search.py` | NN-guided PUCT, deprecated for combat. Retained because `expectimax_search.py` imports its DPW / transposition / RNG-probe machinery. `az_search_debug.py` is a debug artifact. |

`slay-sim/AGENTS.md` says "`expectimax_search.py` is the authoritative combat
search, NOT `az_search.py`" — true within the Python stack, and easy to misread
as a claim about the whole-run pipeline. `tune_search_cma.py` likewise describes
its parameters as "`expectimax_search`'s", but it calls
`sts.set_search_params()` (`tune_search_cma.py:369`) and therefore tunes the
**native** search.

## The model

`WholeRunTransformerPolicyV27` extends `WholeRunTransformerPolicy`. At
dim=96 / layers=2 / heads=4 it is **1,617,935 parameters**; at
dim=192 / layers=3 / heads=6 it is **6,574,511**.

Base class (`whole_run_transformer.py`) — state tokens are a learned CLS-like
summary of 10 fixed scalars, plus one token per deck card, per relic, per
potion, and per map node:

| Component | Shape |
|---|---|
| `card` / `relic` / `potion` / `room` embeddings | 400 / 200 / 100 / 16 × dim |
| `action_content` embedding | 700 × dim — one shared vocabulary: `1+CardId`, `400+RelicId`, `600+PotionId` |
| `event`, `neow_bonus`, `neow_drawback` | 64 / 32 / 16 × dim, zero-init |
| encoder | `nn.TransformerEncoderLayer`, ff = 4×dim, dropout 0, GELU, pre-norm off |
| `score` | (3·dim → dim → 1) over `[state, action, action_content]` |
| `act_score` | 5 residual heads, one per act, zero-init |
| `phase_score` | 20 residual heads (5 acts × 4 floor phases of 6 floors), zero-init |
| `human_score` | zero-init residual, applied **only** on screens 2 and 3 (rewards, boss relic) |
| `value` | (dim → dim → 1) on the state token |

V27 subclass adds, all zero-initialized so they start as an exact no-op:

- **10 decision experts** — one per screen id, gated by `_expert_id()`: screens
  0–8, with 9 reserved for Neow (screen 1 carrying non-zero Neow bonus ids).
- **3 uncertainty heads** — a bootstrap ensemble. Their mean is the production
  logit; their standard deviation is the reported uncertainty. At inference the
  three MLPs are algebraically fused into one linear pair
  (`_refresh_fast_ensemble`) so production pays for one head, not three.
- **6 auxiliary heads** — `next_combat_survival`, `next_combat_hp`,
  `next_rest_reach`, `act_boss_survival`, `next_act_entry_hp`, `terminal_floor`.
- **3 structured adapters** — deck summary (16 features: mean card
  type/rarity/innate/strike/starter/upgraded, deck size, upgrade fraction),
  strategic context (12: HP ratio, max HP, gold, floor, act, floor phase, …),
  action card structure (14 per candidate card).

Note that `human_score` lives in the **base** class, not v27, and that
`act_score`/`phase_score` are additional per-act and per-phase residual heads
that no earlier documentation mentioned.

### Card encoding

Cards are embedded by **raw `CardId` enum value** — `deck_ids` comes straight
from `rep.deck.cards` and is clamped to `[0, 399]`. The model does **not** use
the C++ `NNInterface` one-hot vocabulary; nothing in `lightspeed/` references
`NNInterface` at all. This matters for the card-color bug in
[07-known-issues.md](07-known-issues.md): that bug corrupts the `NNInterface`
vocabulary, which the whole-run lineage never touches.

## Directory layout

```
sts-project/
  README.md                      setup + current state
  AGENTS.md                      how to work in this repo
  FULL_RUN_RL_DESIGN.md          design intent; not implemented
  docs/                          this documentation set

  slay-sim/
    sts/                         pure-Python engine
      cards.py 4600  enemies.py 1910  powers.py 877  combat.py 510  relics.py 253
      value_net.py 175           POWER_VOCAB at :47 — append-only, order is load-bearing
      bridge/                    communication_mod.py, native_recommend.py,
                                 state_mapper.py, predict.py
    lightspeed/
      whole_run_env.py           RL environment over the native GameContext
      whole_run_transformer.py   base policy
      whole_run_transformer_v27.py   experts, uncertainty, auxiliary heads, adapters
      v27_features.py            the structured feature blocks the adapters read
      generate_whole_run_rollouts.py       label generation (single shard)
      parallel_generate_whole_run_rollouts.py   sharded, resumable wrapper
      train_whole_run_v27.py     the training loop
      eval_whole_run_policy.py   paired evaluation harness
      search_config.py           loads / applies / verifies the native search config
      tune_search_cma.py         CMA-ES over the native search parameters
      tuned_search_params.json   the active configuration (29 overrides + fitness config)
      run_label_quality_v31.py   current experiment launcher (arms 300 / 800 / yield)
      run_long_training_v26/v28/v30.py, run_v*.cmd    superseded launchers, kept as provenance
      env.py, ppo.py, policy.py  the older combat-only PPO stack
    stsmod/                      BaseMod overlay (Java) + prebuilt STSPredictor.jar
    tests/                       19 files, 157 tests
    runs/                        checkpoints, datasets, manifests, eval .jsonl, logs

  sts_lightspeed/
    src/combat/                  BattleContext, Actions, Player, Monster, CardManager
    src/game/                    GameContext, Deck, Map, Shop, Neow, SaveFile
    src/sim/search/              Action (32-bit packed), SimpleAgent, ScumSearchAgent2
    include/constants/           card / relic / monster / potion / room data tables
    bindings/slaythespire.cpp    bindings + native MCTS (5,310 lines)
    build/slaythespire.cp313-win_amd64.pyd

  silverbot-reference/           read-only reference fork (Daniel Ziegler)
```

## Conventions that are load-bearing

- `from __future__ import annotations` at the top of every Python file.
- No packaging. Everything runs from `slay-sim/` with `PYTHONPATH` set to
  `../sts_lightspeed/build` and `.`.
- Training and launcher scripts are single-use per experiment; copy and modify
  rather than parameterizing an old one. The `run_*.cmd` files are provenance
  records, not a build system.
- `sts.set_search_params()` mutates unlocked process-global state. Two
  configurations must never be in flight in one process — CMA-ES evaluates each
  candidate in its own worker **process** for exactly this reason
  (`tune_search_cma.py:23-27`).
- `POWER_VOCAB` in `sts/value_net.py:47` is append-only; reordering it silently
  changes what a trained `value_net` means. This applies to that net only, not
  to the whole-run transformers.
- C++: PascalCase types, camelCase methods, UPPER_SNAKE_CASE enums, fixed-size
  containers (`fixed_list.h`) on hot paths, `FOREACH_ACTIONTYPE` macro codegen.
