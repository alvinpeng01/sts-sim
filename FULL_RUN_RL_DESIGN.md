# Full-Run RL Design

Status: initial design  
Scope: Ironclad, complete seeded runs, eventually A20 + Heart  
Primary simulator: this repository's `sts_lightspeed` fork  
Reference: [daniel-ziegler/sts_lightspeed](https://github.com/daniel-ziegler/sts_lightspeed)

## 1. Goal

Train an agent that learns to maximize complete-run success, including:

- combat play;
- path selection;
- card drafting and skipping;
- shops and card removal;
- campfires;
- events and Neow;
- relic and potion choices;
- key collection and Act 4.

The deployed system should use neural inference for overworld strategy and the tuned native
expectimax MCTS for combat. This is an intentional division of labour, based on the project's
own comparative testing: expectimax MCTS currently plays combat materially better than the
available neural checkpoints. Search and heuristics may still be used as teachers to bootstrap
the overworld policy.

The authoritative state must remain one persistent native `GameContext`. An episode must
never synthesize a new deck, HP total, relic set, or potion set between floors.

## 2. What already exists

### In this repository

- `GameContext` already implements the full overworld state machine.
- `GameAction::getAllActionsInState(gc)` already enumerates legal non-combat actions.
- `BattleContext` and `search::Action` provide the combat state and legal decisions.
- `lightspeed.env.IroncladFightEnv` already converts a `BattleContext` into combat
  observations and legal-action features.
- `lightspeed.policy.ActionScoringPolicy` is a trained variable-action combat actor/value
  network.
- `ScumSearchAgent2::playout()` proves that the native engine can advance a complete run.
- `Agent.playout_hybrid()` proves that the custom native combat search and the existing
  overworld state machine can be joined in one run.

### In the Daniel Ziegler reference

Silver Automaton provides a useful, tested design for the missing outer policy:

- expose `GameAction`, `ScreenStateInfo`, and a structured run observation through pybind;
- encode deck cards, relics, potions, map nodes, fixed state, and legal choices as tokens;
- jointly attend over state tokens and candidate-action tokens;
- score exactly the legal actions rather than use a fixed global action vector;
- bootstrap with supervised/self-play data and then train with PPO + GAE;
- collect full seeded runs in parallel and evaluate on paired held-out seeds.

Its current model is a four-layer, 256-dimensional, eight-head pre-norm transformer. Its
choice types are card, path, relic, potion, event/fixed action. It uses a value head and an
auxiliary destination-room classifier for map grounding.

We should port these interfaces and ideas selectively. We should not replace this fork's
engine wholesale: the two C++ trees have diverged substantially, and this fork contains
combat/card/binding work that must be preserved.

## 3. Architecture decision

Use a hierarchical full-run agent with two decision domains:

1. `RunPolicy`: decisions made on non-combat screens.
2. `CombatSolver`: the tuned native expectimax MCTS used inside `BattleContext`.

`RunPolicy` is the trainable RL model. `CombatSolver` is configured from
`lightspeed/tuned_search_params.json`, including its MCTS exploration, chance-node widening,
and heuristic evaluation weights. The neural combat model remains a useful research artifact
and possible later distillation target, but is not on the critical deployment path.

This is preferable to one flat policy for three reasons:

- a run contains far more combat decisions than overworld decisions, so a flat PPO loss
  would drown the scarce drafting/routing signal;
- combat and overworld legal actions have fundamentally different features;
- tuning the overworld policy against a strong, fixed combat controller gives much lower-variance
  credit assignment than training both controllers simultaneously;
- it exploits the project's current strongest demonstrated combat player.

The initial and deployment versions train only `RunPolicy`; the MCTS configuration is frozen
for a run/evaluation suite. Any later combat-policy work must prove itself on paired
pre-battle states before replacing MCTS.

## 4. Environment contract

Add `slay-sim/lightspeed/full_run_env.py`.

```python
class FullRunEnv:
    def reset(
        self,
        seed: int,
        ascension: int,
        heart: bool = False,
    ) -> RunObservation:
        ...

    def step(self, action: RunAction):
        """Execute one player decision."""
        ...

    def snapshot(self) -> RunSnapshot:
        """Serializable state for replay/debugging."""
        ...
```

`FullRunEnv.step()` advances exactly one player decision:

- outside combat: execute one native `GameAction`;
- in combat: execute one native `search::Action`.

Automatic engine transitions that provide no meaningful choice may be advanced internally,
but every skipped transition must be logged. Actions that are "always take" should initially
remain explicit; only remove them after proving that doing so does not hide a real trade-off
(for example, a relic versus the sapphire key is not automatic).

### Required invariants

- Every returned action is valid according to the native engine.
- Executing the same seed and action trace produces the same terminal state.
- A battle starts from the current `GameContext`.
- `BattleContext::exitBattle(gc)` writes HP, potion usage, relic counters, rewards, and
  outcome back into the same `GameContext`.
- No Python code reimplements action legality.
- An empty legal-action set while the run is undecided is an error with a diagnostic dump,
  not an implicit skip.

## 5. Native binding work

Port the smallest useful interface from the reference repository into this fork.

### Required

Expose:

```python
sts.GameAction.get_all_actions_in_state(gc)
action.is_valid(gc)
action.execute(gc)
action.bits
action.kind
action.idx1
action.idx2
action.rewards_action_type

gc.screen_state_info
sts.get_run_observation(gc)
```

Expose enough `ScreenStateInfo` data to describe:

- reward cards, relics, potions, gold, and keys;
- boss relic choices;
- shop inventory and prices;
- card-select candidates and selection type;
- event identity, phase, costs, and offered objects;
- reachable map columns and destination node information;
- campfire choices;
- treasure-room choices.

For combat, expose a safe lifecycle:

```python
bc = sts.begin_battle(gc)  # initialize from the current run
...
sts.finish_battle(bc, gc)  # calls BattleContext::exitBattle
```

The lifecycle should reject beginning a battle from a non-battle screen and finishing an
undecided battle.

### Do not port blindly

Daniel's fork moved and rewrote several core systems, including `GameAction`, action queues,
RNG handling, and battle search. Copying whole C++ files would risk silently discarding this
fork's combat changes. Port binding fields and small helpers against this fork's existing
types instead.

## 6. Observation schema

### Shared persistent-run tokens

`RunPolicy` receives:

- one token per deck card: card ID, upgrade count, bottled flags;
- one token per relic: relic ID and normalized counter/data;
- one token per potion slot: potion ID and empty/occupied state;
- fixed state token(s): current/max HP, gold, floor, act, ascension, keys, potion capacity,
  boss, current screen, card-removal cost;
- one token per map node.

### Map-node features

Each node should include:

- room type;
- absolute `(x, y)`;
- relative position to the current node;
- current/reachable flags;
- outgoing-edge columns;
- burning-elite flag;
- minimum/maximum elites reachable through its forward cone;
- distance to the nearest rest and shop;
- whether the burning elite remains reachable through that option.

The reference implementation found that raw graph tokens alone did not reliably teach
multi-hop routing. Therefore each path candidate should also carry a precomputed forward-cone
summary. These are deterministic features derived from the visible map, not privileged
information.

### Combat observation

Keep the current `IroncladFightEnv` combat representation for the first implementation:

- the 48 scalar state features;
- variable legal-action features;
- card, monster, relic, and potion identities.

Add run context that matters during combat:

- current floor and act;
- current/max run HP;
- whether the fight is normal, elite, boss, or Act 4;
- potion opportunity cost;
- current gold when combat effects can change it.

Avoid injecting future hidden information.

## 7. Typed action schema

All non-combat choices become a variable-length list of typed candidates:

```python
@dataclass(frozen=True)
class RunAction:
    native_action: object
    action_type: ActionType
    object_id: int
    numeric: tuple[float, ...]
    metadata: dict
```

Initial action types:

- `CARD`: obtain, remove, upgrade, transform, duplicate, bottle, skip-related card choices;
- `PATH`: choose a reachable map column;
- `RELIC`: boss relic, shop relic, chest relic/key trade-off;
- `POTION`: obtain, replace, discard, buy;
- `FIXED`: rest, smith, recall, dig, lift, toke, leave, event options, Neow;
- `SHOP`: purchase/leave actions when object identity and price are relevant.

The model scores candidates, not enum slots. Fixed/event actions still require stable semantic
IDs so that "leave", "pay gold", and event-specific choices do not collapse into an
uninterpretable index.

Every candidate must retain its exact native action. After the network selects candidate
`i`, the environment executes that object rather than reconstructing an action from features.

## 8. Run policy network

Start with the reference's set-transformer pattern:

```text
state tokens + candidate-action tokens
                 |
       4 transformer blocks
       dim 256, 8 attention heads
                 |
      per-candidate scalar scorer
                 |
     masked softmax over legal actions
```

Recommended components:

- shared card/relic/potion embedding tables;
- type embedding for every token;
- RMSNorm pre-norm blocks;
- masked mean pooling for the value head;
- destination-room auxiliary head over path candidates;
- optional deck-summary auxiliary heads such as card count and upgrade count only for
  representation validation, not as reward.

The run policy should not initially share the combat state encoder. It may initialize
card/relic/potion embedding rows from the combat checkpoint, but the projection into the
256-dimensional run-token space should be learned separately.

## 9. Critic and temporal abstraction

Use `V_run(s)`, which predicts final run return at non-combat decisions and battle boundaries.

The outer trajectory treats a completed battle as one semi-Markov transition:

```text
pre-battle run state -> complete combat -> post-battle run state
```

This prevents hundreds of combat steps from dominating the run-level GAE sequence. The MCTS
receives the real deck, HP, relics, potions, and encounter from the persistent `GameContext`;
the resulting post-battle state is the next state for `RunPolicy`.

For strict policy-invariant shaping, use a frozen target copy of `V_run` during collection.
Do not let a rapidly changing critic define the reward inside the same optimizer step.

## 10. Reward design

The true task reward is ordered run progress:

- death before Act 3: lowest;
- deeper floors: higher;
- Act 3 completion with insufficient keys: below reaching Act 4;
- Act 4 death: below Heart victory;
- Heart victory: highest.

Use `gamma = 1.0` for the outer run trajectory. With finite episodic runs, this preserves
the objective and makes potential-difference shaping telescope cleanly. Use GAE lambda near
`0.97`.

Initial Heart reward:

```text
floor progress        capped at 0.30
keys after Act 3      0.05 each
Heart victory         +0.60
```

Exact constants are configurable and must satisfy monotonic ordering tests.

Potential-based shaping may include:

- current HP fraction;
- max HP;
- upgrades;
- relic count;
- starter-card removals;
- keys.

Every shaping feature must be applied as `gamma * Phi(s') - Phi(s)`, with tests proving that
the undiscounted shaped return differs from the base return only by the initial-state constant.
Do not reward raw card count, gold acquisition, damage dealt, or relic acquisition directly.

Combat-model distillation may retain the current dense fight reward, but it is a separate
research track. It must not change the primary full-run reward or displace MCTS without a
paired-seed win-rate and runtime comparison.

## 11. Training stages

### Stage 0: correctness and replay

- Expose the native run action API.
- Execute at least 10,000 heuristic full runs.
- Persist seed plus native action bits.
- Replay every sampled trace and compare terminal floor, HP, deck, relics, potions, keys,
  outcome, and RNG counters.
- Build screen/action coverage reports.

No RL training begins until deterministic replay is reliable.

### Stage 1: fix the combat controller

- Load `tuned_search_params.json` into native expectimax MCTS.
- Set a fixed simulation budget and timeout per encounter tier.
- Benchmark the exact configuration on recorded pre-battle states and preserve its results as
  the combat baseline.
- Freeze search parameters for each RL experiment; changing them changes the environment.

### Stage 2: overworld behavior cloning

Generate trajectories using:

- tuned expectimax MCTS for battles;
- `ScumSearchAgent2` or the reference-style heuristic policy outside battle;
- Boltzmann/random perturbations so the dataset is not one deterministic action per state.

Train the run transformer to predict selected actions and final outcomes. This stage is only
an exploration bootstrap.

### Stage 3: run-policy PPO

- Frozen tuned expectimax MCTS.
- Complete full-run collection.
- PPO with GAE on outer decisions/battle boundaries.
- 256-512 games per iteration, adjusted to measured local throughput.
- Separate game seed from policy-sampling seed.
- Parallel CPU simulators with batched inference.
- Advantage normalization across the whole iteration, never independently per run.
- Annealed entropy and learning-rate schedules.

### Stage 4: optional combat distillation research

- Record MCTS actions and values on representative pre-battle states.
- Distill into a neural combat policy only when inference speed is worth the expected quality
  loss.
- Compare neural, MCTS, and hybrid choices on paired states and complete runs.
- Keep MCTS as the deployed default unless the neural policy wins under the same time budget.

### Stage 5: ascension and Heart curriculum

Train on a mixture, not a one-way sequence:

1. A0 Act 1;
2. A0 complete Act 3;
3. A0 Heart;
4. mixed ascensions;
5. increasing mass on A20 Heart.

Promotion is based on held-out paired-seed evaluation. Earlier tasks remain in the mixture.

## 12. Collection and performance

The simulator is CPU-heavy and inference is GPU/batch-friendly. Use:

- many persistent native environments in worker processes;
- one batched inference service;
- immutable policy snapshots during a collection iteration;
- pipelined collection of iteration `N+1` while optimizing iteration `N`;
- bounded queues and timeouts with seed/action dumps on failure.

Persist trajectories in a columnar format such as Parquet:

- seed and sampling seed;
- ascension and Heart mode;
- observation/action tensors or reconstructable native snapshot;
- legal candidate descriptors;
- chosen candidate index;
- native action bits;
- old log probability and value;
- reward, advantage, and return;
- terminal metrics;
- model/checkpoint and engine version.

## 13. Evaluation protocol

Maintain immutable seed suites:

- development seeds;
- held-out validation seeds;
- final test seeds, never used for tuning.

Every comparison is paired on identical seeds and engine version.

Report:

- Act 1/2/3 completion;
- Act 3 and Heart win rate;
- confidence intervals;
- median death floor;
- boss and elite survival;
- HP and potion inventory at battle boundaries;
- card picks/skips/removals/upgrades;
- map, rest, shop, and key decisions;
- inference and simulation time;
- failure/replay-divergence counts.

Run interventions:

- learned versus random routing with all other decisions fixed;
- learned versus heuristic drafting;
- neural combat versus tuned MCTS on recorded pre-battle states;
- fixed MCTS versus any proposed distilled/hybrid replacement.

## 14. Tests required before training

1. Every non-combat `ScreenState` maps all native actions to unique candidates.
2. Candidate-to-native-action round trips preserve action bits.
3. Candidate masking never exposes an invalid action.
4. Map choice tokens identify the correct destination node and forward cone.
5. Reward-screen skip cannot accidentally abandon uncollected mandatory rewards.
6. Sapphire/emerald key trade-offs remain explicit.
7. Battle start/finish preserves persistent state.
8. Same seed + action trace produces the same terminal snapshot.
9. Potential-shaping invariance holds numerically.
10. PPO logits, masks, chosen indices, and stored old log probabilities agree after
    serialization.

## 15. First implementation milestone

Milestone 1 is a non-learning, replayable vertical slice:

```text
GameContext reset
  -> enumerate native overworld actions
  -> encode typed candidates
  -> heuristic outer choice
  -> learned combat-policy battle
  -> finish battle into same GameContext
  -> continue to terminal outcome
  -> save and replay action trace
```

Deliverables:

- pybind run-action and battle-lifecycle APIs;
- `full_run_env.py`;
- typed action/observation dataclasses;
- heuristic full-run driver using the learned combat model;
- deterministic trace serializer/replayer;
- coverage and throughput report over at least 1,000 runs.

Only after this milestone passes should the transformer and PPO collector be added.

## 16. Explicit non-goals for the first milestone

- replacing the native engine with Daniel's fork;
- training all four characters;
- end-to-end differentiability through the simulator;
- a single flat action vocabulary;
- using live-game observations during training;
- claiming A20 competence from isolated-fight win rates.
