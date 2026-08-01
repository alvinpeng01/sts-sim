# Training pipeline

Read out of `generate_whole_run_rollouts.py`, `parallel_generate_whole_run_rollouts.py`
and `train_whole_run_v27.py` on 2026-07-30.

The loop is: play seeded runs with the current policy, spend MCTS on
counterfactual continuations to score each legal action at a handful of
decisions, softmax those scores into a target distribution, then train the next
policy on those distributions. There is no PPO anywhere on the whole-run path;
`ppo.py` and `env.py` belong to the older combat-only stack.

## Stage 1 — label generation

Entry point: `parallel_generate_whole_run_rollouts.py` (shards across worker
processes and merges), which runs `generate_whole_run_rollouts.py` per shard.

### How one label is made

For a state with ≥2 legal actions (`label_state`, `generate_whole_run_rollouts.py:180`):

1. For each `rollout_index` in `range(rollouts)` and each legal action:
   copy the `GameContext`, execute that action, then **continue the run** under
   the current policy for up to `rollout-decisions` decisions, sampling actions
   at `policy-temperature`. All siblings within a rollout index share
   `matched_seed` — common random numbers, so the comparison is paired.
2. Score the continuation (`rollout_score`):
   `0.10·floor_gain + 1.50·act_gain + 0.12·hp_fraction`, `−0.40` on a loss,
   `+3.0` on a victory, `+0.05` per key in Act 4, plus a boss-progress credit
   for how much of a boss's HP a fatal branch removed.
3. Average across rollouts to get a mean per action, and take
   `softmax(means / (label_temperature + mean_standard_error))`. The added
   standard error is what keeps noisy or nearly-tied candidates soft instead of
   collapsing to a brittle one-hot.

A row stores the compacted observation, `target_probabilities`, `mean_scores`,
`standard_errors`, the raw `per_rollout_scores` matrix, the decision type, the
run seed, floor, act, and the priority score.

Cost: one label costs `rollouts × actions` continuations of up to
`rollout-decisions` steps each — roughly 1,400 simulated decisions, each of
which may itself contain a full MCTS-resolved fight.

### Which decisions get labelled

Nine decision types, derived from the screen (`DECISION_TYPES`):
`neow, event, map, rewards, shop, rest, boss_relic, card_select, treasure`.

A decision is **structurally eligible** when it has ≥2 actions, is a requested
type, and sits inside `[--min-act, --max-act]` and `[--min-floor, --max-floor]`.
Eligible decisions are then filtered twice more:

- **Budget caps** — `--per-type`, `--max-labels`, and `--max-labels-per-episode`.
- **Priority sampling** — when `--priority-accept-base < 1`, accept with
  probability `min(1, base · 2^priority_score)` where `priority_score` adds
  2.0 for HP ≤ 35%, 1.0 for floor ≤ 8, 1.0 for act ≥ 2, 1.0 for
  event/shop/rest/boss_relic, **6.0** when the safety filter removed an
  immediately-losing action, and 1.0 when the policy's top two logits are within
  0.5. At `base = 0.10`, an ordinary decision is kept 10% of the time; a
  safety-filtered one is kept with certainty.

Everything not labelled is still *played* at full MCTS cost. The generator
counts `eligible`, `capped` and `priority_rejected` separately precisely so this
discard rate is visible.

**Main-line play is argmax, not sampling.** When a decision is labelled the run
follows the label's argmax; otherwise it follows `argmax(policy_logits)`
(`generate_whole_run_rollouts.py:511-517`). `--policy-temperature` applies only
*inside* counterfactual continuations.

### Generation parameters

| Flag | Meaning | v28 | v31 yield10x |
|---|---|---|---|
| `--combat-sims` | MCTS sims per combat decision | 300 | 300 |
| `--rollouts` | counterfactual continuations per action | 6 | 8 |
| `--rollout-decisions` | max overworld decisions per continuation | 96 | 96 |
| `--policy-temperature` | sampling temperature inside continuations | 1.05 | 1.05 |
| `--label-temperature` | softmax temperature for the target | 0.15 | 0.15 |
| `--priority-accept-base` | base acceptance for non-priority decisions | 0.10 | 0.10 |
| `--max-labels-per-episode` | correlated-label cap; 0 = unlimited | 2 | **12** |
| `--truncate-after` | stop a continuation after N decisions and bootstrap the rest; 0 = play to terminal | 0 | 0 |
| `--harvest-rate` | fraction of continuation decisions kept as extra rows | 0.0 | 0.0 |
| `--trajectory-auxiliary-targets` | attach observed future outcomes | on | on |

### Curriculum shape

Three per-act stages, each producing a train and a validation dataset
(`run_label_quality_v31.py:50-51`):

| Act | Train labels | Train episode cap | Val labels | Val episode cap |
|---|---|---|---|---|
| 1 | 80 | 800 | 15 | 300 |
| 2 | 100 | 2,500 | 20 | 700 |
| 3 | 120 | 10,000 | 25 | 3,000 |

The `yield` arm multiplies all six budgets by `--label-scale` (10 → 800/1000/1200
train, 150/200/250 validation). Legacy v26 datasets — 400/400/240 train and
60/60/60 validation — are always mixed in at training time.

### Auxiliary targets

With `--trajectory-auxiliary-targets`, each retained row is annotated after its
episode finishes with what actually happened: whether the next combat was
survived and at what HP fraction, whether a rest was reached, whether the act
boss was survived, HP on entering the next act, and the normalized terminal
floor. These supervise the six auxiliary heads.

**How well the heads actually fit** (measured 2026-07-30 on v31 over 3,000 rows,
75/25 split). Each head is a single `nn.Linear` on the frozen trunk, so a ridge
probe on the same features gives that head's achievable ceiling. Survival targets
are compared through a sigmoid, since they are trained with
`binary_cross_entropy_with_logits` and emit logits:

| Head | Head R² | Linear-probe ceiling | Gap |
|---|---:|---:|---:|
| next_combat_survival | +0.160 | +0.194 | +0.034 |
| next_combat_hp | +0.400 | +0.458 | +0.058 |
| next_rest_reach | +0.075 | +0.167 | +0.092 |
| act_boss_survival | +0.578 | +0.685 | +0.107 |
| next_act_entry_hp | +0.615 | +0.692 | +0.077 |
| terminal_floor | +0.658 | +0.715 | +0.057 |

The heads sit within 0.03–0.11 R² of their ceiling, so they are reasonably fitted
— *not* collapsed to the marginal the way the policy head is at campfires. Since
these are convex linear problems on a frozen trunk, closing the remaining gap by
solving them in closed form rather than by SGD is free, but the gain is small.

Two things this establishes. `terminal_floor` at R² 0.658 is good enough to
bootstrap a truncated rollout, which is what `--truncate-after` relies on. And
the trunk demonstrably encodes run-trajectory information, which is the
precondition for adding Silverbot's `dest_room` head
(see [08-silverbot-comparison.md](08-silverbot-comparison.md)).

Beware a measurement trap here: restricting to Act 1 rows makes `terminal_floor`
look useless (correlation 0.277) purely through range restriction, and comparing
survival-head logits to 0/1 targets without a sigmoid produces R² values around
−100. Both are artifacts.

### Harvesting (available, off by default)

`--harvest-rate` keeps a fraction of the decisions *inside* continuations as
`(observation, action, return)` rows in a sibling `.harvest.pt`. They are
already simulated, so they are free; they are correlated and slightly off-policy,
so they are wrong for the policy head. Each row's return is measured from **its
own** floor/act, which is what `rollout_score` computes given a different start,
so no extra simulation is needed. Measured yield: **49 rows per label at
`--harvest-rate 0.02`**, i.e. a v31-scale run would produce ~176k rows against
the 4,008 the policy head trains on. Every shipped manifest before `v37_trunc`
has `harvest_rate = 0.0`.

**They are not usable as value-head data when generation is truncated**
(measured 2026-07-31 over all 82,861 rows in `runs/v37_trunc/`; earlier
revisions of this section recommended them for exactly that purpose):

| | n | mean | sd |
|---|---:|---:|---:|
| bootstrapped returns | 37,646 | +1.798 | 1.267 |
| observed returns | 30,794 | −0.096 | 0.375 |

**54.4% of the returns came from the previous model's own `terminal_floor`
head**, so fitting a critic to them re-learns the old critic. The other 46% are
not a random sample of the rest: a branch carries a true terminal return exactly
when it *ended* inside the 20-decision window, and over 99% of those sit at
`−0.4 + 0.1 × floors_gained` — they are deaths. Censoring is perfectly
correlated with the target, and no uncensored held-out subset survives to
validate against. Untruncated harvest (`--truncate-after 0`) does not have this
defect, but no such dataset exists on disk.

For critic data, collect complete runs instead:
`lightspeed/collect_run_value_data.py` labels every overworld state with the
return actually realized from it, which is uncensored and on-policy by
construction, at ~0.90 s per A20 run at 100 sims.

## The critic — fitted 2026-07-31, and it is mostly a depth lookup

`train_whole_run_v27.py` has four loss terms and **none of them touches
`self.value`**. The head has only ever been trained by
`pretrain_whole_run_value.py`, which pays for fresh episodes one at a time. As
shipped in v37 the head scores **R² −172** against on-policy returns: not merely
untrained but on an unrelated scale, so any code reading `value` today is
reading noise.

Fitted on **6,000 v37 runs / 484,486 states** (`collect_run_value_data.py`,
100 sims, seed-disjoint 80/20 split), target = the undiscounted sum of
`WholeRunEnv`'s own step rewards from each state:

| predictor | val R² |
|---|---:|
| v37's head as shipped | −172.16 |
| predict the mean | −0.0001 |
| ridge probe on `state` (linear) | −0.0225 |
| **floor+act+screen only (MLP)** | **+0.2232** |
| lookup table on floor alone | +0.2540 |
| **`state` (96-d, MLP) — the shipped head** | **+0.2973** |
| **`state` ⊕ (floor, act, screen)** | **+0.3202** |

Read the middle rows before the top one. **A lookup table keyed on floor alone
reaches +0.254**, so most of what the fitted head knows is how deep the run is.
The representation is not empty — `state` ⊕ scalars reaches +0.320 against
+0.223 for the scalars alone — but the honest summary is a critic that is ~80%
depth and ~20% state.

Two consequences for PPO. A depth-dominated baseline still removes the largest
variance term, which is what a baseline is for, so this is a usable warm start
rather than a good critic. And the +0.32 configuration needs a head that also
reads the scalars, which the current `value.*` shape cannot; PPO builds its own
critic anyway, so it should take `[state ⊕ floor ⊕ act ⊕ screen]` from the
start. Note also that the linear probe scores **below** predict-the-mean here,
so unlike the auxiliary heads it is a baseline the MLP beats, not a ceiling.

Checkpoint: `runs/whole_run_transformer_v37_critic.pt` — v37 with the fitted
`state`-only head, every other tensor unchanged. The critic PPO actually uses is
`lightspeed/run_critic.py` (`runs/run_critic_v37.pt`, **val R² +0.3208**), which
takes `state ⊕ (floor, act, screen)`; the checkpoint's `value.*` shape cannot.

## On-policy collection (`lightspeed/ppo_collect.py`)

The collection half of run-level PPO, built 2026-07-31. One iteration is N
complete A20 runs under a frozen snapshot; combat stays inside the environment,
so a whole battle is one semi-Markov transition of the outer trajectory. γ = 1.0
and λ = 0.97 per `FULL_RUN_RL_DESIGN.md` §9–10; game seed and sampling seed are
separate streams; truncation at `max_decisions` is **bootstrapped** while a real
ending is not (runs reach at most 204 of 256 decisions today, but a stronger
policy goes deeper).

`--verify N` replays stored observations through the same snapshot and checks
the recorded log-probabilities and values reproduce — max |Δlogp| and |ΔV| both
**0.00e+00** on the first batches. This catches mutated observations, stale
snapshots and train-mode leakage, which otherwise appear only as a silently
wrong PPO ratio.

Measured on the 6-core development machine at 100 sims:

| | |
|---|---|
| an iteration of 256 episodes | **50.7 s** (15,699 transitions, T=0.2) |
| throughput | 5.05 episodes/s, 309 transitions/s |
| transitions per run | 40 (T=1.0) to 61 (T=0.2) |
| batch on disk | 47.5 MB per iteration with observations kept |

At ~51 s per iteration that is **~70 iterations/hour**, so Silverbot's ~2,575
iterations is a **1–2 day** run rather than the multi-week estimate this doc
carried before the loop was built and timed. Read that with one caveat: episodes
get more expensive as the policy improves — a run reaching floor 37 costs
roughly 5–10× one dying at 18 — so late iterations will be minutes, not seconds.

Two results that change how the first iterations should be set up. **Collect
cold**: sampling costs floors steeply because the policy's decision margins are
~0.13 nats (see [07-known-issues.md](07-known-issues.md)), so T=0.2 gives floor
18.12 against T=1.0's 12.94 while still emitting 0.713 nats of entropy —
exploration is nearly free and does not need a warm temperature. And **an RL
curve starts near 18 floors, not at v37's 22.89 greedy number**; the first thing
PPO has to buy back is the sampling penalty.

The critic is off-distribution at the start for the same reason: it was fitted
on argmax runs (mean floor 22.80) and scores explained variance −0.73 on a
T=1.0 batch. That is expected — PPO refits it every iteration — but it means
iteration 1's advantages are poor and the early curve should not be read as
signal. Collecting at T=0.2 largely avoids it: explained variance on that batch
is **+0.361**, matching the critic's own fitted R².

## The update (`lightspeed/ppo_update.py`, `lightspeed/ppo_train.py`)

Clipped surrogate + entropy bonus + critic refit, with the loop driver in
`ppo_train.py`. Configuration forced by measurements rather than convention:

- **Trunk frozen by default** (80.1% of parameters still train). It is 40%
  cheaper per transition — 7.77 ms against 12.89 — and it keeps collection's
  cached `state` features valid, which a moving trunk would silently invalidate.
- **Single-threaded torch**, measured *faster* than six threads (7.77 ms against
  9.75). The model is small enough that intra-op parallelism is pure overhead;
  parallelism belongs at the process level.
- **`--target-kl` 0.01 with early stopping.** With a median decision margin of
  0.129 nats, a step a normal PPO would call small can reorder the argmax
  everywhere. In practice the epoch loop stops after 1–2 epochs.
- **The critic refit is keep-best on held-out transitions**, with the incumbent
  weights as one of the candidates. A degrading critic does not announce itself;
  it biases every later advantage. The first version of this refit *did* degrade
  (val MSE 0.00204 → 0.00223) through AdamW weight decay pulling the output bias
  off a target that sits near −0.99; decay is now off and the guard catches it
  either way. Working refit: 0.00204 → 0.00117, explained variance 0.434.

Loop verified end to end (2 iterations × 48 episodes): sampled floor 17.31 →
18.48, entropy 0.699 → 0.636 as the policy sharpens, KL 0.004–0.008 per epoch,
greedy eval on held-out seeds 23.21 ± 1.05.

**The update, not collection, is the bottleneck** — ~7.8 ms per transition per
epoch against 3.3 ms to generate one. `lightspeed/ppo_parallel.py` sums
gradients across processes to close that gap: workers load the iteration's batch
once, so a step ships only the flattened trainable weights out (5.2 MB) and the
summed gradient back, and each shard divides its loss by the **full** minibatch
so summing shards reproduces the single-process mean exactly.

Measured on a 15,699-transition iteration, one epoch:

| | single process | 6 workers |
|---|---:|---:|
| minibatch 512 | 122 s | 42 s |
| minibatch 1024 | 122 s | 40 s |
| minibatch 2048 | 122 s | **39.5 s** |

**3.1×.** Minibatch size barely matters, so transfer is not the constraint —
per-worker throughput under contention is, the same effect collection shows
(0.84 s/run solo against ~1.2 s effective). Six shards of a 15.7k batch is
already near the practical ceiling of 6 cores.

End to end at 256 episodes / 2 epochs / 6 update workers: collect 47 s, update
60–79 s, **~2.1 min per iteration**.

| config | per iteration | 2,575 iterations |
|---|---:|---:|
| 1 epoch, single process | 2.9 min | 5.2 days |
| 2 epochs, single process | 4.9 min | 8.8 days |
| **2 epochs, 6 workers** | **2.1 min** | **3.8 days** |
| 1 epoch, 6 workers | ~1.5 min | ~2.6 days |

A wrong gradient reduction does not crash — it produces a training curve that
quietly does not move, which looks exactly like a hard task. So
`--verify-gradients` recomputes one minibatch single-process from identical
weights and compares elementwise; the loop runs it automatically on iteration 1.
Measured agreement: **max \|serial − parallel\| 2.4e-07 against a gradient scale
of 0.31**, i.e. float32 summation-order noise.

## Where the wall clock actually goes (profiled 2026-08-01)

Timing collection across simulation budgets separates the two costs, because
only one of them scales with sims:

```
ms per overworld decision = 0.0282 x sims + 5.44
at 100 sims:  combat 2.82 ms (34%)   network 5.44 ms (66%)
```

Since the update is entirely network, a full iteration is:

| | time | |
|---|---:|---|
| collection — network | ~30 s | policy forwards, one decision at a time |
| collection — combat | ~16 s | native MCTS at 100 sims |
| update | ~80 s | forward+backward, one transition at a time |
| **total** | **~126 s** | **~87% neural network, ~13% combat** |

**The bottleneck is the 1.6M-parameter network being run one decision at a
time on CPU, not the simulator.** Two untapped wins, both pure code:

1. **The trunk is computed twice per decision.** `ppo_collect` calls
   `policy(obs)` for logits and then `_state_features(policy, obs)` for the
   critic, which re-runs `_state_tokens`, the encoder and both adapters — work
   the first call already did. That is ~1.9 ms of the 5.44 ms network cost,
   discarded every decision, and the vine and label harnesses repeat the same
   pattern. Fixing it means returning the state token alongside the logits.
2. **The update never batches.** 7.8 ms per transition, sequentially, because
   each state has a variable candidate count and a variable-length token
   sequence. Padding and masking both would let a minibatch go through as one
   batch. This is ~60% of the iteration.

### The engine is already optimized; do not look for time there

`CMakeLists.txt` builds with `-O3 -DNDEBUG -flto` and the cache carries
`STS_PGO=use`, so profile-guided optimization is already on. Measured
throughput: **~74,000 MCTS simulations/second, ~13.5 µs per full playout to the
end of a battle**. The `NDEBUG` comment shows the `sts_asserts` branches in the
hot paths were already dealt with.

What remains is small and the payoff is smaller. `-march=native` appears in a
comment but not in the flags — worth taking on the next rebuild (5–15% on code
like this), but combat is 13% of an iteration, so it buys ~1% of the loop. Even
a *free 2x on the whole engine* buys 6.5%. Tree reuse was separately measured at
1.34x effective budget and rejected. Engine speed does matter more in label
generation, which runs at 300 sims where combat is ~61% of the cost.

### Simulation budget and ascension, measured on cost rather than quality

| sims | s/run | sampled floor (T=0.2) |
|---:|---:|---:|
| 25 | 0.35 | 17.01 |
| 50 | 0.41 | 16.77 |
| **100** | **0.49** | **18.38** |
| 200 | 0.68 | 18.39 |

The knee really is at 100: below it quality falls, above it nothing changes.

**A0 is 65% more expensive per episode than A20, not cheaper** — 1.27 s against
0.77 s, 115 decisions against 81 — because the policy survives to floor 32.5
instead of 23.0, so there are more floors, battles and decisions to pay for.
Easier individual fights do not offset the extra length. What A0 buys is a
non-degenerate reward: A20 has 0 victories in 6,000 runs, so its terminal term
is a constant, while A0 wins ~1/120 here and 13/100 historically for v31.

### Truncation and bootstrapping (available, off by default)

`--truncate-after N` stops each continuation after N decisions and estimates the
remaining value from the `terminal_floor` auxiliary head, mapping the predicted
floor to an act and applying `-loss_penalty` so truncated and terminal branches
share a scale. Harvested rows from a truncated branch are bootstrapped the same
way — scoring them with `rollout_score` would read a non-terminal state and
silently produce wrong returns.

Rationale: playing every continuation to terminal makes each score carry the
variance of an entire run, which is the dominant term in label noise. Truncating
trades that random variance for a *deterministic* state-dependent error that
partially cancels between sibling branches, which Monte Carlo noise cannot.

Measured at N=20 against terminal, 60 labels each, same seed and act range:

| | terminal | truncated @20 |
|---|---:|---:|
| median paired SNR | 0.803 | 0.840 |
| median **absolute** SNR | 0.724 | 0.432 |
| seconds/label | 17.1 | **7.5** |
| SNR per unit compute | 0.194 | **0.307** |

Paired SNR is unchanged within noise; the gain is that labels cost **2.3× less**.
The absolute/paired split confirms the mechanism — bootstrap error inflates each
arm's own variance while the sibling *difference* holds. The saving is best spent
on more rollouts rather than on speed: `--truncate-after 20 --rollouts 18` costs
about what untruncated `--rollouts 8` does.

Caveats: 60 labels, so only the cost difference is solid; the mean best-vs-runner-up
gap shrank slightly (0.177 → 0.163), suggesting mild compression; and no model has
yet been trained on truncated labels.

## Stage 2 — training

`train_whole_run_v27.py`. Per row, four loss terms, scaled by a per-decision-type
frequency weight:

```
type_weight · ( policy_loss
              + anchor_weight    · KL(model ‖ frozen warm-start)
              + ensemble_weight  · mean bootstrap-member cross-entropy
              + auxiliary_weight · (BCE on survival targets, MSE on the rest) )
```

- `policy_loss` is cross-entropy against the soft target.
- The **anchor** is a `deepcopy` of whatever `load_model()` returned. If the warm
  start fails, the anchor is a random reference and the term inverts its purpose
  — this is what happened in v30, and `load_model` now raises on a dim mismatch
  rather than continuing.
- **Ensemble members** each see ~75% of rows, selected by
  `blake2b(seed:floor:decision_type:member)[0] < 192`.
- `type_weight = len(rows) / (n_types · count[type])`.

Optimizer: AdamW, weight decay 1e-4, gradient clipping at norm 1.0, linear
warmup over `--warmup-fraction` (default 0.05) of total steps then cosine decay.

### Batching — changed 2026-07-30

Gradient is accumulated over `--batch` rows (**default 32**) before each step.
Through v31 the effective batch was **1**, one optimizer step per row at
lr 3e-5. Two consequences, both measured:

- The supervised step was close to a no-op. Every version from v27 to v31 moved
  its training policy loss by under 0.02 nats. See
  [06-experiment-log.md](06-experiment-log.md).
- `type_weight` was inert. At batch 1 the weight scales the whole step, and
  AdamW's `m/√v` cancels a constant gradient scale, so the 76× spread across
  decision types (`boss_relic` 29.69 vs `rewards` 0.39) did nothing. Weights only
  bite when they differ *within* an accumulation window.

`--batch 1` reproduces the pre-v31 behaviour exactly.

### Scopes

`--scope` selects which parameter prefixes get `requires_grad`:

| Scope | Trains | Trainable at dim=96 |
|---|---|---|
| `experts` | `decision_experts.`, `uncertainty_heads.` | — |
| `experts-structure` | + the three adapters | — |
| `human-adapter` | `human_score.` only | — |
| `all-v27` | experts + adapters + `auxiliary_heads.` | 394,771 / 1,617,935 (24.4%) |
| `full` | everything, including embeddings, trunk, and score/value heads | 1,617,935 (100%) |

Every scope except `full` **freezes the trunk**. That is correct when the trunk
arrives fitted from the previous version and ruinous when it does not — see
v30 in [05-model-lineage.md](05-model-lineage.md). Use `--scope full` whenever
the trunk is not inherited. The trainer prints its `scope=… trainable=…/…` line
on every run; check it.

### Checkpoint outputs

Two files are written:

- `--out` — the epoch with the lowest validation NLL.
- `<out>_final.pt` — the last epoch's weights.

Both exist because validation NLL is a poor proxy for mean floor on this task,
and has picked the wrong epoch repeatedly. `--checkpoint-every N` additionally
keeps `<out>_ep<N>.pt` so an epoch curve can be selected on paired floors instead.
See [04-evaluation.md](04-evaluation.md).

## Hyperparameters by version

Taken from each run's `RUN` command line in `runs/*_master.log` and from the
manifests, not from prose.

| | v26 | v28 | v30 | v31 yield10x | v32/v33 | v36 |
|---|---|---|---|---|---|---|
| combat_sims | 300 | 300 | 300 | 300 | 300 | 300 |
| rollouts | 8 | 6 | 8 | 8 | 8 | 8 |
| max_labels_per_episode | 2 | 2 | 2 | **12** | 12 | 12 |
| train / val rows | — | 1,308 / 238 | 1,308 / 238 | **4,008 / 778** | 4,008 / 778 | 4,008 / 778 |
| epochs | 40 | 24 | 30 | 24 | 30 | 24 |
| batch | 1 | 1 | 1 | 1 | **32** | 32 |
| lr | 5e-5 | 3e-5 | 5e-5 | 3e-5 | **1e-3** | 1e-3 |
| anchor / ensemble / auxiliary | 0.15/0.25/0.20 | 0.25/0.20/0.30 | 0.25/0.20/0.30 | 0.25/0.20/0.30 | 0.25*/0.20/0.30 | 0.25/0.20/0.30 |
| scope | experts-structure | all-v27 | all-v27 | all-v27 | **full** | all-v27 |
| dim / layers / heads | 96/2/4 | 96/2/4 | **192/3/6** | 96/2/4 | 96/2/4 | 96/2/4 |

\* the v33 `noanchor` arm used `--anchor-weight 0`; the `conditional` arm kept 0.25.

v30's row says 300 sims, not the 800 in its original plan — the budget was
reverted before training to isolate the capacity effect. The 800-sim Act 1 data
that was generated survives in `runs/v30_comparison_800sims/` and still backs the
never-run `--arm 800` quality arm.

## Running it

```bash
cd slay-sim
$env:PYTHONPATH='..\sts_lightspeed\build;.'
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1
python -m lightspeed.run_label_quality_v31 --arm yield --label-scale 10
```

`--dry-run` prints the manifest and the generation commands without running
them. Generation is skipped for any dataset file that already exists, which is
what makes a multi-hour run interruptible — and is also a hazard, see
[07-known-issues.md](07-known-issues.md).

Cost on the 6-core development machine at `--workers 6`: the yield-10x arm took
~2.75 h of generation, 17.4 min of training, 4.6 min of evaluation. Generation
dominates; training never does.
