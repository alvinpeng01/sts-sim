# Documentation index

Rebuilt 2026-07-30 from the working tree: every number here was re-derived from
the code, the checkpoint files, the run manifests, and the evaluation `.jsonl`
outputs currently on disk. Where a claim could not be re-verified, it says so.

| Doc | Covers |
|---|---|
| [01-architecture.md](01-architecture.md) | What the pieces are, which engine is authoritative, how Python and C++ connect |
| [02-training-pipeline.md](02-training-pipeline.md) | Rollout generation → supervised training → checkpoint, with the parameters that matter |
| [03-combat-search.md](03-combat-search.md) | Native expectimax MCTS, the rollout heuristic, terminal evaluation, CMA-ES tuning |
| [04-evaluation.md](04-evaluation.md) | The paired-seed evaluation protocol, its metrics, and its traps |
| [05-model-lineage.md](05-model-lineage.md) | v1 → v36: what each version changed and what it measured |
| [06-experiment-log.md](06-experiment-log.md) | The label-quality/label-count investigation and the fitting experiments in full |
| [07-known-issues.md](07-known-issues.md) | Open defects, deferred defects, and coverage gaps |
| [08-silverbot-comparison.md](08-silverbot-comparison.md) | Measured differences against Silver Automaton (`silverbot-reference/`) |
| [09-live-play-bridge.md](09-live-play-bridge.md) | CommunicationMod bridge, the BaseMod overlay, autobattle |
| [10-other-characters.md](10-other-characters.md) | Silent/Defect/Watcher: what the engine already supports, what the search cannot see, and what starting would cost |
| [11-engine-validation.md](11-engine-validation.md) | Validating the engine against the real game's own bytecode — the oracle, what it verified, what it found |

Design intent, as opposed to what exists, lives in
[`../FULL_RUN_RL_DESIGN.md`](../FULL_RUN_RL_DESIGN.md) — a hierarchical
RunPolicy + CombatSolver architecture that has **not** been implemented.

## Reproducing the numbers

Every measurement cited in these docs has a harness in `slay-sim/lightspeed/`,
under the existing `_*.py` audit-script convention. Run them from `slay-sim/`
with `PYTHONPATH='../sts_lightspeed/build;.'`.

| Script | Answers | Cited by |
|---|---|---|
| `_eval_summary.py` | Paired per-seed deltas between checkpoints in an eval `.jsonl` — the number that decides a promotion, which the eval harness itself does not print | [04](04-evaluation.md), [05](05-model-lineage.md) |
| `_reuse_ceiling.py` | How much of the search tree sits under the chosen action, and how often that action is deterministic — the ceiling on tree reuse | [03](03-combat-search.md), [07](07-known-issues.md) |
| `_relic_uptake.py` | Which boss relics the policy actually ends up holding | [04](04-evaluation.md), [07](07-known-issues.md) |
| `_bridge_intent_audit.py` | How often the live bridge's guessed monster intent matches the telegraphed one, replayed against the real capture | [07](07-known-issues.md), [09](09-live-play-bridge.md) |
| `_class_card_audit.py` | Whether every character's card selects enumerate, whether the enumeration agrees with the validator, and whether each card's `case` sits in the switch its `CardType` routes to | [01](01-architecture.md), [03](03-combat-search.md), [07](07-known-issues.md) |
| `_game_jar_audit.py` | Card damage and monster constants against the game's own `desktop-1.0.jar`, resolved at A0 and A20 — the only first-hand oracle available | [07](07-known-issues.md), [11](11-engine-validation.md) |
| `_card_effect_audit.py` | Which effects a card queues, against the game's `use()` bytecode — behaviour rather than constants. Ironclad fully reconciles; other colours need vocabulary mapping | [11](11-engine-validation.md) |
| `_relic_audit.py` | Relic constants against the game, and relics that sit in a pool but are never read by any code — obtainable and inert | [07](07-known-issues.md), [11](11-engine-validation.md) |
| `_engine_invariants.py` | Properties that must hold whatever cards are played — a battle must not change the master deck, playing a card must not increase its own count, card rewards come from the character's own pool, HP/block/energy stay in range. Needs no oracle, so it covers all four characters | [07](07-known-issues.md), [10](10-other-characters.md) |
| `collect_rollout_policy_data.py`, `train_rollout_policy_net.py`, `_eval_rollout_policy_net.py` | Distil the search into the rollout policy and measure it at **matched wall clock** (the net buys down simulations, so matched-sims comparisons mislead). Refuted 2026-07-31, and the eval harness is reusable for any future rollout-policy change | [03](03-combat-search.md) |
| `compare_tier_combat.py` | Both engines' combat on synthetic tier decks neither was tuned on — the generalization check for combat claims | [03](03-combat-search.md), [08](08-silverbot-comparison.md) |
| `_clairvoyance_cost.py` | What the combat search is worth WITHOUT knowing the draw order — permutes the draw pile before every search, contents unchanged. The honest estimate of live-bridge degradation, since CommunicationMod never reports shuffle order | [07](07-known-issues.md) |
| `replay_human_runs.py` | Recovers a human's OWN decisions -- routing, drafting, campfires, smith targets -- by regenerating his map from his base-35 seed and solving his recorded room sequence back into the node he clicked. 4,044 rows from 97 runs. Diagnostic, not training data (see [07](07-known-issues.md)) | [07](07-known-issues.md) |
| `ppo_parallel.py` | Process-parallel gradient accumulation for the update: workers hold the batch, a step ships weights out and summed gradients back, each shard divides by the full minibatch so the sum reproduces the single-process mean. **3.1×** (122 s → 39.5 s per epoch). `--verify-gradients` checks one minibatch against the single-process gradient — max \|Δ\| 2.4e-07 | [02](02-training-pipeline.md) |
| `ppo_train.py`, `ppo_update.py` | The PPO update half and the loop driver: clipped surrogate, entropy bonus, KL-targeted early stop, and a critic refit that is kept only if held-out MSE improves. Tracks sampled floor (what PPO optimizes) and greedy floor on fixed held-out seeds (what the checkpoint is worth) separately, because they start 4.5 floors apart. Update is the bottleneck at ~8 ms/transition/epoch | [02](02-training-pipeline.md) |
| `ppo_collect.py`, `run_critic.py` | On-policy collection for run-level PPO: N complete runs under a frozen snapshot, GAE at γ=1/λ=0.97, truncation bootstrapped, and `--verify` replaying stored transitions through the snapshot (max \|Δlogp\| 0.00e+00). 50.7 s per 256-episode iteration on 6 cores (~70 iterations/hour). `run_critic.py` is the V(s) it collects against, val R² +0.3208 | [02](02-training-pipeline.md) |
| `collect_run_value_data.py` | Plays complete runs and labels every overworld state with the return actually realized from it — uncensored, on-policy critic data at ~0.84 s/run at 100 sims. Fits the value head `train_whole_run_v27.py` has never trained, and is the collection half of an eventual PPO loop | [02](02-training-pipeline.md) |
| `train_value_from_harvest.py` | Why the free `--harvest-rate` rows **cannot** supply that critic: 54.4% of returns are the old model's own bootstrap, and >99% of the rest are deaths, so censoring correlates perfectly with the target. Kept as the record of a closed option | [02](02-training-pipeline.md) |
| `_advantage_estimators.py` | VinePPO / GRPO / self-competition baselines against a vine Monte-Carlo reference. All null — and the decomposition behind it: **true per-decision advantage sd 0.0096 against episode-return sd 0.084**, needing ~47 rollouts/state for SNR 1 | [06](06-experiment-log.md) |
| `_gumbel_label_probe.py` | Gumbel top-k + sequential halving vs the generator's uniform allocation, scored by simple regret against a 24-rollout reference pool. Uniform wins; the prior's top-2 contains the best action only 66% of the time. Also measures that the prior alone picks best 51.8% while 32 rollouts buys 57.6% | [06](06-experiment-log.md) |
| `_aivat_eval.py` | AIVAT-style control variates on run outcomes. **Unbiased (−0.024 ± 0.082) and worth 0%**: corr(floor, correction) is 0.014 against the 0.153 needed to break even. Doubles as the acceptance test for any future critic — and shows 91% of T=0.2 floor variance is policy sampling, so the ceiling is real | [06](06-experiment-log.md) |
| `_decision_ablation.py` | What each decision type is worth: one arm per type played uniformly at random, everything else on policy. Drafting **−5.85 ± 0.50**, shop −2.44, routing −1.91, and rest/event/card_select/neow/treasure all ~0 (confirmed on a second seed set) | [06](06-experiment-log.md) |
| `_checkpoint_ensemble.py` | Averaging checkpoints' log-probabilities at inference — free variance reduction given 0.129-nat margins. Measured +0.71 ± 0.29 on one 600-seed set and **−0.11 ± 0.30 on another**; refuted | [06](06-experiment-log.md) |
| `_route_bias_probe.py` | Whether *correcting* the two wrong-signed routing coefficients buys floors, as two or three additive logit biases at MAP_SCREEN — nothing trained, so it tests valuation against capability directly. Answer: forcing elites to 2.2/run costs **−1.93 ± 0.72**, and the rest arms do not replicate on a second seed set | [07](07-known-issues.md) |
| `_param_ab.py` | Paired A/B of ONE search parameter against the shipped config, with a standard error from observed paired differences. Defaults to the train split. The measurement that caught four combat nulls on 2026-08-01 | [03](03-combat-search.md) |
| `_probe_card_identity.py` | Whether action IDENTITY carries signal the 8 `nativeActionFeatures` do not — the offline kill-gate that refuted widening the feature vector before any engine change was written | [03](03-combat-search.md) |
| `batched_policy.py` | Batched forward for the whole-run transformer (3.66x on the training step). Nothing else in the project ever ran this model on a batch — `train_whole_run_v27.py`'s `--batch` is gradient accumulation over a per-row loop. Verified against the single-observation path in `tests/test_batched_policy.py` | [02](02-training-pipeline.md) |
| `_route_planner.py`, `_eval_route_planner.py` | Survival-weighted route planning over the act-map DAG, and the paired floor eval that substitutes it for v31's map decisions. Refuted at elite weight 3.0 (−3.68 ± 1.10 floors); the eval harness works for any map-decision rule | [07](07-known-issues.md) |
| `_run_audit.py`, `_room_audit.py`, `_routing_audit.py` | Run economy, per-room outcomes, routing (pre-existing) | [06](06-experiment-log.md) |
| `_overfit_probe.py`, `_overfit_probe_long.py`, `_label_snr.py` | Whether the model *can* fit; label signal-to-noise (pre-existing) | [06](06-experiment-log.md) |

### Three harnesses were removed on 2026-08-01

`eval_heart1_hybrid.py`, `collect_heart1_labels.py` and
`_silverbot_human_deck.py` required an external agent (Daniel Ziegler's Silver
Automaton fork) to be installed, and were removed so that everything tracked in
this repository runs against it alone. **Citations to them elsewhere in these
docs are historical and still accurate about what was measured** — the file is
simply no longer present.

The measurements they produced, all still valid: the layer swap that established
the run policy as the binding constraint (+15.71 ± 3.13 floors,
[03](03-combat-search.md)); the second-agent arm on the human combat benchmark
that turns our HP figure into a bracket ([03](03-combat-search.md),
[08](08-silverbot-comparison.md)); and the routing-coefficient reference
([07](07-known-issues.md)). None of the three is reproducible in-tree any more.
The layer swap is the one that matters — reconstructing it means driving our
`GameContext` with an external overworld policy while
`native_playout_current_battle` owns every combat decision, which holds combat
byte-identical and isolates the run layer.

A claim in these docs without either a file:line reference or a harness here
should be treated as unverified.

## If you read one thing

The state of the project:

- **The binding constraint is the OVERWORLD POLICY, not combat.** Established
  2026-07-31 by layer swap: running Silverbot's `heart1.pt` overworld policy on
  OUR engine and OUR combat (`lightspeed/eval_heart1_hybrid.py`, 24 paired A20
  seeds, combat byte-identical at 100 sims) moved v31's mean floor 21.29 → 37.00
  (**+15.71 ± 3.13, t = 5.02**) and win rate **0% → 12.5%** — the first A20
  victories this stack has ever produced. Silverbot's own full stack on the same
  protocol: floor 39.29, 20.8% — statistically indistinguishable from their
  policy on our combat. The entire measurable gap is the run policy. heart1 is
  the same architecture class as v31 (a trained net, forward pass per overworld
  decision, no search), so this is a **training-quality gap, not a design gap**.
- **Combat simulation budget is not a lever anywhere.** v31 at 15x sims:
  +0.115 ± 0.507 floors, 0 wins either way (n=200). The hybrid at 10x sims:
  −7.25 ± 5.23, wins unchanged (n=12). Meanwhile combat *config* tuning worth
  +6.05 HP/fight on the human benchmark measured **−0.23 ± 0.29 floors** — above
  ~100 sims, benchmark HP does not predict run outcomes. The benchmark remains
  valid for human-relative combat comparison, not as a floor proxy.
- **Our combat search is draw-order clairvoyant** — `run_mcts_search` copies the
  full `BattleContext` including the ordered draw pile, so every simulation sees
  the exact future draws. Silverbot found the same defect in their engine
  (2026-06-03, "WE'VE BEEN CHEATING", measured ≈ **+34pp** win rate at 1k sims),
  removed it, and every comparison above is our clairvoyant search vs their
  honest one. Our apparent combat parity/lead (473/504 vs 463/504 on neutral
  synthetic decks) is propped up by information they deliberately gave up.
  **Measured 2026-07-31 at −3.78 ± 0.84 HP/fight** (`_clairvoyance_cost.py`,
  250 fights): blind to draw order we score −12.25 against their honest −6.58, so
  **their combat is better than ours** and the apparent parity was the cheat. Not
  fixed — and fixing it costs floors rather than gaining them, since combat is
  saturated above ~100 sims. See [07-known-issues.md](07-known-issues.md).

- **Five published methods were tested against this task and all are null**
  (2026-08-01): VinePPO, GRPO, GAZ self-competition, Gumbel AlphaZero allocation,
  and AIVAT variance reduction. The reason is one measurement: the **true
  per-decision advantage has sd 0.0096 against an episode-return sd of 0.084**,
  so one decision moves the outcome by a tenth of a run's standard deviation and
  ~80 decisions account for the whole variance. Reaching SNR 1 on a single
  decision needs ~47 rollouts. That also explains the 0.129-nat logits, the
  0.803 label SNR, the floor-lookup critic, and v37 tying v31 at 2.25x the
  rollouts. See [06](06-experiment-log.md).
- **The wall-clock bottleneck is the network, not the simulator** — an
  iteration is ~87% neural network and ~13% combat, and the engine already runs
  74k MCTS simulations/second with PGO on. The two real speedups are a trunk
  that is computed twice per decision and an update that never batches. See
  [02](02-training-pipeline.md).
- **The routing coefficients are a description of the gap, not a fix target**
  (2026-07-31, `lightspeed/_route_bias_probe.py`). Correcting the wrong-signed
  ELITE preference directly — as a plain logit bias at map screens, nothing
  trained, combat and seeds held identical — raises elite capture 0.85 → 2.23
  per run and **costs −1.93 ± 0.72 floors** (120 paired A20 seeds); pushing
  avoidance further also costs (−0.38 ± 0.23). v31 sits at a local optimum in
  both directions, so elite avoidance is a **rational response to expensive
  combat**, not a valuation error inherited from labels. The rest-side arms
  measured +0.50 ± 0.44 on 120 seeds and **+0.10 to +0.23 (t < 1) on 240 fresh
  seeds** — they did not replicate. See [07](07-known-issues.md).
- **v37 = v31, exactly** (paired, 200 shared seeds: **+0.01 ± 0.51**, t = 0.03).
  The truncated estimator bought 2.25× the rollouts at equal cost and converted
  none of it into policy quality, so paired label SNR was not the binding
  constraint it was taken to be. See [05](05-model-lineage.md).
- **Re-baselined on the post-rebuild engine at n=600** (2026-07-31,
  `runs/sharp_rebaseline_600seeds.jsonl`): v31 − v28 = **+1.14 ± 0.34**
  (t=3.39), v37 − v28 = **+1.54 ± 0.34** (t=4.52), v37 − v31 = +0.40 ± 0.34
  (n.s.). The ordering survives; the magnitudes shrink to ~62% of their
  pre-rebuild size. **v37 is the working baseline** — best on floors (22.89),
  Act 3+ reach (29/600) and W/T/L, though not separated from v31.
- **n=200 is too small to decide anything in this lineage.** The same
  comparison at the project's habitual n=200 gave v31 − v28 = +0.41 ± 0.58
  (t=0.70) and read as "nothing separates"; only the sample size changed. Paired
  sem is ~0.55 at n=200 versus ~0.34 at n=600, so the standard eval cannot
  resolve below ~1.1 floors — **larger than most single-version gains here**.
  Six parallel processes make n=600 a ~6-minute job. See [05](05-model-lineage.md).
- Win rate at A20 is **0%**. At A0, v31 wins 13/100 (pre-fix measurement).
- The binding constraint measured so far is **training label volume**, not model
  capacity and not combat search budget. Both of those were tested and rejected.
- Combat is now measured against **a top human's own decks, relics, potions and
  HP** across 2841 reconstructed fights, with **Silver Automaton as a second arm
  on the identical fights** ([03](03-combat-search.md)). On the 528-fight test
  split, 2026-07-31 tuning moved us from **-11.83 to -5.78** on
  `mean(human_damage - hp_paid)` against silverbot's **-6.58** -- from 1.8x their
  HP surplus to **ahead of them**, while also dying less (78 vs 84) and holding
  more HP per fight (48.67 vs 47.87).
  Search budget is flat across 35x compute, so the ceiling is the rollout policy,
  not the tree.
- **Imitating the human archive is refuted as a training method** (2026-07-31).
  `replay_human_runs.py` recovers 4,044 of Baalorlord's own decisions from his
  seeds; cloning them made the policy WORSE -- routing -15.80 +/- 0.74 floors,
  drafting -5.42 +/- 0.81 -- because the extraction pins his deck, so every
  observation shows states our policy never occupies. The data remains an
  excellent **diagnostic** (his routing coefficients, 73% elite capture, 83%
  smith rate, and the drafting table in [07](07-known-issues.md)) and a poor
  training set. The map-representation hypothesis is separately closed:
  `map_route_features` already computes the `(minE, maxE, distRest)` aggregates
  Silverbot credits. On-policy RL is the remaining path.
- Two measurement bugs found the same day are worth knowing before trusting any
  earlier combat number: the benchmark omitted **potions** (worth 4.7 HP -- he
  drinks one in 38.6% of elites), and the rollout's sampling RNG was seeded from
  `random_device`, so **common random numbers were silently broken** the moment
  `rollout_temperature` left zero. Both fixed.
- An engine bug (`cardOnExit` making Armaments upgrades permanent, fixed
  2026-07-30) inflated every result produced before that date by roughly 3 floors.
  Cross-fix comparisons are invalid; see [07-known-issues.md](07-known-issues.md).
- Three engine bugs were found and fixed on 2026-07-30 while verifying these
  docs: the live bridge never removed a played card from hand, `cardColors[]` had
  8 wrong entries (letting an Ironclad transform return the same card), and the
  `CardColor` pybind enum was missing `BLUE`. See
  [07-known-issues.md](07-known-issues.md).
- Three more on 2026-07-31, all off the Ironclad path and so invisible to every
  number above: five card-select tasks returned an empty action list (a Silent
  native playout segfaulted on it), `SEEK`/`MEDITATE` enumerated actions the
  validator rejects, and four cards had their `case` in a switch
  `BattleContext::useCard` never routes them to, so they silently did nothing.
  The engine implements **all four characters'** cards, contrary to what earlier
  revisions of [01](01-architecture.md), [03](03-combat-search.md) and
  [07](07-known-issues.md) claimed.
- The engine is now checkable against the **real game's own bytecode**
  ([11](11-engine-validation.md)), and every known defect it found is closed:
  four monster behaviours (Lagavulin's Metallicize, Champ's Gloat tiers,
  Darkling's Chomp hit count, Writhing Mass's flail block), seven relics that
  were obtainable and did nothing, and two card energy costs.
- **Every number in these docs predates that.** Six of those fixes change
  Ironclad behaviour, so the situation is the Armaments fix again — the ranking
  will probably survive, the absolute numbers will move.
- **Re-baselined 2026-08-01** (`runs/rebaseline_v28_v31_200seeds_20260801.jsonl`,
  200 paired A20 seeds from 18.9M at 300 sims — the same protocol as the
  pre-rebuild baseline). The prediction held exactly: the ranking survives and
  every absolute number moved down.

  | | pre-rebuild | post-rebuild |
  |---|---:|---:|
  | v28 mean floor | 21.76 | **21.64 ± 0.47** |
  | v31 mean floor | 23.57 | **23.05 ± 0.53** |
  | paired v31 − v28 | +1.80 ± 0.55 | **+1.41 ± 0.50 (t = 2.82)** |
  | A20 wins | 0 / 0 | 0 / 0 |

  v31 remains the best checkpoint, by a margin that shrank about 20%
  (W/T/L 68/87/45, 7 runs reaching Act 3 against v28's 4). **This is now the
  comparison point for any new work.** v37's 23.675 was measured on the pre-fix
  engine and is still not comparable.
