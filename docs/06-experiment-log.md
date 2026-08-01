# Experiment log

The chronological argument behind the current state of the project, with the
measurements each step actually produced. Version-by-version results live in
[05-model-lineage.md](05-model-lineage.md); this file records *why* each
experiment was run and what it settled.

## The question: what is the bottleneck?

By v30 the lineage had a policy that reached about floor 24 of 56 at Ascension 20
and had never won a run. Four candidate bottlenecks were on the table:

1. model capacity,
2. combat search budget (300 sims),
3. training label quality,
4. training label count.

Three of the four have now been measured. Only one paid.

## 1. Capacity — tested, rejected (v30)

4.1× the parameters (1.6M → 6.6M) produced a mean floor of 18.76, well below
v28. But the run was broken in a way that makes it a weak test of capacity: the
warm start failed silently on the dim mismatch, and `--scope all-v27` then froze
the resulting **random** trunk — 76% of the model. See v30 in the lineage doc.

What the log does establish, cleanly, is underfitting rather than overfitting.
Because the targets are soft distributions, the loss has an irreducible floor
equal to their mean entropy: **1.2722 nats** across the 1,308 v28/v30 training
rows, against a uniform baseline of 1.6632. So the learnable range is 0.391 nats.

| Run | train policy, first → last | val_nll first → best → last | Moved |
|---|---|---|---|
| v28 (24 ep) | 1.4274 → 1.4221 | 1.3892 → **1.3888** → 1.3891 | 0.005 nats |
| v30 (30 ep) | 1.4250 → 1.4044 | 1.4063 → **1.4038** → 1.4072 | 0.021 nats |

Both stop 0.13–0.15 nats above the floor with no train/val gap. Memorizing 1,308
rows with 6.6M parameters would drive train toward 1.2722 while val climbed away.
Neither happened.

**Conclusion: through v31 the supervised training step was close to a no-op.**
Whatever each version knew, it inherited from its warm start.

## 2. Search budget — tested, rejected

Identical checkpoints, identical 200 seeds, 300 versus 900 simulations per combat
decision (`whole_run_v28_vs_batched_a20_200seeds_300sims.jsonl` and
`whole_run_v28_v31_v32_a20_200seeds_900sims.jsonl`, both pre-fix):

| Checkpoint | 300 sims | 900 sims | Δ |
|---|---:|---:|---:|
| v28 | 23.70 | 24.66 | +0.96 |
| v31 | 26.30 | 26.51 | +0.21 |
| v32 | 25.22 | 26.07 | +0.85 |

Tripling the budget buys under one floor, and no additional victories. For
comparison, changing the checkpoint at a fixed 900-sim budget (v28 → v31) is
worth +1.85. **The policy is worth roughly twice what tripling the search is.**

The reading this supports: the search has largely converged by 300 sims, so the
ceiling is set by what it converges *to* — the terminal evaluation and the
heuristic weights — not by how long it looks. That argues for tuning and for
heuristic coverage, not for budget.

## 3 & 4. Label quality versus label count — the v31 experiment

`run_label_quality_v31.py` runs one arm per candidate answer, holding everything
else fixed at v28's architecture and hyperparameters so the warm start transfers:

| Arm | Tests | combat_sims | Labels | Generation |
|---|---|---|---|---|
| `--arm 300` | control | 300 | 1× (80/100/120) | none — reuses v30's datasets |
| `--arm 800` | label **quality** | 800 | 1× | Acts 2–3 only |
| `--arm yield --label-scale 10` | label **count** | 300 | 10× (800/1000/1200) | all six stages |

The yield arm raises `--max-labels-per-episode` from 2 to 12 rather than relaxing
`--priority-accept-base`, so the label *distribution* stays comparable and only
the count changes. The tradeoff it accepts: labels from one episode are
correlated, so effective sample size grows more slowly than row count.

**Result: count paid, and the 800-quality arm was never run.**

| Comparison (200 paired seeds, pre-fix) | Δ floors | sem | t |
|---|---:|---:|---:|
| labelq300 − v28 (relabelling at 1×) | +0.66 | 0.40 | 1.65 |
| yield10x − v28 | +2.60 | 0.65 | 4.00 |
| **yield10x − labelq300 (volume alone)** | **+1.95** | 0.65 | 2.99 |

~75% of the gain is label count. The generator had been discarding roughly 99%
of the episodes it paid full MCTS price for: at
`--max-labels-per-episode 2` with `--priority-accept-base 0.10`, Act 3 played up
to 10,000 episodes to keep 120 labels, and every one of those episodes cost a
full set of MCTS-resolved fights whether or not any decision was retained.

Raising labels-per-episode also amortizes the main line, which is why 10× the
labels cost ~2.75 h rather than 10× the original generation time. Act 2 was
cheaper per label than Act 1 (2.14 s vs 3.47 s) — deeper runs expose more
harvestable decisions.

The 800-sim quality arm remains untested. Its Act 1 data exists in
`runs/v30_comparison_800sims/`; Acts 2–3 would take about 1.5 h.

## Why the model would not fit, and what happened when it did

A 200-row overfit probe (`lightspeed/_overfit_probe_long.py`, entropy floor
1.1496 nats) isolates optimization from capacity. Deliberately overfitting a
small subset is the standard discriminator.

| Config | NLL after 400 epochs | Gap above floor |
|---|---:|---:|
| `all-v27`, batch 32, lr 1e-3 | 1.1725 | +0.0229 |
| `full`, batch 32, lr 1e-3 | **1.1581** | **+0.0084** |

Production settings (batch 1, lr 3e-5, frozen trunk) close essentially none of
the gap; batching plus a real learning rate on an unfrozen trunk close nearly all
of it. Ranked by effect: the unfrozen trunk is the largest single factor, batch
size and learning rate second, and removing the anchor is worth about 0.001 nats
— keep it, it costs nothing.

Gradient clipping was investigated and ruled out: median gradient norm is 0.08
and `clip_grad_norm_(1.0)` binds on ~3% of rows.

Note that the `full` config *starts* worse (1.3336 at epoch 1) because unfreezing
perturbs the trunk early, then ends far better. Judging it by its first epoch
would reject it.

### v32: the fit improved and the policy did not

Applying the probe's prescription to the real 4,008-row dataset produced exactly
the predicted fit — train policy loss −0.037 nats, train argmax 40.2% → 51.0% —
and a worse policy than v31 (+1.52 vs v28, against v31's +2.60 on the same file).

Two things went wrong at once:

1. **The first genuine overfitting in the lineage.** Validation NLL bottomed at
   epoch 7 and rose monotonically thereafter while train kept falling. 4,008 rows
   cannot support 1.5M unfrozen parameters. The frozen trunk in v28–v31 had been
   unintended regularization all along.
2. **The probe over-promised.** It reached +0.0084 above floor on 200 rows with
   anchor 0 and no ensemble or auxiliary terms; production on 4,008 rows under
   the full four-term loss with anchor 0.25 reached only +0.0857. The anchor also
   binds far harder once the trunk moves — its KL roughly quadrupled versus v31.
   A small-subset overfit probe answers "can it fit at all", not "what will
   production do".

### v36: batching without the trunk

Isolating the two: batch 32 and lr 1e-3 but `--scope all-v27`. Every epoch
checkpoint lands 0.7–1.0 floors **below** v31, and the curve is flat across
epochs 8/16/24 — so it is not an epoch-selection artifact. Batching alone buys
some fit and costs about a floor.

**Standing conclusion: fitting the training data better does not by itself
produce a better policy on this task.** The binding constraint measured so far is
data volume, and the training step can now consume data — it exhausts 4,008 rows
in about seven epochs.

## The engine bug that reframed everything

On 2026-07-30 a run-economy audit (`lightspeed/_run_audit.py`) reported 7.6
upgrades per run while a campfire audit showed the policy choosing REST at
**137 of 137** campfires and gaining zero upgrades there. Tracing the upgrades
found them landing one at a time on entering MONSTER rooms, correlated with
Armaments in the deck:

| Armaments copies | Upgrades/run |
|---|---|
| 2 | 20 |
| 1 | 11, 17, 1 |
| 0 | 2, 0, 0, 9, 1, 2 |

Cause: `cardOnExit` (`BattleContext.cpp:608`) wrote every *temporary* in-combat
upgrade permanently onto the master deck. Armaments upgrades a card "for the rest
of combat" and `chooseArmamentsCard` correctly keeps that local to the hand — but
the card then exited combat upgraded and this made it permanent. Roughly one free
permanent upgrade per Armaments play. The clause does not exist upstream;
`silverbot-reference`'s `cardOnExit` carries only the Ritual Dagger line.

Measured on the identical checkpoint and 12 seeds, engine as the only variable:
upgrades 7.6 → 1.5, mean floor 28.4 → 25.2. **The 3.2-floor effect is larger than
v31's entire advantage over v28.**

The full 200-seed post-fix rebaseline (v28 21.73, v31 23.57, v33-final 19.62)
confirms the ranking survives while every absolute number moves. It also partly
explains the campfire collapse: with ~6 free upgrades arriving per run, spending
a campfire on SMITH genuinely was low-value *in this engine*. The policy was
optimizing a game we had accidentally made easier, and the behaviour may correct
itself on regenerated labels without a targeted fix.

### The two later engine fixes were performance-neutral

Measured 2026-07-30. The SCRY `getLegalActions` fix and the `cardColors`
correction, both landed after the Armaments fix, together moved nothing on a
200-seed A20 whole-run eval:

| Checkpoint | 20:52 (Armaments fix only) | 23:07 (+SCRY +colorfix) | Δ |
|---|---|---|---|
| v28 | 21.73, act3+ 3 | 21.76, act3+ 3 | +0.03 |
| v31 | 23.57, act3+ 7 | 23.57, act3+ 7 | 0.00 |

Zero victories throughout. v31 being identical to two decimals *and* on Act 3
count is expected rather than suspicious: runs are seed-deterministic, so a
change that never triggers reproduces byte-identically.

Both results are what the mechanics predict. **Scry is a Watcher mechanic and an
Ironclad run never reaches an `InputState::SCRY`**, so that fix cannot show up in
an Ironclad eval — it was only reachable at all because the relic sampler was
handing an Ironclad non-Ironclad relics (since removed from the pool). The color
fix only reaches `getTransformedCard` at p ≈ 1/72 per transform, which is
plausibly exactly v28's +0.03.

Worth stating plainly because it is easy to misread: these were correct fixes
that bought no measurable strength, and that is the expected outcome, not a
disappointment. Neither was a performance lever.

Not separable from this baseline: the card cost/exhaust/innate corrections and
the `nativeImmediateBlockBase` block-table fixes were already in the 20:44 build,
so they are baked into the 21.73 figure. Isolating them would need a purpose-built
binary and is not worth doing for a likely-tiny effect.

## What the policy actually does wrong

From `_room_audit.py` over 60 seeds per checkpoint
(`runs/room_audit_v2.jsonl`, pre-fix):

| | v28 | v31 |
|---|---:|---:|
| Mean floor | 23.37 | 28.12 |
| Elites fought, whole run | 0.90 | 1.08 |
| Elites *available* in Act 1 alone | 3.15 | 3.15 |
| Rooms per run: MONSTER / REST / EVENT / SHOP | 8.6 / 3.9 / 3.7 / 2.3 | 10.3 / 4.9 / 4.2 / 2.8 |

The policy takes roughly one elite per run out of three-plus offered in Act 1
alone. Elites are the main relic source, so runs arrive at act bosses
under-relic'd. v31 improved this (0.90 → 1.08, and Act 2 elites 0.18 → 0.37) but
it remains the clearest behavioural gap, alongside the campfire marginal
described above.

## The labels cannot rank their own actions

Measured 2026-07-30 with `lightspeed/_label_snr.py`, over 60 labels generated at
matched seed and act range. Two signal-to-noise readings per label, using the
stored `per_rollout_scores` matrix:

- **absolute** — each arm's own standard error, as if arms were independent;
- **paired** — the standard error of the *difference* between the best and
  second-best arm, which is what actually decides the label's argmax, and which
  benefits from sibling branches sharing `matched_seed`.

| | |
|---|---:|
| Median paired SNR | **0.803** |
| Labels with paired SNR < 1 | **66.7%** |
| Median absolute SNR | 0.724 |
| Mean gap (best − runner-up) | 0.177 |

**On two-thirds of labelled decisions the measured gap between the best and
second-best action is smaller than the uncertainty in measuring it.** The target
those rows teach is close to a coin flip.

Common random numbers help, but much less than assumed: the paired-to-absolute
ratio implies only ~14% variance reduction versus independent arms. Sibling
branches diverge within a few decisions of the branch point, after which the
shared seed buys nothing.

This retroactively explains the fitting experiments. v32, v33 and v36 each fit
the labels better than v31 and each produced a worse policy — which is what
fitting a mostly-noisy target predicts. v31's near-total failure to train has
been acting as regularization. It also explains why *volume* is the one lever
that paid: more labels average the noise down, while better fitting absorbs more
of it.

**Scaling the current estimator out of this is not affordable.** SE falls as
1/√n, so median SNR 0.8 → 2.0 needs ~6× the rollouts, at a cost per label that
already prevents volume. The answer has to be a different estimator, not a bigger
one — which is what `--truncate-after` is a first attempt at
(see [02-training-pipeline.md](02-training-pipeline.md)).

Caveat: 60 labels. An earlier 30-label sample gave 1.019, so the point estimate
moves. Every generated row now stores `per_rollout_scores`, so this firms up
automatically with the next full run.

## Where the policy's value actually lives (2026-08-01)

`lightspeed/_decision_ablation.py`, v37, 240 paired A20 seeds at 300 sims. One
arm per decision type: that type is played by a uniform-random legal choice,
everything else stays with the policy. The cost in floors is what the network's
preference on that screen is worth.

| arm | mean floor | cost vs baseline | t | decisions/run |
|---|---:|---:|---:|---:|
| baseline | 22.71 | — | — | — |
| **rewards** (drafting) | 16.87 | **−5.85 ± 0.50** | −11.68 | 40.7 |
| **shop** | 20.27 | **−2.44 ± 0.47** | −5.19 | 6.2 |
| **map** (routing) | 20.80 | **−1.91 ± 0.70** | −2.72 | 22.1 |
| boss_relic | 22.28 | −0.43 ± 0.29 | −1.51 | 0.7 |
| treasure | 22.31 | −0.40 ± 0.25 | −1.60 | 1.6 |
| rest | 22.78 | +0.07 ± 0.37 | 0.18 | 3.7 |
| event | 22.86 | +0.15 ± 0.45 | 0.32 | 3.4 |
| card_select | 23.22 | +0.51 ± 0.24 | 2.12 | 1.0 |
| neow | 23.29 | +0.57 ± 0.48 | 1.19 | 1.0 |

**Drafting is what the network is for.** It is worth three times routing, which
is the decision class this project has spent most of its analysis on. It is also
the documented weakness (Clash taken 25 times, Perfected Strike without Strikes),
so it is simultaneously the largest contribution and the largest known defect —
the highest-leverage target rather than a lost cause. Note also that `rewards` is
40.7 of ~80 decisions per run, so on-policy transitions are already dominated by
the decision type that matters most.

**Rest, event, card_select, neow and treasure contribute nothing.** Confirmed on
**400 fresh seeds** (`runs/ablation_confirm.jsonl`): rest −0.30 ± 0.30, neow
−0.15 ± 0.38, card_select +0.00 ± 0.25, event +0.17 ± 0.39. This is consistent
with the campfire marginal collapse and with `_route_bias_probe.py` finding no
rest-side gain, from a third independent direction.

Read that carefully: "worth nothing" means a cheap rule could *match* the
network there, not beat it. It hands over no floors by itself.

### The card_select and neow results were false positives

The first sweep put card_select at +0.51 (t=2.12) and neow at +0.57 — i.e. the
policy appeared to be actively *worse than random* on those screens. Neither
survived fresh seeds. Nine arms were tested, so a t of 2.1 is what multiple
comparisons produce routinely.

## Refuted 2026-08-01: checkpoint ensembling

Since v28/v31/v37 do not separate and decisions turn on a median 0.129-nat
margin, averaging independently-trained logits is the cheapest available
variance reduction and needs no training. `lightspeed/_checkpoint_ensemble.py`
averages members' **log-probabilities** (raw logits carry a per-state additive
offset that would weight members unequally).

| 600 paired seeds | mean floor | vs v37 | t |
|---|---:|---:|---:|
| v37 | 22.89 | — | — |
| v28+v31+v37 | 23.03 | +0.15 ± 0.29 | 0.51 |
| v31+v37 | **23.59** | **+0.71 ± 0.29** | **2.46** |

On **600 fresh seeds** the same comparison gives **−0.11 ± 0.30 (t = −0.38)**;
v37 22.89 → 23.24 and the ensemble 23.59 → 23.13. **It does not replicate.**
Pooled, the effect is roughly +0.30 ± 0.21. Ensembling is not a lever.

### The methodological finding is the durable one

Three separate promising results in two days — rest-side routing biases
(+0.50 ± 0.44), checkpoint ensembling (+0.71 ± 0.29, t=2.46), and card_select
ablation (+0.51, t=2.12) — **all vanished on a second seed set**, while every
null replicated exactly. Seed-set variation alone moves v37's own 600-seed mean
by 0.35 floors. Anything under about +1 floor at n≤600 in this project should be
treated as unmeasured until it has survived a fresh seed set, and the second
measurement costs the same as the first.

## Published methods tested against this task (2026-08-01)

Five methods from the literature, each chosen because its stated failure mode
matched something measured here. All five are null, and the reasons are more
useful than the results.

### The measurement that explains all of them

`lightspeed/_advantage_estimators.py`, 162 vine-estimated states over 60 seeds.
`v_mc_a` and `v_mc_b` are independent Monte-Carlo estimates of the same V(s), so
their difference isolates the estimator's own noise:

| | |
|---|---:|
| SE of one 8-rollout V_mc estimate | 0.0166 |
| observed sd of the A_vine reference | 0.0253 |
| **implied sd of the TRUE per-decision advantage** | **0.0096** |
| episode-return sd | 0.084 |
| rollouts per state for advantage SNR 1 | **47** |

**A single overworld decision moves the outcome by about a tenth of a run's
standard deviation**, and 80 decisions x 0.0096² ≈ the whole return variance —
the outcome is the sum of ~80 small, roughly equal contributions. That one fact
accounts for the 0.129-nat logits, the 0.803 label SNR, the floor-lookup critic,
v37 tying v31 at 2.25x the rollouts, and every non-replication in this file.

### 1-3. VinePPO, GRPO, self-competition — all null

Correlation of each advantage estimator with the vine reference:

| estimator | pearson | spearman | sign |
|---|---:|---:|---:|
| A_critic = G − V_critic (what `ppo_update` does) | −0.087 | −0.036 | 0.438 |
| A_vine = G − V_mc (VinePPO) | −0.046 | −0.052 | 0.481 |
| A_group = G_ep − group mean (GRPO) | −0.055 | −0.010 | 0.488 |
| A_self = G_ep − greedy on same seed (GAZ PTP) | −0.107 | −0.101 | 0.284 |

A first run showed VinePPO at +0.254, which was an artifact: the reference
`Q_mc − V_mc` and the arm `G − V_mc` shared one V estimate, so its noise sat in
both. With disjoint rollout sets the effect vanishes. **Beware any comparison
where the estimator and the yardstick share a term.**

Reference noise caps achievable correlation near 0.37, and all four arms are at
or below zero, so these are genuine nulls rather than attenuation.

### 4. Gumbel AlphaZero label allocation — null, with the mechanism identified

`lightspeed/_gumbel_label_probe.py`, 56 states, reference pools of 24
continuations per action, strategies replayed against the pools at matched
budget. Metric is simple regret against the reference best.

| budget | uniform (current) | sequential halving | gumbel-top2+SH | gumbel-top3+SH |
|---:|---:|---:|---:|---:|
| 8 | **0.0939** | **0.0895** | 0.1021 | 0.1067 |
| 16 | **0.0740** | 0.0754 | 0.0875 | 0.0897 |
| 32 | **0.0527** | 0.0590 | 0.0726 | 0.0685 |

Gumbel top-m needs the prior's top-m to contain the best action. Measured:

| m | reference-best inside prior top-m |
|---:|---:|
| 1 | 0.518 |
| 2 | 0.661 |
| 3 | 0.839 |

with 5.09 legal actions on average. Restricting to top-2 caps the hit rate at
0.66, below what uniform reaches at the same budget. Sequential halving ties
uniform because the tree is only ~5 wide and SH's edge grows with arm count.

**The most useful number in this table is not in it**: the policy's own prior
picks the best action **51.8%** of the time for free, and 32 rollouts per state
buys **57.6%**. Six points for 32x the compute — which is why label volume was
the only lever that ever paid and why better estimators do not.

### 5. AIVAT variance reduction — built, unbiased, and worth 0%

`lightspeed/_aivat_eval.py`. Two findings, and the first one matters most:

**AIVAT cannot apply to this project's standard evaluation at all.** Greedy
policy plus `deterministic_combat` makes a run a deterministic function of its
seed, so the chance term and the known-strategy term are both identically zero.
All variance is seed heterogeneity, and paired seeds already remove 39-43% of it
(rho 0.39-0.43 on `runs/sharp_rebaseline_600seeds.jsonl`). The engine exposes no
RNG reseed, so nature cannot be resampled in a fork regardless.

Applied where randomness does exist — a stochastic policy at T=0.2, 600 runs,
correcting 59.9% of decisions (the rest trigger combat when priced):

| | |
|---|---:|
| unbiasedness (corrected − plain) | **−0.024 ± 0.082** |
| sd, plain → corrected | 6.603 → 6.877 |
| corr(floor, correction) | **+0.014** |
| corr needed to break even | 0.153 |
| best possible reduction at the optimal β | **0.0%** |

The estimator is correct — unbiased to within a tenth of its own standard error
— and the control variate carries no information, so it adds Var(C)=4.06 and
removes nothing.

**The ceiling is not the problem.** Decomposing 50 seeds x 6 sampling seeds:
**91% of the floor variance at T=0.2 is policy sampling**, only ~9% is seed
heterogeneity. A perfect V could remove that 91% — the same magnitude AIVAT
achieved in poker. What blocks it is the value function's inability to rank
candidates *within* a decision: all candidates share a floor, and the critic is
~80% a floor lookup. That is now the fourth independent measurement of the same
defect, alongside the floor-lookup decomposition, the zero advantage
correlation, and the 0.129-nat logit margins.

**The actionable form**: a future critic becomes profitable here the moment
`corr(floor, correction)` clears **0.153**. That is a concrete, cheap acceptance
test for any value-function work, and this harness runs it in four minutes.

## Open threads

- **RUNNING 2026-08-01: `pilot1`, the first on-policy RL run on this stack.**
  `runs/ppo/pilot1/`, launched with
  `--iterations 120 --episodes 256 --epochs 2 --temperature 0.2
  --target-kl 0.01 --eval-every 40`, warm-started from
  `whole_run_transformer_v37_critic.pt` and `run_critic_v37.pt`. ~100 s per
  iteration, so ~3.4 h. Iteration 1 reproduced the baseline exactly (sampled
  floor 18.33 ± 0.41) with the gradient check at 2.4e-07 and KL 0.005/0.003,
  under the 0.01 guard.

  **The bar was set before any data arrived**, which is the discipline the three
  non-replications in this file argue for:

  | outcome at iteration 120 | reading |
  |---|---|
  | sampled floor **> 19.5** | the curve is moving; the ~2x loop speedups become worth building immediately |
  | flat within **±0.5** | floor-progress shaping alone does not drive learning at A20 on this timescale — go to the A0 curriculum or reward-screen exploration, not to more of the same |
  | in between | unresolved at n=256; do not read a trend into it |

  Two config choices are deliberate. **A20**, despite the terminal reward being
  a constant there (0 victories in 6,000 runs), because the pilot's actual
  question is whether floor-progress shaping alone moves the policy; A0 changes
  the task *and* costs 65% more per episode. And **`--target-kl 0.01` unchanged**,
  even though it is in tension with fixing drafting — loosening the KL budget and
  adding reward-screen exploration at the same time would confound the pilot.

  Warm-starting is deliberate too. Randomizing the three decision types the net
  contributes to costs 5.85 + 2.44 + 1.91 ≈ **10 floors**, and Silverbot's own
  agent without its net reaches floor 14-17 against v37's 22.9. "Drafting is
  wrong" is true relative to a top human and false relative to random.

- **Resolved: regenerate labels against the fixed engine.** `runs/v37_trunc/`
  completed; v37 ties v31 exactly (+0.01 ± 0.51 paired pre-rebuild, and
  +0.40 ± 0.34 on the n=600 post-rebuild set). The 2.25x rollout increase bought
  no policy quality. See [05-model-lineage.md](05-model-lineage.md).
- **The 800-sim quality arm.** Never run; ~1.5 h to complete.
- **30× volume.** v34 stopped mid-generation. Volume is the one lever that has
  paid, and it has only been tested at 10×.
- **CMA-ES re-tune.** The current search parameters were tuned against the buggy
  engine dynamics.
- **Label noise — analyzed 2026-07-30, and it is the binding constraint.**
  See "The labels cannot rank their own actions" below. Paired SNR is 0.803 with
  **66.7% of labels below 1.0**; common random numbers help far less than hoped.
- **Heuristic terms that are off.** A dozen scoring terms exist in the search and
  sit at 0.0 in the tuned config — see [03-combat-search.md](03-combat-search.md).
  Tuning them is untested territory that costs no new engine work.
- **TODO (live bridge, not scheduled): feed telegraphed monster intents into the
  reconstruction.** `native_recommend.py` never sets `move_name`, so the engine
  rolls a random plausible move; measured against the capture it matches the real
  intent only **12.5%** of the time, and predicts *zero* incoming damage against
  Snecko and Snake Plant, which makes the search suppress the Defend in hand and
  attack into the hit. Fix is a `(monster_id, intent, move_base_damage,
  move_hits)` → `MonsterMoveId` lookup — 21 distinct pairs across 12 monsters in
  the existing capture, each uniquely fingerprinted by that tuple — then set
  `spec.move_name`. Verifiable by replaying `sts_raw_states.log`; the measurement
  harness doubles as the test. **Affects live play and autobattle only — no
  training or evaluation impact, so it moves no floor numbers.** Full detail in
  [07-known-issues.md](07-known-issues.md) and
  [09-live-play-bridge.md](09-live-play-bridge.md).

## Pipeline hazards

- **Stale shards resume silently.** `parallel_generate_whole_run_rollouts.py`
  resumes any existing shard with rows, and the launcher skips any dataset whose
  merged file already exists (it prints `HAVE <path>`). This is what makes a
  multi-hour run interruptible, and it is also invisible contamination from an
  aborted attempt. Verify the target directory is empty, or pass `--data-dir` to
  a fresh one.
- **`--tag` and `--data-dir` are required when re-running an arm.** The default
  checkpoint name derives from the arm, and an existing dataset is silently
  reused. This is how v32 and v33 ended up as the same run under two names.
- **Scope silently determines what trains.** Always read the
  `scope=… trainable=…/…` line the trainer prints.
- **Head count is unrecoverable from a checkpoint.** See
  [04-evaluation.md](04-evaluation.md).
