# Model lineage

> **Every floor count below predates the 2026-07-31 14:36 engine rebuild** and is
> therefore measured on a different game -- eight engine changes landed, six of
> them altering Ironclad behaviour (Lagavulin, Champ, Darkling, Writhing Mass,
> Toy Ornithopter, Dolly's Mirror). This includes v37's 23.675. Cross-rebuild
> comparisons are invalid for the same reason the Armaments fix invalidated
> everything before it, and run-level numbers are more exposed than combat
> numbers because a boss behaving differently compounds across a whole run.
> See [07-known-issues.md](07-known-issues.md).


Every delta below was recomputed from the `.jsonl` file named beside it, as a
**paired** per-seed mean with its standard error. Numbers from different files
are not comparable to each other: the engine, the search configuration, and the
seed set all changed over the lineage's life. Compare within a file.

## Current state

**Working baseline: `runs/whole_run_transformer_postfix-trunc_a20_v37.pt`**,
re-baselined 2026-07-31 on the post-rebuild engine.
`runs/sharp_rebaseline_600seeds.jsonl` — **600** paired A20 seeds at 300 sims,
all three arms on identical seeds and the same build:

| | mean floor | Act 3+ | wins |
|---|---:|---:|---:|
| v28 | 21.34 ± 0.29 | 14/600 | 0 |
| v31 | 22.49 ± 0.32 | 21/600 | 0 |
| **v37** | **22.89 ± 0.32** | **29/600** | 0 |

| paired | Δ floors | t | W/T/L |
|---|---:|---:|---|
| v31 − v28 | **+1.14 ± 0.34** | 3.39 | 207/258/135 |
| v37 − v28 | **+1.54 ± 0.34** | 4.52 | 217/256/127 |
| v37 − v31 | +0.40 ± 0.34 | 1.16 | 183/253/164 |

**The ordering survives the rebuild; the magnitudes shrink.** v31's advantage
over v28 is +1.14, about 62% of the +1.83 measured pre-rebuild. v37 is the
largest gain over v28 and is at worst equal to v31 — nominally ahead on floors,
Act 3+ reach and W/T/L, but +0.40 ± 0.34 does not separate them. v37 is the
better working baseline on the balance of evidence, not on a decided result.

**A power warning worth more than the numbers.** The first re-baseline of this
same comparison used the project's habitual n=200
(`runs/rebaseline_enginefix_v28_v31_v37_200seeds.jsonl`) and showed
v31 − v28 = +0.41 ± 0.58 (t=0.70) with no pair separating — it read as "the
lineage did not survive the engine fixes". Tripling to n=600 turned that into
+1.14 ± 0.34 (t=3.39). Nothing changed but the sample. At n=200 the paired sem
is ~0.55, so **the standard evaluation cannot resolve anything below about 1.1
floors** — which is larger than most single-version gains in this lineage. Every
promotion decision here was made at that resolution. Prefer n≥600 for any
comparison that decides something; it costs ~6 minutes across six processes.

## v1–v12 — combat-only prototypes

REINFORCE policy gradient over `ActionScoringPolicy`, a per-action feature
encoder with a variable action space. No overworld awareness at all.
Checkpoints are the `lightspeed/checkpoint_*.pt` files. Testbed only.

## v13 — first whole-run transformer

`whole_run_transformer_rollout_replay_a20_v13.pt`. `WholeRunTransformerPolicy`
trained on rollout replay. Production checkpoint until v28.

Against v3 on 1,000 paired seeds: **+0.57 ± 0.17** (10.11 → 10.68). Mean floors
were around 10 in this era; the ~24 of the v28 era reflects later engine, search
and pipeline changes as much as policy quality.

## v14–v21 — incremental structure

Act-2 adapters (v16, v17), an Act-2 residual (v18), boss readiness (v19), late
boss prep (v20), boss progress (v21). Checkpoints are present in `runs/`; none
of these were promoted on a paired evaluation that survives on disk.

## v22 — Baalorlord human prior (rejected)

`whole_run_transformer_baalor_prior_a20_v22.pt`, plus α = 0.1 / 0.25 / 0.5 blend
variants. Fine-tuned on human demonstrations. Held-out NLL on human data
improved 1.645 → 1.140.

The best blend, α = 0.1, on 500 paired seeds vs v13: **−0.09 ± 0.13** — a
non-result. The unblended v22 measured −0.983 floors. `BAALORLORD_DATA.md`'s
decision: do not promote; keep v13.

This is the lineage's first clean demonstration that NLL and floors can move in
opposite directions.

## v23–v25 — routing and search tuning

Route cones (v23), route-tuned (v24), consequences (v25, plus a scoped v25b).
v24 is the checkpoint the v26 comparison uses as its baseline.

## v26 — long-horizon curriculum

`whole_run_transformer_long_horizon_a20_v26.pt`. Act-balanced curriculum,
400/400/240 labels across Acts 1–3, 300 combat sims, 8 rollouts, warm-started
from v24. Launcher `run_long_training_v26.py`.

Against v24 on 500 paired seeds: **+2.86 ± 0.40** (19.61 → 22.47). This is the
**largest paired single-version gain in the lineage** — larger than v31's, which
earlier documentation claimed the record for. v26 also recorded the only A20
victory anywhere in these files (1/500).

## v27 — experts, uncertainty, adapters

`whole_run_transformer_experts_a20_v27.pt`. The architecture expansion described
in [01-architecture.md](01-architecture.md), all residuals zero-initialized.
`--scope experts-structure`.

Against v26 on 500 paired seeds: **+0.80 ± 0.36** (22.25 → 23.05).

## v28 — outcome-supervised curriculum

`whole_run_transformer_outcome_a20_v28.pt`. Warm-started from v27; 80/100/120
labels, 300 sims, 6 rollouts, 24 epochs, lr 3e-5, `--scope all-v27`, legacy v26
datasets mixed in. 1,308 train / 238 validation rows.

Against v27 on 500 paired seeds: **+0.95 ± 0.39** (23.25 → 24.20).

v28 is the baseline every later experiment is measured against, and the rollout
policy every later dataset was generated with.

**Its own training did almost nothing.** Train policy loss moved 1.4274 → 1.4221
over 24 epochs — 0.005 nats — with validation NLL flat at 1.3892 → 1.3891. v28's
advantage over v27 was inherited through the warm start, not produced by its own
optimizer.

## v29 — human adapter (rejected)

`whole_run_transformer_baalor_a20_v29.pt`. `--scope human-adapter`: everything
frozen except `human_score.*`. 100 Baalorlord Ironclad A20 Heart runs,
2,188 train / 556 validation rows, 12 epochs, best val_nll 1.1661.

−2.2 mean floors against v28 on 200 seeds. The smoke variant was neutral at −0.1.
Not promoted.

## v30 — 4× capacity (rejected)

`whole_run_transformer_outcome_a20_v30.pt`, 25.8 MB. dim 96 → 192, layers 2 → 3,
heads 4 → 6: **6,574,511 params**, 4.1× v28. Same 300 sims, 8 rollouts,
30 epochs, lr 5e-5, `--scope all-v27`.

Mean floor **18.76 ± 0.63** over 200 seeds.

**The comparison was never paired.** A single global `--dim 192` was passed to
the evaluation, v28 mismatched, `load_policy` returned `None`, and it was
silently skipped: `whole_run_v28_vs_v30_a20_200seeds_300sims.jsonl` holds 200
rows, all v30. The commonly quoted −5.4 figure compares that 18.76 against a v28
number from a different seed set. It has never been re-run since
`eval_whole_run_policy.py` learned to infer architecture per checkpoint.

**Root cause, and it is not capacity.** `--dim 192` mismatched v28's dim=96, so
the warm start raised `RuntimeError`; the error was swallowed, leaving a random
initialization; `--scope all-v27` then froze that random trunk — 5,010,844 of
6,574,511 parameters — permanently. v30 spent 30 epochs teaching residual heads
to read a random representation.

The training log shows underfitting, not overfitting: train policy loss fell
0.021 nats over 30 epochs and stopped 0.13 above the 1.2722-nat irreducible
floor set by the targets' own entropy, with validation at or below train
throughout.

Fixes that came out of it: `load_model` now raises on a dim mismatch,
`--scope full` exists and trains the trunk, the trainer prints its
trainable/total split, and the evaluator infers architecture per checkpoint.

## v31 — 10× label yield (current best)

`whole_run_transformer_yield10x_a20_v31.pt`. Back to v28's exact architecture so
the warm start transfers. The **only** change is data volume:
`--max-labels-per-episode` 2 → 12, at the same 300 sims and the same v28 rollout
policy. 3,600 new labels, giving **4,008 train / 778 validation** rows with the
legacy v26 sets mixed in — 3.1× v28.

Cost: ~2.75 h generation at 6 workers, 17.4 min training, 4.6 min evaluation.
24 epochs, lr 3e-5, `--scope all-v27`, best_val_nll 1.3311.

| Metric (200 paired seeds, A20, 300 sims, pre-fix) | v28 | v31 |
|---|---:|---:|
| Mean floor | 23.70 | **26.30** |
| Paired delta | — | **+2.60 ± 0.65** |
| W/T/L over seeds | — | 89 / 68 / 43 |
| Act 3+ reach | 10/200 | 22/200 |
| Early loss (≤ floor 10) | 3/200 | 1/200 |
| Victories | 0 | 0 |

Post-fix rebaseline on the same 200 seeds: 21.73 → 23.57, **+1.83 ± 0.55**. The
gain survives the engine fix at about 70% of its original size.

At **Ascension 0**, 100 paired seeds (pre-fix): 35.59 → **39.21**,
**+3.62 ± 0.96**, with victories **6 → 13**. This is the only setting where the
system wins runs at all.

**Attribution.** A control arm, `whole_run_transformer_labelq300_a20_v31.pt`,
relabelled at 1× volume with the same v28 rollout policy:

| Comparison (200 paired seeds, pre-fix) | Δ floors | sem |
|---|---:|---:|
| labelq300 − v28 (relabelling alone) | +0.66 | 0.40 |
| yield10x − v28 (relabelling + 10× volume) | +2.60 | 0.65 |
| **yield10x − labelq300 (volume alone)** | **+1.95** | 0.65 |

Roughly **75% of the gain is label count**, not the fresher rollout policy.
`labelq300` shares the v31 tag but is a control, not a lineage version.

**The fit did not improve.** Policy loss moved 0.005 nats over 24 epochs and
stopped 0.114 above this dataset's 1.2007-nat floor — the same non-result as v28
and v30. The gain came from better labels reaching a frozen-trunk model, not
from the model learning them better.

## v32 / v33 — batching and an unfrozen trunk

`whole_run_transformer_batched_a20_v32.pt` and
`whole_run_transformer_conditional_a20_v33.pt` are **bit-identical** — verified
by hashing all 263 tensors in each. The two runs used the same data, seed, and
hyperparameters (30 epochs, lr 1e-3, batch 32, anchor 0.25, `--scope full`), and
their epoch logs match to four decimals. v33 `conditional` is a re-run of v32,
not a separate experiment.

What v32 changed relative to v31: batch 1 → 32, lr 3e-5 → 1e-3 with
warmup + cosine, and `--scope full` so the transformer trunk trained for the
first time in the lineage (1,617,935 trainable vs 394,771).

**The fit improved exactly as predicted.** Train policy loss fell 1.3229 →
1.2864 (0.037 nats, ~7× v31's entire run) and train argmax rose 40.2% → 51.0%.

**The policy did not.** Two checkpoints came out of this run and they measure
very differently:

| Checkpoint | Eval | Paired vs v28 |
|---|---|---:|
| `batched_a20_v32.pt` (best-val, epoch 7) | pre-fix, 200 seeds | +1.52 ± 0.76 |
| `conditional_a20_v33_final.pt` (last epoch, 29) | post-fix, 200 seeds | −2.12 ± 0.58 |

Those two rows straddle the engine fix, so they do **not** cleanly isolate
best-val versus last-epoch selection. What is clean: v32 at +1.52 sits below
v31's +2.60 on the same pre-fix file, and v33-final at −2.12 sits far below
v31's +1.83 on the same post-fix file. Neither was promoted.

**First real overfitting in the lineage.** Validation NLL bottomed at epoch 7
(1.3315) and rose monotonically to 1.3395 by epoch 29 while train kept falling.
4,008 rows cannot support 1.5M unfrozen parameters. The frozen trunk in v28–v31
had been acting as unintended regularization: those runs never overfit because
76% of the model was held still, not because they were tuned.

A third arm, `noanchor_a20_v33`, ran the same configuration with
`--anchor-weight 0`. Its best val_nll was 1.3326 against conditional's 1.3315 —
the anchor is neither helping nor hurting much. It was never evaluated on floors.

## v34 — 30× yield (abandoned)

`whole_run_scale30_v34_manifest.json` planned 2,400/3,000/3,600 train labels with
`--max-labels-per-episode 0` (unlimited). Generation for Acts 1–2 completed
(`runs/v31_yield30x/`); Act 3 was still running when the run was stopped. No
checkpoint exists. A related `bigdata_v33` manifest at 30× with
`--priority-accept-base 1.0` and `--rollouts 4` was never run either.

The ~6,450 rows that were generated are pre-Armaments-fix and are retained as a
record, not as training data.

## v36 — batching with the trunk frozen (rejected)

`whole_run_transformer_batchfrozen_a20_v36*.pt`. Isolates batching from the
unfrozen trunk: batch 32 and lr 1e-3 as in v32, but `--scope all-v27` as in v31,
24 epochs, `--checkpoint-every 4` so the epoch curve could be swept on floors.

Post-fix, 200 paired seeds, against **v31** as baseline:

| Checkpoint | Mean floor | Paired vs v31 |
|---|---:|---:|
| v31 `yield10x` | 23.57 | — |
| v36 epoch 8 | 22.55 | −1.01 ± 0.51 |
| v36 epoch 16 | 22.85 | −0.71 ± 0.61 |
| v36 final (epoch 24) | 22.55 | −1.01 ± 0.60 |

Every epoch is worse than v31 and the curve is flat, so this is not an
epoch-selection problem. Train policy loss moved 1.3199 → 1.3081 — more than
v31's 0.005 nats, less than v32's 0.037. Batching alone, without the trunk, buys
some fit and costs about one floor.

## v37 — truncated + 18 rollouts (ran; did not beat v31)

`runs/v37_trunc/` — a full regeneration of the v31 label set against the
post-Armaments-fix engine, using the cheaper estimator:
`--combat-sims 300 --rollouts 18 --truncate-after 20 --harvest-rate 0.02
--max-labels-per-episode 12`. That combination costs about what untruncated
`--rollouts 8` does, so it buys ~2.25x the rollouts at equal price, and was
predicted to lift paired SNR from 0.803 toward ~1.2. Generation, training and
evaluation all completed 2026-07-31.

| Metric (200 seeds, A20, 300 sims) | v28 | v37 |
|---|---:|---:|
| Mean floor | 21.76 | **23.675** |
| Median floor | 21 | 22 |
| Act 2+ reach | 121 | 136 |
| Act 3+ reach | 3 | 5 |
| Early loss (<= floor 10) | 6 | **0** |
| Victories | 0 | 0 |

**It did not beat v31.** v31 measures 23.57 against the same v28 baseline on the
same seeds, so v37's +1.92 sits inside noise of v31's +1.81 — the SNR gain did
not convert into policy quality. The early-loss column moving 6 -> 0 is the one
place the distribution clearly changed shape.

**The paired deltas were computed 2026-07-31** and they settle it. Against v28
on its own eval file, v37 is **+1.92 ± 0.47 (t=4.09, W/T/L 87/74/39)** — a real
gain over the baseline. But on `runs/monsterfix_v28_v31_v37_200seeds.jsonl`,
where all three arms share seeds:

| paired, 200 seeds | Δ floors | t | W/T/L |
|---|---:|---:|---|
| v31 − v28 | +1.71 ± 0.53 | 3.20 | 83/70/47 |
| v37 − v28 | +1.73 ± 0.45 | 3.86 | 85/75/40 |
| **v37 − v31** | **+0.01 ± 0.51** | **0.03** | 60/85/55 |

v37 and v31 are the same policy to within a hundredth of a floor. The truncated
estimator bought 2.25× the rollouts at equal cost and **converted none of it**
into policy quality — consistent with the label-SNR diagnosis in
[06-experiment-log.md](06-experiment-log.md), and evidence that paired SNR was
not the binding constraint it was assumed to be. These numbers still predate the
2026-07-31 14:36 rebuild.

The earlier `v37_postfix` attempt (90 labels, untruncated) was abandoned; its
directory is retained but must not be mixed with `v37_trunc`.

## Summary

| Version | Arch | Trained | Paired result | Verdict |
|---|---|---|---|---|
| v13 | 96/2/4 | rollout replay | +0.57 vs v3 (1000 seeds) | superseded |
| v22 | 96/2/4 | human prior | −0.09 vs v13 (500 seeds) | rejected |
| v26 | 96/2/4 | long-horizon curriculum | **+2.86 vs v24** (500 seeds) | superseded |
| v27 | 96/2/4 | experts + adapters | +0.80 vs v26 (500 seeds) | superseded |
| v28 | 96/2/4 | outcome curriculum | +0.95 vs v27 (500 seeds) | baseline |
| v29 | 96/2/4 | human adapter | −2.2 vs v28 | rejected |
| v30 | 192/3/6 | 4× capacity, random frozen trunk | 18.76 mean, **unpaired** | rejected |
| **v31** | 96/2/4 | **10× label volume** | **+2.60 pre-fix / +1.83 post-fix vs v28** | **current best** |
| v32 = v33 | 96/2/4 | batch 32, lr 1e-3, full scope | +1.52 (best-val) / −2.12 (final) | rejected |
| v34 | — | 30× volume | generation abandoned | incomplete |
| v36 | 96/2/4 | batch 32, frozen trunk | −1.01 vs v31 | rejected |
| v37 | 96/2/4 | truncated, 18 rollouts | +1.92 vs v28 (200 seeds, unpaired, pre-engine-fix) | did not beat v31 |
