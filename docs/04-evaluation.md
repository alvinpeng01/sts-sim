# Evaluation

`lightspeed/eval_whole_run_policy.py`. Every result quoted anywhere in these
docs was recomputed from the `.jsonl` files this harness produced.

## The protocol

```bash
cd slay-sim
$env:PYTHONPATH='..\sts_lightspeed\build;.'
python -m lightspeed.eval_whole_run_policy `
  runs/whole_run_transformer_outcome_a20_v28.pt `
  runs/whole_run_transformer_yield10x_a20_v31.pt `
  --runs 200 --seed-base 18_900_000 --sims 300 --ascension 20 `
  --torch-threads 1 --out runs/my_comparison.jsonl
```

Every checkpoint on the command line plays the **same seed range**
(`seed_base + 0 … seed_base + runs-1`) under the **same search configuration**,
with `deterministic_combat` on by default — MCTS search seeds are derived from
the run seed and floor, so combat is reproducible and two checkpoints facing the
same seed face the same fights. That is what makes the comparison paired.

Action selection is **argmax** unless `--sample` is passed. `--temperature`
applies only with `--sample`.

Output is one JSON line per run: checkpoint, seed, floor, act, hp, outcome,
seconds, plus the full combat audit (battles, search/searched/forced decisions,
total simulations, stall and safety counters). A `summary` line follows each
checkpoint with mean/median floor, act2/3/4 reach, early losses, and victories.

## Metrics

**Mean floor reached** is the primary metric. Win rate is the goal but is 0% at
A20 across every checkpoint measured so far, so it carries no signal there; at
A0 it does (v31: 13/100).

Report the **paired** delta, not the difference of means: for each seed take
`floor(B) − floor(A)`, then mean and standard error over seeds. Pairing removes
seed difficulty, which is the dominant variance term. A useful supplement is the
win/tie/loss split over seeds.

Recomputed reference numbers (200 seeds, A20, 300 sims, after the Armaments fix):

| Checkpoint | Mean floor | sem | Max floor | Paired vs v28 |
|---|---:|---:|---:|---:|
| v28 `outcome` | 21.73 | 0.48 | 39 | — |
| v31 `yield10x` | **23.57** | 0.53 | 50 | **+1.83 ± 0.55** |
| v33 `conditional_final` | 19.62 | 0.44 | 33 | −2.12 ± 0.58 |

## Traps this harness has already fallen into

**Architecture must come from the checkpoint, not the command line.** There is
no `--dim/--layers/--heads` flag any more. `checkpoint_architecture()` recovers
`dim` from `state["card.weight"].shape[1]` and `layers` from the highest
`encoder.layers.N` index. A single global `--dim 192` once silenced every
checkpoint that did not match it, which is how the v28 baseline vanished from
the v28-vs-v30 comparison without anyone noticing — that file holds 200 rows,
all v30.

**Head count cannot be recovered from a state dict.** `nn.MultiheadAttention`
packs qkv as `(3·dim, dim)` regardless of head count, so a wrong value loads
cleanly and computes different attention. `HEADS_BY_DIM = {96: 4, 192: 6}` is a
hardcoded map that raises on an unknown dim. Future checkpoints should record
architecture metadata beside the weights.

**Do not select checkpoints on validation NLL.** NLL and mean floor are only
weakly coupled on this task, in both directions:

- v29 improved held-out NLL from 1.645 to 1.140 and **lost** 0.983 floors.
- v31 moved policy loss 0.005 nats — essentially nothing — and **gained** 2.60
  floors, while its trainable modules reorganized 12–44% by relative L2.
- Validation NLL has picked the wrong epoch in three consecutive runs (v30, v32,
  v33), which is why `train_whole_run_v27.py:85-89` now recommends
  `--checkpoint-every` plus a paired floor sweep instead.

The mechanism: NLL is dominated by the many high-entropy decisions whose targets
are near-uniform, while floors depend on a minority of pivotal ones.

**Cross-engine comparisons are invalid.** The Armaments upgrade leak was fixed
on 2026-07-30 at 20:44. Any evaluation run before that timestamp measured a
different game. v28's own mean floor moved 23.70 → 21.73 on the identical 200
seeds across that boundary. Check the mtime of
`sts_lightspeed/build/slaythespire.cp313-win_amd64.pyd` against the eval log's
before comparing two numbers.

**The same marginal collapse shows up at boss relics.** Over 100 A20 seeds with
v31 (`lightspeed/_relic_uptake.py`), 68 runs acquired a boss relic and the
distribution across 21 distinct relics is nearly flat — Busted Crown and Slaver's
Collar lead at 6% of runs, and eight relics including Runic Dome tie at 3%. A
policy with real preferences over boss relics would not produce a near-uniform
histogram. Read alongside the campfire result below, this looks like the same
underlying failure: the network has learned the label *marginal* rather than a
state-conditional preference. It is also why the Runic Dome question resolved as
"not worth fixing" — see [07-known-issues.md](07-known-issues.md).

**Argmax is not a neutral readout.** On campfire decisions the net emits roughly
P(REST)=0.41 / P(SMITH)=0.35 in every state, matching the label marginal — but
argmax turns that into REST 100% of the time, while the labels prefer SMITH in
26% of states. Sampling is available (`--sample`) and measurably worse on floors
at both temperatures tried:

| Readout | Mean floor (200 seeds, A20, 300 sims) |
|---|---:|
| argmax | 26.30 |
| sample, T = 0.5 | 14.82 |
| sample, T = 1.0 | 13.31 |

So argmax's collapse of the marginal is a real distortion *and* still the better
policy — the sampled proportions are not worth preserving at this level of
policy quality.

## Determinism

v28's per-seed floors are bit-identical across evaluation files produced against
the same engine build. If two runs of the same checkpoint on the same seeds
disagree, something in the search configuration or the engine changed.

## Evaluation files on disk

| File | What it is |
|---|---|
| `postfix_v28_v31_v33_a20_200seeds.jsonl` | post-fix rebaseline, the current reference |
| `v36_sweep_a20_200seeds.jsonl` | post-fix, v31 vs three v36 epoch checkpoints |
| `whole_run_v28_vs_yield10x_a20_200seeds_300sims.jsonl` | pre-fix, the +2.60 result |
| `whole_run_v28_vs_labelq300_a20_200seeds_300sims.jsonl` | pre-fix, the 1× relabelling control |
| `whole_run_v28_vs_batched_a20_200seeds_300sims.jsonl` | pre-fix, v32 |
| `whole_run_v28_v31_v32_a20_200seeds_900sims.jsonl` | pre-fix, the 900-sim budget test |
| `a0_v28_v31_v33_100seeds.jsonl` | pre-fix, Ascension 0 |
| `whole_run_v28_vs_v30_a20_200seeds_300sims.jsonl` | **unpaired** — contains only v30 rows |
| `v31_sample_T0.5.jsonl`, `v31_sample_T1.0.jsonl` | pre-fix, sampled readout |
| `room_audit_v28_v31.jsonl` | 60-seed room-level audit |
