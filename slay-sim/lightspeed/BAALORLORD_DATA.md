# Baalorlord demonstration pipeline

This pipeline uses curated public Ironclad A20 Heart run histories as a human
strategic prior. Combat remains entirely owned by the native expectimax MCTS.

## Import

```powershell
$env:PYTHONPATH='<repo>\sts_lightspeed\build'
python -m lightspeed.import_baalorlord_runs `
  --input '<archive>\outputs\baalorlord_ironclad_a20_heart_20runs.jsonl' `
  --out runs/baalorlord_a20_human_train_v1.pt `
  --validation-out runs/baalorlord_a20_human_validation_v1.pt `
  --audit runs/baalorlord_a20_human_split_v1.audit.json
```

The importer currently accepts only decisions with visible alternatives:

- post-combat card rewards and explicit skips;
- Singing Bowl's max-HP choice, represented as the reward skip action;
- boss relic rewards.

It reconstructs the deck and relic sequence from earlier floors. Map state,
potions, Neow, boss identity, relic counters, and exact RNG state are not
available, so every row is marked as approximate state.

Current 20-run result:

- 1,140 floor records;
- 486 card reward demonstrations;
- 40 boss relic demonstrations;
- 428 training decisions from 16 runs;
- 98 validation decisions from four entirely held-out runs.

## Train and evaluate

```powershell
python -m lightspeed.train_whole_run_replay `
  --dataset runs/baalorlord_a20_human_train_v1.pt `
  --validation-dataset runs/baalorlord_a20_human_validation_v1.pt `
  --load runs/whole_run_transformer_rollout_replay_a20_v13.pt `
  --out runs/whole_run_transformer_baalor_prior_a20_v22.pt `
  --epochs 25 --lr 0.0001 --anchor-weight 0.5 `
  --train-scope human-adapter

python -m lightspeed.eval_human_demonstrations `
  --dataset runs/baalorlord_a20_human_validation_v1.pt `
  --checkpoint runs/whole_run_transformer_baalor_prior_a20_v22.pt
```

The zero-initialized `human_score` residual is applied only on ordinary reward
and boss-relic screens. Loading older checkpoints leaves it at zero and
preserves their behavior.

## Initial finding

Direct imitation generalizes to held-out human runs (v13: 32/98 agreement,
v22: 48/98; NLL 1.645 to 1.140), but does not improve our simulator:

- ungated v22: -0.983 mean floors over 300 paired seeds;
- gated 10% prior: +0.060 over 200 exploratory seeds, confidence interval
  spanning zero;
- stronger gated priors were negative.

Do not promote v22. Keep v13 active.

The next step is simulator relabeling: use the human prior to propose card and
boss-relic candidates at native states, evaluate those candidates with matched
deterministic continuations, and train on the resulting soft counterfactual
targets. This preserves the data's coverage while optimizing for the actual
combat policy and simulator.
