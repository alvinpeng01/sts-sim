@echo off
set "PYTHONPATH=C:\Users\Alvin\grok\sts-project\sts_lightspeed\build;C:\Users\Alvin\grok\sts-project\slay-sim"
cd /d "C:\Users\Alvin\grok\sts-project\slay-sim"
python -m lightspeed.eval_whole_run_policy runs/whole_run_transformer_long_horizon_a20_v26.pt runs/whole_run_transformer_experts_a20_v27.pt --runs 500 --seed-base 7600000 --sims 300 --ascension 20 --out runs/whole_run_v26_vs_v27_a20_500seeds_300sims.jsonl > runs/whole_run_v26_vs_v27_a20_500seeds_300sims.log 2> runs/whole_run_v26_vs_v27_a20_500seeds_300sims.err.log
