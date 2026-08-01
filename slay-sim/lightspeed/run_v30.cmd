@echo off
REM V30 training: dim=192 layers=3 heads=6, 300 combat sims. NOT warm-started —
REM the dim mismatch against v28 (dim=96) means this trains from scratch, which
REM train_whole_run_v27.py now rejects outright. Kept for provenance; see
REM docs/05-model-lineage.md. Superseded by run_v31.cmd.
set "PYTHONPATH=C:\Users\Alvin\grok\sts-project\sts_lightspeed\build;C:\Users\Alvin\grok\sts-project\slay-sim"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
cd /d "C:\Users\Alvin\grok\sts-project\slay-sim"
python -m lightspeed.run_long_training_v30 > runs\whole_run_long_v30_master.log 2> runs\whole_run_long_v30_master.err.log
