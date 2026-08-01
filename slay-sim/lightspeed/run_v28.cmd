@echo off
set "PYTHONPATH=C:\Users\Alvin\grok\sts-project\sts_lightspeed\build;C:\Users\Alvin\grok\sts-project\slay-sim"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
cd /d "C:\Users\Alvin\grok\sts-project\slay-sim"
python -m lightspeed.run_long_training_v28 > runs\whole_run_long_v28_master.log 2> runs\whole_run_long_v28_master.err.log
