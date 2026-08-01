@echo off
REM V31 label experiment: dim=96 layers=2 heads=4, warm-start from v28.
REM Every arm uses v28's architecture and hyperparameters. The variable is the
REM training labels: arm 300 is the control (no generation needed), arm 800 tests
REM label quality, arm yield tests label count.
REM Run the control first — it is ~12 min and validates the pipeline before the
REM expensive arms. The yield arm is many hours; see docs/06-experiment-log.md.
set "PYTHONPATH=C:\Users\Alvin\grok\sts-project\sts_lightspeed\build;C:\Users\Alvin\grok\sts-project\slay-sim"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
cd /d "C:\Users\Alvin\grok\sts-project\slay-sim"
python -m lightspeed.run_label_quality_v31 --arm 300 > runs\whole_run_labelq300_v31_master.log 2> runs\whole_run_labelq300_v31_master.err.log
python -m lightspeed.run_label_quality_v31 --arm 800 > runs\whole_run_labelq800_v31_master.log 2> runs\whole_run_labelq800_v31_master.err.log
python -m lightspeed.run_label_quality_v31 --arm yield --label-scale 10 > runs\whole_run_yield10x_v31_master.log 2> runs\whole_run_yield10x_v31_master.err.log
