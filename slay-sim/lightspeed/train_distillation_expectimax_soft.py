"""Soft-launch wrapper around train_distillation_expectimax -- same script,
reduced worker/thread counts (2 PPO workers, 2 distillation workers x 1
tree instead of 6 and 6x2) so it coexists with tune_search_cma.py's own
12-worker CMA-ES run using the rest of the machine's cores, rather than
badly oversubscribing a 12-core machine with two full-tilt jobs at once.
Expect BOTH jobs to run slower than they would alone -- that's the accepted
tradeoff for running them concurrently rather than sequentially.

Run:  PYTHONPATH=".;../sts_lightspeed/build" python -m lightspeed.train_distillation_expectimax_soft
"""

from . import train_distillation_expectimax as base

base.N_WORKERS = 2
base.DISTILL_WORKERS = 2
base.DISTILL_TREES = 1
base.OUT_CHECKPOINT = "lightspeed/checkpoint_distillation_expectimax_soft.pt"
base.LOG_PATH = "lightspeed/train_distillation_expectimax_soft_progress.log"

if __name__ == "__main__":
    base.main()
