"""Repository-relative paths.

Seventeen tracked files hard-coded `C:\\Users\\Alvin\\grok\\sts-project\\...`,
which leaks a developer's home directory and breaks on every other machine. All
of them are derivable from this file's own location, so they are derived here
once instead of being repeated.

Kept dependency-free on purpose. Several callers import this from inside a
`_worker_init` that runs BEFORE `slaythespire` is importable -- its whole job is
to put the native build directory on `sys.path` -- so importing anything heavy
here would be a circular dependency in the workers that need it most.

The `run_v*.cmd` launchers still carry absolute paths and are deliberately left
alone: `AGENTS.md` records them as provenance, the exact command a past
experiment was run with. Rewriting them would falsify that record.
"""
from __future__ import annotations

from pathlib import Path

LIGHTSPEED_DIR = Path(__file__).resolve().parent
SLAY_SIM_DIR = LIGHTSPEED_DIR.parent
PROJECT_ROOT = SLAY_SIM_DIR.parent

RUNS_DIR = SLAY_SIM_DIR / "runs"
# The compiled pybind11 module. Workers spawned on Windows re-import their module
# and do not inherit the parent's sys.path, so each one re-inserts this.
NATIVE_BUILD_DIR = PROJECT_ROOT / "sts_lightspeed" / "build"

# The 100-run, potion-inclusive A20 Heart benchmark with a 60/20/20 split by run.
HUMAN_BENCHMARK = RUNS_DIR / "human_fight_benchmark_100.json"


def native_build_path() -> str:
    """`sys.path` entry for the native engine, as a str for `sys.path.insert`."""
    return str(NATIVE_BUILD_DIR)
