"""One authoritative loader for expectimax search configuration.

The CMA-ES artifact contains two kinds of settings: numeric parameters in
``params`` and selector settings in ``fitness_config``.  Loading only the
former silently changes the search being evaluated or distilled.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


DEFAULT_SEARCH_CONFIG_PATH = Path(__file__).with_name("tuned_search_params.json")


def load_search_config(path: str | Path) -> dict[str, Any]:
    """Load a CMA-ES artifact without dropping selector settings."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "params" not in data:
        raise ValueError(f"{path} does not contain a search parameter block")
    return data


def active_search_config_mismatches(config: Mapping[str, Any]) -> list[str]:
    """Describe differences between a saved config and the native runtime."""
    import slaythespire as sts

    mismatches = []
    params = config.get("params", config)
    actual = dict(sts.get_search_params())
    for key, expected in params.items():
        if key not in actual:
            mismatches.append(f"missing native parameter {key!r}")
        elif not math.isclose(
                float(actual[key]), float(expected), rel_tol=1e-12, abs_tol=1e-12):
            mismatches.append(f"{key}: active={actual[key]!r} expected={expected!r}")
    fitness_config = config.get("fitness_config", {})
    expected_halving = bool(fitness_config.get("seq_halving", False))
    if bool(sts.get_seq_halving()) != expected_halving:
        mismatches.append(
            f"sequential_halving: active={bool(sts.get_seq_halving())} "
            f"expected={expected_halving}")
    if hasattr(sts, "get_state_merging") and bool(sts.get_state_merging()):
        mismatches.append("unsafe compact state merging is enabled")
    if hasattr(sts, "get_rave") and bool(sts.get_rave()):
        mismatches.append("RAVE is enabled but the tuned config was calibrated without it")
    if hasattr(sts, "get_leaf_eval_mode"):
        mode, _ = sts.get_leaf_eval_mode()
        if mode != "rollout":
            mismatches.append(f"leaf_eval_mode: active={mode!r} expected='rollout'")
    return mismatches


def apply_search_config(config: Mapping[str, Any], *, verify: bool = True) -> None:
    """Reset, apply, and optionally verify one complete search configuration."""
    import slaythespire as sts

    if hasattr(sts, "reset_search_config"):
        sts.reset_search_config()
    else:
        # Compatibility with an older extension. Explicit selector resets
        # prevent the most dangerous forms of cross-experiment leakage.
        sts.set_rave(False)
        sts.set_seq_halving(False)
        sts.set_state_merging(False)
        sts.set_leaf_eval_mode("rollout")
        if hasattr(sts, "set_early_act_card_biases"):
            sts.set_early_act_card_biases({})
    params = config.get("params", config)
    sts.set_search_params(params)
    fitness_config = config.get("fitness_config", {})
    sts.set_seq_halving(bool(fitness_config.get("seq_halving", False)))
    if verify:
        mismatches = active_search_config_mismatches(config)
        if mismatches:
            raise RuntimeError(
                "native MCTS configuration verification failed: "
                + "; ".join(mismatches))


def ensure_search_config(path: str | Path = DEFAULT_SEARCH_CONFIG_PATH) -> dict[str, Any]:
    """Load and verify the authoritative config before whole-run combat."""
    config = load_search_config(path)
    # Always reset first: the artifact intentionally contains only tuned
    # overrides, so checking those keys alone cannot detect stale values in
    # unspecified parameters left behind by an earlier experiment.
    apply_search_config(config, verify=True)
    return config


def sequential_halving_enabled(config: Mapping[str, Any] | None) -> bool:
    return bool(config and config.get("fitness_config", {}).get("seq_halving", False))
