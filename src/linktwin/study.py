"""Scenario evaluation: run every method on shared latent trajectories and tabulate efficiency."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .bell import ACTIONS
from .drift import DriftModel
from .methods import (
    SHADOW_GRID,
    LinkConfig,
    TwinConfig,
    make_episode,
    run_fixed,
    run_oracle,
    run_shadow,
    run_twin,
    score,
)

FIXED_NAMES = [f"fixed_{a.name}" for a in ACTIONS]


def evaluate(
    drift: DriftModel,
    cfg: LinkConfig,
    seed: int,
    twins: dict[str, TwinConfig],
    shadow_grid: list[tuple[int, int]] | None = None,
    keep_cycles: bool = False,
) -> list[dict]:
    """Evaluate all methods on the trajectory of ``seed``; one record per method.

    ``twins`` maps a label to a twin configuration.  Estimators believe a readout error of
    ``cfg.assumed_readout_error`` while the hardware has ``cfg.readout_error`` (mismatch knob).
    """
    grid = SHADOW_GRID if shadow_grid is None else shadow_grid
    rng = np.random.default_rng([seed, 0])
    ep = make_episode(drift, cfg, rng)
    rows: list[dict] = []

    def add(name: str, run, c: LinkConfig = cfg) -> None:
        s = score(ep, run, c)
        row = {"method": name, "seed": seed, "efficiency": s["efficiency"],
               "mean_monitor_pairs": s["mean_monitor_pairs"], "outage_rate": s["outage_rate"]}
        if keep_cycles:
            ok = s["oracle"] > 0
            row["cycle_eff"] = (s["delivered"][ok] / s["oracle"][ok]).tolist()
        rows.append(row)

    add("oracle", run_oracle(ep, cfg, rng))
    for i, name in enumerate(FIXED_NAMES):
        add(name, run_fixed(ep, cfg, rng, i))
    add("static_model", run_shadow(ep, cfg, np.random.default_rng([seed, 1]), 10**9, 48))
    for k, (p, m) in enumerate(grid):
        add(f"shadow_P{p}_M{m}", run_shadow(ep, cfg, np.random.default_rng([seed, 2, k]), p, m))
    for k, (label, tw) in enumerate(twins.items()):
        add(label, run_twin(ep, cfg, np.random.default_rng([seed, 3, k]), drift, tw))
    return rows


def record_cfg(cfg: LinkConfig, drift: DriftModel, **extra) -> dict:
    d = {f"cfg_{k}": v for k, v in asdict(cfg).items()}
    d.update({"drift_sigma": drift.sigma, "drift_phi": drift.phi, "drift_jump_prob": drift.jump_prob})
    d.update(extra)
    return d
