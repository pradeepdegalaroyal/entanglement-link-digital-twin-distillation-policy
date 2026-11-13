"""Bell-diagonal states, entanglement-distillation recurrences and the action table.

Bell-diagonal weights are stored as ``p = (A, B, C, D)`` for the Bell states
``(Phi+, Psi-, Psi+, Phi-)`` (the ordering of the DEJMPS recurrence).  All functions are
vectorised over leading axes, so a particle cloud ``(M, 4)`` is processed in one call.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

Array = np.ndarray

# Expected Pauli correlators <ZZ>, <XX>, <YY> as linear maps of p = (A, B, C, D).
CORR_MATRIX = np.array(
    [
        [1.0, -1.0, -1.0, 1.0],  # ZZ
        [1.0, -1.0, 1.0, -1.0],  # XX
        [-1.0, -1.0, 1.0, 1.0],  # YY
    ]
)


def bell_weights(fidelity: Array, dephasing_share: Array) -> Array:
    """Bell-diagonal weights of a dephasing-biased link state.

    ``A = F`` (Phi+), the Phi- weight is ``(1 - F) * w`` and Psi+/Psi- share the rest.
    """
    f = np.asarray(fidelity, dtype=float)
    w = np.asarray(dephasing_share, dtype=float)
    rest = 1.0 - f
    return np.stack([f, rest * (1.0 - w) / 2.0, rest * (1.0 - w) / 2.0, rest * w], axis=-1)


def correlators(p: Array) -> Array:
    """Pauli correlators (ZZ, XX, YY) of Bell-diagonal states, shape ``(..., 3)``."""
    return p @ CORR_MATRIX.T


def weights_from_correlators(c: Array) -> Array:
    """Invert :func:`correlators`; clip to the simplex (plug-in estimate)."""
    zz, xx, yy = c[..., 0], c[..., 1], c[..., 2]
    b = (1.0 - zz - xx - yy) / 4.0
    a = b + (zz + xx) / 2.0
    d = b + (zz + yy) / 2.0
    cc = b + (xx + yy) / 2.0
    p = np.stack([a, b, cc, d], axis=-1)
    p = np.clip(p, 0.0, None)
    return p / np.clip(p.sum(axis=-1, keepdims=True), 1e-12, None)


def depolarise(p: Array, q: float) -> Array:
    """Mix with the maximally mixed state: ``(1 - q) p + q / 4`` (local gate noise)."""
    return (1.0 - q) * p + q / 4.0


def dejmps_round(p: Array) -> tuple[Array, Array]:
    """One DEJMPS round on two copies of a Bell-diagonal state -> (output weights, success prob)."""
    a, b, c, d = (p[..., i] for i in range(4))
    n = (a + b) ** 2 + (c + d) ** 2
    out = np.stack([a * a + b * b, 2 * c * d, c * c + d * d, 2 * a * b], axis=-1) / n[..., None]
    return out, n


def bbpssw_round(p: Array) -> tuple[Array, Array]:
    """One BBPSSW round (inputs are twirled to Werner form first)."""
    f = p[..., 0]
    e = (1.0 - f) / 3.0
    n = f * f + 2 * f * e + 5 * e * e
    f2 = (f * f + e * e) / n
    e2 = (1.0 - f2) / 3.0
    out = np.stack([f2, e2, e2, e2], axis=-1)
    return out, n


class Action(NamedTuple):
    """A distillation policy: ``rounds`` repetitions of ``protocol`` (``none`` = use raw pairs)."""

    protocol: str
    rounds: int

    @property
    def name(self) -> str:
        return "raw" if self.rounds == 0 else f"{self.protocol}x{self.rounds}"


ACTIONS: tuple[Action, ...] = (
    Action("none", 0),
    Action("bbpssw", 1),
    Action("bbpssw", 2),
    Action("dejmps", 1),
    Action("dejmps", 2),
    Action("dejmps", 3),
)


def run_action(p: Array, action: Action, gate_noise: float = 0.0) -> tuple[Array, Array]:
    """Apply an action; return (output weights, expected output pairs per raw pair).

    Each round first mixes both input copies with the maximally mixed state (local gate noise
    ``gate_noise``), then applies the ideal recurrence.  A round consumes two inputs and
    succeeds with probability ``n``, so it yields ``n / 2`` output pairs per input pair.
    """
    yield_per_raw = np.ones(p.shape[:-1])
    for _ in range(action.rounds):
        p = depolarise(p, gate_noise)
        step = dejmps_round if action.protocol == "dejmps" else bbpssw_round
        p, n = step(p)
        yield_per_raw = yield_per_raw * n / 2.0
    return p, yield_per_raw


def action_values(p: Array, f_req: float, gate_noise: float = 0.0) -> Array:
    """Value per raw pair of every action, shape ``(..., n_actions)``.

    A delivered pair counts only if its fidelity meets the application threshold ``f_req``.
    """
    cols = []
    for act in ACTIONS:
        out, y = run_action(p, act, gate_noise)
        cols.append(y * (out[..., 0] >= f_req))
    return np.stack(cols, axis=-1)


def teleportation_fidelity(f: Array) -> Array:
    """Average teleportation fidelity through a state with singlet fraction ``f``: (2f + 1) / 3."""
    return (2.0 * np.asarray(f) + 1.0) / 3.0
