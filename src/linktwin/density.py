"""Explicit four-qubit density-matrix simulation used to validate the closed-form recurrences."""
from __future__ import annotations

import numpy as np

from .bell import Array

_I = np.eye(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_S2 = 1 / np.sqrt(2)
# Bell states in the order (Phi+, Psi-, Psi+, Phi-) used by the recurrences.
BELL = np.array(
    [
        [_S2, 0, 0, _S2],
        [0, _S2, -_S2, 0],
        [0, _S2, _S2, 0],
        [_S2, 0, 0, -_S2],
    ],
    dtype=complex,
)


def bell_diagonal_state(p: Array) -> Array:
    """4x4 density matrix with Bell weights ``p``."""
    return sum(p[i] * np.outer(BELL[i], BELL[i].conj()) for i in range(4))


def _rx(theta: float) -> Array:
    return np.cos(theta / 2) * _I - 1j * np.sin(theta / 2) * _X


def _op(n: int, ops: dict[int, Array]) -> Array:
    out = np.array([[1.0 + 0j]])
    for q in range(n):
        out = np.kron(out, ops.get(q, _I))
    return out


def _cnot(n: int, c: int, t: int) -> Array:
    p0 = np.diag([1, 0]).astype(complex)
    p1 = np.diag([0, 1]).astype(complex)
    return _op(n, {c: p0}) + _op(n, {c: p1, t: _X})


def distil_once(p: Array, protocol: str) -> tuple[Array, float]:
    """Run one distillation round on two copies of the Bell-diagonal state ``p``.

    Qubit order ``[a1, b1, a2, b2]``.  For ``dejmps`` Alice applies Rx(pi/2) and Bob Rx(-pi/2)
    on both pairs; for ``bbpssw`` the input is first twirled to Werner form.  Then bilateral
    CNOTs (pair 1 controls pair 2), Z measurement of pair 2, keep coincident outcomes.
    Returns the Bell weights of the kept pair 1 and the success probability.
    """
    if protocol == "bbpssw":
        e = (1 - p[0]) / 3
        p = np.array([p[0], e, e, e])
    rho1 = bell_diagonal_state(p)
    rho = np.kron(rho1, rho1)
    if protocol == "dejmps":
        u = _op(4, {0: _rx(np.pi / 2), 2: _rx(np.pi / 2), 1: _rx(-np.pi / 2), 3: _rx(-np.pi / 2)})
        rho = u @ rho @ u.conj().T
    cn = _cnot(4, 0, 2) @ _cnot(4, 1, 3)
    rho = cn @ rho @ cn.conj().T
    keep = np.zeros((16, 16), dtype=complex)
    for k in (0b00, 0b11):  # (a2, b2) outcomes equal
        proj = np.zeros(16, dtype=complex)
        for idx in range(16):
            if (idx & 0b11) == k:
                proj[idx] = 1.0
        keep += np.diag(proj)
    rho_k = keep @ rho @ keep
    succ = float(np.real(np.trace(rho_k)))
    r = rho_k.reshape(4, 4, 4, 4)  # (pair1, pair2, pair1', pair2')
    red = np.einsum("ikjk->ij", r) / succ
    weights = np.array([np.real(BELL[i].conj() @ red @ BELL[i]) for i in range(4)])
    return weights, succ
