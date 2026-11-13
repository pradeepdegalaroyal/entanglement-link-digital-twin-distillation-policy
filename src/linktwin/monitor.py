"""Pair-consuming monitoring: matched-basis ZZ/XX/YY measurements and their likelihood."""
from __future__ import annotations

import numpy as np

from .bell import CORR_MATRIX, Array, weights_from_correlators


def readout_attenuation(readout_error: float) -> float:
    """Correlator attenuation ``(1 - 2e)^2`` when each of the two readouts flips with prob ``e``."""
    return (1.0 - 2.0 * readout_error) ** 2


def measure(p_true: Array, pairs_per_basis: int, readout_error: float, rng: np.random.Generator) -> Array:
    """Counts of ``+1`` outcomes in each of the three bases (shape ``(3,)``)."""
    c = readout_attenuation(readout_error) * (CORR_MATRIX @ p_true)
    prob_plus = np.clip((1.0 + c) / 2.0, 0.0, 1.0)
    return rng.binomial(pairs_per_basis, prob_plus)


def plug_in_weights(counts: Array, n_per_basis: int, assumed_readout_error: float = 0.0) -> Array:
    """Point estimate of the Bell weights from accumulated counts."""
    c = (2.0 * counts / n_per_basis - 1.0) / readout_attenuation(assumed_readout_error)
    return weights_from_correlators(np.clip(c, -1.0, 1.0))


def log_likelihood(
    p_particles: Array, counts: Array, n_per_basis: int, assumed_readout_error: float = 0.0
) -> Array:
    """Binomial log-likelihood of ``counts`` for each particle's Bell weights, shape ``(M,)``."""
    c = readout_attenuation(assumed_readout_error) * (p_particles @ CORR_MATRIX.T)
    pi = np.clip((1.0 + c) / 2.0, 1e-9, 1 - 1e-9)
    return (counts * np.log(pi) + (n_per_basis - counts) * np.log1p(-pi)).sum(axis=-1)
