"""Latent drift of a dephasing-biased entanglement link (Ornstein-Uhlenbeck plus jumps)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bell import Array, bell_weights


def sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-x))


def theta_to_weights(z: Array) -> Array:
    """Map latent ``z = (z1, z2)`` to Bell weights: ``F = 0.5 + 0.5 sigmoid(z1)``, ``w = sigmoid(z2)``."""
    return bell_weights(0.5 + 0.5 * sigmoid(z[..., 0]), sigmoid(z[..., 1]))


@dataclass(frozen=True)
class DriftModel:
    """Parameters of the latent drift process (per cycle)."""

    mu: tuple[float, float] = (0.0, 1.0)
    phi: float = 0.95
    sigma: float = 0.20  # volatility of the fidelity coordinate z1
    sigma2: float = 0.05  # volatility of the dephasing-share coordinate z2
    jump_prob: float = 0.02
    jump_size: float = 0.8

    def step(self, z: Array, rng: np.random.Generator) -> Array:
        """Advance latent state(s) by one cycle; ``z`` has shape ``(..., 2)``."""
        mu = np.asarray(self.mu)
        eps = rng.standard_normal(z.shape) * np.array([self.sigma, self.sigma2])
        z_new = mu + self.phi * (z - mu) + eps
        jump = (rng.random(z.shape[:-1]) < self.jump_prob) * rng.choice([-1.0, 1.0], size=z.shape[:-1])
        z_new[..., 0] += jump * self.jump_size
        return z_new

    def stationary_sd(self) -> float:
        return self.sigma / np.sqrt(1 - self.phi**2)

    def trajectory(self, n_cycles: int, rng: np.random.Generator) -> Array:
        """Latent path ``(n_cycles, 2)`` started from the stationary law of the OU part."""
        z = np.asarray(self.mu) + rng.standard_normal(2) * np.array([self.sigma, self.sigma2]) / np.sqrt(
            1 - self.phi**2
        )
        out = np.empty((n_cycles, 2))
        for t in range(n_cycles):
            out[t] = z
            z = self.step(z, rng)
        return out
