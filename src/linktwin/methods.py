"""Link-management methods: oracle, fixed policies, static model, periodic shadow, digital twin.

Every method receives the same latent trajectory and returns, per cycle, the action it used and the
number of raw pairs it spent on monitoring; :func:`score` turns that into delivered value.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bell import ACTIONS, Array, action_values
from .drift import DriftModel, theta_to_weights
from .bell import CORR_MATRIX
from .monitor import log_likelihood, measure, plug_in_weights, readout_attenuation

A_RAW = 0
A_DEJMPS2 = [a.name for a in ACTIONS].index("dejmpsx2")


@dataclass(frozen=True)
class LinkConfig:
    """Operating point of one simulated link."""

    n_pairs: int = 96  # raw pairs generated per cycle
    f_req: float = 0.90  # application fidelity threshold
    gate_noise: float = 0.02  # local depolarising noise per distillation input
    readout_error: float = 0.0  # true per-qubit readout flip probability
    assumed_readout_error: float = 0.0  # what the estimators believe
    n_cycles: int = 150


@dataclass
class Episode:
    """Ground truth for one trajectory."""

    p_true: Array  # (T, 4)
    v_true: Array  # (T, n_actions) value per raw pair


def make_episode(drift: DriftModel, cfg: LinkConfig, rng: np.random.Generator) -> Episode:
    z = drift.trajectory(cfg.n_cycles, rng)
    p = theta_to_weights(z)
    return Episode(p, action_values(p, cfg.f_req, cfg.gate_noise))


@dataclass
class Run:
    actions: Array  # (T,) chosen action index
    monitor_pairs: Array  # (T,) raw pairs consumed by monitoring


def score(ep: Episode, run: Run, cfg: LinkConfig) -> dict[str, float | Array]:
    """Delivered value, oracle value and derived efficiency of a run."""
    t = np.arange(len(run.actions))
    delivered = np.maximum(cfg.n_pairs - run.monitor_pairs, 0) * ep.v_true[t, run.actions]
    oracle = cfg.n_pairs * ep.v_true.max(axis=1)
    eff = delivered.sum() / max(oracle.sum(), 1e-12)
    return {
        "efficiency": float(eff),
        "mean_monitor_pairs": float(run.monitor_pairs.mean()),
        "outage_rate": float(np.mean((delivered <= 0) & (oracle > 0))),
        "delivered": delivered,
        "oracle": oracle,
    }


# ----------------------------------------------------------------------------- simple methods
def run_oracle(ep: Episode, cfg: LinkConfig, rng: np.random.Generator) -> Run:
    return Run(ep.v_true.argmax(axis=1), np.zeros(len(ep.v_true)))


def run_fixed(ep: Episode, cfg: LinkConfig, rng: np.random.Generator, action: int) -> Run:
    n = len(ep.v_true)
    return Run(np.full(n, action), np.zeros(n))


def _best_action(p_hat: Array, cfg: LinkConfig) -> int:
    return int(np.argmax(action_values(p_hat, cfg.f_req, cfg.gate_noise)))


def run_shadow(ep: Episode, cfg: LinkConfig, rng: np.random.Generator, period: int, m_cal: int) -> Run:
    """Digital shadow: recalibrate every ``period`` cycles with ``m_cal`` pairs, keep the plug-in action.

    ``period`` larger than the horizon gives the static digital model (one calibration at t = 0).
    """
    n = len(ep.p_true)
    acts = np.zeros(n, dtype=int)
    mon = np.zeros(n)
    action = A_RAW
    n_b = max(m_cal // 3, 1)
    for t in range(n):
        if t % period == 0:
            counts = measure(ep.p_true[t], n_b, cfg.readout_error, rng)
            action = _best_action(plug_in_weights(counts, n_b, cfg.assumed_readout_error), cfg)
            mon[t] = 3 * n_b
        acts[t] = action
    return Run(acts, mon)


SHADOW_GRID = [(p, m) for p in (1, 2, 4, 8, 16, 32) for m in (12, 24, 48, 96)]


# ----------------------------------------------------------------------------------- the twin
@dataclass(frozen=True)
class TwinConfig:
    """Hyper-parameters of the digital twin (filter + sequential monitoring)."""

    n_particles: int = 400
    batch_pairs: int = 6  # raw pairs per monitoring batch (2 per basis)
    m_min: int = 6  # minimum monitoring per cycle (guard against stopping on the prior alone)
    m_max: int = 60
    lookahead: tuple[int, ...] = (6, 12, 24)  # monitoring sizes (raw pairs) assessed by pre-posterior analysis
    n_preposterior: int = 48  # Monte Carlo draws per pre-posterior evaluation
    use_memory: bool = True  # carry the posterior across cycles (False = flat prior every cycle)
    adaptive: bool = True  # sequential stopping (False = fixed m_min pairs every cycle)
    drift_scale: float = 1.0  # assumed / true volatility ratio (model-mismatch knob)
    jumps_in_model: bool = True
    extra: dict = field(default_factory=dict)


def _net_gain(
    w: Array, v: Array, p_part: Array, cfg: LinkConfig, tw: TwinConfig, used: int, rng: np.random.Generator
) -> float:
    """Best expected net gain (value units) of monitoring ``s`` more pairs, over ``s`` in the look-ahead set.

    Pre-posterior analysis: draw a true state from the current posterior, simulate the counts,
    update, take the Bayes action, and score it on the drawn state.  The gain of monitoring is the
    expected value of the improved decision on the remaining pairs minus the value of the pairs burnt.
    """
    remaining = cfg.n_pairs - used
    now = float((w @ v).max()) * remaining
    idx = rng.choice(len(w), size=tw.n_preposterior, p=w)
    c_true = readout_attenuation(cfg.readout_error) * (p_part[idx] @ CORR_MATRIX.T)
    pi_true = np.clip((1 + c_true) / 2, 0, 1)
    c_model = readout_attenuation(cfg.assumed_readout_error) * (p_part @ CORR_MATRIX.T)
    pi = np.clip((1 + c_model) / 2, 1e-9, 1 - 1e-9)
    lp, lq = np.log(pi), np.log1p(-pi)
    best = -np.inf
    for s in tw.lookahead:
        if s > remaining:
            continue
        n_b = s // 3
        counts = rng.binomial(n_b, pi_true)
        ll = counts @ lp.T + (n_b - counts) @ lq.T  # (K, M)
        post = w * np.exp(ll - ll.max(axis=1, keepdims=True))
        post /= post.sum(axis=1, keepdims=True)
        gain = (post @ v).max(axis=1)  # posterior-expected value of the Bayes action
        # the Bayes action is scored on the drawn truth, not on its own posterior mean
        chosen = (post @ v).argmax(axis=1)
        realised = v[idx, chosen]
        best = max(best, float(realised.mean()) * (remaining - s) - now)
        del gain
    return best


def run_twin(
    ep: Episode,
    cfg: LinkConfig,
    rng: np.random.Generator,
    drift: DriftModel,
    tw: TwinConfig = TwinConfig(),
) -> Run:
    """Digital twin: particle filter over the latent link state, Bayes-optimal action, adaptive monitoring."""
    n = len(ep.p_true)
    m_part = tw.n_particles
    model = DriftModel(
        mu=drift.mu,
        phi=drift.phi,
        sigma=drift.sigma * tw.drift_scale,
        sigma2=drift.sigma2 * tw.drift_scale,
        jump_prob=drift.jump_prob if tw.jumps_in_model else 0.0,
        jump_size=drift.jump_size,
    )
    sd = np.array([model.sigma, model.sigma2]) / np.sqrt(1 - model.phi**2)
    prior = lambda: np.asarray(model.mu) + rng.standard_normal((m_part, 2)) * sd  # noqa: E731
    z = prior()
    acts = np.zeros(n, dtype=int)
    mon = np.zeros(n)
    nb_batch = tw.batch_pairs // 3
    for t in range(n):
        if t > 0:
            z = model.step(z, rng) if tw.use_memory else prior()
        p_part = theta_to_weights(z)
        v = action_values(p_part, cfg.f_req, cfg.gate_noise)  # (M, A)
        vstar = v.max(axis=1)
        logw = np.zeros(m_part)
        counts = np.zeros(3)
        n_per = 0
        used = 0
        while True:
            w = np.exp(logw - logw.max())
            w /= w.sum()
            ev = w @ v
            a_star = int(np.argmax(ev))
            if used >= tw.m_min:
                if not tw.adaptive or used >= tw.m_max:
                    break
                if _net_gain(w, v, p_part, cfg, tw, used, rng) <= 0.0:
                    break
            c = measure(ep.p_true[t], nb_batch, cfg.readout_error, rng)
            logw += log_likelihood(p_part, c, nb_batch, cfg.assumed_readout_error)
            used += 3 * nb_batch
        acts[t], mon[t] = a_star, used
        # resample every cycle so that the particles represent the posterior
        if True:
            idx = rng.choice(m_part, size=m_part, p=w)
            z = z[idx] + rng.standard_normal((m_part, 2)) * 0.02
    return Run(acts, mon)
