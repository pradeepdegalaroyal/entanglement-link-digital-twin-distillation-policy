import numpy as np

from linktwin.drift import DriftModel
from linktwin.methods import (A_DEJMPS2, LinkConfig, TwinConfig, make_episode, run_fixed, run_oracle, run_shadow,
                              run_twin, score)
from linktwin.monitor import measure, plug_in_weights
from linktwin.bell import bell_weights


def test_oracle_efficiency_is_one_and_monitoring_costs():
    cfg = LinkConfig(n_cycles=20)
    ep = make_episode(DriftModel(), cfg, np.random.default_rng(0))
    assert score(ep, run_oracle(ep, cfg, np.random.default_rng(1)), cfg)["efficiency"] == 1.0
    s = score(ep, run_shadow(ep, cfg, np.random.default_rng(1), 1, 48), cfg)
    assert s["efficiency"] < 1.0 and s["mean_monitor_pairs"] == 48


def test_plug_in_estimate_converges():
    p = bell_weights(0.8, 0.7)
    rng = np.random.default_rng(0)
    n = 200000
    est = plug_in_weights(measure(p, n, 0.0, rng), n)
    assert np.allclose(est, p, atol=5e-3)


def test_twin_runs_and_respects_budget():
    cfg = LinkConfig(n_cycles=12)
    drift = DriftModel()
    ep = make_episode(drift, cfg, np.random.default_rng(3))
    tw = TwinConfig(n_particles=100, n_preposterior=8, m_max=24)
    run = run_twin(ep, cfg, np.random.default_rng(4), drift, tw)
    assert run.monitor_pairs.min() >= tw.m_min and run.monitor_pairs.max() <= tw.m_max + tw.batch_pairs
    assert 0.0 <= score(ep, run, cfg)["efficiency"] <= 1.0
    fixed = score(ep, run_fixed(ep, cfg, np.random.default_rng(0), A_DEJMPS2), cfg)
    assert fixed["mean_monitor_pairs"] == 0
