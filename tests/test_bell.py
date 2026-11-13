import numpy as np
import pytest

from linktwin.bell import (ACTIONS, action_values, bbpssw_round, bell_weights, correlators, dejmps_round,
                           run_action, weights_from_correlators)
from linktwin.density import distil_once


@pytest.mark.parametrize("f,w", [(0.7, 0.3), (0.8, 0.9), (0.95, 0.5)])
def test_dejmps_matches_density_matrix(f, w):
    p = bell_weights(f, w)
    sim, s_sim = distil_once(p, "dejmps")
    out, s = dejmps_round(p)
    assert np.allclose(sim, out, atol=1e-10)
    assert s_sim == pytest.approx(float(s))


@pytest.mark.parametrize("f,w", [(0.7, 0.3), (0.85, 0.9)])
def test_bbpssw_fidelity_and_success_match_density_matrix(f, w):
    p = bell_weights(f, w)
    sim, s_sim = distil_once(p, "bbpssw")
    out, s = bbpssw_round(p)
    assert sim[0] == pytest.approx(float(out[0]))
    assert s_sim == pytest.approx(float(s))


def test_correlator_round_trip_and_normalisation():
    p = bell_weights(np.array([0.6, 0.8, 0.95]), np.array([0.2, 0.7, 0.9]))
    assert np.allclose(p.sum(-1), 1.0)
    assert np.allclose(weights_from_correlators(correlators(p)), p)


def test_distillation_improves_dephased_state_and_noise_hurts():
    p = bell_weights(0.8, 0.9)
    clean, y = run_action(p, ACTIONS[4], 0.0)
    noisy, _ = run_action(p, ACTIONS[4], 0.05)
    assert clean[0] > p[0] > 0.5 and noisy[0] < clean[0] and 0 < y < 1


def test_action_values_zero_when_threshold_unreachable():
    p = bell_weights(0.55, 0.5)
    assert np.all(action_values(p, 0.99, 0.02) == 0)
