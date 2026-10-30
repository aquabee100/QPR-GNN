import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qprgnn.queueing import mm1k_sojourn


def _numeric(lam, mu, K):
    p = (lam / mu) ** np.arange(K + 1); p /= p.sum()
    L = (np.arange(K + 1) * p).sum()
    return L / (lam * (1 - p[-1])), p[-1]


def test_closed_form_matches_stationary_distribution():
    for rho in (0.2, 0.7, 0.95, 1.0, 1.3, 3.0):
        for K in (1, 4, 32, 200):
            T, PK = mm1k_sojourn(rho * 1000.0, 1000.0, K, return_loss=True)
            Tn, PKn = _numeric(rho * 1000.0, 1000.0, K)
            assert np.isclose(T, Tn, rtol=1e-6) and np.isclose(PK, PKn, rtol=1e-6, atol=1e-12), (rho, K)


def test_large_buffer_tends_to_mm1_and_overload_stays_finite():
    lam, mu = 600.0, 1000.0
    assert np.isclose(mm1k_sojourn(lam, mu, 10_000), 1 / (mu - lam), rtol=1e-6)
    T = mm1k_sojourn(5 * mu, mu, 50)
    assert np.isfinite(T) and T < 51 / mu        # at most K+1 service times


if __name__ == "__main__":
    test_closed_form_matches_stationary_distribution(); test_large_buffer_tends_to_mm1_and_overload_stays_finite()
    print("queueing tests passed")
