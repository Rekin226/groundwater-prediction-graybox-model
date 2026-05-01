import numpy as np
from subsidence.sub_metrics import kge, kge_components, kge_on_rate


def test_kge_perfect_match_is_one():
    y = np.linspace(0, 1, 100)
    assert abs(kge(y, y) - 1.0) < 1e-9


def test_kge_components_returns_four():
    y_obs = np.array([1.0, 2, 3, 4, 5], float)
    y_sim = y_obs + 0.1
    out = kge_components(y_obs, y_sim)
    assert set(out) == {"kge", "r", "alpha", "beta", "bias"}


def test_kge_on_rate_uses_first_difference():
    # Use non-constant rates so r is well-defined (std > 0 in both series).
    # np.diff(np.cumsum(np.linspace(1, 2, 10))) == np.linspace(1, 2, 10)[1:]
    # which is a monotonically increasing sequence — std > 0.
    # The original test used np.cumsum(np.ones(10)), whose diff is all-ones
    # (constant, sd=0), making r undefined (NaN) and kge NaN.
    y_obs = np.cumsum(np.linspace(1.0, 2.0, 10))
    y_sim = np.cumsum(np.linspace(1.0, 2.1, 10))
    k_rate = kge_on_rate(y_obs, y_sim)
    # rate is non-constant in both → r finite; alpha ≈ 1; beta slightly off
    assert k_rate > 0.5
