"""Deterministic unit fixtures; these are not empirical assignment results."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize


@pytest.fixture(scope="module")
def mpt():
    path = Path(__file__).resolve().parents[1] / "mpt.py"
    if not path.is_file():
        from types import SimpleNamespace
        return SimpleNamespace()
    spec = importlib.util.spec_from_file_location("mpt_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def market():
    return (np.array([0.05, 0.11, 0.17]),
            np.array([[0.04, 0.008, 0.006],
                      [0.008, 0.09, 0.012],
                      [0.006, 0.012, 0.16]]))


def test_estimate_annualizes_sample_log_moments_without_mutation(mpt):
    prices = pd.DataFrame({"A": [100., 110., 99., 111.],
                           "B": [50., 49., 55., 54.]},
                          index=pd.date_range("2020-01-01", periods=4))
    before = prices.copy(deep=True)
    logs, mu, cov = mpt.estimate(prices, periods=12)
    expected = np.log(prices / prices.shift(1)).iloc[1:]
    pd.testing.assert_frame_equal(logs, expected)
    assert_allclose(mu, expected.mean().to_numpy() * 12)
    assert_allclose(cov, expected.cov().to_numpy() * 12)
    pd.testing.assert_frame_equal(prices, before)


@pytest.mark.parametrize("bad", [
    pd.DataFrame({"A": [1., np.nan, 2., 3.]}),
    pd.DataFrame({"A": [1., np.inf, 2., 3.]}),
    pd.DataFrame({"A": [1., 0., 2., 3.]}),
    pd.DataFrame({"A": [1., -2., 2., 3.]}),
    pd.DataFrame({"A": ["a", "b", "c"]}),
    pd.DataFrame({"A": [1., 2.]}),
    pd.DataFrame(),
    pd.DataFrame({"A": [1., 1., 1., 1.]}),
    pd.DataFrame([[1., 1.], [2., 2.], [3., 3.], [5., 5.]]),
    pd.DataFrame({"A": [1., 2., 3.]}, index=[0, 0, 1]),
    pd.DataFrame({"A": [1., 2., 3.]}, index=[2, 0, 1]),
    pd.DataFrame([[1., 2.], [2., 3.], [3., 5.]], columns=["A", "A"]),
])
def test_estimate_rejects_invalid_prices_without_filling(mpt, bad):
    with pytest.raises(ValueError):
        mpt.estimate(bad)


@pytest.mark.parametrize("periods", [0, -1, np.nan, np.inf, True])
def test_estimate_rejects_invalid_annualization(mpt, periods):
    with pytest.raises(ValueError):
        mpt.estimate(pd.DataFrame({"A": [1., 2., 3., 5.]}), periods)


@pytest.mark.parametrize("target", [None, 0.02, 0.12, 0.24])
def test_short_minimum_variance_matches_analytic_and_independent_scipy(mpt, market, target):
    mu, cov = market
    weights = mpt.min_variance(mu, cov, target=target)
    ones = np.ones(len(mu))
    basis = ones[:, None] if target is None else np.column_stack([ones, mu])
    rhs = np.array([1.]) if target is None else np.array([1., target])
    solved = np.linalg.solve(cov, basis)
    expected = solved @ np.linalg.solve(basis.T @ solved, rhs)
    oracle = minimize(lambda w: w @ cov @ w, ones / len(mu),
                      method="SLSQP", constraints={"type": "eq",
                      "fun": lambda w: basis.T @ w - rhs},
                      options={"ftol": 1e-13, "maxiter": 2000})
    assert oracle.success
    assert_allclose(weights, expected, atol=1e-10)
    assert_allclose(weights, oracle.x, atol=2e-6)
    assert_allclose(weights.sum(), 1., atol=1e-10)
    if target is not None:
        assert_allclose(weights @ mu, target, atol=1e-10)


@pytest.mark.parametrize("bad_mu,bad_cov", [
    ([0.1, 0.2], [[1., 2.], [2., 1.]]),
    ([0.1, 0.2], [[1., 1.], [1., 1.]]),
    ([0.1, 0.2], [[1., 0.1], [0.2, 1.]]),
    ([0.1, np.nan], np.eye(2)),
    ([0.1, 0.2], [[1., 0.], [0., np.inf]]),
    ([0.1, 0.2], np.eye(3)),
    ([[0.1, 0.2]], np.eye(2)),
    ([], np.empty((0, 0))),
])
def test_optimization_rejects_invalid_moments(mpt, bad_mu, bad_cov):
    with pytest.raises(ValueError):
        mpt.min_variance(bad_mu, bad_cov)


def test_constant_means_have_only_one_feasible_target(mpt):
    mu, cov = np.full(3, 0.1), np.diag([1., 2., 3.])
    expected = mpt.min_variance(mu, cov)
    assert_allclose(mpt.min_variance(mu, cov, target=0.1), expected)
    with pytest.raises(ValueError, match="target"):
        mpt.min_variance(mu, cov, target=0.2)
    with pytest.raises(ValueError, match="target"):
        mpt.min_variance(mu, cov, target=np.nan)


@pytest.mark.parametrize("target", [None, 0.05, 0.075, 0.12, 0.17])
def test_long_only_minimum_variance_is_feasible_and_dominates_candidates(mpt, market, target):
    mu, cov = market
    w = mpt.min_variance(mu, cov, long_only=True, target=target)
    assert_allclose(w.sum(), 1., atol=1e-9)
    assert w.min() >= -1e-10
    if target is None:
        candidates = np.random.default_rng(193).dirichlet(np.ones(3), size=10000)
    else:
        assert_allclose(w @ mu, target, atol=1e-9)
        # Enumerate the full feasible line segment independently of SLSQP.
        middle = np.linspace(0, 1, 10001)
        high = (target - mu[0] - middle * (mu[1] - mu[0])) / (mu[2] - mu[0])
        candidates = np.column_stack([1 - middle - high, middle, high])
        candidates = candidates[(candidates.min(axis=1) >= -1e-12)]
    variances = np.einsum("ij,jk,ik->i", candidates, cov, candidates)
    assert w @ cov @ w <= variances.min() + 1e-9
    short = mpt.min_variance(mu, cov, target=target)
    assert short @ cov @ short <= w @ cov @ w + 1e-9


def test_long_only_mvp_active_bound_and_duplicate_endpoint(mpt):
    mu, cov = np.array([.08, .16]), np.array([[.01, .018], [.018, .04]])
    assert mpt.min_variance(mu, cov).min() < 0
    assert_allclose(mpt.min_variance(mu, cov, True), [1., 0.], atol=1e-9)
    mu, cov = np.array([.05, .15, .15]), np.diag([.02, .04, .09])
    assert_allclose(mpt.min_variance(mu, cov, True, .15), [0., 9/13, 4/13], atol=1e-7)
    for target in [.01, .2]:
        with pytest.raises(ValueError, match="target"):
            mpt.min_variance(mu, cov, True, target)


def test_minimum_variance_rejects_failed_or_infeasible_solver(mpt, market, monkeypatch):
    from types import SimpleNamespace
    mu, cov = market
    for result in [SimpleNamespace(success=False, message="test failure", x=np.ones(3)/3),
                   SimpleNamespace(success=True, message="fake success", x=np.zeros(3)),
                   SimpleNamespace(success=True, message="fake success", x=np.array([2., -1., 0.])),
                   SimpleNamespace(success=True, message="fake success", x=np.full(3, np.nan))]:
        monkeypatch.setattr(mpt, "minimize", lambda *a, **k: result)
        with pytest.raises(RuntimeError, match="optimization") as error:
            mpt.min_variance(mu, cov, True)
        assert type(error.value) is RuntimeError


def test_short_tangency_matches_analytic_sharpe_bound(mpt, market):
    mu, cov = market
    excess = mu - .03
    expected = np.linalg.solve(cov, excess)
    expected /= expected.sum()
    w = mpt.tangency(mu, cov)
    assert_allclose(w, expected, atol=1e-10)
    assert_allclose(w.sum(), 1, atol=1e-10)
    sharpe = w @ excess / np.sqrt(w @ cov @ w)
    assert_allclose(sharpe, np.sqrt(excess @ np.linalg.solve(cov, excess)))


@pytest.mark.parametrize("mu", [[.01, .02], [.04, .02], [.03, .03]])
def test_short_tangency_rejects_nonpositive_normalization(mpt, mu):
    with pytest.raises(ValueError, match="positive-excess tangency"):
        mpt.tangency(mu, np.eye(2))


@pytest.mark.parametrize("mu", [[.05, .11, .17], [-.08, .04, .15], [.029, .03, .0301]])
def test_long_tangency_dominates_random_feasible_portfolios(mpt, market, mu):
    _, cov = market
    mu = np.array(mu)
    w = mpt.tangency(mu, cov, long_only=True)
    assert_allclose(w.sum(), 1, atol=1e-9)
    assert w.min() >= -1e-10
    excess = mu - .03
    assert w @ excess > 0
    candidates = np.vstack([np.eye(3), np.random.default_rng(23).dirichlet(np.ones(3), size=50000)])
    risk = np.sqrt(np.einsum("ij,jk,ik->i", candidates, cov, candidates))
    assert w @ excess / np.sqrt(w @ cov @ w) >= np.max(candidates @ excess / risk) - 1e-8
    # Separate nonconvex oracle started from equal weights, not the QP output.
    oracle = minimize(lambda v: -(v @ excess) / np.sqrt(v @ cov @ v), np.ones(3)/3,
                      bounds=[(0, 1)] * 3, method="SLSQP",
                      constraints={"type": "eq", "fun": lambda v: v.sum() - 1},
                      options={"ftol": 1e-13, "maxiter": 2000})
    assert oracle.success
    assert_allclose(w, oracle.x, atol=2e-5)


def test_long_tangency_rejects_no_positive_excess(mpt):
    with pytest.raises(ValueError, match="positive excess"):
        mpt.tangency([.01, .03], np.eye(2), long_only=True)


def test_tangency_rejects_solver_failure(mpt, market, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(mpt, "minimize", lambda *a, **k: SimpleNamespace(
        success=False, message="forced failure", x=np.ones(3)))
    with pytest.raises(RuntimeError, match="optimization"):
        mpt.tangency(*market, long_only=True)


def test_portfolio_stats_use_assignment_log_moment_convention(mpt, market):
    mu, cov = market
    w = np.array([1.2, -.5, .3])
    result = mpt.portfolio_stats(w, mu, cov)
    assert set(result) == {"return", "volatility", "sharpe", "gross_exposure"}
    assert_allclose(result["return"], w @ mu)
    assert_allclose(result["volatility"], np.sqrt(w @ cov @ w))
    assert_allclose(result["sharpe"], (w @ mu - .03) / np.sqrt(w @ cov @ w))
    assert_allclose(result["gross_exposure"], 2.)


@pytest.mark.parametrize("w", [[.2, .2, .2], [1, 0], [np.nan, 0, 1], [[1, 0, 0]]])
def test_portfolio_stats_reject_invalid_weights(mpt, market, w):
    with pytest.raises(ValueError, match="weights"):
        mpt.portfolio_stats(w, *market)


@pytest.mark.parametrize("rf", [np.nan, np.inf, -1, -2])
def test_risk_free_must_be_valid_for_all_apis(mpt, market, rf):
    with pytest.raises(ValueError, match="rf"):
        mpt.portfolio_stats([1, 0, 0], *market, rf=rf)
    with pytest.raises(ValueError, match="rf"):
        mpt.tangency(*market, rf=rf)


@pytest.mark.parametrize("long_only", [False, True])
def test_frontier_is_upper_branch_starting_at_mvp(mpt, market, long_only):
    mu, cov = market
    result = mpt.frontier(mu, cov, long_only=long_only, points=31)
    assert list(result.columns) == ["return", "volatility"]
    assert len(result) == 31
    mvp = mpt.min_variance(mu, cov, long_only)
    assert_allclose(result.iloc[0], [mvp @ mu, np.sqrt(mvp @ cov @ mvp)], atol=1e-8)
    assert_allclose(result["return"].iloc[-1], mu.max())
    assert np.all(np.diff(result["return"]) >= -1e-10)
    assert np.all(np.diff(result["volatility"]) >= -1e-8)
    for row in result.itertuples(index=False, name=None):
        w = mpt.min_variance(mu, cov, long_only, row[0])
        assert_allclose(row[1], np.sqrt(w @ cov @ w), atol=1e-8)
        unrestricted = mpt.min_variance(mu, cov, target=row[0])
        assert np.sqrt(unrestricted @ cov @ unrestricted) <= row[1] + 1e-8


def test_frontier_max_return_controls_range_and_short_has_no_asset_cap(mpt, market):
    result = mpt.frontier(*market, points=17, max_return=.3)
    assert_allclose(result["return"].iloc[-1], .3)
    assert len(result) == 17
    with pytest.raises(ValueError, match="max_return"):
        mpt.frontier(*market, long_only=True, max_return=.3)
    with pytest.raises(ValueError, match="max_return"):
        mpt.frontier(*market, max_return=.01)
    with pytest.raises(ValueError, match="max_return"):
        mpt.frontier(*market, max_return=np.nan)


@pytest.mark.parametrize("points", [0, 1, -1, 2.5, True])
def test_frontier_rejects_invalid_point_count(mpt, market, points):
    with pytest.raises(ValueError, match="points"):
        mpt.frontier(*market, points=points)


@pytest.mark.parametrize("long_only", [False, True])
def test_constant_mean_frontier_is_single_location(mpt, long_only):
    result = mpt.frontier(np.full(3, .1), np.eye(3), long_only, points=4)
    assert_allclose(result["return"], .1)
    assert_allclose(result["volatility"], np.sqrt(1/3))


@pytest.mark.parametrize("weights", [np.array([.6, .4]), np.array([1.5, -.5])])
def test_backtest_exact_simple_returns_with_frozen_daily_rebalancing(mpt, weights):
    simple = pd.DataFrame({"A": [-.2, .1, .03, -.02], "B": [.1, -.05, -.01, .04]},
                          index=pd.date_range("2022-01-01", periods=4))
    logs, original_weights = np.log1p(simple), weights.copy()
    daily, metrics = mpt.backtest(logs, weights, periods=12)
    expected = simple.to_numpy() @ weights
    assert isinstance(daily, pd.Series)
    assert daily.index.equals(logs.index)
    assert_allclose(daily, expected, atol=1e-15)
    assert not np.allclose(daily, np.expm1(logs.to_numpy() @ weights))
    assert_allclose(weights, original_weights)
    wealth = np.r_[1., np.cumprod(1 + expected)]
    volatility = expected.std(ddof=1) * np.sqrt(12)
    assert set(metrics) == {"annual_return", "volatility", "sharpe", "cagr", "total_return", "max_drawdown"}
    assert_allclose(metrics["annual_return"], expected.mean() * 12)
    assert_allclose(metrics["volatility"], volatility)
    assert_allclose(metrics["sharpe"], (expected.mean() - ((1.03)**(1/12) - 1)) * 12 / volatility)
    assert_allclose(metrics["cagr"], wealth[-1] ** (12 / len(expected)) - 1)
    assert_allclose(metrics["total_return"], wealth[-1] - 1)
    assert_allclose(metrics["max_drawdown"], (wealth / np.maximum.accumulate(wealth) - 1).min())


def test_backtest_counts_first_day_loss_in_drawdown(mpt):
    daily, metrics = mpt.backtest(pd.DataFrame({"A": np.log1p([-.2, .1])}), [1.])
    assert_allclose(daily, [-.2, .1])
    assert_allclose(metrics["max_drawdown"], -.2)


def test_backtest_zero_volatility_sharpe_is_undefined(mpt):
    _, metrics = mpt.backtest(pd.DataFrame({"A": [0., 0., 0.]}), [1.])
    assert metrics["volatility"] == 0
    assert np.isnan(metrics["sharpe"])
    assert metrics["total_return"] == 0
    assert metrics["max_drawdown"] == 0


@pytest.mark.parametrize("loss", [-.5, -.6])
def test_backtest_rejects_bankruptcy(mpt, loss):
    logs = np.log1p(pd.DataFrame({"A": [loss, .1], "B": [0., 0.]}))
    with pytest.raises(ValueError, match="bankrupt"):
        mpt.backtest(logs, [2., -1.])


@pytest.mark.parametrize("bad", [
    pd.DataFrame({"A": [0., np.nan]}), pd.DataFrame({"A": [0., np.inf]}),
    pd.DataFrame({"A": [0.]}), pd.DataFrame(),
    pd.DataFrame({"A": [0., .1]}, index=[1, 0]),
    pd.DataFrame({"A": [0., .1]}, index=[1, 1]),
])
def test_backtest_rejects_invalid_return_data(mpt, bad):
    with pytest.raises(ValueError):
        mpt.backtest(bad, [1.])


def test_backtest_rejects_bad_parameters_and_nonfinite_wealth(mpt):
    logs = pd.DataFrame({"A": [0., .01], "B": [.02, -.01]})
    for w in [[.2, .2], [1], [np.nan, 1]]:
        with pytest.raises(ValueError, match="weights"):
            mpt.backtest(logs, w)
    for rf in [-1, np.nan]:
        with pytest.raises(ValueError, match="rf"):
            mpt.backtest(logs, [.5, .5], rf=rf)
    with pytest.raises(ValueError, match="periods"):
        mpt.backtest(logs, [.5, .5], periods=0)
    with pytest.raises(ValueError, match="finite"):
        mpt.backtest(pd.DataFrame({"A": [1000., 0.]}), [1.])


def test_frozen_out_of_sample_weights_and_prefix_do_not_depend_on_future(mpt):
    rng = np.random.default_rng(391)
    train = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(.001, .01, (160, 3)), axis=0)))
    _, mu, cov = mpt.estimate(train)
    learned = mpt.min_variance(mu, cov, long_only=True)
    holdout = pd.DataFrame(rng.normal(0, .01, (25, 3)))
    changed = holdout.copy()
    changed.iloc[12:] = .05
    for weights in [learned, np.full(3, 1/3)]:
        original = weights.copy()
        baseline, _ = mpt.backtest(holdout, weights)
        perturbed, _ = mpt.backtest(changed, weights)
        prefix, _ = mpt.backtest(holdout.iloc[:12], weights)
        assert_allclose(baseline.iloc[:12], perturbed.iloc[:12])
        assert_allclose(baseline.iloc[:12], prefix)
        assert_allclose(baseline, np.expm1(holdout).to_numpy() @ weights)
        assert_allclose(weights, original)


@pytest.mark.parametrize("bad", [np.array([.1 + .2j, .2]), np.array([True, False])])
def test_optimization_rejects_nonreal_numeric_inputs(mpt, bad):
    with pytest.raises(ValueError, match="real"):
        mpt.min_variance(bad, np.eye(2))
    with pytest.raises(ValueError, match="real"):
        mpt.portfolio_stats(bad, [.1, .2], np.eye(2))
    with pytest.raises(ValueError, match="real"):
        mpt.min_variance([.1, .2], np.diag(bad))


@pytest.mark.parametrize("bad", [True, 1 + 1j, None, [12]])
def test_scalar_parameters_reject_invalid_types(mpt, market, bad):
    with pytest.raises(ValueError, match="rf"):
        mpt.tangency(*market, rf=bad)
    with pytest.raises(ValueError, match="periods"):
        mpt.estimate(pd.DataFrame({"A": [1., 2., 3.]}), periods=bad)


@pytest.mark.parametrize("seed", range(8))
def test_random_markets_match_independent_optimization(mpt, seed):
    rng = np.random.default_rng(seed)
    n = 5
    factor = rng.normal(size=(n, n))
    cov = (factor @ factor.T + np.eye(n)) / 100
    mu = rng.uniform(.04, .25, n)
    target = float(np.median(mu))
    for long_only in [False, True]:
        w = mpt.min_variance(mu, cov, long_only, target)
        oracle = minimize(lambda v: v @ cov @ v, np.ones(n)/n,
                          method="SLSQP", bounds=[(0, 1)]*n if long_only else None,
                          constraints={"type": "eq", "fun": lambda v: np.array([v.sum()-1, v @ mu-target])},
                          options={"ftol": 1e-13, "maxiter": 2000})
        assert oracle.success
        assert_allclose(w, oracle.x, atol=3e-6)
        assert_allclose(w.sum(), 1., atol=1e-9)
        assert_allclose(w @ mu, target, atol=1e-9)
        permutation = rng.permutation(n)
        permuted = mpt.min_variance(mu[permutation], cov[np.ix_(permutation, permutation)], long_only, target)
        assert_allclose(permuted, w[permutation], atol=3e-6)
        scaled = mpt.min_variance(mu, cov * 1e-8, long_only, target)
        assert_allclose(scaled, w, atol=3e-6)
        if long_only:
            tangency = mpt.tangency(mu, cov, long_only=True)
            candidates = rng.dirichlet(np.ones(n), size=5000)
            scores = candidates @ (mu-.03) / np.sqrt(np.einsum("ij,jk,ik->i", candidates, cov, candidates))
            assert tangency @ (mu-.03) / np.sqrt(tangency @ cov @ tangency) >= scores.max() - 1e-8


def test_solver_success_does_not_override_target_feasibility(mpt, market, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(mpt, "minimize", lambda *a, **k: SimpleNamespace(
        success=True, message="wrong target", x=np.full(3, 1/3)))
    with pytest.raises(RuntimeError, match="optimization"):
        mpt.min_variance(*market, long_only=True, target=.15)


def test_backtest_default_daily_risk_free_and_nonzero_constant_return(mpt):
    daily_rf = (1.03)**(1/252) - 1
    simple = np.array([.01, -.005, .002])
    _, result = mpt.backtest(pd.DataFrame({"A": np.log1p(simple)}), [1.])
    expected_sharpe = (simple.mean() - daily_rf) / simple.std(ddof=1) * np.sqrt(252)
    assert_allclose(result["sharpe"], expected_sharpe)
    _, flat = mpt.backtest(pd.DataFrame({"A": np.full(3, np.log1p(.001))}), [1.])
    assert flat["volatility"] == 0
    assert np.isnan(flat["sharpe"])
