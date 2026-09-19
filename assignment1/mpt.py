"""Log-moment MPT estimation and exact simple-return portfolio evaluation.

This module is self-contained so its source can be embedded in the submission
notebook. Annual expected log returns are an MPT approximation, not exact
portfolio growth rates. Optimization subtracts the assignment's annual rf
convention directly; realized performance instead uses exact simple returns.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _real_array(value, name):
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real numeric values")
    return array.astype(float)


def _scalar(value, name):
    array = _real_array(value, name)
    if array.ndim != 0 or not np.isfinite(array):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(array)


def _periods(periods):
    periods = _scalar(periods, "periods")
    if not np.isfinite(periods) or periods <= 0:
        raise ValueError("periods must be a positive finite number")
    return periods


def _frame(data, name, min_rows):
    if not isinstance(data, pd.DataFrame) or data.shape[1] == 0 or len(data) < min_rows:
        raise ValueError(f"{name} must be a nonempty DataFrame with at least {min_rows} rows")
    if not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be unique and increasing")
    if not data.columns.is_unique:
        raise ValueError(f"{name} columns must be unique")
    if any(not pd.api.types.is_numeric_dtype(dtype) or
           pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_complex_dtype(dtype)
           for dtype in data.dtypes):
        raise ValueError(f"{name} must contain real numeric values")
    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite and complete; no filling is performed")
    return values


def _moments(mu, cov):
    mu, cov = _real_array(mu, "mu"), _real_array(cov, "cov")
    if mu.ndim != 1 or mu.size == 0 or not np.isfinite(mu).all():
        raise ValueError("mu must be a nonempty finite vector")
    if cov.shape != (mu.size, mu.size) or not np.isfinite(cov).all():
        raise ValueError("cov must be a finite square matrix matching mu")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-14):
        raise ValueError("cov must be symmetric")
    cov = (cov + cov.T) / 2
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues[0] <= np.finfo(float).eps * mu.size * max(eigenvalues[-1], 0):
        raise ValueError("cov must be numerically positive definite")
    return mu, cov


def estimate(prices, periods=252):
    """Return daily log returns, annual log means, and sample covariance.

    Reject missing/nonpositive prices, unordered or duplicate observations,
    duplicate assets, and singular sample covariance. Never forward fill.
    """
    periods = _periods(periods)
    values = _frame(prices, "prices", min_rows=3)
    if (values <= 0).any():
        raise ValueError("prices must be strictly positive")
    log_returns = np.log(prices.astype(float)).diff().iloc[1:]
    mu, cov = _moments(log_returns.mean().to_numpy() * periods,
                       log_returns.cov().to_numpy() * periods)
    return log_returns, mu, cov


def _check_weights(weights, n, long_only=False):
    weights = _real_array(weights, "weights")
    if weights.shape != (n,) or not np.isfinite(weights).all():
        raise ValueError("weights must be a finite vector matching the assets")
    if abs(weights.sum() - 1) > 1e-8:
        raise ValueError("weights must sum to one; implicit normalization is forbidden")
    if long_only and weights.min() < -1e-9:
        raise ValueError("long-only weights cannot be negative")
    return weights


def _quadratic_program(cov, basis, rhs, start):
    """Convex nonnegative QP with checked solver status and feasibility."""
    scaled_cov = cov / np.max(np.diag(cov))
    result = minimize(lambda w: w @ scaled_cov @ w, start,
                      jac=lambda w: 2 * scaled_cov @ w,
                      method="SLSQP", bounds=[(0., None)] * len(start),
                      constraints={"type": "eq", "fun": lambda w: basis.T @ w - rhs,
                                   "jac": lambda w: basis.T},
                      options={"ftol": 1e-13, "maxiter": 2000})
    weights = np.asarray(result.x, dtype=float)
    if (not result.success or weights.shape != start.shape or
            not np.isfinite(weights).all() or weights.min() < -1e-9 or
            np.max(np.abs(basis.T @ weights - rhs)) > 1e-8):
        raise RuntimeError(f"optimization failed or infeasible: {result.message}")
    return weights


def min_variance(mu, cov, long_only=False, target=None):
    """Fully invested minimum variance weights, optionally at target log return.

    Short-allowed solutions use equality-constrained linear algebra. A target
    selects either branch of the minimum-variance locus (frontier selects the
    efficient upper branch separately).
    """
    mu, cov = _moments(mu, cov)
    if target is not None:
        target = float(target)
        if not np.isfinite(target):
            raise ValueError("target must be finite")
        if np.ptp(mu) == 0:
            if abs(target - mu[0]) > 1e-12:
                raise ValueError("target is infeasible when all means are equal")
            target = None
    ones = np.ones(mu.size)
    basis, rhs = ones[:, None], np.array([1.])
    if target is not None:
        # Centering/scaling prevents avoidable cancellation in the 2x2 solve.
        center, scale = mu.mean(), np.ptp(mu)
        basis = np.column_stack([ones, (mu - center) / scale])
        rhs = np.array([1., (target - center) / scale])
    if long_only:
        start = ones / mu.size
        if target is not None:
            lo, hi = int(np.argmin(mu)), int(np.argmax(mu))
            if target < mu[lo] or target > mu[hi]:
                raise ValueError("target is outside the long-only feasible return range")
            if target == mu[lo] or target == mu[hi]:
                # At endpoints, optimize among all assets tied at that return.
                eligible = mu == target
                weights = np.zeros(mu.size)
                weights[eligible] = min_variance(mu[eligible], cov[np.ix_(eligible, eligible)], True)
                return weights
            fraction = (target - mu[lo]) / (mu[hi] - mu[lo])
            start = np.zeros(mu.size)
            start[lo], start[hi] = 1 - fraction, fraction
        weights = _quadratic_program(cov, basis, rhs, start)
    else:
        solved = np.linalg.solve(cov, basis)
        weights = solved @ np.linalg.solve(basis.T @ solved, rhs)
    _check_weights(weights, mu.size, long_only)
    if target is not None and abs(weights @ mu - target) > 1e-8:
        raise RuntimeError("minimum-variance target constraint was not satisfied")
    return weights


def _risk_free(rf):
    rf = _scalar(rf, "rf")
    if not np.isfinite(rf) or rf <= -1:
        raise ValueError("rf must be finite and greater than -1")
    return rf


def tangency(mu, cov, rf=.03, long_only=False):
    """Maximum positive-excess MPT Sharpe weights at annual risk-free rate rf.

    For shorts, reject nonpositive normalization: no finite conventional
    positive-excess tangency portfolio exists. For long-only weights, solve
    the convex transformed problem min z'cov z subject to (mu-rf)'z=1, z>=0,
    then normalize z to unit budget. No nonconvex ratio optimization is used.
    """
    mu, cov = _moments(mu, cov)
    excess = mu - _risk_free(rf)
    if long_only:
        best = int(np.argmax(excess))
        if excess[best] <= 0:
            raise ValueError("no long-only portfolio has positive excess return")
        # Scale z by max(excess) to keep the equivalent equality well-scaled.
        # If y solves this QP, z=y/max(excess) satisfies excess'z=1.
        scaled_excess = excess / excess[best]
        start = np.zeros(mu.size)
        start[best] = 1.
        direction = _quadratic_program(cov, scaled_excess[:, None], np.array([1.]), start)
    else:
        direction = np.linalg.solve(cov, excess)
        denominator = direction.sum()
        tolerance = np.finfo(float).eps * mu.size * np.abs(direction).sum()
        if denominator <= tolerance:
            raise ValueError("no finite conventional positive-excess tangency: nonpositive normalization")
    weights = direction / direction.sum()
    _check_weights(weights, mu.size, long_only)
    if weights @ excess <= 0:
        raise RuntimeError("optimization did not produce positive excess return")
    return weights


def portfolio_stats(weights, mu, cov, rf=.03):
    """Annual log-moment MPT statistics; not exact realized growth statistics."""
    mu, cov = _moments(mu, cov)
    weights = _check_weights(weights, mu.size)
    rf = _risk_free(rf)
    mean = float(weights @ mu)
    volatility = float(np.sqrt(weights @ cov @ weights))
    return {"return": mean, "volatility": volatility,
            "sharpe": (mean - rf) / volatility,
            "gross_exposure": float(np.abs(weights).sum())}


def frontier(mu, cov, long_only=False, points=100, max_return=None):
    """Upper efficient branch as a DataFrame with return/volatility columns.

    Begin at the global MVP and end at max_return (default: the larger of
    the highest asset mean and MVP mean). Short-allowed frontiers are unbounded;
    max_return selects a plotting range, not an economic upper bound. Identical
    asset means yield repeated copies of the sole efficient location.
    """
    mu, cov = _moments(mu, cov)
    if isinstance(points, (bool, np.bool_)) or not isinstance(points, (int, np.integer)) or points < 2:
        raise ValueError("points must be an integer at least 2")
    mvp = min_variance(mu, cov, long_only)
    start = float(mvp @ mu)
    stop = max(start, float(mu.max())) if max_return is None else float(max_return)
    if not np.isfinite(stop) or stop < start - 1e-12:
        raise ValueError("max_return must be finite and not below the MVP return")
    if long_only and stop > mu.max() + 1e-12:
        raise ValueError("max_return exceeds the long-only feasible return range")
    if long_only:
        start, stop = np.clip([start, stop], mu.min(), mu.max())
    targets = np.linspace(start, stop, points)
    risks = []
    for i, target in enumerate(targets):
        w = mvp if i == 0 else min_variance(mu, cov, long_only, target)
        risks.append(float(np.sqrt(w @ cov @ w)))
    return pd.DataFrame({"return": targets, "volatility": risks})


def backtest(log_returns, weights, rf=.03, periods=252):
    """Evaluate frozen target weights rebalanced daily, with no trading costs.

    The caller supplies out-of-sample asset log returns in the SAME asset order
    as the training weights, including the first test day's return relative to
    the last training close. This function never estimates or changes weights.
    Daily portfolio simple return is expm1(asset log returns) @ weights, NOT
    expm1(weighted asset log returns). Borrow/short costs are not modeled.

    annual_return is arithmetic daily mean times periods; volatility uses ddof=1;
    Sharpe annualizes daily excess over (1+rf)**(1/periods)-1. CAGR compounds exact
    daily returns. max_drawdown is nonpositive and includes initial wealth=1.
    Zero volatility gives NaN Sharpe. Nonpositive wealth raises ValueError.
    """
    periods, rf = _periods(periods), _risk_free(rf)
    values = _frame(log_returns, "log_returns", min_rows=2)
    weights = _check_weights(weights, values.shape[1])
    with np.errstate(over="ignore", invalid="ignore"):
        daily = np.expm1(values) @ weights
    if not np.isfinite(daily).all():
        raise ValueError("portfolio simple returns must remain finite")
    if (daily <= -1).any():
        raise ValueError("portfolio is bankrupt: a daily return is at or below -100%")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        wealth = np.r_[1., np.cumprod(1 + daily)]
        cagr = float(np.expm1(np.log1p(daily).mean() * periods))
    if not np.isfinite(wealth).all() or not np.isfinite(cagr):
        raise ValueError("wealth and CAGR must remain finite")
    if (wealth <= 0).any():
        raise ValueError("portfolio is bankrupt or wealth underflowed to zero")
    volatility = float(daily.std(ddof=1) * np.sqrt(periods))
    mean = float(daily.mean())
    rf_daily = float(np.expm1(np.log1p(rf) / periods))
    metrics = {"annual_return": mean * periods, "volatility": volatility,
               "sharpe": (mean - rf_daily) * periods / volatility if volatility > 0 else float("nan"),
               "cagr": cagr, "total_return": float(wealth[-1] - 1),
               "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1))}
    return pd.Series(daily, index=log_returns.index, name="portfolio_return"), metrics
