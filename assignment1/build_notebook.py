"""Generate the submission notebook with tested functions and an embedded snapshot.

Run from the repository's .venv. The notebook is self-contained: its offline
fallback is the exact gzipped CSV, not simulated or regenerated market data.
"""
from pathlib import Path
import base64
import gzip
import hashlib
import json
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))


md(r"""
# FIN6131 Assignment 1 — Mean–Variance Portfolio Selection
**Nine diversified ETFs · five years · unrestricted versus long-only investing**

This notebook implements the complete analysis in the accompanying three-page
LaTeX report. **Run All** uses a frozen Yahoo Finance adjusted-close snapshot
and needs no network or companion files. The exact CSV is embedded below for
standalone execution; the repository also archives the provider's raw JSON.

- Analysis window: **2021-09-19 ≤ return date < 2026-09-19**.
- Training: dates before **2024-09-19**; test: dates on/after that boundary.
- First price is the prior trading close (2021-09-17), used only to calculate
  the first in-window return. No missing prices are filled.
- 252 sessions per year; annual risk-free rate **3%**; reproducibility seed **6131**.
- Full-sample estimates are descriptive, **not** the weights used in the backtest.

**Return convention.** The assignment requests annualized log-return moments.
We therefore optimize with $\mu=252\bar r$ and $\Sigma=252\widehat{\mathrm{Cov}}(r)$,
using $(w^\top\mu-0.03)/\sqrt{w^\top\Sigma w}$ as its conventional Sharpe objective.
Weighted asset log returns are only a first-order approximation to portfolio
returns, especially under leverage. Realized out-of-sample wealth instead uses
exact weighted **simple** returns. The CAL below is a line in this approximate
mean–variance coordinate system, not an exact log-growth identity.
""")
code("""
from pathlib import Path
import base64, gzip, hashlib, io, json, sys, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from IPython.display import display, Markdown
from scipy.optimize import minimize

SEED = 6131
rng = np.random.default_rng(SEED)
np.random.seed(SEED)
RF, PERIODS = 0.03, 252
START, SPLIT, END = pd.Timestamp('2021-09-19'), pd.Timestamp('2024-09-19'), pd.Timestamp('2026-09-19')
TICKERS = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'LQD', 'GLD', 'VNQ']
REFRESH = False  # Explicit opt-in; provider revisions may change all conclusions.
ROOT = Path.cwd()
if (ROOT / 'assignment1' / 'data').exists():
    ROOT = ROOT / 'assignment1'
OUT = ROOT / ('outputs_refresh' if REFRESH else 'outputs')
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
pd.set_option('display.max_columns', 20)
plt.rcParams.update({'figure.dpi': 130, 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.bbox': 'tight'})
print('Interpreter:', sys.executable)
print('Python:', platform.python_version(), '| NumPy:', np.__version__, '| pandas:', pd.__version__)
""")
md("""
## 1. Download, provenance, and data validation

The initial `yfinance.download` call was blocked from this machine. The same
provider's public chart endpoint was retrieved in a real browser on September
19, 2026, preserving its `adjclose` field. Every raw response has a recorded URL,
retrieval timestamp, and SHA-256 in `data/metadata.json`. Yahoo adjusted closes
account for splits and distributions [3]; they are a historical total-return proxy,
not raw tradable execution prices.

The cell below offers a **real yfinance refresh** (`auto_adjust=False`, selecting
`Adj Close` explicitly). Default execution validates the local CSV or uses the
byte-identical embedded snapshot. Refresh results go into `outputs_refresh/`;
they do not silently replace the report's data or results. Network errors stop
refresh instead of silently substituting another sample.
""")
raw = (ROOT / 'data/adjusted_close.csv').read_bytes()
encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
digest = hashlib.sha256(raw).hexdigest()
code(f"""
# Exact archived CSV, compressed only to keep the standalone notebook small.
SNAPSHOT_SHA256 = {digest!r}
SNAPSHOT_GZIP_BASE64 = {encoded!r}

if REFRESH:
    import yfinance as yf
    fetched = yf.download(TICKERS, start='2021-09-17', end='2026-09-19',
                          interval='1d', auto_adjust=False, back_adjust=False,
                          repair=False, progress=False, threads=False)
    if fetched is None or fetched.empty or 'Adj Close' not in fetched.columns.get_level_values(0):
        raise RuntimeError('Yahoo download failed: keep REFRESH=False for the verified snapshot.')
    prices = fetched['Adj Close'].reindex(columns=TICKERS)
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    print('REFRESHED DATA — differs potentially from the submitted report.')
else:
    local = ROOT / 'data/adjusted_close.csv'
    csv_bytes = local.read_bytes() if local.exists() else gzip.decompress(base64.b64decode(SNAPSHOT_GZIP_BASE64))
    assert hashlib.sha256(csv_bytes).hexdigest() == SNAPSHOT_SHA256, 'Snapshot checksum mismatch'
    prices = pd.read_csv(io.BytesIO(csv_bytes), index_col='Date', parse_dates=True, float_precision='round_trip')
    print('Verified CSV SHA-256:', SNAPSHOT_SHA256)

assert list(prices.columns) == TICKERS
assert isinstance(prices.index, pd.DatetimeIndex)
assert prices.index.is_monotonic_increasing and not prices.index.has_duplicates
assert np.isfinite(prices.to_numpy()).all() and (prices.to_numpy() > 0).all()
assert prices.index[0] == pd.Timestamp('2021-09-17') and prices.index[-1] == pd.Timestamp('2026-09-18')
assert len(prices) > 1200, 'Insufficient five-year history'
print(f'{{len(prices)}} complete prices per ETF; {{int(prices.isna().sum().sum())}} missing observations')
display(prices.iloc[[0, -1]].round(3))
""")
md(r"""
## 2. Estimation and numerical methods

For each asset, $r_{i,t}=\log(P_{i,t}/P_{i,t-1})$. The sample covariance uses
$T-1$ in its denominator. The covariance must be positive definite; the notebook
does not hide an ill-conditioned problem by silently adding a ridge penalty.

For target return $m$, solve
\[
\min_w\;\tfrac12 w^\top\Sigma w\quad\text{s.t.}\quad
\mathbf1^\top w=1,\quad\mu^\top w=m,
\]
with $w\in\mathbb R^9$ (shorts allowed), or additionally $w_i\ge0$ (long-only).
The **global minimum variance portfolio** drops the target-return constraint.
Only the branch above each regime's MVP is efficient.

Write $A=\mathbf1^\top\Sigma^{-1}\mathbf1$,
$B=\mathbf1^\top\Sigma^{-1}\mu$, $C=\mu^\top\Sigma^{-1}\mu$, and $D=AC-B^2$.
The unrestricted formulas are
\[
w_G=\Sigma^{-1}\mathbf1/A,\quad
w(m)=\Sigma^{-1}\!\left[\frac{C-Bm}{D}\mathbf1+\frac{Am-B}{D}\mu\right],\quad
\sigma^2(m)=\frac{Am^2-2Bm+C}{D}.
\]
We cross-check these results against independent SLSQP minimization.

For the tangency portfolio, maximize $(w^\top\mu-r_f)/\sigma(w)$.
The unrestricted solution is $w_T=\Sigma^{-1}(\mu-r_f\mathbf1)/
[\mathbf1^\top\Sigma^{-1}(\mu-r_f\mathbf1)]$, provided the denominator is positive.
The long-only case uses the globally valid convex transformation
$\min_{z\ge0}z^\top\Sigma z$ subject to $(\mu-r_f\mathbf1)^\top z=1$,
then $w=z/(\mathbf1^\top z)$. This requires a positive attainable excess return.
All solver statuses and constraint residuals are checked.
""")
# Embed the tested source so submitting this one notebook is sufficient.
code((ROOT / 'mpt.py').read_text())
code("""
log_returns, mu, cov = estimate(prices)
log_returns = log_returns.loc[(log_returns.index >= START) & (log_returns.index < END)]
# Explicitly recompute moments on the analysis window, not the extra prior close.
mu = PERIODS * log_returns.mean().to_numpy()
cov = PERIODS * log_returns.cov().to_numpy()
train = log_returns.loc[log_returns.index < SPLIT]
test = log_returns.loc[log_returns.index >= SPLIT]
assert len(train) + len(test) == len(log_returns)
assert train.index.max() < test.index.min()
mu_is, cov_is = PERIODS * train.mean().to_numpy(), PERIODS * train.cov().to_numpy()
asset_stats = pd.DataFrame({'Asset class': ['US large-cap', 'Nasdaq-100', 'US small-cap',
    'Developed ex-US equity', 'Emerging equity', 'Long US Treasuries',
    'Investment-grade credit', 'Gold', 'US REITs'],
    'Annual log mean': mu, 'Annual volatility': np.sqrt(np.diag(cov))}, index=TICKERS)
asset_stats['Sharpe proxy'] = (asset_stats['Annual log mean'] - RF) / asset_stats['Annual volatility']
display(asset_stats.round(4))
display(pd.DataFrame(cov, index=TICKERS, columns=TICKERS).round(6))
print(f'Return observations: full={len(log_returns)}, training={len(train)}, test={len(test)}')
print('Covariance condition number:', np.linalg.cond(cov))
asset_stats.to_csv(OUT / 'asset_statistics.csv')
pd.DataFrame(cov, index=TICKERS, columns=TICKERS).to_csv(OUT / 'annual_covariance.csv')
log_returns.to_csv(OUT / 'daily_log_returns.csv')
""")
code("""
# Dependence, rather than a count of tickers, determines diversification.
fig, ax = plt.subplots(figsize=(7.2, 4.5), layout='constrained')
corr = log_returns.corr()
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(9), TICKERS); ax.set_yticks(range(9), TICKERS)
for i in range(9):
    for j in range(9):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                color='white' if abs(corr.iloc[i, j]) > .65 else 'black', fontsize=8)
fig.colorbar(im, ax=ax, label='Daily log-return correlation')
ax.set_title('Diversification depends on covariance, not ticker count')
fig.savefig(FIG / 'correlations.pdf')
plt.show()
""")
md("""
## 3. Full-sample optimization

`U` denotes budget-constrained but **unrestricted short selling**; `L` denotes
**long-only** risky-asset weights. “Unconstrained” does not remove the budget
condition. Gross exposure is the sum of absolute weights and can exceed 100%
in regime U. No additional leverage limit or borrowing fee is imposed.
""")
code("""
def solve_portfolios(mean, covariance):
    \"\"\"Solve both objectives under both short-sale regimes.\"\"\"
    return {'U-MVP': min_variance(mean, covariance),
            'L-MVP': min_variance(mean, covariance, long_only=True),
            'U-TP': tangency(mean, covariance, rf=RF),
            'L-TP': tangency(mean, covariance, rf=RF, long_only=True)}

full_weights = solve_portfolios(mu, cov)
full_stats = pd.DataFrame({name: portfolio_stats(w, mu, cov, RF)
                           for name, w in full_weights.items()}).T
weights_table = pd.DataFrame(full_weights, index=TICKERS)
display((weights_table * 100).round(2).rename_axis('Weight (%)'))
display(full_stats.round(4))
weights_table.to_csv(OUT / 'full_sample_weights.csv')
full_stats.to_csv(OUT / 'full_sample_statistics.csv')

# The unrestricted frontier is unbounded in target return; plot a finite range
# that includes its tangency point. The long-only frontier ends at max(mu).
upper = max(float(mu.max()), float(full_stats.loc['U-TP', 'return'])) * 1.12
front_u = frontier(mu, cov, max_return=upper, points=180)
front_l = frontier(mu, cov, long_only=True, points=140)
front_u.to_csv(OUT / 'frontier_unrestricted.csv', index=False)
front_l.to_csv(OUT / 'frontier_long_only.csv', index=False)
""")
md(r"""
## 4. Efficient frontiers and capital allocation lines

For risky tangency portfolio $T$, allocation $y\ge0$ to it and $1-y$ to cash gives
$\mu_C=r_f+y(\mu_T-r_f)$ and $\sigma_C=y\sigma_T$, hence
$\mu_C=r_f+S_T\sigma_C$. We show **both** regimes' CALs for comparison.
Solid segments correspond to $0\le y\le1$; dashed extensions assume borrowing
at the same 3% rate. Long-only here restricts the **risky sleeve**, not cash
borrowing. If negative cash holdings were prohibited too, the L-CAL would end
at L-TP. Neither expected return nor a CAL guarantees realized performance.
""")
code("""
COLORS = {'U-MVP': '#1e40af', 'L-MVP': '#c2410c', 'U-TP': '#1e40af', 'L-TP': '#c2410c'}
fig, ax = plt.subplots(figsize=(9.5, 5.2), layout='constrained')
ax.plot(front_u['volatility'], front_u['return'], color='#1e40af', lw=2, label='Shorts allowed')
ax.plot(front_l['volatility'], front_l['return'], color='#c2410c', lw=2, ls='-.', label='Long-only')
xmax = max(front_u['volatility'].max(), np.sqrt(np.diag(cov)).max()) * 1.05
for name in ['U-TP', 'L-TP']:
    row = full_stats.loc[name]
    xx = np.linspace(0, row['volatility'], 80)
    ax.plot(xx, RF + row['sharpe'] * xx, color=COLORS[name], alpha=.7, lw=1.2, label=name + ' CAL')
    xx = np.linspace(row['volatility'], xmax, 80)
    ax.plot(xx, RF + row['sharpe'] * xx, color=COLORS[name], alpha=.65, lw=1.2, ls='--')
for name, row in full_stats.iterrows():
    ax.scatter(row['volatility'], row['return'], marker='*' if 'TP' in name else 's',
               s=130 if 'TP' in name else 45, color=COLORS[name], edgecolor='white', zorder=5)
    if name == 'U-TP':
        ax.annotate(name, (row['volatility'], row['return']), xytext=(8, -10), textcoords='offset points', fontsize=9)
ax.scatter(np.sqrt(np.diag(cov)), mu, color='#334155', s=24, label='Individual ETFs', zorder=4)
ax.scatter([0], [RF], color='black', s=20, zorder=6)
ax.annotate('Risk-free 3%', (0, RF), xytext=(5, -14), textcoords='offset points', fontsize=8)
ax.set(xlabel='Annualized volatility', ylabel='Annualized mean log return (MPT proxy)',
       title='Efficient frontiers and capital allocation lines', xlim=(-.008, xmax),
       ylim=(min(mu.min() - .025, -.03), upper * 1.07))
ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.grid(alpha=.18)
ax.legend(loc='lower right', fontsize=8, frameon=False)

# A zoom keeps asset and long-only labels readable despite unrestricted leverage.
zoom = ax.inset_axes([.08, .51, .43, .45])
zoom.plot(front_u['volatility'], front_u['return'], color='#1e40af', lw=1.5)
zoom.plot(front_l['volatility'], front_l['return'], color='#c2410c', lw=1.5, ls='-.')
zoom.scatter(np.sqrt(np.diag(cov)), mu, color='#334155', s=15)
offsets = {'SPY':(-23,1), 'QQQ':(5,3), 'IWM':(5,-5), 'EFA':(-20,3),
           'EEM':(5,-5), 'TLT':(5,0), 'LQD':(5,-8), 'GLD':(5,0), 'VNQ':(5,0)}
for ticker, x, y in zip(TICKERS, np.sqrt(np.diag(cov)), mu):
    zoom.annotate(ticker, (x,y), xytext=offsets[ticker], textcoords='offset points', fontsize=7)
for name in ['U-MVP', 'L-MVP', 'L-TP']:
    row = full_stats.loc[name]
    zoom.scatter(row['volatility'], row['return'], marker='*' if 'TP' in name else 's',
                 s=65 if 'TP' in name else 25, color=COLORS[name], zorder=5)
    zoom.annotate(name, (row['volatility'],row['return']), xytext=(-8, -12 if 'MVP' in name else 6),
                  textcoords='offset points', fontsize=7, color=COLORS[name])
zoom.set(xlim=(.035, .27), ylim=(mu.min()-.025, mu.max()+.045), title='Asset / long-only detail')
zoom.title.set_fontsize(8)
zoom.xaxis.set_major_formatter(PercentFormatter(1, decimals=0)); zoom.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
zoom.tick_params(labelsize=7); zoom.grid(alpha=.15)
fig.savefig(FIG / 'frontiers.pdf'); fig.savefig(FIG / 'frontiers.png', dpi=180)
plt.show()
""")
md(r"""
## 5. Three-year training / two-year holdout

Estimate weights **only** from returns dated before 2024-09-19, then freeze those
target weights for the whole holdout. All five strategies rebalance daily to
these targets, including the equal-weight benchmark (each ETF has weight $1/9$).
This is a constant-mix backtest, **not** a buy-and-hold fixed-share backtest.

At each test date, $R_{p,t}=\sum_iw_i[\exp(r_{i,t})-1]$ and
$W_t=\prod_{s\le t}(1+R_{p,s})$. For the first test return the denominator is the
last training close; it is not dropped or used to re-estimate weights.
We report annual arithmetic mean, volatility, excess-return Sharpe, CAGR,
total return, and maximum drawdown. The daily cash rate is
$(1.03)^{1/252}-1$; realized Sharpe is
$\sqrt{252}\,\overline{(R_p-R_{f,d})}/s(R_p)$.
CAGR uses $252/N$ as the annualization exponent. Initial wealth 1 is included
in drawdown peaks so an immediate loss is not missed.

**Frictionless assumptions:** no taxes, turnover costs, bid–ask spreads, margin
calls, stock-loan fees, or differing debit/credit rates. Dividends are represented
through adjusted returns. Short positions earn/pay the same adjusted total
returns symmetrically. This is optimistic, particularly for highly levered U-TP.
""")
code("""
is_weights = solve_portfolios(mu_is, cov_is)
is_stats = pd.DataFrame({name: portfolio_stats(w, mu_is, cov_is, RF)
                         for name, w in is_weights.items()}).T
is_weights['Equal'] = np.ones(len(TICKERS)) / len(TICKERS)
oos_daily, oos_metrics = {}, {}
for name, w in is_weights.items():
    oos_daily[name], oos_metrics[name] = backtest(test, w, rf=RF)
oos = pd.DataFrame(oos_metrics).T
realized = pd.DataFrame(oos_daily)
display((pd.DataFrame(is_weights, index=TICKERS) * 100).round(2).rename_axis('Training weight (%)'))
display(is_stats.round(4))
display(oos.round(4))
pd.DataFrame(is_weights, index=TICKERS).to_csv(OUT / 'in_sample_weights.csv')
is_stats.to_csv(OUT / 'in_sample_statistics.csv')
oos.to_csv(OUT / 'out_of_sample_statistics.csv')
realized.to_csv(OUT / 'out_of_sample_daily_simple_returns.csv')

wealth = (1 + realized).cumprod()
wealth = pd.concat([pd.DataFrame(1., index=[train.index[-1]], columns=wealth.columns), wealth])
fig, ax = plt.subplots(figsize=(9.5, 3.6), layout='constrained')
styles = {'U-MVP':('#1e40af', ':'), 'L-MVP':('#c2410c', ':'),
          'U-TP':('#1e40af', '-'), 'L-TP':('#c2410c', '-'), 'Equal':('#111827', '--')}
for name in wealth:
    color, style = styles[name]
    ax.plot(wealth.index, wealth[name], label=name, color=color, ls=style, lw=1.6)
ax.set(xlabel='Holdout date', ylabel='Wealth from $1', title='Frozen training weights: exact simple-return backtest')
ax.legend(ncol=5, fontsize=8, loc='upper left', frameon=False); ax.grid(alpha=.18)
fig.savefig(FIG / 'oos_wealth.pdf'); fig.savefig(FIG / 'oos_wealth.png', dpi=180)
plt.show()
""")
md("""
## 6. Verification, sensitivity, and limitations

The next cell verifies budget/bounds, common-target frontier dominance, analytic
versus numerical solutions, and the separation of training from test data.
A seeded random feasible-portfolio check supplements (but does not replace) the
convex optimality argument for long-only tangency. Log-risk-free-rate conversion
is shown as a sensitivity check; the submitted baseline remains exactly 3% as
specified. The regression tests also check infeasible targets and drawdowns.
""")
code("""
checks = {}
for name, w in full_weights.items():
    assert np.isclose(w.sum(), 1., atol=1e-8)
    if name.startswith('L'):
        assert w.min() >= -1e-8
for name, w in is_weights.items():
    assert np.isclose(w.sum(), 1., atol=1e-8)

# Feasible-set inclusion implies U risk <= L risk at every common target.
for target in np.linspace(full_stats.loc['L-MVP', 'return'], mu.max(), 25):
    wu = min_variance(mu, cov, target=float(target))
    wl = min_variance(mu, cov, long_only=True, target=float(target))
    assert wu @ cov @ wu <= wl @ cov @ wl + 1e-8
checks['frontier_dominance'] = True

# Independent numerical cross-check of closed-form unrestricted QP solutions.
errors = []
for target in [None, float(full_stats.loc['L-MVP', 'return']), float(mu.max())]:
    analytic = min_variance(mu, cov, target=target)
    constraints = [{'type':'eq', 'fun':lambda w: w.sum()-1, 'jac':lambda w: np.ones(len(w))}]
    if target is not None:
        constraints.append({'type':'eq', 'fun':lambda w, t=target: w @ mu-t, 'jac':lambda w: mu})
    fit = minimize(lambda w: w @ cov @ w, np.ones(9)/9, jac=lambda w: 2*cov@w,
                   constraints=constraints, method='SLSQP', options={'ftol':1e-13, 'maxiter':2000})
    assert fit.success, fit.message
    errors.append(float(np.max(np.abs(fit.x-analytic))))
assert max(errors) < 1e-4
checks['independent_solver_agreement'] = True
checks['max_independent_weight_error'] = max(errors)

# Changing only holdout returns cannot change training estimates or weights.
perturbed = log_returns.copy()
perturbed.loc[perturbed.index >= SPLIT] *= -2
alternative_train = perturbed.loc[perturbed.index < SPLIT]
alternative_weights = solve_portfolios(PERIODS*alternative_train.mean().to_numpy(), PERIODS*alternative_train.cov().to_numpy())
assert all(np.allclose(alternative_weights[k], is_weights[k]) for k in alternative_weights)
checks['no_lookahead'] = True

random_weights = rng.dirichlet(np.ones(9), size=10000)
random_sharpes = (random_weights@mu-RF) / np.sqrt(np.einsum('ij,jk,ik->i', random_weights, cov, random_weights))
assert full_stats.loc['L-TP','sharpe'] >= random_sharpes.max()-1e-8
checks['long_only_random_check'] = True
rf_log_weights = tangency(mu, cov, rf=np.log1p(RF), long_only=True)
checks['rf_log_sensitivity_max_weight_change'] = float(np.max(np.abs(rf_log_weights-full_weights['L-TP'])))
print(json.dumps(checks, indent=2))
""")
code(r"""
# Save exact machine-readable results and LaTeX tables from this same execution.
result = {'snapshot_sha256': SNAPSHOT_SHA256, 'refreshed': REFRESH,
          'sample_sizes': {'full':len(log_returns), 'in_sample':len(train), 'out_of_sample':len(test)},
          'dates': {'first_return':str(log_returns.index[0].date()), 'last_return':str(log_returns.index[-1].date()),
                    'train_end':str(train.index[-1].date()), 'test_start':str(test.index[0].date())},
          'full_sample': full_stats.to_dict(orient='index'),
          'full_weights': {k:v.tolist() for k,v in full_weights.items()},
          'in_sample': is_stats.to_dict(orient='index'),
          'in_sample_weights': {k:v.tolist() for k,v in is_weights.items()},
          'out_of_sample': oos.to_dict(orient='index'),
          'covariance_condition_number': float(np.linalg.cond(cov)), 'checks':checks}
(OUT / 'results.json').write_text(json.dumps(result, indent=2)+'\n')

# Pure string formatting avoids a separate report data-entry step.
def latex_table(frame, formats, headers, path, index_label="Portfolio"):
    lines = [r'\begin{tabular}{l' + 'r'*len(frame.columns) + '}', r'\toprule',
             ' & '.join([index_label] + headers) + r' \\', r'\midrule']
    for label, row in frame.iterrows():
        lines.append(' & '.join([str(label)] + [fmt(value) for fmt,value in zip(formats,row)]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    path.write_text('\n'.join(lines)+'\n')

pct = lambda v: f'{100*v:.2f}'
num = lambda v: f'{v:.3f}'
latex_table(full_stats[['return','volatility','sharpe','gross_exposure']],
            [pct,pct,num,num], [r'$\mu$ (\%)',r'$\sigma$ (\%)','Sharpe','Gross ($\times$)'], OUT/'full_stats.tex')
latex_table(weights_table[['U-MVP','L-MVP','U-TP','L-TP']], [pct]*4,
            ['U-MVP','L-MVP','U-TP','L-TP'], OUT/'weights.tex', index_label='ETF')
latex_table(oos[['cagr','volatility','sharpe','max_drawdown','total_return']],
            [pct,pct,num,pct,pct], [r'CAGR (\%)',r'Vol. (\%)','Sharpe',r'Max DD (\%)',r'Total (\%)'], OUT/'oos_stats.tex')

best_sharpe = oos['sharpe'].idxmax()
display(Markdown(f'''### Empirical summary
- The largest realized holdout Sharpe belongs to **{best_sharpe}** ({oos.loc[best_sharpe,'sharpe']:.3f}).
- Unrestricted tangency gross exposure is **{full_stats.loc['U-TP','gross_exposure']:.2f}×** full-sample and
  **{is_stats.loc['U-TP','gross_exposure']:.2f}×** when fitted on training data.
- U-TP Sharpe falls from the training log-moment proxy **{is_stats.loc['U-TP','sharpe']:.3f}** to
  realized simple-return holdout Sharpe **{oos.loc['U-TP','sharpe']:.3f}**; these use explicitly different conventions.
- The holdout is one historical regime, not a statistical guarantee. Mean estimates,
  covariance instability, selected surviving ETFs, fat tails, and omitted financing /
  trading costs limit the result. Long-only constraints reduce the feasible set and
  may act as regularization, but do not guarantee better future performance.
- All full-sample weights, training weights, moments, daily realized returns, tables,
  and figures are exported into `{OUT.name}/`.
'''))
""")
md(r"""
## References and reproducibility

1. Markowitz, H. (1952). “Portfolio Selection.” *The Journal of Finance*, 7(1), 77–91.
   https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1952.tb01525.x
2. yfinance, `download` API documentation (date exclusivity and adjustment options).
   https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
3. Yahoo Finance, “What is the adjusted close?”
   https://help.yahoo.com/kb/SLN28256.html
4. Yahoo Finance historical chart responses, retrieved 2026-09-19. The repository's
   `data/raw/manifest.json` lists all nine exact query URLs, dates, and response files;
   `data/metadata.json` adds checksums. The standalone notebook carries the verified CSV.

Exact package versions are in the repository's `requirements.lock.txt`.
All portfolio estimates, plots, and report tables come from this notebook's real
execution. Synthetic fixtures appear only in the unit tests, never as market data.
The implementation and draft narrative were prepared with AI assistance; review
and disclose that assistance according to the course's submission policy.
""")
nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name':'Python 3 (FIN6131 repository .venv)', 'language':'python', 'name':'python3'},
    'language_info': {'name':'python', 'version':'3.11.16'},
})
# Stable cell IDs reduce meaningless Git diffs across rebuilds.
for i, cell in enumerate(nb.cells):
    cell['id'] = f'fin6131-{i:02d}'
nbf.validate(nb)
# Tests generate into a temporary directory without overwriting the submission.
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT / 'portfolio_analysis.ipynb')
output = parser.parse_args().output
output.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, output)
print(f'Generated {len(cells)} cells with the exact embedded snapshot.')
