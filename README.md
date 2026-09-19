# FIN6131 — Portfolio and Asset Management

Homework projects for FIN6131 at CUHK-Shenzhen.

## Assignment 1: Mean–variance portfolio selection

### Submission files

- [`assignment1/portfolio_analysis.ipynb`](assignment1/portfolio_analysis.ipynb) — executed, commented notebook with tested numerical functions and an embedded, checksum-verified price snapshot. Runs without companion files or network access.
- [`assignment1/report.pdf`](assignment1/report.pdf) — three-page analytical report.
- [`assignment1/report.tex`](assignment1/report.tex) — editable LaTeX source.

ETFs: **SPY, QQQ, IWM, EFA, EEM, TLT, LQD, GLD, VNQ**. Return dates: **2021-09-20 through 2026-09-18**. The nominal five-calendar-year window begins 2021-09-19; its previous trading close is retained to compute the first return. **1,255 complete returns** split into **754 training / 501 holdout** observations at 2024-09-19.

### Main results

| Training-fitted strategy | Holdout CAGR | Volatility | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| Unrestricted MVP | 7.53% | 5.52% | 0.808 | −5.93% |
| Long-only MVP | 9.19% | 8.27% | 0.747 | −6.72% |
| Unrestricted tangency | 61.18% | 56.83% | 1.075 | −64.99% |
| Long-only tangency | 28.15% | 18.98% | 1.247 | −18.38% |
| Equal weight | 14.75% | 12.55% | 0.924 | −11.48% |

The unrestricted frontier looks better in sample, but its training tangency portfolio requires **14.68× gross exposure**. Long-only tangency has the highest holdout Sharpe, not the lowest risk. The report discusses leverage, concentration, estimation error, and omitted financing/trading costs. Historical research results are not investment advice.

### Reproduce with the repository-local environment

Python **3.11.16** was used. All Python dependencies and execution use **`.venv/` inside this repository**, excluded from Git; system Python is not modified.

```bash
cd /home/weiqi/Desktop/CUHKSZ/FIN6131-Portf-and-Asset-Mngt
make setup  # Create .venv if missing; install exact locked dependencies using uv
make all    # Test, execute notebook, compile LaTeX, verify artifacts
```

The Makefile uses `.venv/bin/python` explicitly. Notebook execution pins the kernel to that interpreter and shuts it down afterward. The kernel binds to `127.0.0.1`; recent ipykernel versions may print a generic unencrypted-local-TCP warning.

```bash
make data         # Reconstruct CSV/checksums from archived real responses; no network
make test         # 98 tests, including standalone notebook execution
make notebook     # Regenerate and execute the final notebook
make report       # Compile LaTeX using existing tables and figures
make verify       # Check data hashes, execution, results, and PDF
make clean-cache  # Remove LaTeX caches; preserve deliverables and data
```

`make test` generates and executes an isolated notebook in a temporary directory, leaving the submitted notebook untouched. For lasting changes, edit `build_notebook.py` and `mpt.py`, not the generated notebook.

For interactive use, open the notebook in your notebook editor and select this repository's `.venv/bin/python`. Keep `REFRESH=False` to reproduce the submission. `REFRESH=True` attempts a real Yahoo download for the same fixed dates and writes to `outputs_refresh/`, not the submitted results. A blocked/empty download fails explicitly. Refreshing does not update the report's prose or dates automatically.

LaTeX requires `latexmk`, `pdflatex`, and packages loaded by `report.tex`: `geometry`, `fontenc`, `lmodern`, `amsmath`, `amssymb`, `booktabs`, `graphicx`, `microtype`, `caption`, `hyperref`, and `enumitem`. Python versions are pinned in `requirements.lock.txt`.

### Methodological conventions

- Annualized daily **log-return** means and sample covariances; 252 sessions/year and annual risk-free rate 3%.
- Unrestricted portfolios still satisfy the budget condition. They permit shorting without a gross-leverage cap; long-only portfolios have nonnegative risky-asset weights.
- Both regimes have MVP/tangency solutions. Unrestricted formulas are checked against SciPy SLSQP; long-only tangency uses a convex transformed quadratic program.
- One chart shows both efficient frontiers, nine assets, four optimal portfolios, and both CALs. Dashed CAL extensions assume borrowing at 3%. Risky-sleeve long-only constraints do not forbid negative cash holdings.
- Optimized log-moment Sharpe/CAL calculations are conventional **approximations**, not exact portfolio log-growth identities.
- Holdout weights use training data only. Target weights are frozen, with **daily rebalancing** for every strategy, including equal weight. This is not fixed-share buy-and-hold.
- Realized wealth compounds exact weighted **simple** returns. Realized Sharpe converts annual cash interest to its daily simple rate. Drawdowns include initial wealth.
- No trading costs, short-borrow fees, taxes, or margin calls. These omissions are especially material for unrestricted tangency.

### Data provenance

Direct `yfinance` and local Yahoo chart requests were blocked. The public Yahoo chart endpoint was accessible through a real browser; all nine genuine responses were archived on **2026-09-19**. The `adjclose` field supplies adjusted prices. No synthetic market data, forward filling, or silent row removal are used.

- [`data/raw/manifest.json`](assignment1/data/raw/manifest.json): exact URLs and retrieval timestamps.
- [`data/metadata.json`](assignment1/data/metadata.json): dates, assets, missingness counts, raw-response and CSV hashes.
- CSV SHA-256: `8baf117bd68b1703cd1c71c7884f3111fe697409cf628f78a18176c3a54651cc`.
- Yahoo can revise historical adjustments; the frozen snapshot, not a later download, reproduces the submission.

### Repository layout

| Path | Purpose |
|---|---|
| `assignment1/portfolio_analysis.ipynb` | Executed standalone submission notebook |
| `assignment1/report.tex`, `assignment1/report.pdf` | Report source and compiled PDF |
| `assignment1/mpt.py` | Validated estimation, optimization, and backtesting |
| `assignment1/data_io.py`, `assignment1/data/` | Snapshot construction, adjusted CSV, metadata, raw JSON |
| `assignment1/build_notebook.py` | Embeds the tested core and archived CSV |
| `assignment1/execute_notebook.py` | Execution using repository Python |
| `assignment1/outputs/` | Full/training weights, moments, daily holdout returns, metrics, LaTeX tables |
| `assignment1/outputs/figures/` | Frontier, correlation, and holdout-wealth plots |
| `assignment1/tests/` | Numerical unit tests and data/notebook integration tests |
| `assignment1/verify_artifacts.py` | Hashes, execution, independent metric recomputation, PDF checks |
| `assignment1/references/ledger.json` | Retrieved methodological/data-documentation URLs |
| `requirements.txt`, `requirements.lock.txt` | Direct dependencies and exact executed environment |
| `Makefile`, `pytest.ini` | Build targets and test configuration |
| `.gitignore` | Excludes environment/caches/editor locks; tracks submission PDFs explicitly |

### Report outline and verification

1. **Data, estimation, and theory:** sample construction, constraints, analytic formulas, tangency/CAL, log-return caveat.
2. **Full-sample interpretation:** frontier chart, weights/statistics, diversification versus leverage.
3. **Robustness and limitations:** chronological holdout, benchmark comparison, wealth chart, model limitations.

Verified: **98 tests pass**, all **10 code cells execute**, PDF has **3 pages**, and no unresolved references or LaTeX warnings/overflow. Independent optimizer weights agree within **3.23e-7**. Perturbing holdout data does not change training weights. Synthetic unit-test fixtures are separate from the real empirical data.

Code and draft prose were prepared with AI assistance. Review and disclose assistance according to course policy before submission. No student name or ID has been invented; add your own if required. Git version control is local; nothing is automatically pushed.

*Maintainer note:* README mathematics should use standard LaTeX commands rather than document-specific preamble macros.
