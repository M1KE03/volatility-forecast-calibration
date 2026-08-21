# Trusting the Error Bars: Calibration of Frequentist vs Bayesian Volatility Forecasts

**Status: scaffold only. No results exist yet. Every function in `src/` is an unimplemented stub.**

## Research question

For one-day-ahead S&P 500 volatility, are the predictive intervals of a frequentist
GARCH(1,1)-t model and its Bayesian counterpart calibrated at their nominal levels out
of sample, and does that calibration survive high-volatility regimes?

The methodological core is a comparison of a **frequentist plug-in predictive
distribution** against a **Bayesian posterior-predictive distribution** built from the
*same* GARCH likelihood. The two differ only in whether parameter uncertainty is
integrated over. The question is whether doing so improves predictive-interval
calibration.

## Models compared

| # | Model | Role |
|---|---|---|
| 1 | Yesterday's volatility | Naive baseline |
| 2 | EWMA / RiskMetrics | Industry baseline |
| 3 | GARCH(1,1)-t, frequentist MLE | Plug-in predictive |
| 4 | GARCH(1,1)-t, Bayesian | Posterior predictive |

Models 3 and 4 share one log-likelihood implementation (`src/models.py`), so that
"the same underlying likelihood" is a property of the code rather than a claim.

## Locked design decisions

| Item | Value |
|---|---|
| Asset | SPY |
| Frequency | Daily |
| Data source | Yahoo Finance via `yfinance` |
| Main sample | 2014-01-01 to 2025-06-30 |
| Initial training period | 2014-01-02 to 2016-12-30 |
| Out-of-sample period | 2017-01-03 to 2025-06-30 |
| Returns | Daily log returns from adjusted close |
| Point-forecast proxy | Parkinson variance from daily high/low |
| Calibration target | Observed daily returns |
| Forecast horizon | 1 trading day |
| Refit cadence | Every 21 trading days |
| Intervals evaluated | Two-sided 90%, 95%, 99%; one-sided 99% VaR |
| Primary point loss | QLIKE |
| Secondary point loss | MSE on the variance scale |
| Primary comparison test | Diebold-Mariano, with caveats |
| Uncertainty quantification | Stationary block bootstrap, mean block length 20, 1,000 replications, fixed seed |
| Regimes | Lagged VIX close: calm < 15, normal 15-25, stressed > 25 |

Explicitly **out of scope** in the core version: LSTM, Transformer, stochastic
volatility, macro variables, multiple assets, dashboards.

The full decision register, including decisions made during scaffolding and the
items still open, is in [`research_log.md`](research_log.md).

## Quantities that must not be conflated

This project involves several distinct objects that are easy to blur together. They
are kept distinct in code and in the report:

- **Observed returns** -- realised daily log returns. The target for interval
  calibration and VaR.
- **Parkinson volatility proxy** -- a high/low range estimator of *intraday* variance.
  The target for point-forecast loss only. It is **not** the same quantity the models
  forecast (see the caveat below).
- **Conditional variance forecast** -- the model's one-day-ahead variance.
- **Predictive interval** -- a two-sided interval for the *return*, not for the variance.
- **VaR forecast** -- a one-sided lower quantile of the return distribution.
- **Parameter uncertainty** -- uncertainty about GARCH coefficients. Present in the
  Bayesian posterior predictive; absent from the frequentist plug-in.
- **Innovation uncertainty** -- uncertainty from the Student-t shock. Present in both.

### Known caveat: Parkinson is not on the forecast target's scale

Parkinson variance estimates the variance of the *intraday continuous* price path. The
GARCH models forecast the variance of the *close-to-close* return, which also contains
the overnight gap. Parkinson is therefore biased low relative to the forecast target,
and QLIKE's proxy-robustness property assumes a conditionally unbiased proxy. This
affects the **point-forecast loss only**; the interval-calibration and VaR results are
evaluated against observed returns and are unaffected. Handling is recorded as open
item D2 in `research_log.md`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Requires Python 3.12. No system C/C++ compiler is needed.

## Usage

```bash
python run_all.py --help            # list pipeline stages
python run_all.py --stage data      # run one stage
python run_all.py --all             # run the full pipeline
```

All stages currently raise `NotImplementedError`. This is intentional at this point.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── run_all.py             # single entry point
├── research_log.md        # decision register + changelog
├── data/
│   ├── raw/               # committed price snapshots + SHA-256 manifest
│   └── processed/         # derived, gitignored, regenerable
├── src/
│   ├── data.py            # download, caching, returns, proxy, regimes
│   ├── models.py          # shared likelihood, MLE, MCMC, baselines
│   ├── backtest.py        # walk-forward loop, refit schedule
│   ├── evaluation.py      # losses, coverage tests, DM
│   └── bootstrap.py       # stationary block bootstrap
├── notebooks/             # thin presentation layer only
├── tests/
├── figures/
└── report/report.md
```

## Reproducibility

- Dependencies are pinned in `requirements.txt`.
- Raw price data is committed as CSV with a SHA-256 manifest, because Yahoo Finance
  revises history and a download script alone is not reproducible.
- All stochastic components take an explicit seed.
- `run_all.py` is the only supported entry point. Notebooks read cached artefacts and
  contain no reusable logic.
