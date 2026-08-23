# Trusting the Error Bars: Calibration of Frequentist vs Bayesian Volatility Forecasts

**Status: Stage 1 complete. The walk-forward harness is built and its look-ahead audit
passes; both baselines produce a complete forecast table over the 2,134-day evaluation
window. Neither GARCH model exists yet: `evaluation.py` and `bootstrap.py` are still
stubs and the GARCH half of `models.py` is stubbed. Stage 2 (frequentist GARCH) is
next.**

Stage numbers follow [`docs/project1-implementation-plan.md`](docs/project1-implementation-plan.md),
the governing plan.

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

The plan of work is [`docs/project1-implementation-plan.md`](docs/project1-implementation-plan.md):
eight stages, a ~15-20h budget, and an ordered cut list. The full decision register,
including decisions made during scaffolding and the items still open, is in
[`research_log.md`](research_log.md); §1.5 there records the 2026-08-23 switch to this
plan and the retirement of the earlier one, which was deleted at the switch and
survives only in git history (see `git log -- docs/implementation_plan.md`).

For a problem-oriented view of the same history — every difficulty encountered, why it
mattered, and the fix now in the repository — see
[`docs/problems-and-solutions.md`](docs/problems-and-solutions.md). It covers the
look-ahead hazards, the proxy scale mismatch, the environment and toolchain work, and a
set of test-integrity lessons that generalise beyond this project.

**Picking this up cold?** [`docs/handoff.md`](docs/handoff.md) is the entry point: what
exists, what is next, which rules are not revisable, and the specific traps that do damage
while looking fine.

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
evaluated against observed returns and are unaffected. The governing plan handles this
by reporting the squared-return proxy as a robustness check at Stage 6, and the caveat
belongs in the results section of the report, not only the methods section.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Requires Python 3.12 **and a C/C++ compiler on `PATH`** — PyMC's PyTensor backend needs
one. `pip install -r requirements.txt` does not supply it, so it is the one part of the
environment that is not reproducible from that file alone.

This machine uses **MinGW-w64 GCC 16.2.0** (UCRT, x86_64) from the
[WinLibs](https://winlibs.com/) release `16.2.0posix-14.0.0-ucrt-r1`, unpacked to
`C:\Users\micha\toolchains\mingw64` and added to the user `PATH`. It is portable: no
administrator rights are needed, and uninstalling means deleting that directory. Verify
with:

```bash
g++ --version                 # expect 16.2.0
python -c "import pytensor; print(pytensor.config.cxx)"
```

MSVC Build Tools works equally well if you already have it. Provenance, the verified
SHA-256, and why this replaced the earlier compiler-free design are in `research_log.md`
§1.6 (decision B1-R).

## Usage

```bash
python run_all.py --help            # list pipeline stages
python run_all.py --stage data      # build the analysis frame (implemented)
python run_all.py --stage eda       # ARCH-LM, Ljung-Box, ADF + Stage 0 figures (implemented)
python run_all.py --stage backtest  # walk-forward loop + the two baselines (implemented)
python run_all.py --all             # run the full pipeline
```

`--stage data` uses the committed snapshot in `data/raw/` and makes no network call;
pass `--refresh` to re-download, which deliberately replaces that snapshot. It prints the
full data quality report and writes `data/processed/analysis_frame.csv`.

The `evaluate` and `figures` stages still raise `NotImplementedError`.

## The scale convention

Every variance forecast in this project and the evaluation proxy are on the
**close-to-close return-variance scale**. This is what makes the four models comparable
to each other and their intervals comparable to observed returns.

The Parkinson estimator is built from the intraday high/low range and so sees no part of
the overnight move, which for SPY is a large share of daily variance. It understates
close-to-close variance by roughly a third. The correction is a single constant estimated
on the warm-up sample only and frozen:

```
c = mean(r^2) / mean(sigma^2_P) = 1.517318      (2014-01-02 .. 2016-12-30, n = 756)
```

Without it the error would pull in two directions at once: the RW baseline would be
unfairly advantaged on QLIKE (its units match the raw proxy) while its intervals came out
~23% too narrow, so it would win the point-forecast table and fail the calibration table
for reasons having nothing to do with forecasting. Separately, QLIKE's proxy-robustness
(Patton 2011) presumes an unbiased proxy.

**What `c` does not do.** It removes the systematic level error but does *not* make the
proxy conditionally unbiased -- within the warm-up alone the quarterly ratio ranges from
1.18 to 1.94, and the overnight share plausibly co-moves with regime. The proxy is
approximately unbiased *on average*; proxy-robustness is approached, not restored. Stage
6 re-runs the QLIKE ranking on raw Parkinson to show the ranking does not hinge on `c`.

## The look-ahead audit

The single test the whole project's credibility rests on. It runs the backtest, corrupts
every observation from a date `t` onward, re-runs, and asserts the forecasts are
bit-identical before `t` -- at Volmageddon, the largest COVID drawdown day, and a 2022
selloff.

A precise distinction is built into it. Corrupting from `t` must leave the *forecast*
columns unchanged up to and including `t`, because the forecast for `t` is built from
data through `t-1`. It must **not** leave the *evaluation* columns at `t` unchanged --
`log_return` and its `pit` are functions of day `t` itself. Asserting the stronger claim
would be asserting something false, and would have to be weakened later, which is how a
look-ahead test quietly stops testing anything.

`test_corruption_actually_changes_the_future` sits beside it, because everything above
would also hold for a backtest that ignored its input entirely.

```bash
pytest tests/test_backtest.py -q
```

### What the EDA establishes

Diagnostics are computed on the **training window only** (2014-01-02 to 2016-12-30).
Justifying the model class with a statistic computed over the out-of-sample period would
let the evaluation window argue for the model later evaluated on it -- a mild look-ahead,
but the exact species this project exists to detect. Full-sample values are printed as
labelled descriptive context and justify nothing.

| Diagnostic | Training-window result |
|---|---|
| ARCH-LM (Engle), lags 5 / 10 / 22 | p = 1.5e-23 / 1.8e-21 / 1.0e-17 — rejects at every lag |
| Ljung-Box, **squared** returns | p = 3.4e-44 / 1.1e-51 / 1.8e-47 — rejects |
| Ljung-Box, **raw** returns | p = 0.45 / 0.62 / 0.26 — no rejection |
| ADF on returns | -27.39, p < 1e-300 — stationary |
| Excess kurtosis | 2.46 |

The contrast between the two Ljung-Box rows is the point, not the first row alone:
abundant structure in the second moment, none detectable in the first. That is the regime
in which a conditional-variance model earns its place. The excess kurtosis is the
independent empirical warrant for the Student-t innovation.

A known limitation, recorded up front: over the **full** sample the raw-return Ljung-Box
does reject (p = 7.1e-14), as short-horizon autocorrelation rises in crises. The constant
mean is applied identically to all four forecasters so it cannot bias the comparison, but
it is a real simplification over the evaluation period and is carried into the report's
limitations section.

### The analysis frame

2,890 trading days, 2014-01-02 to 2025-06-30: 756 training rows (2014-01-02 to
2016-12-30) and 2,134 out-of-sample rows (2017-01-03 to 2025-06-30), implying 102 refits
at the locked 21-day cadence. Regime counts on the lagged VIX close are calm 1,198,
normal 1,317, stressed 375 — the stressed regime is 13% of the sample, which is thin for
the 99% level and for the 99% VaR and is why every regime table reports `n`.

Raw data is downloaded from 2013-12-01, one month before the sample start, so that the
lagged quantities are defined on the first in-sample row. The buffer feeds the lag only
and never reaches the analysis frame (decision D8 in `research_log.md`).

## Repository layout

```
.
├── README.md
├── requirements.txt
├── run_all.py             # single entry point
├── research_log.md        # decision register + changelog
├── docs/
│   ├── handoff.md                        # START HERE: state, next actions, traps
│   ├── project1-implementation-plan.md   # governing plan: stages, budget, cut list
│   └── problems-and-solutions.md         # every difficulty hit so far, and its fix
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
