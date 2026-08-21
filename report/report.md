# Trusting the Error Bars: Calibration of Frequentist vs Bayesian Volatility Forecasts

> **UNPOPULATED SCAFFOLD.** This file contains section headings only. It holds no
> results, and no numbers may be written into it — not even illustrative or
> placeholder ones — until the corresponding pipeline stage has been implemented and
> validated. Prose written against invented numbers has a way of surviving into the
> final draft.

---

## 1. Introduction

*Motivation; why interval calibration rather than point accuracy is the question of
interest for risk management.*

## 2. Research question

*Statement of the question and of the specific comparison: a frequentist plug-in
predictive versus a Bayesian posterior predictive built from the same GARCH likelihood,
differing only in whether parameter uncertainty is integrated over.*

## 3. Data

*SPY and ^VIX, daily, 2014-01-01 to 2025-06-30. Source and caching. Data quality
findings from the pipeline's own report, including any dropped dates.*

### 3.1 Quantities and their distinct roles

*Observed returns; the Parkinson proxy; conditional variance forecasts; predictive
intervals; VaR forecasts; parameter uncertainty; innovation uncertainty.*

### 3.2 The Parkinson proxy scale caveat

*The proxy measures intraday variance while the models forecast close-to-close
variance including the overnight gap. Consequences for QLIKE, and why interval
calibration and VaR results are unaffected. See open item D2.*

## 4. Models

### 4.1 Baselines

*Yesterday's volatility; EWMA/RiskMetrics with the fixed RiskMetrics decay factor.*

### 4.2 Frequentist GARCH(1,1)-t

*Specification; estimation; the plug-in predictive and what it assumes away.*

### 4.3 Bayesian GARCH(1,1)-t

*The same likelihood; priors (open item D4, to be frozen before any out-of-sample
result is computed); the sampler; convergence diagnostics; the posterior predictive as
a mixture over draws.*

## 5. Backtest design

*Initial training block; expanding versus rolling estimation window (open item D1);
refit every 21 trading days with daily filtering between refits; the look-ahead
invariants and how they are tested.*

## 6. Evaluation methodology

*QLIKE and variance MSE against the proxy; interval coverage, PIT, and VaR backtests
against observed returns; Diebold-Mariano with its caveats; the stationary block
bootstrap.*

## 7. Results

### 7.1 Point-forecast accuracy

### 7.2 Interval calibration, full sample

### 7.3 VaR backtests

### 7.4 Calibration by volatility regime

*Calm, normal, and stressed subsamples from the lagged VIX. Subsample sizes reported
alongside every statistic; the stressed regime is small and the tests underpowered.*

## 8. Discussion

*What the evidence does and does not support. A lower loss in one sample is not
evidence of a better model; conclusions must rest on the bootstrap intervals, and an
interval containing zero is a reportable finding rather than a failure.*

## 9. Limitations

*Single asset; single horizon; one GARCH family; proxy scale mismatch; multiplicity
across the pairwise comparisons; the sample period's specific volatility episodes.*

## 10. Reproduction

*Environment, pinned dependencies, seeds, the committed raw-data snapshot and its
manifest, and the exact commands to regenerate every table and figure.*

## References
