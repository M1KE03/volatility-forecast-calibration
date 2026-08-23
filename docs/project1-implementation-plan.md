# Project 1 — Implementation Plan
## Trusting the Error Bars: Calibration of Frequentist vs Bayesian Volatility Forecasts

**Research question:** For one-day-ahead S&P 500 volatility, are the predictive intervals of a frequentist GARCH(1,1)-t and its Bayesian counterpart calibrated at their nominal levels out of sample — and does that calibration survive high-volatility regimes?

**Budget:** ~15–20 hours over ~1 week. Stage hours below sum to 16.5–21.5h; the cut list at the end tells you what to drop if you run over.

---

## 0. One design refinement (read this first)

The design turn proposed Bayesian **stochastic volatility** as the Bayesian model. For the implementation plan I'm firming that up to **Bayesian GARCH(1,1)-t** as the primary Bayesian model, with SV demoted to a stretch goal. Two reasons, and the first one makes the science *better*, not just easier:

1. **It isolates the mechanism.** Frequentist GARCH-t and Bayesian GARCH-t share the identical likelihood. The *only* difference between their predictive intervals is that the frequentist version plugs in point estimates (parameters treated as known) while the Bayesian posterior predictive integrates over parameter uncertainty. Any calibration difference you find is therefore attributable to exactly one cause. With SV you'd be comparing across model classes, and you couldn't say whether differences came from parameter uncertainty or from the latent-vol specification. This controlled comparison is the cleanest possible version of the research question — and the easiest to defend in an interview.
2. **It removes the schedule risk.** Given posterior draws of (ω, α, β, ν), the GARCH variance recursion is deterministic — you compute σ²ₜ₊₁ per draw from observed returns with no latent-state filtering. One monthly posterior fit gives you exact daily one-step-ahead predictive distributions for the whole month. SV would have required a filtering approximation between refits, which was the riskiest component of the original design.

The model lineup is therefore: **RW-in-vol, EWMA/RiskMetrics (with normal intervals — the "naive UQ" baseline), GARCH(1,1)-t (MLE, plug-in intervals), Bayesian GARCH(1,1)-t (posterior predictive intervals).** Four forecasters, two of which are baselines. No LSTM.

---

## 1. Locked decisions

Fix these now and write them into the report's protocol section before producing any results. Changing them after seeing results is the look-ahead you're trying to avoid.

| Decision | Value | Rationale |
|---|---|---|
| Asset | SPY, daily OHLCV via `yfinance` | Liquid, long history, index-level dynamics |
| Sample | 2014-01-01 → 2025-06-30 | ~2,900 trading days |
| Warm-up / initial training | 2014-01-02 → 2016-12-30 (~755 obs) | First fit uses this; expanding window after |
| Evaluation window | 2017-01-03 → 2025-06-30 (~2,130 obs) | Contains Feb-2018 "Volmageddon", COVID-2020, 2022 bear market |
| Returns | log returns from adjusted close | Standard |
| Volatility proxy (point-forecast target) | Parkinson: σ²ₚ,ₜ = [ln(Hₜ/Lₜ)]² / (4 ln 2) | ~5× less noisy than r²ₜ; invariant to multiplicative price adjustment (H/L ratio); free from OHLC |
| Calibration target | realized returns rₜ (not the proxy) | Interval coverage is evaluated on what you observe — sidesteps the latent-vol problem entirely |
| Refit cadence | every 21 trading days, expanding window, all models | Monthly is standard; sensitivity-checked in Stage 6 |
| Interval levels | two-sided 90%, 95%, 99%; one-sided 99% VaR (left tail) | 99% VaR is where models differ most and where Basel-style backtesting lives |
| Point losses | QLIKE (headline), MSE (secondary), both on variance scale vs Parkinson | QLIKE is robust to noisy proxies (Patton 2011) and penalizes vol under-prediction asymmetrically |
| DM test | on QLIKE differentials, h=1, Harvey–Leybourne–Newbold correction | Loss differential CIs additionally via block bootstrap |
| Bootstrap | stationary block bootstrap (Politis–Romano), mean block length 20, 1,000 replications, fixed seed | iid bootstrap is invalid under serial dependence — say so in the report |
| Regime definition | lagged VIX close (day t−1): calm < 15, normal 15–25, stressed > 25 | Fixed ex-ante thresholds, zero estimation, no look-ahead by construction. Sensitivity: terciles of trailing 21-day Parkinson vol (through t−1, thresholds fixed on warm-up sample) |
| Seeds & env | fixed seeds everywhere; pinned `requirements.txt`; one entry-point script | Reproducibility is part of the deliverable |

---

## 2. Repo structure

```
project-1-volatility-uq/
├── README.md                 ← 90-second version: question, headline figure, key finding
├── requirements.txt          ← pinned versions
├── run_all.py                ← reproduces every result and figure from scratch
├── data/                     ← cached raw pull + processed series (or a download script)
├── src/
│   ├── data.py               ← download, cleaning, returns, Parkinson proxy
│   ├── models.py             ← the four forecasters behind one common interface
│   ├── backtest.py           ← rolling harness (the no-look-ahead machinery)
│   ├── evaluation.py         ← QLIKE/MSE, DM, PIT, Kupiec, Christoffersen, coverage
│   └── bootstrap.py          ← stationary block bootstrap utilities
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_results.ipynb      ← all headline tables/figures, generated from src/
│   └── 03_robustness.ipynb
├── figures/
└── report/report.md (or pdf) ← the 2-page mini-paper
```

The `src/` + thin-notebooks split matters: it reads as software engineering discipline, and it's what makes `run_all.py` possible.

---

## 3. Stages

### Stage 0 — Scaffold, data, EDA (1.5–2h)
- Repo scaffold, env, pinned deps (`numpy`, `pandas`, `scipy`, `statsmodels`, `arch`, `pymc`/`arviz`, `matplotlib`, `yfinance`).
- Pull SPY OHLCV; hygiene checks: missing days vs NYSE calendar, zero/negative highs-lows, split handling. Cache the raw pull to disk so results don't drift if Yahoo revises data.
- Compute log returns and daily Parkinson variance. Pull VIX (^VIX) closes for regime labels.
- EDA: returns plot showing volatility clustering; ACF of r²ₜ; **ARCH-LM test** (the motivating test — this is your "why GARCH exists" evidence); Ljung-Box on squared returns; ADF on returns as one honest line ("stationary, as expected — the interesting structure is in the second moment").
- **Done when:** `data.py` produces the analysis dataset deterministically; EDA notebook shows clustering + test results.

### Stage 1 — Backtest harness + baselines (2.5–3h)
Build the harness *before* any interesting model — it's the component where look-ahead bugs live.
- Define the forecaster interface: given data through day t, return for day t+1 a variance forecast and a predictive return distribution (as parameters or as quantile/CDF functions). Every model, including baselines, goes through it.
- Rolling loop: expanding window, refit flags every 21 days, forecast every day. Store all forecasts in one tidy DataFrame (date, model, σ², quantiles at all needed levels, CDF evaluated at realized return — i.e. the PIT value).
- Implement RW-in-vol (yesterday's Parkinson) and EWMA (λ = 0.94, RiskMetrics convention, normal predictive distribution).
- **Look-ahead audit:** a test asserting the forecast for t+1 changes only when data ≤ t changes; check the day-t proxy is never an input to the day-t+1 forecast, only an evaluation target. Keep this test in the repo — mention it in the README.
- **Done when:** both baselines produce a full forecast table over the evaluation window and the audit passes.

### Stage 2 — Frequentist GARCH(1,1)-t (2–2.5h)
- Fit with `arch` (constant mean, GARCH(1,1), Student-t). Plug into the harness with monthly refits.
- Predictive distribution: t with ν̂ degrees of freedom scaled by σ̂ₜ₊₁ — plug-in, parameters treated as known. That's the point; state it.
- Diagnostics on the warm-up fit: standardized residuals — Ljung-Box on levels and squares, QQ plot vs fitted t. Fit a normal-innovation variant too (ablation for Stage 6: normal vs t is the cheap lever on 99% tail coverage).
- Track parameter estimates across refits (stability plot — nice diagnostic, ~free).
- **Done when:** GARCH forecast table complete; residual diagnostics recorded; you can articulate what the plug-in assumption ignores.

### Stage 3 — Bayesian GARCH(1,1)-t (3.5–4.5h) ← the one stage with real risk
- Same likelihood as Stage 2, now with priors over (ω, α, β, ν) and MCMC. Weakly informative priors; enforce positivity and α+β < 1 via priors/transforms; document them.
- Implementation route A (preferred): PyMC with the GARCH recursion via `pytensor.scan` (worked examples of Bayesian GARCH in PyMC exist — start from one).
- Route B (fallback, ~guaranteed to work): write the GARCH-t log-likelihood in plain NumPy and sample with `emcee` or a simple Metropolis sampler. The likelihood is ~15 lines; you don't need gradients for a 4-parameter model. This fallback costs you nothing scientifically.
- Convergence: R-hat < 1.01, adequate ESS, divergence check (route A); report these like a grown-up in the write-up.
- Forecasting: per posterior draw, run the (deterministic) variance recursion on observed returns to get σ²ₜ₊₁, giving a mixture-of-t predictive distribution. Quantiles/CDF by Monte Carlo over draws (500–1,000 thinned draws is plenty). Refit monthly like the others.
- **Done when:** convergence diagnostics pass and the Bayesian forecast table is complete. Sanity check: Bayesian intervals should be ≥ frequentist widths, most visibly early in the sample and at 99% — if not, hunt for a bug before proceeding.

### Stage 4 — Evaluation layer (3–4h) ← the intellectual core
- **Point accuracy:** QLIKE and MSE vs Parkinson for all four models; DM tests (HLN) on all pairs vs EWMA and GARCH-vs-Bayesian; block-bootstrap CIs on loss differentials. Expected result: models are hard to separate — that's the setup for the calibration act, not a failure.
- **PIT:** uₜ = Fₜ(rₜ) for each model (you stored this in the harness). Histograms (20 bins) per model; note formally that KS-uniformity p-values are approximate here because parameters are estimated — knowing that caveat is worth more than the test itself.
- **Coverage tables:** empirical vs nominal at 90/95/99 two-sided and 99% VaR, per model.
- **Kupiec POF** (unconditional coverage) and **Christoffersen** (independence + conditional coverage) on 99% VaR hit sequences. Christoffersen is the money test: it detects *clustered* failures, the failure mode that ruins risk models.
- **Hit-sequence figure:** timeline of 99% VaR breaches per model with stress periods shaded — this is likely your headline figure.
- **Done when:** one results notebook produces every table and figure above from the stored forecast tables alone.

### Stage 5 — Regime-conditional analysis (2–2.5h)
- Label each evaluation day by lagged-VIX regime (thresholds from §1). Report regime sample sizes up front — stressed days are a minority and readers should see that.
- Per regime: QLIKE ranking, coverage at each level, 99% VaR breach rate, with stationary-block-bootstrap CIs on each per-regime coverage estimate (crisis subsamples are small; error bars are mandatory, and their width is itself on-theme).
- Headline figure candidate: coverage vs nominal by regime, one panel per model — the plot that shows whether 99% means 99% when VIX > 25.
- Sensitivity: repeat with trailing-realized-vol terciles; relegate to robustness notebook.
- **Done when:** the regime figure + table exist with bootstrap CIs and an explicit no-look-ahead statement for the regime definition.

### Stage 6 — Robustness & ablations (1.5–2h)
- Proxy sensitivity: recompute QLIKE/MSE rankings with r²ₜ instead of Parkinson; note whether rankings move (they may — that's the Patton point in action).
- Innovation ablation: GARCH-normal vs GARCH-t coverage at 99% (expect normal to fail hard — cheap, decisive, interview-friendly).
- Refit-cadence spot check: rerun GARCH at 63-day refits; confirm conclusions don't hinge on cadence.
- **Done when:** each check is a short notebook section with a one-sentence verdict.

### Stage 7 — Write-up & polish (3–4h)
- **Report (2 pages), mini-paper arc:** (1) question — risk management consumes intervals, not point forecasts; (2) data & target — the latent-vol problem and proxy choice; (3) protocol — written as if pre-registered; (4) point-accuracy results — models hard to separate, motivating the calibration turn; (5) calibration results — PIT, coverage, Kupiec/Christoffersen, with the parameter-uncertainty mechanism explaining differences; (6) regime results — does 99% survive stress; (7) limitations & model risk — what unconditional backtests hide, what this design couldn't detect (one asset, one horizon, ~2 stress episodes, proxy noise).
- **README:** question in one sentence, headline figure, three-bullet findings, repro instructions (`pip install -r requirements.txt && python run_all.py`), repo map.
- Figure pass: consistent style, labeled axes, stress shading, captions that state the takeaway.
- Final `run_all.py` run from a clean clone.
- **Done when:** a stranger gets the finding from the README in 90 seconds and can reproduce every figure with one command.

---

## 4. Suggested week layout

| Day | Stages | Hours |
|---|---|---|
| 1 | Stage 0 + start Stage 1 | 3 |
| 2 | Finish Stage 1 + Stage 2 | 3–3.5 |
| 3–4 | Stage 3 (Bayesian) | 3.5–4.5 |
| 5 | Stage 4 | 3–4 |
| 6 | Stage 5 + Stage 6 | 3.5–4.5 |
| 7 | Stage 7 | 3–4 |

Checkpoint rule: if Stage 3 isn't producing forecasts by the end of day 4, switch to route B (emcee/Metropolis) immediately rather than debugging PyMC further.

## 5. Cut list (in order, if over budget)

1. SV stretch goal (already out of the core plan).
2. Refit-cadence sensitivity (Stage 6, last bullet).
3. Trailing-vol regime sensitivity (keep the VIX version).
4. MSE as secondary loss (keep QLIKE only).
5. **Never cut:** the look-ahead audit, Christoffersen, bootstrap CIs on regime coverage, or the limitations section — these are the maturity signals the whole project exists to send.

## 6. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| PyMC GARCH (scan) fiddliness | Medium | Route B fallback (NumPy likelihood + emcee); decision deadline end of day 4 |
| Bayesian ≈ frequentist everywhere | Medium | Still a finding: "at n≈750+, parameter uncertainty moves intervals less than innovation-distribution choice" — the normal-vs-t ablation then carries the calibration story |
| Few stressed-regime observations | Certain | Bootstrap CIs, report n per regime, honest language |
| Yahoo data revisions/gaps | Low | Cache raw pull in repo (or download script + checksum) |
| Scope creep (more assets/models) | Self-inflicted | The locked-decisions table is the contract; new ideas go to a `future-work.md` |

## 7. Interview ammunition (know these cold)

- Why QLIKE: proxy-robust loss (Patton 2011) — with a noisy volatility proxy, MSE rankings can be distorted; QLIKE's ranking is consistent, and it penalizes under-predicting vol more than over-predicting, matching risk-management asymmetry.
- Why Christoffersen beats Kupiec: correct *average* coverage can hide breaches that cluster in crises; independence of the hit sequence is the real requirement.
- The one-cause comparison: identical likelihood, plug-in vs posterior predictive — any interval difference is parameter uncertainty, full stop.
- Why block bootstrap: forecast errors are serially dependent; iid resampling destroys that dependence and understates uncertainty.
- Why coverage is evaluated on returns, not the proxy: returns are observed; volatility never is.

## 8. Free resources

- Patton (2011), "Volatility forecast comparison using imperfectly volatility proxies" — the QLIKE/proxy argument (working-paper version free online).
- `arch` docs (bashtage) — GARCH fitting, forecasting, and its own volatility notebooks.
- PyMC example gallery — Bayesian time-series examples incl. stochastic volatility (useful even for the GARCH build).
- Christoffersen (1998), "Evaluating interval forecasts" — the conditional-coverage framework.
- Politis & Romano (1994) stationary bootstrap — `arch.bootstrap.StationaryBootstrap` implements it, so read for the idea, not the code.
- Hyndman & Athanasopoulos, *FPP3* (free online) — the chapter on forecast evaluation protocol, for the rolling-origin discipline.
