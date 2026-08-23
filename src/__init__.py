"""Calibration of frequentist vs Bayesian one-day-ahead volatility forecasts for SPY.

All reusable logic lives in this package. Notebooks are a thin presentation layer and
must not define analysis logic of their own.

Modules
-------
data
    Download, caching, integrity verification, returns, the Parkinson proxy, and VIX
    regime assignment.
eda
    The diagnostics that establish the project's premise: Engle ARCH-LM, Ljung-Box on
    squared and raw returns, ADF, and the ACF of squared returns.
figures
    Figure rendering and house style. Figures are written by ``run_all.py``, never by a
    notebook.
models
    The shared GARCH(1,1)-t log-likelihood, the frequentist MLE, the Bayesian posterior
    sampler, predictive distributions, and the two baselines.
backtest
    The walk-forward loop: refit schedule, daily filtering, forecast production.
evaluation
    Point losses, interval coverage diagnostics, VaR backtests, Diebold-Mariano.
bootstrap
    Stationary block bootstrap for uncertainty around evaluation differences.
"""

__all__ = ["data", "eda", "figures", "models", "backtest", "evaluation", "bootstrap"]

__version__ = "0.1.0.dev0"
