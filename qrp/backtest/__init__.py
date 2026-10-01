"""Long-only backtesting engine (Phase 3).

Use run_backtest() to simulate a signal strategy day by day, then
compute_metrics() for a plain-English performance report.
"""

from __future__ import annotations

from qrp.backtest.engine import (
    BacktestResult,
    load_price_matrix,
    load_signal_matrix,
    rebalance_dates,
    run_backtest,
)
from qrp.backtest.metrics import compute_metrics
from qrp.backtest.strategies import top_n_by_signal

__all__ = [
    "BacktestResult",
    "compute_metrics",
    "load_price_matrix",
    "load_signal_matrix",
    "rebalance_dates",
    "run_backtest",
    "top_n_by_signal",
]
