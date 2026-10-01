"""Unit tests for the Phase 2 signal framework.

All tests use synthetic price data with known properties -- no network.
"""

import numpy as np
import pandas as pd

from qrp.signals import default_registry
from qrp.signals.quant import LowVolatility, Momentum, Trend
from qrp.signals.registry import SignalRegistry


def make_bars(closes) -> pd.DataFrame:
    """Build an OHLCV DataFrame (DatetimeIndex, oldest first) from closes."""
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="B")
    c = pd.Series(np.asarray(closes, dtype=float), index=idx)
    return pd.DataFrame(
        {
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "volume": 1_000_000,
        }
    )


def rising(n: int) -> np.ndarray:
    """Steadily rising prices: +0.1% per day."""
    return 100.0 * 1.001 ** np.arange(n)


def flat(n: int) -> np.ndarray:
    return np.full(n, 100.0)


# --- registry ---------------------------------------------------------------

def test_registry_round_trip():
    reg = SignalRegistry()
    sig = Momentum()
    reg.register(sig)
    assert reg.get("momentum") is sig
    names = [name for name, _ in reg.list_signals()]
    assert names == ["momentum"]


def test_registry_get_unknown_raises():
    reg = SignalRegistry()
    try:
        reg.get("nope")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown signal")


def test_default_registry_has_three_signals():
    reg = default_registry()
    assert {n for n, _ in reg.list_signals()} == {
        "momentum",
        "trend",
        "low_volatility",
    }


# --- momentum ---------------------------------------------------------------

def test_momentum_rising_is_positive():
    vals = Momentum().compute(make_bars(rising(400)))
    assert pd.notna(vals.iloc[-1])
    assert vals.iloc[-1] > 0


def test_momentum_flat_is_zero_not_nan():
    vals = Momentum().compute(make_bars(flat(400)))
    assert vals.iloc[-1] == 0.0


# --- trend ------------------------------------------------------------------

def test_trend_rising_is_positive():
    vals = Trend().compute(make_bars(rising(400)))
    assert pd.notna(vals.iloc[-1])
    assert vals.iloc[-1] > 0


def test_trend_flat_is_zero():
    vals = Trend().compute(make_bars(flat(400)))
    assert vals.iloc[-1] == 0.0


# --- low volatility ---------------------------------------------------------

def test_low_vol_constant_price_no_crash():
    # std of zero returns is 0 -> signal 0.0, never inf/NaN.
    vals = LowVolatility().compute(make_bars(flat(200)))
    last = vals.iloc[-1]
    assert pd.notna(last)
    assert last == 0.0
    assert np.isfinite(vals.dropna()).all()


def test_low_vol_volatile_is_negative():
    rng = np.random.default_rng(0)
    closes = 100.0 * np.cumprod(1 + rng.normal(0, 0.02, 300))
    vals = LowVolatility().compute(make_bars(closes))
    assert pd.notna(vals.iloc[-1])
    assert vals.iloc[-1] < 0


# --- lookback / NaN handling ------------------------------------------------

def test_short_history_yields_nan_not_garbage():
    df = make_bars(rising(60))  # shorter than every signal's lookback (min 64)
    for sig in (Momentum(), Trend(), LowVolatility()):
        vals = sig.compute(df)
        assert len(vals) == len(df)
        assert vals.isna().all(), f"{sig.name} should be all-NaN on short history"


def test_compute_output_shape_and_index():
    df = make_bars(rising(300))
    for sig in (Momentum(), Trend(), LowVolatility()):
        vals = sig.compute(df)
        assert isinstance(vals, pd.Series)
        assert vals.index.equals(df.index)
        assert vals.dtype == float
