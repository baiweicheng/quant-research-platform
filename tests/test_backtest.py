"""Tests for the long-only backtesting engine (Phase 3).

All tests use SYNTHETIC data (no network). They pin down the engine's most
important correctness properties:
  1. A perfect-foresight signal makes money (the engine can capture an edge).
  2. Higher trading costs never help (costs are actually charged).
  3. NO LOOKAHEAD: a signal that "knows" about a price jump dated T can only
     trade at T+1 prices — the jump itself is never captured.
  4. Long-only invariant: target weights are never negative and never sum to
     more than 1 on any rebalance date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qrp.backtest.engine import rebalance_dates, run_backtest
from qrp.backtest.costs import CostModel, FlatCostModel, SpreadCostModel
from qrp.backtest.metrics import compute_metrics
from qrp.backtest.strategies import top_n_by_signal

INITIAL = 100_000.0


def _trend_setup(periods: int = 120):
    """A climbs steadily, B is flat, C falls steadily; foresight signal."""
    dates = pd.bdate_range("2024-01-02", periods=periods)
    n = len(dates)
    prices = pd.DataFrame(
        {
            "A": 100.0 * (1.005 ** np.arange(n)),
            "B": np.full(n, 100.0),
            "C": 100.0 * (0.995 ** np.arange(n)),
        },
        index=dates,
    )
    # "Perfect foresight": the signal on date d is the 20-day forward return.
    # (Synthetic only — a real signal could never know this.)
    foresight = prices.shift(-20) / prices - 1.0
    return prices, foresight


def test_perfect_foresight_is_profitable():
    prices, foresight = _trend_setup()
    res = run_backtest(
        prices,
        foresight,
        top_n_by_signal("foresight", 1),
        frequency="monthly",
        cost_bps=0.0,
        initial_capital=INITIAL,
    )
    assert res.equity["portfolio_value"].notna().all()
    assert res.equity["portfolio_value"].iloc[-1] > INITIAL


def test_higher_costs_reduce_return():
    dates = pd.bdate_range("2024-01-02", periods=120)
    n = len(dates)
    prices = pd.DataFrame(
        {
            "A": 100.0 * (1.002 ** np.arange(n)),
            "B": 100.0 * (1.001 ** np.arange(n)),
        },
        index=dates,
    )
    # Alternating signal forces real turnover every month.
    parity = (dates.month % 2).to_numpy().astype(float)
    signals = pd.DataFrame({"A": parity, "B": 1.0 - parity}, index=dates)

    cheap = run_backtest(
        prices, signals, top_n_by_signal("alt", 1),
        frequency="monthly", cost_bps=0.0, initial_capital=INITIAL,
    )
    pricey = run_backtest(
        prices, signals, top_n_by_signal("alt", 1),
        frequency="monthly", cost_bps=500.0, initial_capital=INITIAL,
    )
    assert len(cheap.trades) > 2  # sanity: turnover actually happened
    assert (
        cheap.equity["portfolio_value"].iloc[-1]
        > pricey.equity["portfolio_value"].iloc[-1]
    )


def test_no_lookahead_on_jump():
    """A signal dated T that 'knows' about a jump on T must trade at T+1."""
    dates = pd.bdate_range("2024-01-02", periods=60)
    jump_day = pd.Timestamp("2024-02-01")  # a month-start => a rebalance date
    assert jump_day in dates
    x = np.where(dates >= jump_day, 200.0, 100.0)
    prices = pd.DataFrame({"X": x, "Y": np.full(len(dates), 100.0)}, index=dates)

    # The signal "knows" the jump, but only with the jump day's date on it.
    sig_x = pd.Series(np.nan, index=dates)
    sig_x.loc[jump_day] = 1.0
    signals = pd.DataFrame({"X": sig_x, "Y": np.nan}, index=dates)

    res = run_backtest(
        prices,
        signals,
        top_n_by_signal("jump", 1),
        frequency="monthly",
        cost_bps=0.0,
        initial_capital=INITIAL,
    )
    buys = res.trades[(res.trades["ticker"] == "X") & (res.trades["action"] == "BUY")]
    assert not buys.empty, "expected the strategy to buy X after the jump signal"
    # The trade happens strictly after the jump date, at the post-jump price.
    assert (buys["date"] > jump_day).all()
    assert np.allclose(buys["price"].to_numpy(), 200.0)
    # The jump itself (100 -> 200) was NOT captured: bought at 200, still 200.
    assert res.equity["portfolio_value"].iloc[-1] == pytest.approx(INITIAL)


def test_long_only_invariant():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=120)
    tickers = ["A", "B", "C", "D"]
    data = {}
    for t in tickers:
        rets = rng.normal(0.0005, 0.02, len(dates))
        data[t] = 100.0 * np.cumprod(1.0 + rets)
    prices = pd.DataFrame(data, index=dates)
    # Point-in-time signal by construction (uses only past prices).
    signals = pd.DataFrame(
        {t: prices[t].pct_change(20).fillna(0.0) for t in tickers}, index=dates
    )

    res = run_backtest(
        prices,
        signals,
        top_n_by_signal("s", 2),
        frequency="monthly",
        cost_bps=10.0,
        initial_capital=INITIAL,
    )
    assert res.target_weights, "expected at least one rebalance"
    for day, weights in res.target_weights.items():
        assert all(w >= 0 for w in weights.values()), f"negative weight on {day}"
        assert sum(weights.values()) <= 1.0 + 1e-9, f"weights sum > 1 on {day}"
    assert (res.equity["portfolio_value"] >= 0).all()
    assert res.equity["portfolio_value"].notna().all()


def test_rebalance_date_schedules():
    dates = pd.bdate_range("2024-01-02", periods=90)
    monthly = rebalance_dates(dates, "monthly")
    weekly = rebalance_dates(dates, "weekly")
    # First trading day of each calendar month present in the data.
    assert monthly[0] == pd.Timestamp("2024-01-02")
    assert monthly[1] == pd.Timestamp("2024-02-01")
    assert monthly[2] == pd.Timestamp("2024-03-01")
    assert len(weekly) > len(monthly)
    with pytest.raises(ValueError):
        rebalance_dates(dates, "yearly")


def test_metrics_explanations_present():
    prices, foresight = _trend_setup()
    res = run_backtest(
        prices,
        foresight,
        top_n_by_signal("foresight", 1),
        frequency="monthly",
        cost_bps=10.0,
        initial_capital=INITIAL,
    )
    metrics = compute_metrics(res.equity, res.trades)
    expected = {
        "final_value", "total_return", "cagr", "volatility", "sharpe",
        "max_drawdown", "turnover", "win_rate", "num_trades",
    }
    assert expected <= set(metrics.keys())
    for key, m in metrics.items():
        # Every metric carries a plain-English explanation for non-experts.
        assert {"value", "label", "what", "why", "good"} <= set(m.keys()), key
        assert m["label"] and m["what"] and m["why"] and m["good"], key
    assert metrics["final_value"]["value"] > 0


# ---------------------------------------------------------------------------
# Transaction cost models (Feature 2): FlatCostModel and SpreadCostModel.
# ---------------------------------------------------------------------------


def _synthetic_bars(dates, close, high_low_pct=0.01, volume=1_000_000.0):
    """Fake OHLCV bars: close series with symmetric high/low ranges."""
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * (1.0 + high_low_pct),
            "low": close * (1.0 - high_low_pct),
            "close": close,
            "volume": np.full(len(dates), volume),
        },
        index=pd.DatetimeIndex(dates),
    )


def test_flat_cost_model_rate():
    model = FlatCostModel(10.0)
    assert isinstance(model, CostModel)
    assert model.name == "flat"
    day = pd.Timestamp("2024-03-01")
    # 10 bps = 0.10% = 0.001 of notional, regardless of ticker/day/size.
    assert model.rate("AAA", day, 50_000.0) == pytest.approx(0.001)
    assert model.rate("ZZZ", day, 1.0) == pytest.approx(0.001)
    assert FlatCostModel(0.0).rate("AAA", day, 1.0) == pytest.approx(0.0)
    assert FlatCostModel(25.5).rate("AAA", day, 1.0) == pytest.approx(0.00255)
    with pytest.raises(ValueError):
        FlatCostModel(-1.0)


def test_spread_rates_within_bounds():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=60)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, len(dates)))
    model = SpreadCostModel({"AAA": _synthetic_bars(dates, close, 0.02)})
    assert model.name == "spread"
    for day in (dates[5], dates[20], dates[-1]):
        for trade_value in (100.0, 50_000.0, 5_000_000.0):
            rate = model.rate("AAA", day, trade_value)
            assert 0.0 <= rate <= 0.02, (day, trade_value, rate)


def test_spread_charges_more_for_volatile_stock():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-02", periods=60)
    n = len(dates)
    calm = _synthetic_bars(dates, np.full(n, 100.0), high_low_pct=0.005)
    wild_close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, n))
    wild = _synthetic_bars(dates, wild_close, high_low_pct=0.03)
    model = SpreadCostModel({"CALM": calm, "WILD": wild})
    day = dates[-1]
    trade_value = 100_000.0
    calm_rate = model.rate("CALM", day, trade_value)
    wild_rate = model.rate("WILD", day, trade_value)
    assert 0.0 <= calm_rate <= 0.02
    assert 0.0 <= wild_rate <= 0.02
    # Same ~$100 price level, but the jumpy stock costs more to trade.
    assert wild_rate > calm_rate


def test_spread_falls_back_with_too_little_history():
    dates = pd.bdate_range("2024-01-02", periods=1)
    model = SpreadCostModel({"AAA": _synthetic_bars(dates, [100.0])})
    # One day of history is not enough to estimate anything: neutral 10 bps.
    assert model.rate("AAA", dates[0], 10_000.0) == pytest.approx(0.001)
    # Unknown tickers and empty histories fall back too, never crash.
    assert model.rate("NOPE", dates[0], 10_000.0) == pytest.approx(0.001)
    assert SpreadCostModel({}).rate("AAA", dates[0], 1.0) == pytest.approx(0.001)


def test_spread_rate_never_peeks_at_future_bars():
    rng = np.random.default_rng(13)
    dates = pd.bdate_range("2024-01-02", periods=60)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, len(dates)))
    full = _synthetic_bars(dates, close, high_low_pct=0.015)
    cutoff = dates[29]
    truncated = full.loc[:cutoff]
    model_with_future = SpreadCostModel({"AAA": full})
    model_without_future = SpreadCostModel({"AAA": truncated})
    # The rate on the cutoff day must be bit-identical whether or not later
    # bars exist in the database.
    assert model_with_future.rate("AAA", cutoff, 25_000.0) == (
        model_without_future.rate("AAA", cutoff, 25_000.0)
    )


def test_flat_model_is_default_and_results_unchanged():
    prices, foresight = _trend_setup()
    strategy = top_n_by_signal("foresight", 1)
    default = run_backtest(
        prices, foresight, strategy,
        frequency="monthly", cost_bps=10.0, initial_capital=INITIAL,
    )
    explicit_flat = run_backtest(
        prices, foresight, strategy,
        frequency="monthly", cost_bps=10.0, initial_capital=INITIAL,
        cost_model=FlatCostModel(10.0),
    )
    assert default.cost_model_name == "flat"
    assert explicit_flat.cost_model_name == "flat"
    # Identical inputs => identical outputs: the new cost_model plumbing
    # did not change a single number for the default flat model.
    pd.testing.assert_frame_equal(default.equity, explicit_flat.equity)
    pd.testing.assert_frame_equal(default.trades, explicit_flat.trades)


def test_zero_trades_means_zero_costs():
    prices, _ = _trend_setup()
    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    do_nothing = lambda sig, day: {}  # noqa: E731 - tiny test strategy

    for model in (None, FlatCostModel(10.0),
                  SpreadCostModel({t: _synthetic_bars(prices.index, prices[t])
                                   for t in prices.columns})):
        res = run_backtest(
            prices, signals, do_nothing,
            frequency="monthly", cost_bps=500.0, initial_capital=INITIAL,
            cost_model=model,
        )
        assert res.trades["cost"].sum() == 0.0
        assert res.equity["portfolio_value"].iloc[-1] == pytest.approx(INITIAL)


def test_spread_cost_model_runs_in_backtest():
    rng = np.random.default_rng(21)
    dates = pd.bdate_range("2024-01-02", periods=120)
    prices = pd.DataFrame(
        {
            "A": 100.0 * np.cumprod(1.0 + rng.normal(0.002, 0.02, len(dates))),
            "B": 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, len(dates))),
        },
        index=dates,
    )
    bars = {
        t: _synthetic_bars(dates, prices[t].to_numpy(), high_low_pct=0.02)
        for t in prices.columns
    }
    signals = pd.DataFrame(
        {t: prices[t].pct_change(20).fillna(0.0) for t in prices.columns},
        index=dates,
    )
    res = run_backtest(
        prices, signals, top_n_by_signal("s", 1),
        frequency="monthly", initial_capital=INITIAL,
        cost_model=SpreadCostModel(bars),
    )
    assert res.cost_model_name == "spread"
    assert len(res.trades) > 0, "expected real turnover"
    assert (res.trades["cost"] >= 0).all()
    notional = res.trades["shares"] * res.trades["price"]
    assert (res.trades["cost"] <= notional * 0.02 + 1e-9).all()


def test_cli_cost_model_spread(tmp_path, capsys):
    """The --cost-model spread flag runs end to end on a tiny synthetic DB."""
    from qrp.backtest.cli import main as cli_main
    from qrp.data.store import upsert_bars
    from qrp.signals.store import upsert_signals

    db = tmp_path / "spread_test.db"
    dates = pd.bdate_range("2024-01-02", periods=90)
    rng = np.random.default_rng(31)
    for t in ("AAA", "BBB"):
        close = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.015, len(dates)))
        upsert_bars(
            pd.DataFrame(
                {
                    # Stored as YYYY-MM-DD strings, like the real pipeline.
                    "Date": dates.strftime("%Y-%m-%d"),
                    "Open": close,
                    "High": close * 1.01,
                    "Low": close * 0.99,
                    "Close": close,
                    "Volume": np.full(len(dates), 1_000_000),
                }
            ),
            t,
            path=db,
        )
        upsert_signals(
            pd.Series(rng.normal(0.0, 1.0, len(dates)), index=dates),
            t,
            "momentum",
            path=db,
        )

    rc = cli_main(
        [
            "--db", str(db),
            "--tickers", "AAA,BBB",
            "--signal", "momentum",
            "--cost-model", "spread",
            "--start", "2024-01-02",
            "--end", "2024-05-10",
            "--rebalance", "monthly",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "spread" in out


def test_rebalance_dates_weekly_chronological_across_years():
    """Regression: weekly labels once sorted as TEXT ("W10" < "W2"), so only
    ~11 rebalances ever fired and weekly backtests silently became
    buy-and-hold. Dates must come out in calendar order."""
    dates = pd.bdate_range("2014-01-02", "2016-12-30")
    weekly = rebalance_dates(dates, "weekly")
    assert weekly == sorted(weekly), "weekly rebalance dates must be chronological"
    # ~156 weeks over 3 years; crosses the W9/W10 boundary and year ends.
    assert 140 <= len(weekly) <= 165
    gaps = pd.Series(weekly).diff().dropna().dt.days
    assert (gaps >= 4).all() and (gaps <= 10).all(), "expected ~7-day spacing"


def test_rebalance_dates_daily_is_every_trading_day():
    dates = pd.bdate_range("2024-01-02", periods=60)
    daily = rebalance_dates(dates, "daily")
    assert daily == list(dates)
    with pytest.raises(ValueError):
        rebalance_dates(dates, "yearly")


def test_weekly_backtest_rebalances_throughout():
    """End-to-end: a weekly backtest must keep rebalancing for years, not
    just the first ~11 weeks."""
    dates = pd.bdate_range("2015-01-02", "2020-12-31")
    n = len(dates)
    prices = pd.DataFrame(
        {
            "A": 100.0 * (1.001 ** np.arange(n)),
            "B": 100.0 * (0.999 ** np.arange(n)),
        },
        index=dates,
    )
    # Signal flips every quarter so every rebalance should trade.
    quarter = (dates.quarter % 2).to_numpy().astype(float)
    signals = pd.DataFrame({"A": quarter, "B": 1.0 - quarter}, index=dates)
    res = run_backtest(
        prices, signals, top_n_by_signal("q", 1),
        frequency="weekly", cost_bps=0.0, initial_capital=INITIAL,
    )
    rb_years = {d[:4] for d in res.target_weights}
    assert {"2015", "2017", "2020"} <= rb_years
    assert len(res.trades) > 40, "far more than the ~11 early rebalances"


def test_daily_rebalance_no_churn_without_signal_changes():
    """Daily re-evaluation + drift buffer: a static signal must not churn."""
    dates = pd.bdate_range("2024-01-02", periods=120)
    n = len(dates)
    prices = pd.DataFrame(
        {"A": 100.0 * (1.0005 ** np.arange(n)), "B": np.full(n, 100.0)},
        index=dates,
    )
    signals = pd.DataFrame(
        {"A": np.full(n, 1.0), "B": np.full(n, 0.5)}, index=dates
    )
    res = run_backtest(
        prices, signals, top_n_by_signal("s", 1),
        frequency="daily", cost_bps=10.0, initial_capital=INITIAL,
        drift_tolerance=0.02,
    )
    # One initial buy; the unchanging signal never drifts enough to re-trade.
    buys = res.trades[res.trades["action"] == "BUY"]
    assert len(buys) <= 2


def _old_corwin_schultz_loop(hist):
    """Reference: the pre-vectorization Python loop (audit 3.3)."""
    import math

    highs = hist["high"].to_numpy(dtype=float)
    lows = hist["low"].to_numpy(dtype=float)
    betas, gammas = [], []
    for h1, l1, h2, l2 in zip(highs[:-1], lows[:-1], highs[1:], lows[1:]):
        if not all(math.isfinite(v) and v > 0 for v in (h1, l1, h2, l2)):
            continue
        if min(l1, l2) <= 0:
            continue
        betas.append(math.log(h1 / l1) ** 2 + math.log(h2 / l2) ** 2)
        gammas.append(math.log(max(h1, h2) / min(l1, l2)) ** 2)
    if not betas:
        return 0.0
    beta = sum(betas) / len(betas)
    gamma = sum(gammas) / len(gammas)
    denom = 3.0 - 2.0 * math.sqrt(2.0)
    alpha = ((math.sqrt(2.0 * beta) - math.sqrt(beta)) / denom
             - math.sqrt(gamma / denom))
    return max(2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha)), 0.0)


def test_corwin_schultz_vectorized_matches_loop():
    """Audit 3.3: vectorized Corwin-Schultz == old loop to 1e-12."""
    import numpy as np
    import pandas as pd

    from qrp.backtest.costs import SpreadCostModel

    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(50):
        n = int(rng.integers(2, 60))
        h = 100.0 + np.abs(rng.normal(0, 2, n))
        l = 100.0 - np.abs(rng.normal(0, 2, n))
        if trial % 3 == 0:
            h[rng.integers(0, n)] = np.nan
        if trial % 4 == 0:
            l[rng.integers(0, n)] = 0.0
        hist = pd.DataFrame({"high": h, "low": l, "close": (h + l) / 2})
        diff = abs(
            _old_corwin_schultz_loop(hist)
            - SpreadCostModel._corwin_schultz_spread(hist)
        )
        worst = max(worst, diff)
        assert diff < 1e-12, (trial, diff)
    assert worst < 1e-12


def test_corwin_schultz_vectorized_is_faster():
    """Audit 3.3: the vectorized estimator is substantially faster."""
    import time

    import numpy as np
    import pandas as pd

    from qrp.backtest.costs import SpreadCostModel

    rng = np.random.default_rng(1)
    n = 252
    h = 100 + np.abs(rng.normal(0, 2, n))
    l = 100 - np.abs(rng.normal(0, 2, n))
    hist = pd.DataFrame({"high": h, "low": l, "close": (h + l) / 2})
    reps = 300
    t0 = time.perf_counter()
    for _ in range(reps):
        _old_corwin_schultz_loop(hist)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(reps):
        SpreadCostModel._corwin_schultz_spread(hist)
    t_new = time.perf_counter() - t0
    assert t_old > 0.1, "synthetic data too small to time reliably"
    assert t_old / t_new >= 5.0, f"speedup only {t_old / t_new:.1f}x"


def test_prefilled_signal_matrix_matches_per_date_ffill():
    """Audit 2.6/1.2: pre-filling once == per-date ffill, but faster."""
    import time

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=1250)
    tickers = [f"S{i:03d}" for i in range(200)]
    vals = rng.normal(size=(1250, 200))
    vals[rng.random(size=(1250, 200)) < 0.10] = np.nan
    signals = pd.DataFrame(vals, index=dates, columns=tickers)
    rb_dates = dates[::21][:60]

    t0 = time.perf_counter()
    olds = [signals.loc[:d].ffill().iloc[-1] for d in rb_dates]
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    filled = signals.ffill()
    news = [filled.loc[:d].iloc[-1] for d in rb_dates]
    t_new = time.perf_counter() - t0

    for a, b in zip(olds, news):
        pd.testing.assert_series_equal(a, b)
    assert t_old / t_new >= 5.0, f"speedup only {t_old / t_new:.1f}x"
