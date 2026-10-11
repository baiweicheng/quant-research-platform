"""Tests for the Oct-2026 backtest-realism build (audit fixes).

Covers:
  (a) point-in-time S&P 500 membership spot-checks on known changes;
  (b) the engine never buys a stock before it joined the index;
  (c) a delisted holding is closed at its last real price without crashing;
  (d) new defaults: 3% max weight, spread costs, 2013 data start, monthly
      registry presets;
  (e) paper trading is untouched by the preset changes (still manual);
  (f) the Compare "Strategy defaults" table renders all nine strategies;
  plus the Compare frequency-override dropdown (uniform override + cache
  key invalidation).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import qrp.ui.helpers as H
from qrp.backtest.engine import last_valid_dates, run_backtest
from qrp.backtest.strategies import top_n_by_signal
from qrp.data.sp500_history import (
    eligible_on,
    history_available,
    is_known,
    make_universe_filter,
    members_on,
)
from qrp.data.store import upsert_bars
from qrp.signals.store import upsert_signals
from qrp.strategies.registry import STRATEGIES


# ---------------------------------------------------------------------------
# (a) membership table spot-checks
# ---------------------------------------------------------------------------

def test_history_available():
    assert history_available(), "sp500_history.csv must ship with the repo"


def test_2024_addition_absent_from_2020_universe():
    # Palantir joined 2024-09-23.
    assert "PLTR" not in members_on("2020-06-01")
    assert "PLTR" in members_on("2025-06-01")
    assert not eligible_on("PLTR", "2020-06-01")
    assert eligible_on("PLTR", "2025-06-01")


def test_2023_removal_present_then_absent():
    # Activision Blizzard was acquired and left 2023-10-18.
    assert "ATVI" in members_on("2020-06-01")
    assert "ATVI" not in members_on("2024-01-01")
    assert "ATVI" not in members_on("2025-06-01")


def test_bankruptcy_removal():
    # Lehman Brothers filed for bankruptcy 2008-09-16.
    assert "LEH" in members_on("2008-01-01")
    assert "LEH" not in members_on("2009-01-01")


def test_universe_size_sane():
    assert 480 <= len(members_on("2015-06-01")) <= 530
    assert 480 <= len(members_on("2020-06-01")) <= 530
    assert 480 <= len(members_on("2025-06-01")) <= 530


def test_unknown_tickers_pass_through():
    # Custom tickers the table never heard of are always eligible.
    assert not is_known("ZZZZ_FAKE")
    assert eligible_on("ZZZZ_FAKE", "2020-06-01")
    uf = make_universe_filter(["AAA", "ZZZZ_FAKE"])
    assert uf is not None
    assert uf(pd.Timestamp("2020-06-01")) == {"AAA", "ZZZZ_FAKE"}


def test_ticker_rename_continuous():
    # Meta renamed FB->META in 2022; membership is one continuous interval.
    assert "META" in members_on("2015-06-01")
    assert "META" in members_on("2025-06-01")
    assert "FB" not in members_on("2025-06-01")


# ---------------------------------------------------------------------------
# (b) engine: no lookahead through the membership filter
# ---------------------------------------------------------------------------

def _two_stock_setup(periods=90):
    dates = pd.bdate_range("2024-01-02", periods=periods)
    prices = pd.DataFrame({"AAA": 100.0, "BBB": 100.0}, index=dates)
    # BBB always scores higher — without a filter it would be bought first.
    signals = pd.DataFrame({"AAA": 0.1, "BBB": 0.9}, index=dates)
    return dates, prices, signals


def test_universe_filter_blocks_future_members():
    dates, prices, signals = _two_stock_setup()
    mid = pd.Timestamp("2024-02-15")
    uf = lambda day: {"AAA", "BBB"} if day >= mid else {"AAA"}  # noqa: E731
    res = run_backtest(
        prices, signals, top_n_by_signal("s", 1),
        frequency="monthly", cost_bps=0.0, universe_filter=uf,
    )
    bbb_trades = res.trades[res.trades["ticker"] == "BBB"]
    assert not bbb_trades.empty  # BBB is bought once eligible...
    assert (bbb_trades["date"] >= mid).all()  # ...but never before
    # AAA was held in the early stretch instead.
    assert not res.trades[res.trades["ticker"] == "AAA"].empty


def test_no_filter_keeps_old_behavior():
    dates, prices, signals = _two_stock_setup()
    res = run_backtest(
        prices, signals, top_n_by_signal("s", 1),
        frequency="monthly", cost_bps=0.0,
    )
    first = res.trades.iloc[0]
    assert first["ticker"] == "BBB"  # highest scorer, bought immediately


# ---------------------------------------------------------------------------
# (c) delisted holdings close at the last real price
# ---------------------------------------------------------------------------

def test_delisted_position_closed_at_last_price():
    dates = pd.bdate_range("2024-01-02", periods=60)
    # BBB's real history ends at dates[39]; the loader would forward-fill.
    raw = [100.0 + i for i in range(40)] + [float("nan")] * 20
    prices = pd.DataFrame({"AAA": 100.0, "BBB": raw}, index=dates).ffill()
    last_valid = {"AAA": dates[-1], "BBB": dates[39]}

    def always_bbb(_signals, _day):
        return {"BBB": 0.5}

    res = run_backtest(
        prices, pd.DataFrame(index=dates), always_bbb,
        frequency="monthly", cost_bps=0.0, last_valid=last_valid,
    )
    bbb = res.trades[res.trades["ticker"] == "BBB"]
    buys = bbb[bbb["action"] == "BUY"]
    sells = bbb[bbb["action"] == "SELL"]
    assert not buys.empty
    assert (buys["date"] <= dates[39]).all(), "never buy after delisting"
    assert not sells.empty, "the delisted holding must be closed"
    # Closed at the last real price (139.0), not a forward-filled ghost.
    assert sells["price"].iloc[-1] == pytest.approx(139.0)
    # Fully out by the end: every bought share was sold.
    assert sells["shares"].sum() == pytest.approx(buys["shares"].sum())


def test_last_valid_dates_helper():
    dates = pd.bdate_range("2024-01-02", periods=10)
    valid = pd.DataFrame(
        {
            "AAA": [True] * 10,
            "BBB": [True] * 6 + [False] * 4,
            "CCC": [False] * 10,
        },
        index=dates,
    )
    out = last_valid_dates(valid)
    assert out["AAA"] == dates[-1]
    assert out["BBB"] == dates[5]
    assert "CCC" not in out


# ---------------------------------------------------------------------------
# (d) new defaults
# ---------------------------------------------------------------------------

def test_spread_is_default_cost_model():
    for fn in (H.run_strategy_backtest_for_ui, H.run_backtest_for_ui):
        assert (
            inspect.signature(fn).parameters["cost_model"].default == "spread"
        ), fn.__name__
    assert (
        inspect.signature(H.compare_strategies_for_ui)
        .parameters["cost_model"]
        .default
        == "spread"
    )


def test_backtest_cli_defaults():
    from qrp.backtest.cli import parse_args as bt_args
    from qrp.data.cli import parse_args as data_args

    assert bt_args([]).cost_model == "spread"
    assert data_args([]).start == "2013-01-01"


def test_backtest_cli_default_max_weight_is_5pct():
    """The global default max weight is 5% (~20 stocks)."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "qrp.backtest.cli", "--help"],
        capture_output=True, text=True, cwd=".",
    ).stdout
    assert "default: 0.05" in out
    assert "~20 stocks" in out


def test_registry_all_monthly():
    for name, s in STRATEGIES.items():
        assert s.rebalance == "monthly", name


def test_compare_frequency_override_and_cache_key():
    base = dict(
        strategy_names=["momentum"], tickers=["AAA"],
        start="2020-01-01", end="2021-01-01", capital=100_000.0,
        max_weight=0.03, cost_bps=10.0, execution="open",
    )
    keys = {
        H.compare_cache_key(**base),
        H.compare_cache_key(**base, frequency="weekly"),
        H.compare_cache_key(**base, frequency="monthly"),
        H.compare_cache_key(**base, cost_model="flat"),
    }
    assert len(keys) == 4, "frequency/cost-model must change the cache key"


# ---------------------------------------------------------------------------
# (e) paper trading untouched
# ---------------------------------------------------------------------------

def test_paper_trading_independent_of_registry():
    for mod in ("qrp/execution/paper.py", "qrp/execution/rebalance.py"):
        src = (Path(__file__).resolve().parent.parent / mod).read_text()
        assert "qrp.strategies" not in src, mod
    # Paper trading is manual: no rebalancing frequency comes from presets.
    sig = inspect.signature(H.resolve_paper_timing)
    assert "frequency" not in sig.parameters
    assert "rebalance" not in sig.parameters


# ---------------------------------------------------------------------------
# (f) strategy defaults table
# ---------------------------------------------------------------------------

def test_strategy_preset_rows_all_nine():
    rows = H.strategy_preset_rows()
    assert len(rows) == 14
    assert set(rows[0].keys()) == {
        "Strategy", "Stocks", "Rebalance", "Weighting", "Max weight",
    }
    by_name = {r["Strategy"]: r for r in rows}
    momentum = by_name["12-1 Month Cross-Sectional Momentum"]
    assert momentum["Stocks"] == "~50"
    assert momentum["Rebalance"] == "Monthly"
    assert momentum["Weighting"] == "Equal weight"
    assert momentum["Max weight"] == "2%"
    for r in rows:
        assert r["Rebalance"] == "Monthly"


def test_realism_footnote_plain_english():
    note = H.realism_footnote(cost_model_name="spread")
    for phrase in (
        "How realistic are these numbers?",
        "next day's open",
        "dividends are credited",
        "market impact",
        "bankruptcies",
        "Past performance never predicts the future",
    ):
        assert phrase in note, phrase
    assert "traceback" not in note.lower()


# ---------------------------------------------------------------------------
# Compare frequency override applies to every strategy
# ---------------------------------------------------------------------------

def _seed_compare_db(path) -> None:
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    for ticker, base in (("AAA", 100.0), ("BBB", 50.0)):
        upsert_bars(
            pd.DataFrame(
                {
                    "Date": dates.strftime("%Y-%m-%d"),
                    "Open": base,
                    "High": base * 1.01,
                    "Low": base * 0.99,
                    "Close": base,
                    "Volume": 1_000,
                }
            ),
            ticker,
            path,
        )
    for sig in ("momentum", "trend", "low_volatility"):
        upsert_signals(pd.Series([0.5] * 6, index=dates), "AAA", sig, path)
        upsert_signals(pd.Series([0.3] * 6, index=dates), "BBB", sig, path)


def test_frequency_override_uniform(tmp_path, monkeypatch):
    db = str(tmp_path / "f.db")
    _seed_compare_db(db)
    captured: list[str] = []
    real_run = H.run_backtest

    def spy(prices, signals, strategy, **kw):
        captured.append(kw.get("frequency"))
        return real_run(prices, signals, strategy, **kw)

    monkeypatch.setattr(H, "run_backtest", spy)
    for name in ("momentum", "trend", "low_volatility"):
        H._race_one_strategy(
            name, ["AAA", "BBB"], "2024-01-02", "2024-01-08",
            100_000.0, 0.5, 10.0, db, frequency="weekly",
        )
    assert captured == ["weekly"] * 3
    # Without the override each strategy keeps its registry preset.
    captured.clear()
    H._race_one_strategy(
        "momentum", ["AAA", "BBB"], "2024-01-02", "2024-01-08",
        100_000.0, 0.5, 10.0, db, frequency=None,
    )
    assert captured == ["monthly"]


# ---------------------------------------------------------------------------
# (e) bankruptcy-to-zero (backlog #7)
# ---------------------------------------------------------------------------

def test_likely_bankrupt_heuristic():
    from qrp.backtest.engine import likely_bankrupt

    dates = pd.bdate_range("2023-01-02", periods=300)
    # Wipeout: 100 -> 0.5 (99.5% below the high) -> bankrupt.
    wiped = pd.DataFrame(
        {"T": list(np.linspace(100.0, 100.0, 200))
         + list(np.linspace(100.0, 0.5, 100))},
        index=dates,
    )
    assert likely_bankrupt("T", dates[299], wiped) is True
    # Distressed takeover: 100 -> 4 (96% down) -> NOT bankrupt (AUD-05:
    # the 99% bar keeps real deal proceeds from being zeroed).
    distressed = pd.DataFrame(
        {"T": list(np.linspace(100.0, 100.0, 200))
         + list(np.linspace(100.0, 4.0, 100))},
        index=dates,
    )
    assert likely_bankrupt("T", dates[299], distressed) is False
    # Merger-like: ends at 85% of its high -> not bankrupt.
    merged = pd.DataFrame(
        {"T": list(np.linspace(100.0, 100.0, 200))
         + list(np.linspace(100.0, 85.0, 100))},
        index=dates,
    )
    assert likely_bankrupt("T", dates[299], merged) is False
    # Too little history -> conservative False (old behavior).
    short = pd.DataFrame({"T": [100.0, 4.0]}, index=dates[:2])
    assert likely_bankrupt("T", dates[1], short) is False
    # Unknown ticker -> False, never crashes.
    assert likely_bankrupt("NOPE", dates[299], wiped) is False


def _crash_delisting_setup():
    dates = pd.bdate_range("2023-01-02", periods=300)
    # BBB: flat at 100 for 200 days, crashes to 0.5 (99.5% down), then
    # delists. AUD-05: the 99% bar means a 96%-down distressed takeover
    # now exits at the deal price; only a true wipeout goes to zero.
    bbb = [100.0] * 200 + list(np.linspace(100.0, 0.5, 60)) + [float("nan")] * 40
    prices = pd.DataFrame({"AAA": 100.0, "BBB": bbb}, index=dates).ffill()
    last_valid = {"AAA": dates[-1], "BBB": dates[259]}
    return dates, prices, last_valid


def test_bankrupt_delisting_written_to_zero():
    dates, prices, last_valid = _crash_delisting_setup()

    def always_bbb(_signals, _day):
        return {"BBB": 0.5}

    res = run_backtest(
        prices, pd.DataFrame(index=dates), always_bbb,
        frequency="monthly", cost_bps=0.0, last_valid=last_valid,
    )
    bbb = res.trades[res.trades["ticker"] == "BBB"]
    sells = bbb[bbb["action"] == "SELL"]
    assert not sells.empty
    final = sells.iloc[-1]
    assert final["price"] == 0.0, "bankruptcy writes the position to zero"
    assert "bankruptcy" in str(final["reason"])
    # No phantom proceeds: the wipeout adds no cash.
    assert sells["shares"].sum() == pytest.approx(
        bbb[bbb["action"] == "BUY"]["shares"].sum()
    )


def test_merger_like_delisting_still_exits_at_last_close():
    dates = pd.bdate_range("2023-01-02", periods=300)
    # CCC: mild dip to 85, then delists (a take-private, not a wipeout).
    ccc = [100.0] * 200 + list(np.linspace(100.0, 85.0, 60)) + [float("nan")] * 40
    prices = pd.DataFrame({"AAA": 100.0, "CCC": ccc}, index=dates).ffill()
    last_valid = {"AAA": dates[-1], "CCC": dates[259]}

    def always_ccc(_signals, _day):
        return {"CCC": 0.5}

    res = run_backtest(
        prices, pd.DataFrame(index=dates), always_ccc,
        frequency="monthly", cost_bps=0.0, last_valid=last_valid,
    )
    ccc_trades = res.trades[res.trades["ticker"] == "CCC"]
    sells = ccc_trades[ccc_trades["action"] == "SELL"]
    assert not sells.empty
    assert sells["price"].iloc[-1] == pytest.approx(85.0), (
        "a non-wipeout delisting still exits at the last real close"
    )


def test_distressed_takeover_exits_at_deal_price_not_zero():
    """AUD-05: a 96%-down delisting is a distressed takeover, not a
    bankruptcy — the position exits at the $4 deal price, not $0."""
    dates = pd.bdate_range("2023-01-02", periods=300)
    # DDD: flat at 100, slides to 4.0 (96% down), then delists.
    ddd = [100.0] * 200 + list(np.linspace(100.0, 4.0, 60)) + [float("nan")] * 40
    prices = pd.DataFrame({"AAA": 100.0, "DDD": ddd}, index=dates).ffill()
    last_valid = {"AAA": dates[-1], "DDD": dates[259]}

    def always_ddd(_signals, _day):
        return {"DDD": 0.5}

    res = run_backtest(
        prices, pd.DataFrame(index=dates), always_ddd,
        frequency="monthly", cost_bps=0.0, last_valid=last_valid,
    )
    ddd_trades = res.trades[res.trades["ticker"] == "DDD"]
    sells = ddd_trades[ddd_trades["action"] == "SELL"]
    assert not sells.empty
    assert sells["price"].iloc[-1] == pytest.approx(4.0), (
        "a 96%-down delisting exits at the deal price, not zero"
    )
    assert "bankruptcy" not in str(sells["reason"].iloc[-1]).lower()
