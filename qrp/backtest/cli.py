"""CLI: run a long-only backtest of a signal strategy and print the results.

Example:
    python -m qrp.backtest.cli --signal momentum --max-weight 0.1 --rebalance monthly
    python -m qrp.backtest.cli --signal trend --max-weight 0.2 --start 2024-01-01 --cost-bps 5
    python -m qrp.backtest.cli --strategy trending_value --tickers all

Position sizing has a single knob: --max-weight. The number of stocks held
is implied by it (n = round(1 / max_weight), so 0.1 means ~10 stocks).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date

import pandas as pd

from qrp.backtest.costs import SpreadCostModel
from qrp.backtest.engine import load_price_matrix, load_signal_matrix, run_backtest
from qrp.backtest.metrics import compute_metrics
from qrp.backtest.strategies import WEIGHTING_METHODS, weighted_top_n
from qrp.data.store import DEFAULT_DB
from qrp.data.yahoo import normalize_ticker
from qrp.portfolio.construction import METHOD_DESCRIPTIONS, implied_stock_count
from qrp.signals import default_registry
from qrp.signals.store import list_tickers, load_bars
from qrp.strategies import display_name, get_strategy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest a long-only signal strategy."
    )
    p.add_argument(
        "--strategy",
        default=None,
        help=(
            "Named strategy from the registry (e.g. trending_value, "
            "value_momentum, quality_lowvol). Uses the strategy's presets "
            "for signal/weighting/rebalance/max-weight/drift-tolerance; "
            "any of those flags given explicitly still wins. See the "
            "Strategies page in the dashboard to compare them."
        ),
    )
    # Defaults are None so an explicit flag can be told apart from "not
    # given" — the strategy preset (or the classic default) fills the gaps.
    p.add_argument(
        "--signal",
        default=None,
        help="Signal name to rank stocks by (default: momentum)",
    )
    p.add_argument(
        "--weighting",
        default=None,
        choices=sorted(WEIGHTING_METHODS),
        help=(
            "How to size positions: equal (even split), signal (more money to "
            "higher scores), inverse_vol (more money to calmer stocks). "
            "Default: equal"
        ),
    )
    p.add_argument(
        "--max-weight",
        type=float,
        default=None,
        help=(
            "No single stock may exceed this fraction of capital, e.g. 0.1 = "
            "10%% (default: 0.1). This one setting also decides how many "
            "stocks are held: n = round(1 / max-weight), so 0.1 means "
            "~10 stocks."
        ),
    )
    p.add_argument(
        "--rebalance",
        default=None,
        choices=["daily", "weekly", "monthly"],
        help=(
            "How often the portfolio is re-evaluated: daily (the default — "
            "the drift buffer still decides whether anything actually "
            "trades), weekly, or monthly."
        ),
    )
    p.add_argument(
        "--drift-tolerance",
        type=float,
        default=None,
        help=(
            "Skip trades that would move a position by less than this "
            "fraction of portfolio value, e.g. 0.02 = 2%%. Avoids paying "
            "trading costs on tiny wobbles. Default: the strategy's own "
            "setting with --strategy, otherwise 0.0 (trade at every "
            "rebalance, exactly as before)."
        ),
    )
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD (inclusive)")
    p.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="End date YYYY-MM-DD (default: today)",
    )
    p.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help="Trading cost in basis points per trade, 10 = 0.10%% (default: 10). "
        "Used with --cost-model flat.",
    )
    p.add_argument(
        "--cost-model",
        default="flat",
        choices=["flat", "spread"],
        help=(
            "'flat' charges a fixed fee per trade (10 = 0.10%% — the default, "
            "keeps old results identical); 'spread' estimates the real cost "
            "from each stock's recent high-low price ranges plus a "
            "volatility-based impact for large trades — more realistic, "
            "needs enough price history."
        ),
    )
    p.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Starting cash in dollars (default: 100000)",
    )
    p.add_argument(
        "--tickers",
        default="all",
        help="Comma-separated US tickers, or 'all' in the db (default: all)",
    )
    p.add_argument("--db", default=str(DEFAULT_DB), help="SQLite path")
    return p.parse_args(argv)


def _fmt_pct(value: float) -> str:
    if value != value:  # NaN
        return "n/a"
    return f"{value:+.2%}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry = default_registry()

    # Resolve the strategy: a named preset, optionally tweaked by explicit
    # flags. Without --strategy, the classic single-signal defaults apply.
    strategy = None
    if args.strategy:
        try:
            strategy = get_strategy(args.strategy)
        except KeyError:
            from qrp.strategies import list_strategies

            available = ", ".join(n for n, _ in list_strategies())
            print(f"Unknown strategy {args.strategy!r}. Available: {available}")
            return 1
        tweaks: dict = {}
        if args.signal:
            tweaks["signals"] = [(args.signal, 1.0)]
        if args.weighting:
            tweaks["weighting"] = args.weighting
        if args.rebalance:
            tweaks["rebalance"] = args.rebalance
        if args.max_weight is not None:
            tweaks["max_weight"] = args.max_weight
        if args.drift_tolerance is not None:
            tweaks["drift_tolerance"] = args.drift_tolerance
        if tweaks:
            strategy = replace(strategy, **tweaks)

    signal_name = args.signal or (
        strategy.signals[0][0] if strategy and len(strategy.signals) == 1 else None
    )
    if signal_name is None and strategy is None:
        signal_name = "momentum"
    if signal_name is not None:
        try:
            registry.get(signal_name)
        except KeyError:
            available = ", ".join(n for n, _ in registry.list_signals())
            print(f"Unknown signal {signal_name!r}. Available: {available}")
            return 1

    weighting_key = args.weighting or (strategy.weighting if strategy else "equal")
    rebalance = args.rebalance or (strategy.rebalance if strategy else "daily")
    max_weight = (
        args.max_weight
        if args.max_weight is not None
        else (strategy.max_weight if strategy else 0.10)
    )
    # One sizing knob: the stock count is implied by the max-weight cap
    # (n = round(1 / max_weight)), so --top/--max-positions are gone.
    n = implied_stock_count(max_weight)
    # Without --strategy, drift_tolerance stays 0.0 (the engine default):
    # classic backtests behave exactly as they did before strategies
    # existed. Named strategies bring their own preset instead.
    drift_tolerance = (
        args.drift_tolerance
        if args.drift_tolerance is not None
        else (strategy.drift_tolerance if strategy else 0.0)
    )

    if args.tickers.strip().lower() == "all":
        tickers = list_tickers(args.db)
    else:
        tickers = [normalize_ticker(t) for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print("No tickers found (is daily_bars populated?).")
        return 1

    prices = load_price_matrix(tickers, start=args.start, end=args.end, path=args.db)
    if prices.empty:
        print("No price data in range.")
        return 1

    method = WEIGHTING_METHODS[weighting_key]
    vol_matrix = None
    if method == "inverse_volatility":
        # The stored low_volatility signal IS the negative of recent price
        # volatility, so negating it gives point-in-time volatilities.
        low_vol = load_signal_matrix(
            "low_volatility", tickers, start=args.start, end=args.end, path=args.db
        )
        if low_vol.empty:
            print(
                "No stored low_volatility values in range — needed for "
                "inverse-vol weighting. Run the signals CLI first."
            )
            return 1
        vol_matrix = -low_vol

    if strategy is not None and signal_name is None:
        # Multi-signal strategy: blend the signals' cross-sectional ranks.
        from qrp.strategies import make_strategy_fn

        strategy_fn = make_strategy_fn(
            strategy, tickers, start=args.start, end=args.end, db_path=args.db
        )
        label = f"strategy '{display_name(strategy.name)}'"
        what = f"Blended signals: {', '.join(s for s, _ in strategy.signals)}"
        # No signal matrix to load here: the strategy function carries its
        # own pre-loaded point-in-time matrices, and the engine only needs
        # the date grid — an empty frame with the right index aligns it.
        signals = pd.DataFrame(index=prices.index)
    else:
        signals = load_signal_matrix(
            signal_name, tickers, start=args.start, end=args.end, path=args.db
        )
        if signals.empty:
            print(f"No stored values for signal {signal_name!r} in range.")
            return 1
        strategy_fn = weighted_top_n(
            signal_name, n, method=method, max_weight=max_weight,
            vol_matrix=vol_matrix,
        )
        label = f"top {n} by '{signal_name}' ({weighting_key} weighting)"
        what = f"Weighting: {METHOD_DESCRIPTIONS[method]}"

    # Cost model: "flat" is the default flat fee per trade (old behavior,
    # unchanged); "spread" estimates each trade's cost from the stock's own
    # recent high-low ranges plus a volatility-based impact for large trades.
    cost_model = None
    cost_label = f"cost {args.cost_bps:.1f} bps/trade"
    if args.cost_model == "spread":
        bars = {
            t: load_bars(t, start=args.start, end=args.end, path=args.db)
            for t in tickers
        }
        cost_model = SpreadCostModel(bars)
        cost_label = "cost model 'spread' (estimated from recent high-low ranges)"

    result = run_backtest(
        prices,
        signals,
        strategy_fn,
        frequency=rebalance,
        cost_bps=args.cost_bps,
        initial_capital=args.capital,
        drift_tolerance=drift_tolerance,
        cost_model=cost_model,
    )
    metrics = compute_metrics(result.equity, result.trades)

    first_day = result.equity["date"].iloc[0].strftime("%Y-%m-%d")
    last_day = result.equity["date"].iloc[-1].strftime("%Y-%m-%d")
    print(f"Backtest: {label}, {rebalance} rebalance, "
          f"{first_day} -> {last_day}")
    print(f"Universe: {', '.join(prices.columns)} "
          f"({len(prices.columns)} stocks, {cost_label}, "
          f"max weight {max_weight:.0%} (~{n} stocks), "
          f"drift tolerance {drift_tolerance:.1%})")
    if len(prices.columns) < 10:
        print("Note: this tiny universe is only for proving the engine runs — "
              "results on a handful of stocks are illustrative, not meaningful.")
    print(what)
    print()

    order = [
        "final_value", "total_return", "cagr", "volatility", "sharpe",
        "max_drawdown", "turnover", "win_rate", "num_trades",
    ]
    for key in order:
        m = metrics[key]
        value = m["value"]
        if key == "final_value":
            shown = f"${value:,.2f}"
        elif key in ("sharpe", "num_trades"):
            shown = f"{value:.2f}" if value == value else "n/a"
        else:
            shown = _fmt_pct(value)
        print(f"{m['label']}: {shown}")
        print(f"    What it means: {m['what']}")
        print(f"    Why it matters: {m['why']}")
        print(f"    What good looks like: {m['good']}")
        print()

    print("Honest reminder: a backtest shows what WOULD have happened, using "
          "today's knowledge of the past. It does not guarantee anything about "
          "the future — markets change, and costs in real life are often higher.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
