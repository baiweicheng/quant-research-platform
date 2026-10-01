# Quantitative Research Platform

A personal quantitative research platform for **US stocks, long positions only**.

## Scope

- Universe: US-listed equities only
- Direction: long only (no shorting, no leverage in early phases)
- Data: free sources only (Phase 1 uses Yahoo Finance daily bars — no API key required)

## Phased roadmap

Work proceeds in small, reviewable phases. Each phase lands as a commit on `main`.

1. **Data ingestion** ✅ (this phase) — download free daily OHLCV bars and store them locally in SQLite.
2. **Signals** ✅ (this phase) — a signal registry plus quantitative signals (momentum, trend, low volatility); qualitative/news-sentiment signals later.
3. **Backtesting** — long-only backtest engine with realistic transaction costs and performance metrics.
4. **Portfolio construction + risk safeguards** — turn signals into target weights, with hard safeguards (position caps, max positions, drawdown guard, order sanity checks).
5. **Execution** — broker interface; starts with paper trading only. Live orders require explicit per-order approval. (Robinhood wiring is a later step in this phase.)
6. **UI dashboard** — a single interface to orchestrate data, signals, backtests, portfolio, risk, and trade review.

## Project layout

```
qrp/
  data/
    yahoo.py   # download daily OHLCV bars from Yahoo Finance
    store.py   # SQLite storage (data/market.db)
    cli.py     # CLI: python -m qrp.data.cli --tickers AAPL,MSFT --start 2020-01-01
  signals/     # Phase 2: signal framework + quantitative signals
    base.py      # Signal base class
    registry.py  # SignalRegistry (register/get/list)
    quant.py     # momentum, trend, low_volatility
    store.py     # SQLite storage + loaders (signals table)
    cli.py       # CLI: python -m qrp.signals.cli --tickers AAPL,MSFT
  backtest/    # Phase 3 (placeholder)
  portfolio/   # Phase 4 (placeholder)
  risk/        # Phase 4 (placeholder)
  execution/   # Phase 5 (placeholder)
  ui/          # Phase 6 (placeholder)
tests/
  test_yahoo.py
  test_signals.py
```

## Quick start

```bash
pip install -r requirements.txt
python -m qrp.data.cli --tickers AAPL,MSFT,NVDA --start 2023-01-01
```

Bars are stored in `data/market.db` (SQLite, git-ignored).

```bash
python -m qrp.signals.cli --tickers AAPL,MSFT,NVDA
```

Computed signals are stored in the same database (`signals` table).

## Signals explained

Every stock gets a score on each signal below. The scores are built so you
can compare stocks against each other: **a higher score means the stock looks
more attractive to buy and hold.** These are classic starting points used by
investment researchers — not guarantees. Phase 3 (backtesting) will check how
they would actually have performed in the past.

- **Momentum** — *Has this stock been going up?* We look at how much the
  price rose over the past year, leaving out the most recent month (very
  short-term jumps often fade quickly). Stocks on a strong run tend to keep
  climbing for a while, so a high momentum score means the stock has been a
  steady winner lately.
- **Trend** — *Is the price above its long-term average?* We compare today's
  price with the average price over the past 10 months. When a stock trades
  above that average, it is usually in a healthy uptrend. A high trend score
  means the uptrend is intact.
- **Low volatility** — *How bumpy has the ride been?* We measure how much
  the price has jumped around day to day over the past 3 months. Calmer
  stocks have historically delivered steadier returns than roller-coaster
  ones, so a high score favors the steadier stock.

## Running on Mac

```bash
git clone https://github.com/baiweicheng/quant-research-platform.git
cd quant-research-platform
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m qrp.data.cli --tickers AAPL,MSFT,NVDA --start 2023-01-01
```

Requires Python 3.10+. Only `requests`, `pandas`, and the Python standard library
(`sqlite3`) are used — no platform-specific dependencies or paths.

## Safety notes

- Nothing in this repo places real orders. Phase 5 will default to paper trading.
- No credentials are stored in this repo. Broker credentials, when needed, go through a secure vault — never in code or chat.
