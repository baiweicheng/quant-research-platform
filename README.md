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
3. **Backtesting** ✅ (this phase) — long-only backtest engine with realistic transaction costs and performance metrics.
4. **Portfolio construction + risk safeguards** ✅ (this phase) — turn signals into target weights, with hard safeguards (position caps, max positions, drawdown guard, order sanity checks).
5. **Execution** ✅ (this phase) — broker interface plus paper trading: a simulated
   account (real math, fake money) that rebalances toward a target portfolio
   with every order passing the risk safeguards. No live broker is connected;
   live orders require explicit per-order approval. (Robinhood wiring is a later step.)
6. **UI dashboard** ✅ (this phase) — a friendly web UI over the whole pipeline:
   data, signals, backtests, portfolio, paper trading and risk safeguards,
   with plain-English explanations everywhere (see qrp/ui/ and "Running the
   dashboard" below).

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
  backtest/    # Phase 3: long-only backtesting engine
    engine.py    # daily simulation, no-lookahead trade timing, costs
    strategies.py  # top_n_by_signal: hold top-N ranked stocks equally
    metrics.py   # performance metrics with plain-English explanations
    cli.py       # CLI: python -m qrp.backtest.cli --signal momentum --top 10
  portfolio/   # Phase 4: target portfolio construction
    construction.py  # equal / signal / inverse-vol weighting + safety caps
    cli.py           # CLI: python -m qrp.portfolio.cli --signal momentum --method signal
  risk/        # Phase 4: risk management safeguards
    guards.py    # RiskConfig, check_order, check_portfolio, drawdown guard
  execution/   # Phase 5: paper-trading execution
    broker.py    # BrokerClient abstract base class (seam for a future real broker)
    paper.py     # PaperBroker: simulated account, real math, zero real dollars
    rebalance.py # turn a target portfolio into an order list (sells before buys)
    cli.py       # CLI: python -m qrp.execution.cli --dry-run
  ui/          # Phase 6: Streamlit dashboard (app.py) + UI-agnostic helpers
tests/
  test_yahoo.py
  test_signals.py
  test_backtest.py
  test_portfolio_risk.py
  test_execution.py
  test_ui.py
```

## Running the dashboard

The whole platform is also available as a friendly web dashboard — no
command line needed:

```bash
pip install -r requirements.txt
streamlit run qrp/ui/app.py
```

Then open the address it prints (usually http://localhost:8501) in your
browser. The sidebar has seven pages:

- **Home** — what the platform does and the recommended workflow.
- **Market data** — download free daily prices for the tickers you care
  about, and see what is stored.
- **Signals** — score stocks (momentum, trend, low volatility), with each
  signal explained in plain English.
- **Backtest** — replay history with trading costs included; every metric
  comes with a plain-English explanation.
- **Portfolio** — turn scores into "how much of each stock" target
  weights, with safety caps and a plain-English dollar breakdown.
- **Paper trading** — practice with $100,000 of play money: preview what a
  rebalance *would* trade (including blocked orders), execute it in the
  simulated account, or reset and start over.
- **Risk** — review the safety settings and run the checks on any target
  portfolio.

Everything the dashboard does, you can also do from the command line (see
Quick start above) — the dashboard is a friendlier window into the same
pipeline, not a separate system.

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

```bash
python -m qrp.backtest.cli --signal momentum --top 10 --rebalance monthly
```

This simulates buying the 10 highest-ranked stocks each month and prints a
performance report where every number is explained in plain English.

## Backtesting explained

A **backtest** is a simulation: it replays history and asks, "what would have
happened if I had followed this strategy?" It is the standard way to check
whether an idea had merit *before* trusting it with real money — but it is
not a crystal ball (more on that below).

How this engine keeps the simulation honest:

- **No peeking into the future.** On each monthly rebalance day, the strategy
  only sees signal values from that day or earlier, and its trades execute at
  the *next* day's closing price. In real life you cannot trade on today's
  signal at today's closing price (the market is already closed by the time
  you know it), so the engine does not let you.
- **Long only.** The portfolio only ever buys stocks; it never bets against
  them and never borrows money to invest more than it has.
- **Costs are real.** Every buy and sell pays a configurable trading cost
  (default 0.10% of the traded amount), because strategies that trade a lot
  can look great until costs eat the profit.
- **Cash is cash.** Money not invested simply sits as cash.

What the report tells you, in plain terms: how much the portfolio grew
overall, the equivalent steady yearly growth rate, how bumpy the ride was,
how much return you got per unit of bumpiness, the worst fall from a peak,
how much trading the strategy did, and how often months ended in the green.
Each number comes with an explanation of what it means and what "good" looks
like — run the CLI above to see them.

An honest warning: a backtest shows what *would* have happened, using
today's knowledge of the past. Markets change, popular strategies stop
working once everyone uses them, and real-life costs (taxes, wider price
gaps on big orders) are usually higher than the simulation assumes. Treat
backtest results as a filter for bad ideas, not proof of future profits.

> Note: the sample database only contains a few stocks (AAPL, MSFT, NVDA),
> so the example output only proves the engine runs — results on a handful
> of stocks are illustrative, not meaningful.

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

## Portfolio and risk explained

A **target portfolio** is simply the answer to "how much of each stock should
I hold right now?" — ticker by ticker, as a percentage of your money. There
are three ways to decide the percentages:

- **Equal weight** — split the money evenly across the highest-scoring
  stocks. Simple, hard to mess up, and no single stock can dominate. A good
  default.
- **Signal weight** — give more money to higher-scoring stocks (stocks with
  negative scores get nothing). Use this when you trust the ranking and want
  your strongest ideas to count the most.
- **Inverse volatility** — give more money to calmer stocks and less to jumpy
  ones. Use this when you want a gentler ride, accepting that you may give up
  some upside.

**Diversification** here means: don't put all your eggs in one basket. The
platform enforces it automatically — no single stock may exceed 20% of the
portfolio, and you never hold more than 20 stocks. Tiny "dust" positions
under 1% are dropped so the portfolio doesn't fill up with meaningless $50
holdings. Whatever can't be placed stays as **cash** — the portfolio never
forces money into stocks just to stay invested.

**Risk safeguards** are automatic checks that run before any order goes out:

- *Can you afford it?* An order that costs more than your cash is refused —
  you can never accidentally spend money you don't have.
- *Is one stock taking over?* A buy that would push a stock above the 20%
  cap is refused.
- *Does the price look wrong?* An order priced more than 5% away from the
  stock's last close is refused — this catches typos and stale prices.
- *Are you selling what you don't own?* Selling more shares than you hold is
  refused (no shorting, ever).
- *Drawdown stop:* if the portfolio's value falls more than 15% below its own
  peak, **new buying is paused** until you review and lift the halt. Selling
  always still works — the guard never traps you in a falling position, and
  it never sells anything by itself.

Try it: `python -m qrp.portfolio.cli --signal momentum --method signal`

## Paper trading explained

**Paper trading** means practicing with fake money. The account, the orders,
the fees, and the risk checks are all real math — but no real dollars move
and nothing touches a real brokerage. It is the flight simulator before the
real plane: the place to prove the system behaves before risking anything.

- *The account:* starts with $100,000 of play money. Your cash, share
  positions, and trade history are saved in the local database, so the
  account persists between runs.
- *Rebalancing:* `python -m qrp.execution.cli --dry-run` shows what WOULD be
  traded today (and which orders the safety checks would block) without
  trading anything. Drop `--dry-run` to actually execute — in the paper
  account only.
- *Every order passes the risk safeguards first.* A rejected order changes
  nothing and tells you why in plain English.
- *Starting over:* `python -m qrp.execution.cli --reset` wipes the paper
  account back to fresh play money.

**What is NOT connected:** there is no live broker in this repo — no
Robinhood, no real orders, no credentials. Paper trading is the default and
the only mode. Connecting a real broker is a future step and will only
happen with your explicit approval.

Try it:
```bash
python -m qrp.execution.cli --dry-run --signal momentum --method signal
python -m qrp.execution.cli --signal momentum --method signal  # paper trade
```

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

- Nothing in this repo places real orders. Trade execution is paper-trading
  only: simulated money against historical prices, with every order passing
  the risk safeguards. A live broker connection does not exist in this code
  and will only be built with your explicit approval.
- No credentials are stored in this repo. Broker credentials, when needed, go through a secure vault — never in code or chat.
