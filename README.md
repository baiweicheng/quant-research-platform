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
7. **Fundamentals data (SEC EDGAR)** ✅ (this phase) — free company
   financials (revenue, profit, assets, debt, ...) from the SEC's public
   filings, stored with true filing dates so backtests never peek at the
   future (see "Fundamentals data (SEC EDGAR) explained" below).
8. **Institutional strategies** ✅ (this phase) — three named strategies
   blending fundamentals with price patterns (Trending Value, Value+Momentum,
   Quality+Low-Volatility), a strategy registry with plain-English
   comparisons, a strategy picker on the Backtest/Portfolio/Paper-trading
   pages, weekly rebalancing, and drift buffers that skip tiny churn trades
   (see "Institutional strategies explained" below).

## Project layout

```
qrp/
  data/
    yahoo.py   # download daily OHLCV bars from Yahoo Finance
    edgar.py   # download company fundamentals from SEC EDGAR (free, no key)
    store.py   # SQLite storage (data/market.db)
    cli.py     # CLI: python -m qrp.data.cli --tickers AAPL,MSFT --start 2020-01-01
    edgar_cli.py  # CLI: python -m qrp.data.edgar_cli --tickers AAPL,MSFT,NVDA
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
    cli.py       # CLI: python -m qrp.backtest.cli --signal momentum --max-weight 0.1
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
  test_edgar.py
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
browser. The sidebar has nine pages:

- **Home** — what the platform does and the recommended workflow.
- **Market data** — download free daily prices for the tickers you care
  about, and see what is stored. "Check for new filings" re-downloads only
  what's new since your last check (companies file quarterly, so weekly
  checks are plenty), and the "Last checked" table shows when each company
  was last checked.
- **Signals** — score stocks (momentum, trend, low volatility), with each
  signal explained in plain English.
- **Strategies** — read what each ready-made strategy does and compare them
  side by side.
- **Compare strategies** — run every strategy over the exact same stocks,
  dates and starting cash and see which did best, with a plain-English
  winner summary.
- **Backtest** — replay history with trading costs included; every metric
  comes with a plain-English explanation. Choose between a simple flat fee
  per trade or a realistic cost estimated from each stock's price ranges.
- **Backtest wizard** — a guided five-step first backtest: what a backtest
  is, pick a strategy, pick dates, run, and read the results in plain
  English.
- **Portfolio** — turn scores into "how much of each stock" target
  weights, with safety caps and a plain-English dollar breakdown.
- **Paper trading** — practice with $100,000 of play money: preview what a
  rebalance *would* trade (including blocked orders), execute it in the
  simulated account, or reset and start over. Every trade and every
  previewed order carries a plain-English "Why" explaining the reason.
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
python -m qrp.backtest.cli --signal momentum --max-weight 0.1 --rebalance monthly
```

This simulates buying the 10 highest-ranked stocks each month and prints a
performance report where every number is explained in plain English.

## Fundamentals data (SEC EDGAR) explained

**What fundamentals are.** Beyond prices, every US public company must file
official financial reports with the SEC (the government agency overseeing
the stock market): the annual **10-K** and quarterly **10-Q**. These reports
contain the company's *fundamentals* — plain facts like how much money it
made (net income), how much it sold (revenue), what it owns (assets), what
it owes (debt), and how many shares exist. The SEC publishes all of it for
free through the **EDGAR** database, with no API key required.

**How to get them.** The ticker map
(`https://www.sec.gov/files/company_tickers.json`) translates a ticker like
`AAPL` into the SEC's company ID (CIK); the companyfacts endpoint
(`https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`) returns
every reported number for that company, all the way back.

```bash
python -m qrp.data.edgar_cli --tickers AAPL,MSFT,NVDA
```

Facts land in the same `data/market.db` (`fundamentals` table). Because
companies don't all use the same accounting labels, the pipeline tries
several label spellings per fact, prefers annual (10-K) reports, and keeps
the latest filing when a number was later restated.

**Why filing dates matter (no peeking at the future).** A company's fiscal
year often ends months before it files the report — Apple’s year ends in
September but the 10-K arrives in late October/November. A strategy looking
at "2023 earnings" in March 2024 may only use numbers *filed* on or before
that March date. Every stored fact therefore carries its real SEC filing
date, and the query helpers (`facts_as_of`) only return facts filed on or
before the date you ask about. Backtests built any other way quietly cheat.

**Known limitations.** Some companies reorganize and their history splits
across SEC IDs (e.g. a successor holding company with only recent filings);
the CLI prints each ticker's fact count and date range so thin history is
visible. Financial firms and utilities report different balance-sheet line
items, so ratio-based strategies typically exclude them.

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

Two more simplifications to keep in mind:

- **Price-only returns.** The engine replays daily closing prices, so
  dividends are not modeled — a high-dividend strategy will look slightly
  worse here than it would in real life.
- **Survivorship bias.** The database only contains stocks you chose to
  ingest, which are usually stocks that still exist today. Companies that
  went bankrupt or were delisted — the worst performers — tend to be missing
  from history, which can flatter the results.

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

## Institutional strategies explained

A **strategy** is a complete, named recipe: which signal(s) rank the
stocks, how positions are sized, and how often the portfolio is
rebalanced. The three strategies below blend **fundamentals** (company
numbers from SEC filings — how cheap and profitable the business is)
with **technicals** (price patterns — what's been climbing, what's calm).
They are based on approaches used by professional investment firms,
adapted here for long-only investing with free data. Compare them on the
dashboard's **Strategies** page, and run any of them from the Backtest
page (or `python -m qrp.backtest.cli --strategy <name>`).

- **Trending Value** — *Cheap AND climbing.* Ranks stocks on cheapness
  (book value, earnings and cash flow versus price) blended with 6-month
  price momentum. Value tells you *what* to buy; a rising price tells you
  the market is starting to agree — sidestepping "value traps", cheap
  stocks that just keep sinking.
- **Value + Momentum** — *The best of both.* Half the ranking comes from
  cheapness, half from 12-month price strength. Cheap stocks and winning
  stocks tend to take turns leading the market, so owning both smooths
  the ride. This is the classic institutional blend.
- **Quality + Low Volatility** — *The defensive one.* Ranks stocks on
  profitability (gross profit versus assets) blended with price calmness.
  Profitable, steady businesses that hold up better when markets get
  rough — at the cost of lagging when speculative stocks rally.

Two mechanics keep these strategies cheap to run:

- **Weekly rebalancing** — the portfolio is adjusted back to its targets
  once a week (the original single-signal approaches rebalance monthly).
  You can compare weekly vs monthly in the backtest to see whether the
  extra trading earns its keep.
- **Drift buffers** — a trade is skipped when moving a position to its
  target would shift less than 2% of the portfolio's value (the
  `--drift-tolerance` flag, only for named strategies by default). In
  plain English: don't pay trading costs to fix a wobble nobody would
  notice. Classic (non-strategy) backtests default to 0.0 so they trade at
  every rebalance, exactly as they always have — pass the flag explicitly
  (e.g. `--drift-tolerance 0.02`) to skip tiny trades there too.

One convention to be aware of: **Trending Value measures momentum over the
full trailing 6 months**, including the most recent one — that's how
O'Shaughnessy's published version works. The platform's standalone
`momentum` signal instead skips the most recent month (12-1) to avoid
short-term price wobbles contaminating the reading. Same word, slightly
different yardstick — both choices are on purpose.

Honest caveats: the published results behind these ideas come from
academic long-short portfolios (which also bet *against* bad stocks —
something this platform never does), so expect roughly a third to a half
of the headline numbers in a long-only version. And every backtest here
is only as honest as its data: fundamentals are used strictly by SEC
filing date, so a strategy never sees a number before it was published.

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
- *Timing is honest.* A paper rebalance trades at a trading day's close
  using only signals dated *strictly before* that day — a signal is not
  known until after its day's closing bell, so using it for that day's
  close would be peeking at the future. This is the same no-peeking rule
  the backtest engine uses, so paper-trading results are directly
  comparable to backtest results.
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

**Easiest way — no Terminal needed:** after cloning, double-click
**`Run Dashboard.command`** in the project folder. The first time, it sets
everything up by itself (it downloads its own Python 3.12 via `uv` — your
Mac's Python, including Anaconda, is never touched), then opens the dashboard
in your browser. Every later double-click just starts it. To stop the
dashboard, press Ctrl+C in the window that opened.

> **If macOS says the file "could not be executed because you do not have
> appropriate access privileges":** run this once in Terminal from the project
> folder — the executable permission doesn't survive the download, so it needs
> one manual grant. After that, double-clicking works forever:
> ```bash
> chmod +x "Run Dashboard.command"
> ```

If you prefer the Terminal (same thing the double-click does):

```bash
git clone https://github.com/baiweicheng/quant-research-platform.git
cd quant-research-platform
curl -LsSf https://astral.sh/uv/install.sh | sh   # one-time: installs uv
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m qrp.data.cli --tickers AAPL,MSFT,NVDA --start 2023-01-01
# launch the dashboard:
.venv/bin/python -m streamlit run qrp/ui/app.py
```

Works on Apple Silicon and Intel Macs with a recent macOS. Dependencies are
`requests`, `pandas`, `numpy`, and `streamlit`, pinned to exact versions in
`requirements.txt` — no platform-specific dependencies or paths. Developers
who want to run the test suite also install the dev extras:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt  # pytest — only needed for running tests
.venv/bin/python -m pytest tests/
```

## Safety notes

- Nothing in this repo places real orders. Trade execution is paper-trading
  only: simulated money against historical prices, with every order passing
  the risk safeguards. A live broker connection does not exist in this code
  and will only be built with your explicit approval.
- No credentials are stored in this repo. Broker credentials, when needed, go through a secure vault — never in code or chat.
