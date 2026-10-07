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
   blending fundamentals with price patterns (Trending Value
   (O'Shaughnessy), Value + Momentum Composite (AQR-Style), Quality + Low
   Volatility Composite), a strategy registry with plain-English
   comparisons, a strategy picker on the Backtest/Portfolio/Paper-trading
   pages, weekly rebalancing, and drift buffers that skip tiny churn trades
   (see "Strategies explained" below).

## Project layout

```
qrp/
  data/
    yahoo.py   # download daily OHLCV bars from Yahoo Finance
    edgar.py   # download company fundamentals from SEC EDGAR (free, no key)
    store.py   # SQLite storage (data/market.db)
    cli.py     # CLI: python -m qrp.data.cli --tickers AAPL,MSFT --start 2013-01-01
    edgar_cli.py  # CLI: python -m qrp.data.edgar_cli --tickers AAPL,MSFT,NVDA
    french.py  # P1: free daily Fama-French factors from Ken French's data library
               # CLI: python -m qrp.data.french
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
    metrics.py   # P1: historical VaR/CVaR (95%/99%) + 1-day volatility
    attribution.py  # P1: Fama-French factor attribution (alpha, betas, R²)
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
  test_risk_reporting.py  # P1: VaR/CVaR, factor attribution, French parser, stress windows
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

Keeping data fresh: the dashboard's Market data page has a **Refresh new
data** button — one click downloads only what arrived since your last pull
(new daily prices and new SEC filings, never a full re-download) and fills
any pending paper orders unlocked by the new prices. The dated Download
form above it stays for full fresh starts.

```bash
python -m qrp.backtest.cli --signal momentum --max-weight 0.05 --rebalance monthly
```

This simulates buying the ~20 highest-ranked stocks and prints a
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

**TTM normalization (flow metrics are comparable across filings).** A 10-Q
reports a quarter (or year-to-date), a 10-K reports a full year — comparing
them raw would rank a 3-month profit against a 12-month profit. Before any
ratio is computed, flow metrics (net income, gross profit, operating cash
flow) are rebuilt into trailing-twelve-months values from quarterly
increments, using only filings that existed on each date (restatements apply
from their own filing date onward). Point-in-time snapshots (equity, assets,
shares outstanding) are used as filed. Because signal values changed with
this fix, **recompute your fundamental signals** (Signals page) after
updating — old stored values mixed raw quarters with full years.

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

**The market benchmark.** Every backtest and strategy comparison is shown next
to two baselines. **Market benchmark (buy-and-hold)** buys every one of *your*
stocks in equal amounts on day one and never trades — computed from the same
downloaded prices over the same dates, with each stock's dividends credited on
their ex-dates (total return, on the same basis as the strategies). **S&P 500 (SPY)** puts everything into
SPY, the fund tracking the 500 largest US companies, on day one and never
trades — the "just own the whole market" alternative. (Tick "Also download
SPY" on the Market data page to get its prices; the backtest CLI accepts it
as a ticker too.) A strategy that cannot beat sitting still is not earning
its complexity.

The Backtest page also compares each run against the S&P 500 directly:
**alpha** (extra yearly return beyond what its market sensitivity explains),
**beta** (how strongly it moves with the market), **information ratio**
(consistency of outperformance), **tracking error** (how far its day-to-day
returns stray from the market), and **up/down-market capture** (how much of
rallies it captures, how much of crashes it suffers) — each explained in
plain English on the page.

How this engine keeps the simulation honest:

- **No peeking into the future.** On each rebalance day, the strategy
  only sees signal values from that day or earlier, and its trades execute
  at the *next* trading day — never on the day itself. In real life you
  cannot trade on today's signal at today's prices (the market is already
  closed by the time you know it), so the engine does not let you.
- **No survivorship peeking either.** The backtest universe on any date is
  only the stocks that were actually in the S&P 500 on that date, rebuilt
  from a dated log of every index addition and removal (Wikipedia's change
  log, checked into `qrp/data/sp500_history.csv`). A stock can never be
  bought before it joined the index, and a stock that leaves the index is
  sold at the next rebalance. A delisted holding is closed at its last
  available price.
- **When trades fill.** Trade execution is a setting, defaulting to
  **next day's open** — matching the real workflow of running the rebalance
  at night, staging the orders, and filling them at the next morning's
  open (like market-on-open orders). The equity curve is still marked at
  each day's close, so a position bought at the open earns open-to-close
  that day. **Next day's close** is kept as an option (the older
  convention) — pick it on the Backtest page or pass
  `--execution close` on the backtest command line. Results saved before
  this setting existed all used next-day-close, so they will differ
  slightly from new runs; the difference is small (overnight price gaps),
  but re-run anything you want to compare side by side.
- **Open-spread caveat.** The realistic cost model estimates an *average*
  daily spread, while spreads right at the 9:30am open tend to be wider
  than average. Under next-day-open execution it may therefore slightly
  understate trading costs — stated plainly rather than hidden behind a
  made-up multiplier.
- **Long only.** The portfolio only ever buys stocks; it never bets against
  them and never borrows money to invest more than it has.
- **Costs are real.** Every buy and sell pays a trading cost estimated from
  the stock's recent high-low price ranges (the spread-based model — the
  default, since strategies that trade a lot can look great until costs eat
  the profit). A simple flat fee per trade stays available as an option.
- **Cash is cash.** Money not invested simply sits as cash.

What the report tells you, in plain terms: how much the portfolio grew
overall, the equivalent steady yearly growth rate, how bumpy the ride was,
how much return you got per unit of bumpiness, the worst fall from a peak,
how much trading the strategy did (one-way turnover: selling everything and
buying a whole new portfolio counts as 100%, not 200%), and how often months
ended in the green.
Each number comes with an explanation of what it means and what "good" looks
like — run the CLI above to see them.

**The Risk section** answers "how bad could a bad day be?" — the question
the headline numbers don't ask directly:

- **Bad-day loss threshold (VaR 95%)** — on one of the worst 5% of days,
  the strategy lost at least this much. Said the other way: 95 days out
  of 100, the daily loss stayed below this number.
- **Average loss on the worst days (CVaR 95%)** — the average loss across
  those worst 5% of days. VaR tells you where the bad days start; this
  tells you how bad they typically were once you were in one.
- The 99% versions of both, for the truly rare ugly days, plus the
  **typical daily wobble** (1-day volatility) to put them in context.

These come straight from the strategy's own past daily returns — no fancy
assumed distribution. The worst fall and yearly bumpiness shown there are
the same numbers as the metric cards above, reused so the two always agree.

**Stress windows.** Above the date inputs, three one-click presets jump to
famous rough patches — **COVID crash (2020)**, **2022 bear market** — or
back to **Full history**. And **Run all three stress windows** replays the
exact same backtest over all three periods and shows them side by side, so
you can see at a glance whether an idea only worked in good times. A window
your downloaded data doesn't cover says so instead of failing.

## What explains a strategy's returns? (factor attribution)

Academic research found that most of what stocks do can be explained by a
handful of broad **tilts**: the market as a whole, small vs big companies,
cheap vs expensive stocks (value), profitable vs unprofitable companies,
careful vs aggressive spenders, and momentum (recent winners vs losers).
These are the **Fama-French factors**, and Professor Ken French publishes
them as a **free daily dataset** from his data library at Dartmouth
(no key needed):

    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

On the **Market data** page, press **Download factor data** once to fetch
it (re-downloading refreshes it). The **Compare strategies** page then adds
a *"What explains each strategy's returns?"* section: each strategy's daily
returns are regressed against the six tilts, and you get, per strategy:

- **Alpha** — the yearly return left over after the tilts are accounted
  for. "Was there anything clever beyond riding known patterns?"
- **The tilts it leans on** — e.g. "leans toward cheaper (value) stocks
  (beta +0.62)", with a t-stat; beyond ±2 counts as convincing rather
  than noise.
- **R²** — what share of the strategy's day-to-day moves the tilts explain
  together (0.90 = "mostly just the tilts").

Each strategy gets a one-paragraph plain-English reading, e.g. *"It leans
on 2 classic tilts: leans toward cheaper (value) stocks (beta +0.62);
leans toward recent winners (momentum) (beta +0.41). After accounting for
those tilts, there's no extra return left over — the tilts explain it all.
The classic tilts explain most of its day-to-day moves (R² = 0.85)."*

If the factor data isn't downloaded, the section says so plainly instead of
erroring. You can also fetch it from the command line:

    python -m qrp.data.french

An honest warning: a backtest shows what *would* have happened, using
today's knowledge of the past. Markets change, popular strategies stop
working once everyone uses them, and real-life costs (taxes, wider price
gaps on big orders) are usually higher than the simulation assumes. Treat
backtest results as a filter for bad ideas, not proof of future profits.

Two more simplifications to keep in mind:

- **Total returns (dividends included).** The engine replays daily closing
  prices *and* credits dividend cash to the portfolio on each payout date
  (amount × shares held), so the equity curve is a true total return — a
  high-dividend strategy gets full credit for its payouts. Dividends are
  downloaded automatically with prices; if your database was built before
  this feature, re-run the price download to backfill them. (Paper trading
  collects dividends too — press "Collect dividends" on the Paper trading
  page after downloading fresh data.)
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

- **12-1 Month Cross-Sectional Momentum** (`momentum`) — *Has this stock
  been going up?* We look at how much the
  price rose over the past year, leaving out the most recent month (very
  short-term jumps often fade quickly). Stocks on a strong run tend to keep
  climbing for a while, so a high momentum score means the stock has been a
  steady winner lately.
- **200-Day Moving-Average Trend Filter** (`trend`) — *Is the price above
  its long-term average?* We compare today's
  price with the average price over the past 10 months. When a stock trades
  above that average, it is usually in a healthy uptrend. A high trend score
  means the uptrend is intact. The **Trend strategy** adds an absolute rule
  on top: stocks trading *below* their 200-day average are excluded
  outright — never ranked, never bought — so in a broad downturn the
  portfolio can sit partly (or even entirely) in cash instead of holding
  the least-bad losers.
- **Low Volatility (3-Month)** (`low_volatility`) — *How bumpy has the ride
  been?* We measure how much
  the price has jumped around day to day over the past 3 months. Calmer
  stocks have historically delivered steadier returns than roller-coaster
  ones, so a high score favors the steadier stock.

## Strategies explained

A **strategy** is a complete, named recipe: which signal(s) rank the
stocks, how positions are sized, and how often the portfolio is
rebalanced. The nine strategies below are listed in the order they were
added to the platform — the three price-only originals, then the three
fundamental blends, then the three newest research ideas. They are based
on approaches used by professional investment firms, adapted here for
long-only investing with free data. Compare them on the dashboard's
**Strategies** page — each one shows an expandable "How it is
calculated" section with the exact step-by-step math — and run any of
them from the Backtest page (or `python -m qrp.backtest.cli --strategy
<name>`, using the short name in parentheses).

**Original technical strategies** (prices only):

- **12-1 Month Cross-Sectional Momentum** (`momentum`) — *The classic
  trend-rider.* Holds the past year's biggest winners, skipping the most
  recent month. Calculated as: close(21 trading days ago) ÷ close(252
  trading days ago) − 1 per stock, percentile-ranked across all stocks;
  top ~50 held equally (2% cap each), rebalanced monthly — roughly the top
  tenth of the index, matching academic momentum studies.
- **200-Day Moving-Average Trend Filter** (`trend`) — *The uptrend gate.*
  Holds stocks trading above their 200-day average price. Calculated as:
  (close − 200-day average) ÷ 200-day average, with an absolute cutoff —
  stocks below their average are excluded outright and that slice sits in
  cash, so a broad downturn can leave the portfolio mostly in cash; the
  survivors are ranked and the top ~25 held equally (4% cap each),
  rebalanced monthly.
- **Low Volatility (3-Month)** (`low_volatility`) — *The calm one.* Holds
  the calmest stocks of the past 3 months. Calculated as: the negative of
  the standard deviation of daily returns over the trailing 63 trading
  days (so calmer = higher score), percentile-ranked; the calmest ~25 held
  equally (4% cap each), rebalanced monthly.

**Fundamental strategies** (prices + company financials from SEC filings):

- **Trending Value (O'Shaughnessy)** (`trending_value`) — *Cheap AND
  climbing.* Ranks stocks on cheapness (book-to-market, earnings yield and
  cash-flow yield versus price) blended with 6-month price momentum. Value
  tells you *what* to buy; a rising price tells you the market is starting
  to agree — sidestepping "value traps", cheap stocks that just keep
  sinking. Calculated as: average of the value-composite percentile rank
  and the 6-month-return percentile rank; signal-weighted positions, 5%
  cap each (~20 stocks).
- **Value + Momentum Composite (AQR-Style)** (`value_momentum`) — *The best
  of both.* Half the ranking comes from cheapness, half from 12-month
  price strength. Cheap stocks and winning stocks tend to take turns
  leading the market, so owning both smooths the ride. This is the classic
  institutional blend. Calculated as: 50% value-composite rank + 50%
  12-1-month momentum rank; signal-weighted, 5% cap each.
- **Quality + Low Volatility Composite** (`quality_lowvol`) — *The
  defensive one.* Ranks stocks on profitability (gross profit versus
  assets) blended with price calmness. Profitable, steady businesses that
  hold up better when markets get rough — at the cost of lagging when
  speculative stocks rally. Calculated as: 50% profitability rank + 50%
  3-month low-volatility rank; signal-weighted, 5% cap each.

**New research strategies** (from a research paper, implemented from the
original academic definitions):

- **Continuous Momentum (Frog-in-the-Pan)** (`continuous_momentum`) —
  *The smooth climbers.* Among past winners, prefers stocks that rose on
  many small up days over ones that spiked on a few big jumps — smooth
  climbs tend to persist while spikes snap back (Da, Gurun & Warachka
  2014). Calculated as: information discreteness =
  sign(12-month return) × (% down days − % up days); winners ranked by
  lowest discreteness; top ~50 held with inverse-volatility sizing (2%
  cap each), rebalanced monthly.
- **Intraday Institutional Flow (Overnight Divergence)**
  (`overnight_intraday`) — *The quiet accumulators.* Buys stocks with
  steady daytime buying that the overnight crowd hasn't chased yet —
  patient institutional buying leaves a daytime footprint. Calculated as:
  compounded intraday (open→close) return minus compounded overnight
  (close→open, dividend-adjusted) return over ~6 months, Z-scored across
  stocks; top ~50 held equally (2% cap each), rebalanced monthly.
- **Twin Momentum (Price + Earnings Surprise)** (`twin_momentum`) —
  *The double-confirmed.* Requires both a strong price trend AND a real
  earnings surprise: a rising price backed by an earnings beat is much
  sturdier than a hollow spike. Calculated as: Z-scored 12-1-month price
  momentum plus Z-scored standardized unexpected diluted EPS (from 10-Q
  filings, point-in-time); both legs must beat their average or the stock
  is skipped; top ~40 held equally (2.5% cap each), rebalanced monthly.
  Needs the diluted-EPS tag — re-run the fundamentals download after
  pulling this version.

Two mechanics keep these strategies cheap to run:

- **Monthly re-evaluation** — every strategy's preset rebalances monthly
  (the academic convention; daily and weekly stay available everywhere).
  The check is cheap because of the next mechanic:
- **Drift buffers** — a trade is skipped when moving a position to its
  target would shift less than 2% of the portfolio's value (the
  `--drift-tolerance` flag, and a matching slider on the Backtest page).
  In plain English: don't pay trading costs to fix a wobble nobody would
  notice. Monthly checks plus the buffer mean the portfolio reacts to real
  changes without churning on noise. The Compare page can also force one
  rebalancing frequency on all strategies at once, so you can see what
  weekly versus monthly costs net of fees.

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
filing date, so a strategy never sees a number before it was published;
index membership is point-in-time, so it never buys a stock before it
joined the index. What backtests still don't model: market impact at
large size, and bankruptcies that go to zero (a delisted stock is closed
at its last price, but a true wipeout can't be detected automatically).

## Portfolio and risk explained

A **target portfolio** is simply the answer to "how much of each stock should
I hold right now?" — ticker by ticker, as a percentage of your money. There
are four ways to decide the percentages:

- **Equal weight** — split the money evenly across the highest-scoring
  stocks. Simple, hard to mess up, and no single stock can dominate. A good
  default.
- **Signal weight** — the best-scoring stock gets the full max-weight
  allowance and the rest get proportionally less, so stronger ideas count
  more (stocks with negative scores get nothing). Because of the cap, some
  money may stay in cash when scores are very concentrated.
- **Inverse volatility** — give more money to calmer stocks and less to jumpy
  ones. Use this when you want a gentler ride, accepting that you may give up
  some upside.
- **Optimized (mean-variance)** — a math optimizer picks the weights to get
  the most expected return for the risk taken, instead of following a fixed
  recipe. See "The optimizer, in plain English" below.

**Diversification** here means: don't put all your eggs in one basket. The
platform enforces it automatically — no single stock may exceed 10% of the
portfolio by default (adjustable up to 50%), and you never hold more than 20 stocks. Tiny "dust" positions
under 1% are dropped so the portfolio doesn't fill up with meaningless $50
holdings. Whatever can't be placed stays as **cash** — the portfolio never
forces money into stocks just to stay invested.

**Risk safeguards** are automatic checks that run before any order goes out:

- *Can you afford it?* An order that costs more than your cash is refused —
  you can never accidentally spend money you don't have.
- *Is one stock taking over?* A buy that would push a stock above the
  max-weight cap (10% by default) is refused.
- *Does the price look wrong?* A **buy** order priced more than 5% away from
  the stock's last close is refused — this catches typos and stale prices.
  **Sells are exempt on purpose:** in a crash you must always be able to get
  out, even far below the last close.
- *Are you selling what you don't own?* Selling more shares than you hold is
  refused (no shorting, ever).
- *Drawdown stop:* if the portfolio's value falls more than 15% below its own
  peak, **new buying is paused** until you review and lift the halt. Selling
  always still works — the guard never traps you in a falling position, and
  it never sells anything by itself.

Try it: `python -m qrp.portfolio.cli --signal momentum --method signal`

### The optimizer, in plain English

On the Target portfolio page (and in paper trading), you can choose
**Position sizing → Optimized (mean-variance)** instead of one of the three
fixed recipes above. Here's what it does, in one paragraph: it picks weights
to get the most expected return for the risk taken, while limiting trading
and staying close to the market (the S&P 500).

The details, still in plain English:

- **Expected return** — each stock's signal score is turned into a rough
  guess: a stock scoring one notch above average is assumed to beat the
  average stock by about 10% a year. This is a transparent rule of thumb
  about the signal's edge, *not* a measured forecast. It only tells the
  optimizer to prefer higher-scoring stocks.
- **Risk** — the optimizer learns how stocks move together from the last
  year of daily prices, using a technique called shrinkage (Ledoit-Wolf)
  that keeps the estimate stable when there are hundreds of stocks. That
  stops it from loading up on stocks that all fall together.
- **The limits it must respect** — fully invested (weights add to 100%),
  long-only (never bet against a stock), no stock over 5% (the platform's
  usual cap), and a **tracking-error budget** (default 5% a year): how far
  the portfolio's returns may stray from the S&P 500's.
- **The trading brake** — a turnover penalty (default: as if every dollar
  traded cost 0.2%) means the optimizer only trades when the improvement
  is worth it. On the Paper trading page it measures this against your
  actual current holdings.
- **What you see** — expected annual return, predicted annual volatility,
  tracking error vs the S&P 500, estimated turnover, and a before/after
  weight table per stock.

Two honest caveats. First, the optimizer is only as good as its inputs: if
the signal scores are wrong, it will confidently build the wrong portfolio
(the research literature calls an unconstrained optimizer an "error
maximizer" — the caps, the shrinkage, and the tracking-error budget are
what keep it honest). Second, the S&P 500 benchmark is size-weighted when
your SEC fundamentals data includes shares outstanding, and falls back to
equal-weighted when it doesn't — the page tells you which one it used.

Try it: `python -m qrp.portfolio.cli --signal momentum --method optimized`

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
- *The nightly workflow (dashboard):* the Paper trading page stages a
  rebalance as **pending orders** — decided from data through the latest
  close, but nothing trades yet, so the account value doesn't move. The
  orders fill automatically at the **next market open** once new price data
  arrives (the page checks on every visit; the Market data page's *Refresh
  new data* button fills them right after downloading). Pending orders can
  be cancelled any time before they fill, and a refresh never fills the
  same order twice.
- *Every order passes the risk safeguards first.* A rejected order changes
  nothing and tells you why in plain English.
- *Timing is honest.* A paper rebalance's orders fill at the trading day's
  OPEN by default — like real market-on-open orders staged the night before
  — using only signals dated *strictly before* that day. A signal is not
  known until after its day's closing bell, so using it for that day's open
  or close would be peeking at the future. This is the same no-peeking rule
  the backtest engine uses, so paper-trading results are directly
  comparable to backtest results.
- *Trading costs match the backtests.* The dashboard's paper trading prices
  each fill with the spread cost model (estimated from the stock's recent
  price ranges, priced off the day before the fill) — the same default the
  backtests use.
- *Starting over:* `python -m qrp.execution.cli --reset` wipes the paper
  account back to fresh play money.

**What is NOT connected:** no live (real-money) broker — no Robinhood, no
real orders. Live trading is intentionally not exposed in the app; it comes
later behind a dedicated safety review, and only with your explicit
approval. Paper trading is the default and the only mode.

Try it:
```bash
python -m qrp.execution.cli --dry-run --signal momentum --method signal
python -m qrp.execution.cli --signal momentum --method signal  # paper trade
```

## Alpaca paper trading

The dashboard can paper-trade through **Alpaca's paper-trading simulator**
instead of only the built-in one: real market mechanics (opening auctions,
real order handling) with fake money. Setup:

1. Sign up at [alpaca.markets](https://alpaca.markets) (free) and open the
   **paper** trading dashboard.
2. Generate a **paper** API key pair (key ID + secret key) — free.
3. In this dashboard, open the **Brokers** page, paste the two keys into the
   password-masked fields, and press **Save keys**, then **Test
   connection**. A successful test shows your paper account's equity, cash,
   and buying power in plain English.

Your keys are stored in `data/alpaca_paper_keys.json` on your own computer
only — the file is readable just by you (mode 0600) and is gitignored, so
it never lands in the repo. They are sent only to Alpaca, never logged.

**How orders map to Alpaca.** The platform stages orders at night for the
next market open. Every staged order is submitted to Alpaca immediately as a
**market-on-open order** (`time_in_force="opg"`) for the next trading day's
open — Alpaca queues it overnight and fills it in the opening auction. That
is exactly the platform's "fill at the next open" semantic, and it matches
what the backtests assume. There is deliberately no "fill now at market"
option: staging during trading hours still fills at the next open.

Two Alpaca realities are handled for you:

- *Whole shares only.* Alpaca accepts fractional shares only on plain day
  orders, not on market-on-open orders. So the dashboard rounds each staged
  order **down** to whole shares (buys can't overshoot cash, sells can't
  overshoot what's held) and says so; the leftover stays as cash. Orders
  under one whole share are skipped with a note.
- *Rejected orders stay staged.* If Alpaca can't take a market-on-open
  order right now (for example, one staged between 9:28am and 7pm ET, when
  Alpaca doesn't accept them), the order is kept in a local staging queue
  — shown on the Paper trading page under "Waiting to go to Alpaca" — and
  sent automatically for the next valid open. It is never silently
  converted into an immediate fill.

When Alpaca paper keys are saved and the connection tests OK, the Paper
trading page drives Alpaca by default: Alpaca account snapshot, positions,
open orders (cancellable), staging, and recent orders. The built-in local
simulated broker stays available behind the *"Local simulated broker
(offline testing)"* expander — handy for offline testing and for replaying
past dates, which Alpaca can't do. When Alpaca isn't connected, the local
broker remains the visible default, exactly as before.

**Paper and live APIs are identical** except for the base URL and the keys,
so the eventual live switch is a key swap — but the app only ever talks to
the paper endpoint (`https://paper-api.alpaca.markets`). The adapter raises
on Alpaca's live server (`https://api.alpaca.markets`) unless a
Python-level caller explicitly opts in with `allow_live=True`; nothing in
the dashboard or its helpers can pass that flag, so live trading stays
gated to a later phase behind the safety checks for real money.

Verify the integration any time (read-only — it never places an order):

```bash
python -m qrp.execution.alpaca --verify
```

It checks authentication, the account, the market clock, and positions, and
prints plain-English results.

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
