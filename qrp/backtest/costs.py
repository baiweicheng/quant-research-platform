"""Transaction cost models for the backtest engine (Phase 3, Feature 2).

Plain-English overview
----------------------
Every time you buy or sell a stock, a little money disappears. Brokers are
not the main culprit — the market itself takes a cut in two ways:

1. **The bid-ask spread.** At any moment a stock has two prices: the *bid*
   (what buyers are offering to pay) and the *ask* (what sellers are asking
   for). The ask is always a bit higher than the bid. When you buy, you pay
   the ask; when you sell, you receive the bid — so every round trip loses
   roughly the gap between them. For a stock quoted at $100.00 / $100.05,
   the spread is $0.05, and each trade loses about half of that ($0.025)
   versus the "true" middle price.

2. **Market impact.** If you try to buy a huge number of shares at once,
   you eat through all the sellers at the current price and start paying
   higher and higher prices to fill your order. Small orders barely move
   the price; enormous ones do. The effect grows with the stock's jumpiness
   (volatility) and with how large your order is compared to a normal day's
   trading volume.

The two models in this file
---------------------------
- ``FlatCostModel`` — the simple, predictable one: every trade loses a
  fixed percentage of its value (e.g. 10 basis points = 0.10%). This is
  the engine's default, and every backtest ever run with it keeps exactly
  the same results.
- ``SpreadCostModel`` — the realistic one: it estimates the spread from
  each stock's own recent daily high-low price ranges (the Corwin-Schultz
  method, explained below) and adds a volatility-based market-impact
  charge that grows for larger trades. More realistic, but it needs enough
  price history to work.

How the spread estimate works (Corwin-Schultz)
---------------------------------------------
You cannot see the bid-ask spread in daily price data — the database only
stores one price per day. But the *high-low range* of each day carries a
fingerprint of the spread: part of every day's range comes from the true
price drifting around, and part comes from the price bouncing between the
bid and the ask.

Corwin and Schultz (2012) showed how to separate the two using *pairs of
consecutive days*:

- Over two days, the true price can drift further (up to two days' worth
  of movement), while the bid-ask bounce is still just one spread wide.
- By comparing each single day's high-low range (``beta``) against the
  two-day combined range (``gamma``), the math backs out what fraction of
  the range was bounce rather than drift.
- The formulas below turn that fraction into a spread estimate. A
  negative estimate is impossible in reality, so it is floored at zero.

The volatility / market-impact part
-----------------------------------
- Volatility (``sigma``) is the standard deviation of daily price changes
  (measured as log returns) over the trailing window. A stock whose price
  jumps 3% on a typical day has higher sigma than one that drifts 0.3%.
- Your order's *participation rate* is your share count divided by the
  median daily trading volume (how many shares normally change hands).
  Trading 1% of a day's volume barely registers; trading a whole day's
  volume moves the price against you.
- The impact charge is ``0.5 * sigma * sqrt(participation)``: it grows
  with volatility and with the square root of your order size (doubling
  your order less than doubles the impact), and it is capped at 1%.

Honesty rules baked into this model
-----------------------------------
- **Strictly no peeking into the future.** Every number — the spread
  estimate, the volatility, the volume median, the reference price — is
  computed only from trading days dated on or before the trade day.
  Asking "what did this cost on day T?" gives the same answer whether or
  not the database also contains days after T (there is a test for this).
- **Not enough history -> honest fallback.** With fewer than two usable
  days of history there is no way to estimate a spread or a volatility.
  Rather than pretend the trade is free (0%) or impossibly expensive,
  the model charges a neutral 0.10% (10 basis points) — the same as the
  platform's flat default — until enough history accumulates.
- **Hard cap.** No single trade is ever charged more than 2% of its
  value, no matter what the formulas say. Real markets have limits too,
  and a cap keeps one bad data day from wrecking a whole backtest.
- **Spreads are widest right at the open.** The Corwin-Schultz estimator
  uses each day's high-low range, so it estimates the AVERAGE spread
  across the whole trading day. Trades that fill right at the 9:30am
  open — the platform's default "next-day-open" execution — tend to pay
  wider spreads than that average (the open is the noisiest,
  fastest-moving part of the day). So under next-day-open execution this
  model may slightly understate real trading costs — a small built-in
  optimism. No fudge factor is applied: the caveat is stated plainly
  rather than hiding behind an invented open-spread multiplier.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from qrp.data.yahoo import normalize_ticker


class CostModel(ABC):
    """How much of a trade's value is lost to transaction costs.

    ``rate(ticker, day, trade_value)`` returns the *fraction* of the
    trade's notional value lost to costs on that single trade — e.g.
    0.001 means "0.10% of the traded amount". It is called once per
    trade (buys and sells alike).
    """

    name: str = "base"

    @abstractmethod
    def rate(self, ticker: str, day: pd.Timestamp, trade_value: float) -> float:
        """Fraction of notional lost to costs on one trade."""
        raise NotImplementedError


class FlatCostModel(CostModel):
    """Fixed percentage cost on every trade, regardless of stock or size.

    This is the engine's default: ``FlatCostModel(10)`` charges 0.10% of
    the traded value on every buy and sell. Simple, predictable, and the
    reason old backtests never change when new models are added.
    """

    name = "flat"

    def __init__(self, bps: float) -> None:
        if bps < 0:
            raise ValueError("bps cannot be negative")
        self.bps = float(bps)

    def rate(self, ticker: str, day: pd.Timestamp, trade_value: float) -> float:
        return self.bps / 10_000.0

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"FlatCostModel(bps={self.bps})"


class SpreadCostModel(CostModel):
    """Volatility/spread-based costs estimated from each stock's history.

    For a trade in ``ticker`` on ``day``, the cost is:

    ``rate = min(max(half_spread + market_impact, 0), 0.02)``

    where ``half_spread`` is half the Corwin-Schultz bid-ask spread
    estimated from the trailing ``window`` trading days (dates <= day),
    and ``market_impact`` is ``0.5 * sigma * sqrt(participation)``
    (capped at 1%), with ``sigma`` the trailing volatility of daily
    returns and ``participation`` the order size relative to median
    daily volume. See the module docstring for the plain-English story.

    With fewer than 2 usable days of history the rate falls back to
    0.001 (10 basis points) — documented in the module docstring.
    """

    name = "spread"

    # Neutral charge when there is not enough history to estimate anything.
    FALLBACK_RATE = 0.001  # 10 basis points, same as the flat default
    # No trade is ever charged more than this fraction of its value.
    MAX_RATE = 0.02  # 2%
    # The market-impact piece alone is capped here.
    MAX_IMPACT = 0.01  # 1%

    def __init__(self, bars: dict[str, pd.DataFrame], window: int = 21) -> None:
        """Args:
            bars: ticker -> DataFrame with a DatetimeIndex (oldest first)
                and open/high/low/close/volume columns, e.g. as returned
                by ``qrp.signals.store.load_bars``.
            window: how many trailing trading days feed the estimates
                (default 21, about one calendar month).
        """
        if window < 2:
            raise ValueError("window must be at least 2")
        self._window = int(window)
        self._bars = {normalize_ticker(t): df for t, df in bars.items()}

    # -- internal helpers -------------------------------------------------

    def _history(self, ticker: str, day: pd.Timestamp) -> pd.DataFrame:
        """Trailing ``window`` trading days dated on or before ``day``.

        Only days the trader could actually have seen: anything dated
        after ``day`` is excluded, so estimates can never peek ahead.
        """
        bars = self._bars.get(normalize_ticker(ticker))
        if bars is None or bars.empty:
            return pd.DataFrame()
        hist = bars.loc[bars.index <= day].tail(self._window)
        if hist.empty:
            return hist
        # A day only counts if its high/low/close are real, positive numbers.
        valid = hist.dropna(subset=["high", "low", "close"])
        return valid[
            (valid["high"] > 0) & (valid["low"] > 0) & (valid["close"] > 0)
        ]

    @staticmethod
    def _corwin_schultz_spread(hist: pd.DataFrame) -> float:
        """Bid-ask spread from consecutive daily high-low pairs.

        Implements Corwin & Schultz (2012): for each consecutive pair of
        days, beta_i captures the two single-day ranges and gamma_i the
        combined two-day range; averaging them and solving gives the
        spread. Pairs with bad high/low values are skipped. Returns the
        full spread (the model charges half of it); never negative.

        Vectorized over the day-pairs with numpy (no Python loop).
        """
        h = hist["high"].to_numpy(dtype=float)
        l = hist["low"].to_numpy(dtype=float)
        if len(h) < 2:
            return 0.0
        h1, l1, h2, l2 = h[:-1], l[:-1], h[1:], l[1:]
        valid = (
            np.isfinite(h1) & np.isfinite(l1)
            & np.isfinite(h2) & np.isfinite(l2)
            & (h1 > 0) & (l1 > 0) & (h2 > 0) & (l2 > 0)
        )
        two_day_high = np.maximum(h1, h2)
        two_day_low = np.minimum(l1, l2)
        valid &= two_day_low > 0
        if not valid.any():
            return 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            beta_i = np.log(h1 / l1) ** 2 + np.log(h2 / l2) ** 2
            gamma_i = np.log(two_day_high / two_day_low) ** 2
        beta_i = beta_i[valid]
        gamma_i = gamma_i[valid]
        # (log of a positive finite ratio is always finite, so the
        # valid mask already guarantees finite beta_i/gamma_i.)
        beta = float(beta_i.mean())
        gamma = float(gamma_i.mean())
        sqrt2 = math.sqrt(2.0)
        denom = 3.0 - 2.0 * sqrt2
        alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) / denom - math.sqrt(
            gamma / denom
        )
        spread = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
        return max(spread, 0.0)

    @staticmethod
    def _trailing_volatility(hist: pd.DataFrame) -> float:
        """Std dev of daily log returns over the history window.

        A stock that typically jumps 3% a day gets a bigger number than
        one that drifts 0.3%. Returns 0.0 when it cannot be computed.
        """
        closes = hist["close"].astype(float)
        log_returns = (closes.apply(math.log)).diff().dropna()
        if len(log_returns) < 2:
            return 0.0
        sigma = float(log_returns.std())  # sample std (pandas default)
        return sigma if math.isfinite(sigma) else 0.0

    # -- public API ---------------------------------------------------------

    def rate(self, ticker: str, day: pd.Timestamp, trade_value: float) -> float:
        """Estimated cost fraction for one trade (see class docstring)."""
        hist = self._history(ticker, day)
        if len(hist) < 2:
            # Not enough history to estimate a spread or a volatility —
            # charge the neutral fallback instead of guessing wildly.
            return self.FALLBACK_RATE

        # 1) Half the bid-ask spread: you lose about half the gap between
        #    the buy price and the sell price on every trade.
        half_spread = self._corwin_schultz_spread(hist) / 2.0

        # 2) Market impact: big orders move the price against you, more so
        #    for jumpy stocks. Participation is your size versus a normal
        #    day's trading; the square root means doubling the order less
        #    than doubles the impact.
        close_on_day = float(hist["close"].iloc[-1])
        sigma = self._trailing_volatility(hist)
        adv = hist["volume"].median()
        adv = float(adv) if pd.notna(adv) and adv > 0 else 1.0
        participation = (trade_value / close_on_day) / max(adv, 1.0)
        impact = 0.5 * sigma * math.sqrt(min(max(participation, 0.0), 1.0))
        impact = min(impact, self.MAX_IMPACT)

        total = half_spread + impact
        return min(max(total, 0.0), self.MAX_RATE)
