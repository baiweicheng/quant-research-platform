"""Tests for P5: data quality (checks, FIGI mapping, backup/restore).

Style notes: fixed ``asof`` dates keep the staleness tests deterministic;
network fetchers are always faked (``fetch_actions`` / ``fetch_bars`` are
injected by design); the OpenFIGI HTTP layer is faked too.
"""

from datetime import date

import pandas as pd
import pytest

from qrp.data import backup as B
from qrp.data import figi as F
from qrp.data import quality as Q
from qrp.data.store import (
    get_security_id,
    init_db,
    upsert_bars,
    upsert_dividends,
    upsert_security_ids,
)

ASOF = date(2024, 4, 2)  # a Tuesday; deterministic "today" for tests


def _frame(dates, closes):
    closes = list(closes)
    return pd.DataFrame(
        {
            "Date": [d.strftime("%Y-%m-%d") for d in dates],
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(dates),
        }
    )


@pytest.fixture
def db(tmp_path):
    """DB with two clean tickers (gentle drift, no outliers)."""
    path = tmp_path / "q.db"
    dates = pd.bdate_range("2024-01-02", "2024-03-29")
    n = len(dates)
    upsert_bars(
        _frame(dates, (100.0 + i * 0.05 for i in range(n))), "AAA", path
    )
    upsert_bars(
        _frame(dates, (50.0 - i * 0.02 for i in range(n))), "BBB", path
    )
    return path


# --- date helpers -------------------------------------------------------------

def test_trading_days_between_skips_weekends():
    fri = date(2024, 3, 29)  # Friday
    assert Q.trading_days_between(fri, date(2024, 4, 1)) == 1  # Monday
    assert Q.trading_days_between(fri, date(2024, 4, 2)) == 2
    assert Q.trading_days_between(date(2024, 4, 1), date(2024, 4, 1)) == 0


def test_expected_latest_trading_day_rolls_back_weekend():
    assert Q.expected_latest_trading_day(date(2024, 4, 6)) == date(2024, 4, 5)
    assert Q.expected_latest_trading_day(date(2024, 4, 2)) == date(2024, 4, 2)


# --- staleness ------------------------------------------------------------------

def test_stale_ticker_flagged(db):
    from qrp.data.store import connect_db

    dates = pd.bdate_range("2024-01-02", "2024-01-29")
    upsert_bars(_frame(dates, (10.0 for _ in dates)), "STALE", db)
    with connect_db(db) as conn:
        out = Q.check_staleness(conn, ["STALE"], asof=ASOF)
    assert len(out) == 1
    assert out[0].severity == "warning"
    assert out[0].ticker == "STALE"
    assert "trading days behind" in out[0].summary
    assert out[0].action  # tells the user what to do


def test_fresh_tickers_no_stale_findings(db):
    from qrp.data.store import connect_db

    with connect_db(db) as conn:
        assert Q.check_staleness(conn, asof=ASOF) == []


def test_ticker_with_no_bars_is_info(db):
    from qrp.data.store import connect_db

    with connect_db(db) as conn:
        out = Q.check_staleness(conn, ["GHOST"], asof=ASOF)
    assert len(out) == 1 and out[0].severity == "info"


# --- outliers ---------------------------------------------------------------------

def test_outlier_flagged(db):
    dates = pd.bdate_range("2024-01-02", "2024-03-29")
    closes = [200.0 + i * 0.1 for i in range(len(dates))]
    closes[30] *= 1.5  # +50% single-day jump
    upsert_bars(_frame(dates, closes), "OUTL", db)
    from qrp.data.store import connect_db

    with connect_db(db) as conn:
        out = Q.check_outliers(conn, ["OUTL"])
    warns = [f for f in out if f.severity == "warning"]
    assert warns, "expected the +50% day to be flagged"
    assert "+50" in warns[0].summary and "%" in warns[0].summary
    assert "never" in warns[0].detail.lower() or "only" in warns[0].detail.lower()


def test_clean_db_no_outlier_findings(db):
    from qrp.data.store import connect_db

    with connect_db(db) as conn:
        assert Q.check_outliers(conn) == []


def test_impossible_bar_is_error(db):
    dates = pd.bdate_range("2024-01-02", "2024-01-31")
    df = _frame(dates, (300.0 for _ in dates))
    df.loc[5, "High"] = 10.0
    df.loc[5, "Low"] = 20.0
    upsert_bars(df, "BAD", db)
    from qrp.data.store import connect_db

    with connect_db(db) as conn:
        out = Q.check_outliers(conn, ["BAD"])
    errs = [f for f in out if f.severity == "error"]
    assert any("high" in f.summary.lower() and "low" in f.summary.lower() for f in errs)


def test_close_outside_range_is_error(db):
    dates = pd.bdate_range("2024-01-02", "2024-01-31")
    df = _frame(dates, (300.0 for _ in dates))
    df.loc[5, "Close"] = 999.0  # above the day's high
    upsert_bars(df, "BADRANGE", db)
    from qrp.data.store import connect_db

    with connect_db(db) as conn:
        out = Q.check_outliers(conn, ["BADRANGE"])
    assert any(f.severity == "error" for f in out)


def test_nonpositive_close_is_error(db):
    dates = pd.bdate_range("2024-01-02", "2024-01-31")
    df = _frame(dates, (300.0 for _ in dates))
    df.loc[5, "Close"] = 0.0
    upsert_bars(df, "ZEROPX", db)
    from qrp.data.store import connect_db

    with connect_db(db) as conn:
        out = Q.check_outliers(conn, ["ZEROPX"])
    assert any(
        f.severity == "error" and "non-positive" in f.summary for f in out
    )


def test_cross_sectional_info_flag(tmp_path):
    """A +15% move on a calm day fires the info-level median check."""
    path = tmp_path / "x.db"
    dates = pd.bdate_range("2024-01-02", "2024-02-29")
    n = len(dates)
    for i in range(20):
        upsert_bars(_frame(dates, (100.0 + i + j * 0.01 for j in range(n))),
                    f"T{i:02d}", path)
    closes = [100.0 + j * 0.01 for j in range(n)]
    closes[20] *= 1.15
    upsert_bars(_frame(dates, closes), "JUMPER", path)
    from qrp.data.store import connect_db

    with connect_db(path) as conn:
        out = Q.check_outliers(conn, ["JUMPER"] + [f"T{i:02d}" for i in range(20)])
    infos = [
        f for f in out
        if f.severity == "info" and f.ticker == "JUMPER" and "median" in f.summary
    ]
    assert infos, "expected the off-median move to be noted"


# --- corporate actions --------------------------------------------------------------

def _actions_db(tmp_path):
    path = tmp_path / "a.db"
    dates = pd.bdate_range("2024-01-02", "2024-06-28")
    n = len(dates)
    closes = [100.0 + i * 0.1 for i in range(n)]
    upsert_bars(_frame(dates, closes), "SPLT", path)
    return path, dates, closes


def _prices_df(dates, closes, adjcloses):
    return pd.DataFrame(
        {
            "Date": [d.strftime("%Y-%m-%d") for d in dates],
            "Close": closes,
            "AdjClose": adjcloses,
        }
    )


def _div_prices(dates, closes, ex, ratio_before):
    """Synthetic prices with a dividend-implied ratio step at ex-date."""
    adj = [
        c * ratio_before if d.strftime("%Y-%m-%d") < ex else c
        for d, c in zip(dates, closes)
    ]
    return _prices_df(dates, closes, adj)


def test_plausible_split_no_warning(tmp_path):
    """A sane listed split near real price data: no warning (info note only)."""
    from qrp.data.store import connect_db

    path, dates, closes = _actions_db(tmp_path)
    splits = pd.DataFrame(
        {"date": ["2024-03-15"], "numerator": [2.0], "denominator": [1.0]}
    )
    fake = lambda t, s, e: {
        "splits": splits,
        "dividends": pd.DataFrame(columns=["ex_date", "amount"]),
        "prices": _prices_df(dates, closes, closes),
    }
    with connect_db(path) as conn:
        out = Q.check_corporate_actions(
            conn, ["SPLT"], fetch_actions=fake, asof=date(2024, 7, 2)
        )
    assert [f for f in out if f.severity == "warning"] == []
    # ...but the run notes that splits can't be price-verified.
    assert any("can't be verified from split-adjusted" in f.detail for f in out)


def test_unusual_split_factor_flagged(tmp_path):
    """A 1000:1 listed split is implausible: warn with the SEC note."""
    from qrp.data.store import connect_db

    path, dates, closes = _actions_db(tmp_path)
    splits = pd.DataFrame(
        {"date": ["2024-03-15"], "numerator": [1000.0], "denominator": [1.0]}
    )
    fake = lambda t, s, e: {
        "splits": splits,
        "dividends": pd.DataFrame(columns=["ex_date", "amount"]),
        "prices": _prices_df(dates, closes, closes),
    }
    with connect_db(path) as conn:
        out = Q.check_corporate_actions(
            conn, ["SPLT"], fetch_actions=fake, asof=date(2024, 7, 2)
        )
    warns = [f for f in out if f.severity == "warning"]
    assert any("unusual 1000-for-1" in f.summary for f in warns)
    assert any("SEC" in f.detail for f in warns)


def test_split_with_no_nearby_price_data_is_info(tmp_path):
    """Split ex-date far from any price data: info, not warning."""
    from qrp.data.store import connect_db

    path, dates, closes = _actions_db(tmp_path)
    splits = pd.DataFrame(
        {"date": ["1999-06-01"], "numerator": [2.0], "denominator": [1.0]}
    )
    fake = lambda t, s, e: {
        "splits": splits,
        "dividends": pd.DataFrame(columns=["ex_date", "amount"]),
        "prices": _prices_df(dates, closes, closes),
    }
    with connect_db(path) as conn:
        out = Q.check_corporate_actions(
            conn, ["SPLT"], fetch_actions=fake, asof=date(2024, 7, 2)
        )
    assert any(
        f.severity == "info" and "no price data near that date" in f.summary
        for f in out
    )


def test_declared_dividend_with_matching_step_no_warning(tmp_path):
    """Declared $5 dividend and the ratio steps by ~D/C: quiet."""
    from qrp.data.store import connect_db

    path, dates, closes = _actions_db(tmp_path)
    upsert_dividends(
        pd.DataFrame({"ex_date": ["2024-02-15"], "amount": [5.00]}), "SPLT", path
    )
    divs = pd.DataFrame({"ex_date": ["2024-02-15"], "amount": [5.00]})
    fake = lambda t, s, e: {
        "splits": pd.DataFrame(columns=["date", "numerator", "denominator"]),
        "dividends": divs,
        "prices": _div_prices(dates, closes, "2024-02-15", 0.95),
    }
    with connect_db(path) as conn:
        out = Q.check_corporate_actions(
            conn, ["SPLT"], fetch_actions=fake, asof=date(2024, 7, 2)
        )
    assert [f for f in out if f.severity == "warning"] == []


def test_declared_dividend_missing_step_warns(tmp_path):
    """Declared $5 dividend but the ratio doesn't step: warn."""
    from qrp.data.store import connect_db

    path, dates, closes = _actions_db(tmp_path)
    divs = pd.DataFrame({"ex_date": ["2024-02-15"], "amount": [5.00]})
    fake = lambda t, s, e: {
        "splits": pd.DataFrame(columns=["date", "numerator", "denominator"]),
        "dividends": divs,
        "prices": _prices_df(dates, closes, closes),  # flat ratio: no step
    }
    with connect_db(path) as conn:
        out = Q.check_corporate_actions(
            conn, ["SPLT"], fetch_actions=fake, asof=date(2024, 7, 2)
        )
    warns = [f for f in out if f.severity == "warning"]
    assert any("doesn't show the expected step" in f.summary for f in warns)
    assert any("SEC" in f.detail for f in warns)


def test_undeclared_dividend_step_warns(tmp_path):
    """A 5% ratio step with no declared dividend: warn about a missing payout."""
    from qrp.data.store import connect_db

    path, dates, closes = _actions_db(tmp_path)
    fake = lambda t, s, e: {
        "splits": pd.DataFrame(columns=["date", "numerator", "denominator"]),
        "dividends": pd.DataFrame(columns=["ex_date", "amount"]),
        "prices": _div_prices(dates, closes, "2024-02-15", 0.95),
    }
    with connect_db(path) as conn:
        out = Q.check_corporate_actions(
            conn, ["SPLT"], fetch_actions=fake, asof=date(2024, 7, 2)
        )
    warns = [f for f in out if f.severity == "warning"]
    assert any("no dividend is listed" in f.summary for f in warns)


def test_missing_dividend_warning(tmp_path):
    """Yahoo lists a dividend the DB doesn't have: warn."""
    from qrp.data.store import connect_db

    path, dates, closes = _actions_db(tmp_path)
    divs = pd.DataFrame({"ex_date": ["2024-02-10"], "amount": [0.50]})
    fake = lambda t, s, e: {
        "splits": pd.DataFrame(columns=["date", "numerator", "denominator"]),
        "dividends": divs,
        "prices": _prices_df(dates, closes, closes),
    }
    with connect_db(path) as conn:
        out = Q.check_corporate_actions(
            conn, ["SPLT"], fetch_actions=fake, asof=date(2024, 7, 2)
        )
    assert any(
        f.severity == "warning" and "$0.50" in f.summary and "isn't in the database" in f.summary
        for f in out
    )


def test_stored_dividend_not_listed_is_info(tmp_path):
    """DB has a dividend Yahoo no longer lists: info, not warning."""
    from qrp.data.store import connect_db

    path, dates, closes = _actions_db(tmp_path)
    upsert_dividends(
        pd.DataFrame({"ex_date": ["2024-02-10"], "amount": [0.50]}), "SPLT", path
    )
    fake = lambda t, s, e: {
        "splits": pd.DataFrame(columns=["date", "numerator", "denominator"]),
        "dividends": pd.DataFrame(columns=["ex_date", "amount"]),
        "prices": _prices_df(dates, closes, closes),
    }
    with connect_db(path) as conn:
        out = Q.check_corporate_actions(
            conn, ["SPLT"], fetch_actions=fake, asof=date(2024, 7, 2)
        )
    assert any(
        f.severity == "info" and "no longer lists" in f.summary for f in out
    )


def test_actions_network_failure_is_info_not_crash(tmp_path):
    from qrp.data.store import connect_db

    path, _, _ = _actions_db(tmp_path)

    def boom(t, s, e):
        raise ConnectionError("nope")

    with connect_db(path) as conn:
        out = Q.check_corporate_actions(
            conn, ["SPLT"], fetch_actions=boom, asof=date(2024, 7, 2)
        )
    assert len(out) == 1 and out[0].severity == "info"
    assert "skipped" in out[0].summary


# --- cross-vendor ----------------------------------------------------------------------

def test_cross_vendor_mismatch_warns(db):
    def fake_bars(ticker, start, end):
        return pd.DataFrame(
            {"Date": ["2024-03-29"], "Close": [9999.0]}  # way off stored
        )

    from qrp.data.store import connect_db

    with connect_db(db) as conn:
        out = Q.check_cross_vendor(conn, ["AAA"], fetch_bars=fake_bars)
    warns = [f for f in out if f.severity == "warning"]
    assert any("doesn't match a fresh download" in f.summary for f in warns)


def test_cross_vendor_agrees_no_warning(db):
    def fake_bars(ticker, start, end):
        return pd.DataFrame({"Date": [start], "Close": [114.4]})

    from qrp.data.store import connect_db

    with connect_db(db) as conn:
        # stored AAA close on 2024-03-29 = 100 + 62*0.05 = 103.1; use it exactly
        row = conn.execute(
            "SELECT close FROM daily_bars WHERE ticker='AAA' ORDER BY date DESC LIMIT 1"
        ).fetchone()
        close = float(row[0])
        out = Q.check_cross_vendor(
            conn, ["AAA"],
            fetch_bars=lambda t, s, e: pd.DataFrame({"Date": [s], "Close": [close]}),
        )
    assert [f for f in out if f.severity == "warning"] == []


def test_cross_vendor_limitation_always_documented(db):
    from qrp.data.store import connect_db

    with connect_db(db) as conn:
        out = Q.check_cross_vendor(conn, ["AAA"], fetch_bars=None)
    assert any(
        f.severity == "info" and "free data plans" in f.summary for f in out
    )


# --- report orchestration -----------------------------------------------------------------

def test_report_structure_and_programmatic_access(db):
    rep = Q.run_data_health_checks(
        db, asof=ASOF, fetch_actions=None, fetch_bars=None, network_sample=0
    )
    assert rep.tickers_checked == 2
    assert rep.checked_at  # ISO timestamp present
    counts = rep.counts()
    assert set(counts) == {"info", "warning", "error"}
    assert rep.worst_severity() in ("ok", "info", "warning", "error")
    d = rep.as_dict()
    assert d["tickers_checked"] == 2 and "findings" in d
    assert isinstance(rep.plain_english_summary(), str)
    # Every finding carries the fields P6 needs.
    for f in rep.findings:
        assert f.check and f.severity in ("info", "warning", "error")
        assert f.summary


def test_report_no_false_positives_on_clean_data(db):
    rep = Q.run_data_health_checks(
        db, asof=ASOF, fetch_actions=None, fetch_bars=None, network_sample=0
    )
    assert rep.worst_severity() == "info"  # only the "skipped" notes
    assert not rep.has_errors() and not rep.has_warnings()


def test_network_sampling_flagged_first(tmp_path):
    """Flagged tickers get the network checks before the sample fills."""
    path = tmp_path / "s.db"
    dates = pd.bdate_range("2024-01-02", "2024-03-29")
    n = len(dates)
    for i in range(10):
        upsert_bars(_frame(dates, (100.0 + i for _ in range(n))), f"S{i:02d}", path)
    closes = [50.0 for _ in range(n)]
    closes[10] *= 1.6
    upsert_bars(_frame(dates, closes), "ZZZ", path)  # flagged by outlier check

    seen = []

    def fake_actions(ticker, start, end):
        seen.append(ticker)
        return {
            "splits": pd.DataFrame(columns=["date", "numerator", "denominator"]),
            "dividends": pd.DataFrame(columns=["ex_date", "amount"]),
            "prices": _prices_df(dates, [100.0] * n, [100.0] * n),
        }

    rep = Q.run_data_health_checks(
        path, asof=ASOF, fetch_actions=fake_actions, fetch_bars=None,
        network_sample=3,
    )
    assert "ZZZ" in seen, "flagged ticker must be in the network sample"
    assert len(seen) == 3
    assert any("ran on 3 of 11 tickers" in f.summary for f in rep.findings)


def test_run_checks_missing_db(tmp_path):
    rep = Q.run_data_health_checks(tmp_path / "nope.db", asof=ASOF)
    assert rep.tickers_checked == 0
    assert any("No database yet" in f.summary for f in rep.findings)


# --- FIGI -----------------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self._payload = payload if payload is not None else []
        self.headers = headers or {}
        import json as _json

        self.text = _json.dumps(self._payload)[:200]

    def json(self):
        return self._payload


def _figi_item(figi, name="ACME INC"):
    return {
        "data": [
            {
                "figi": figi,
                "name": name,
                "ticker": "X",
                "compositeFIGI": "BBG000000001",
                "shareClassFIGI": "BBG000000002",
                "securityType": "Common Stock",
                "marketSector": "Equity",
            }
        ]
    }


def test_figi_batches_ten_per_request_and_throttles():
    calls, sleeps = [], []

    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        return _FakeResp(payload=[_figi_item("BBG000B9XRY4") for _ in json])

    client = F.OpenFigiClient(
        post=fake_post, sleep=sleeps.append, min_interval_s=2.6
    )
    out = client.map_tickers([f"T{i:03d}" for i in range(25)])
    assert [len(b) for b in calls] == [10, 10, 5]
    assert all(v is not None and v.figi == "BBG000B9XRY4" for v in out.values())
    # throttle sleeps before the 2nd and 3rd request
    assert len([s for s in sleeps if s > 2.0]) == 2


def test_figi_429_retry_honors_retry_after():
    seq = [
        _FakeResp(status=429, headers={"Retry-After": "0.5"}),
        _FakeResp(payload=[_figi_item("BBG000BPH459")]),
    ]
    calls, sleeps = [], []

    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        return seq.pop(0)

    client = F.OpenFigiClient(
        post=fake_post, sleep=sleeps.append, min_interval_s=0
    )
    out = client.map_tickers(["MSFT"])
    assert out["MSFT"].figi == "BBG000BPH459"
    assert 0.5 in sleeps


def test_figi_500_retries_then_marks_batch_failed():
    def fake_post(url, json=None, timeout=None):
        return _FakeResp(status=503)

    client = F.OpenFigiClient(
        post=fake_post, sleep=lambda s: None, min_interval_s=0, max_retries=2
    )
    out = client.map_tickers(["AAA", "BBB"])
    assert out == {"AAA": None, "BBB": None}


def test_figi_error_payload_and_invalid_figi_are_none():
    def fake_post(url, json=None, timeout=None):
        return _FakeResp(
            payload=[{"error": "No identifier found."}, _figi_item("XX")]
        )

    client = F.OpenFigiClient(
        post=fake_post, sleep=lambda s: None, min_interval_s=0
    )
    out = client.map_tickers(["DEAD", "WEIRD"])
    assert out["DEAD"] is None and out["WEIRD"] is None


def test_figi_dash_ticker_retries_with_slash():
    """BRK-B (Yahoo spelling) falls back to BRK/B (FIGI spelling)."""
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append([j["idValue"] for j in json])
        if json[0]["idValue"] == "BRK-B":
            return _FakeResp(payload=[{"error": "No identifier found."}])
        return _FakeResp(payload=[_figi_item("BBG000DWG505", "BERKSHIRE HATHAWAY INC-CL B")])

    client = F.OpenFigiClient(
        post=fake_post, sleep=lambda s: None, min_interval_s=0
    )
    out = client.map_tickers(["BRK-B"])
    assert calls[0] == ["BRK-B"] and calls[1] == ["BRK/B"]
    assert out["BRK-B"] is not None
    assert out["BRK-B"].figi == "BBG000DWG505"


def test_refresh_figi_mapping_caches_in_db(tmp_path):
    path = tmp_path / "f.db"
    init_db(path)
    dates = pd.bdate_range("2024-01-02", "2024-01-10")
    frame = _frame(dates, (1.0 for _ in dates))
    upsert_bars(frame, "AAA", path)
    upsert_bars(frame, "BBB", path)

    net = {"n": 0}

    def fake_post(url, json=None, timeout=None):
        net["n"] += 1
        return _FakeResp(
            payload=[_figi_item("BBG000B9XRY4", "AAA INC"),
                     _figi_item("BBG000BPH459", "BBB INC")]
        )

    client = F.OpenFigiClient(
        post=fake_post, sleep=lambda s: None, min_interval_s=0
    )
    r1 = F.refresh_figi_mapping(path, client=client)
    assert r1.mapped == 2 and net["n"] == 1
    assert "Mapped 2 ticker(s)" in r1.message()
    # second run: everything cached, no HTTP
    r2 = F.refresh_figi_mapping(path, client=client)
    assert net["n"] == 1
    assert "already mapped" in r2.message()
    # stored rows are valid
    row = get_security_id("AAA", path)
    assert row["figi"] == "BBG000B9XRY4"
    assert row["name"] == "AAA INC"


def test_refresh_figi_mapping_failed_tickers_reported(tmp_path):
    path = tmp_path / "f2.db"
    init_db(path)
    dates = pd.bdate_range("2024-01-02", "2024-01-10")
    upsert_bars(_frame(dates, (1.0 for _ in dates)), "DEAD", path)

    def fake_post(url, json=None, timeout=None):
        return _FakeResp(payload=[{"error": "No identifier found."}])

    client = F.OpenFigiClient(
        post=fake_post, sleep=lambda s: None, min_interval_s=0
    )
    r = F.refresh_figi_mapping(path, client=client)
    assert r.mapped == 0 and r.failed == ["DEAD"]
    assert "couldn't map 1" in r.message()


def test_upsert_security_ids_idempotent(tmp_path):
    path = tmp_path / "s.db"
    row = {
        "ticker": "AAA", "figi": "BBG000B9XRY4", "composite_figi": "BBG000000001",
        "share_class_figi": "BBG000000002", "name": "AAA INC",
        "security_type": "Common Stock", "mapped_at": "2024-01-01T00:00:00+00:00",
    }
    assert upsert_security_ids([row], path) == 1
    assert upsert_security_ids([row], path) == 1  # re-map: no duplicate
    assert get_security_id("AAA", path)["figi"] == "BBG000B9XRY4"


def test_figi_mapping_status(tmp_path):
    path = tmp_path / "st.db"
    init_db(path)
    dates = pd.bdate_range("2024-01-02", "2024-01-10")
    frame = _frame(dates, (1.0 for _ in dates))
    upsert_bars(frame, "AAA", path)
    upsert_bars(frame, "BBB", path)
    upsert_security_ids(
        [{"ticker": "AAA", "figi": "BBG000B9XRY4", "mapped_at": "x"}], path
    )
    st = F.figi_mapping_status(path)
    assert st["mapped"] == 1 and st["total"] == 2
    assert st["unmapped"] == ["BBB"]


def test_valid_figi():
    assert F._valid_figi("BBG000B9XRY4")
    assert not F._valid_figi("XX")
    assert not F._valid_figi(None)
    assert not F._valid_figi("AAA000B9XRY4")  # wrong prefix


# --- backup/restore ------------------------------------------------------------------------------

def _priced_db(tmp_path, name="b.db"):
    path = tmp_path / name
    init_db(path)
    dates = pd.bdate_range("2024-01-02", "2024-01-31")
    upsert_bars(_frame(dates, (10.0 for _ in dates)), "AAA", path)
    return path


def test_backup_roundtrip_and_verify(tmp_path):
    db = _priced_db(tmp_path)
    r = B.create_backup(db, keep=5)
    assert r.ok and r.path is not None
    assert "Backed up to" in r.message
    ok, msg = B.verify_backup(r.path)
    assert ok and "integrity check" in msg
    infos = B.list_backups(db)
    assert len(infos) == 1 and infos[0].name == r.path.name


def test_backup_retention_prunes_oldest(tmp_path):
    import time as _time

    db = _priced_db(tmp_path)
    for _ in range(7):
        B.create_backup(db, keep=5)
        _time.sleep(1.1)
    infos = B.list_backups(db)
    assert len(infos) == 5, "retention keeps the newest 5"


def test_restore_replaces_data_and_saves_safety_copy(tmp_path):
    from qrp.data.store import connect_db, count_bars

    db = _priced_db(tmp_path)
    n_before = count_bars("AAA", path=db)
    old = B.create_backup(db, keep=5)
    dates = pd.bdate_range("2024-02-01", "2024-02-29")
    upsert_bars(_frame(dates, (99.0 for _ in dates)), "AAA", db)
    assert count_bars("AAA", path=db) > n_before
    res = B.restore_backup(old.path, db)
    assert res.ok
    assert "safety copy" in res.message or "saved first" in res.message
    assert count_bars("AAA", path=db) == n_before
    safety = [i for i in B.list_backups(db) if i.name.startswith("pre-restore-")]
    assert len(safety) == 1
    with connect_db(db) as conn:
        (n,) = conn.execute("SELECT COUNT(*) FROM daily_bars").fetchone()
    assert n == n_before


def test_restore_missing_file_plain_english(tmp_path):
    db = _priced_db(tmp_path)
    res = B.restore_backup("missing.db", db)
    assert not res.ok
    assert "no backup file" in res.message.lower()
    assert "traceback" not in res.message.lower()


def test_restore_corrupt_file_plain_english_and_changes_nothing(tmp_path):
    from qrp.data.store import count_bars

    db = _priced_db(tmp_path)
    bad = B.backup_dir(db) / "corrupt.db"
    B.backup_dir(db).mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"this is not a database")
    before = count_bars("AAA", path=db)
    res = B.restore_backup(bad, db)
    assert not res.ok
    assert "corrupted" in res.message or "readable" in res.message
    assert count_bars("AAA", path=db) == before, "corrupt restore must change nothing"


def test_backup_no_db_plain_english(tmp_path):
    r = B.create_backup(tmp_path / "nope.db")
    assert not r.ok and "no database" in r.message.lower()


def test_backup_cli(tmp_path, capsys):
    db = _priced_db(tmp_path)
    assert B.main(["--backup", "--db", str(db)]) == 0
    assert "Backed up to" in capsys.readouterr().out
    assert B.main(["--list", "--db", str(db)]) == 0
    assert "market-" in capsys.readouterr().out
    infos = B.list_backups(db)
    assert B.main(["--verify", str(infos[0].path), "--db", str(db)]) == 0
    assert "integrity check" in capsys.readouterr().out
    assert B.main(["--restore", "missing.db", "--db", str(db)]) == 1
    assert "no backup file" in capsys.readouterr().out.lower()


# --- UI helpers ------------------------------------------------------------------------------------

def test_ui_helper_wrappers(tmp_path, monkeypatch):
    from qrp.ui import helpers as H

    db = _priced_db(tmp_path)
    msg = H.backup_now_for_ui(db)
    assert "Backed up to" in msg
    infos = H.backup_list_for_ui(db)
    assert len(infos) == 1 and infos[0]["name"].endswith(".db")

    # restore requires explicit confirmation — never one click
    not_confirmed = H.restore_backup_for_ui(infos[0]["name"], db, confirmed=False)
    assert "confirmation" in not_confirmed.lower()
    confirmed = H.restore_backup_for_ui(infos[0]["name"], db, confirmed=True)
    assert "Restored from" in confirmed

    # figi helpers degrade gracefully without network
    status = H.figi_mapping_status_for_ui(db)
    assert status["mapped"] == 0 and status["total"] == 1


def test_run_checks_for_ui_uses_real_fetchers_but_never_raises(tmp_path, monkeypatch):
    """Even with the network unreachable, the UI entry point completes."""
    from qrp.ui import helpers as H

    db = _priced_db(tmp_path)
    monkeypatch.setattr(
        "qrp.data.yahoo.fetch_actions",
        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline")),
    )
    monkeypatch.setattr(
        "qrp.data.yahoo.fetch_daily_bars",
        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline")),
    )
    rep = H.run_data_health_checks_for_ui(db)
    assert rep.tickers_checked == 1
    assert isinstance(rep.worst_severity(), str)


# --- dashboard page (AppTest) ----------------------------------------------------------------------

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture
def seeded_default_db():
    """Prices in the repo's default DB location, cleaned up afterwards."""
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "data" / "market.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()
    dates = pd.bdate_range("2024-01-02", "2024-03-29")
    n = len(dates)
    upsert_bars(_frame(dates, (100.0 + i * 0.05 for i in range(n))), "AAA", p)
    upsert_bars(_frame(dates, (50.0 - i * 0.02 for i in range(n))), "BBB", p)
    yield p
    if p.exists():
        p.unlink()


def _app():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return AppTest.from_file(str(root / "qrp" / "ui" / "app.py"))


def test_data_health_page_runs_checks_and_renders(seeded_default_db, monkeypatch):
    """Data health page: button runs checks (faked), findings render plainly."""
    from qrp.ui import helpers as H

    canned = Q.DataHealthReport(
        checked_at="2024-04-02T00:00:00+00:00",
        tickers_checked=2,
        findings=[
            Q.Finding(
                check="stale",
                severity="warning",
                ticker="AAA",
                summary="AAA's newest price is from 2024-01-29 — 5 trading days behind.",
                detail="detail text",
                action="re-download",
            )
        ],
    )
    monkeypatch.setattr(
        H, "run_data_health_checks_for_ui", lambda db_path=None: canned
    )
    at = _app()
    at.session_state[H.NAV_PAGE_KEY] = "Data health"
    at.run()
    assert not at.exception
    btn = next(b for b in at.button if b.label == "Run data-health checks")
    btn.click().run()
    assert not at.exception
    assert any("2" in str(m.value) for m in at.metric), "tickers-checked metric"
    assert any(
        "newest price is from 2024-01-29" in str(w.value) for w in at.warning
    ), "the canned warning renders"


def test_data_health_page_backup_buttons(seeded_default_db):
    """Back up now creates a backup; the restore button needs confirmation."""
    from qrp.ui import helpers as H

    at = _app()
    at.session_state[H.NAV_PAGE_KEY] = "Data health"
    at.run()
    assert not at.exception
    btn = next(b for b in at.button if b.label == "Back up now")
    btn.click().run()
    assert not at.exception
    assert any("Backed up to" in str(s.value) for s in at.success)
    # restore button is disabled until its confirmation checkbox is ticked
    restore_btns = [b for b in at.button if b.label == "Restore"]
    assert restore_btns and all(b.disabled for b in restore_btns)
