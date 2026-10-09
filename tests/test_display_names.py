"""Display-name consistency: formal names everywhere, old labels nowhere.

The strategy registry and the signal registry each own a formal,
user-facing display name. Every picker, table, summary, and CLI listing
must route through those registries — never title-case a key at the call
site. This module pins that contract with a blocklist of legacy labels.
"""

from qrp.signals import default_registry
from qrp.strategies import STRATEGIES, display_name as strategy_display_name

# Legacy informal labels that must never appear as a user-facing display
# name. (Covers the vague originals plus the wrong "12-Month" variant
# corrected earlier — the formal name is "Low Volatility (3-Month)".)
LEGACY_LABELS = {
    "Momentum",
    "Trend",
    "Low Volatility",
    "Low Volatility (12-Month)",
    "Value Composite",
    "Trending Value",
    "Profitability",
    "Quality Lowvol",
    "Value Momentum",
    "Quality + Low Volatility",
    "Continuous Momentum",
    "Overnight Intraday",
    "Twin Momentum",
}


def _signal_display_names() -> dict[str, str]:
    reg = default_registry()
    return {name: reg.display_name(name) for name, _ in reg.list_signals()}


def _strategy_display_names() -> dict[str, str]:
    return {name: strategy_display_name(name) for name in STRATEGIES}


def test_every_signal_has_formal_name():
    reg = default_registry()
    for name, _ in reg.list_signals():
        sig = reg.get(name)
        assert sig.formal_name.strip(), f"signal {name!r} is missing formal_name"


def test_every_strategy_has_formal_name():
    for name, s in STRATEGIES.items():
        assert s.formal_name.strip(), f"strategy {name!r} is missing formal_name"


def test_no_legacy_labels_in_strategy_display_names():
    names = _strategy_display_names()
    for key, label in names.items():
        assert label not in LEGACY_LABELS, (
            f"strategy {key!r} displays legacy label {label!r}"
        )


def test_no_legacy_labels_in_signal_display_names():
    names = _signal_display_names()
    for key, label in names.items():
        assert label not in LEGACY_LABELS, (
            f"signal {key!r} displays legacy label {label!r}"
        )


def test_signal_display_names_come_from_registry_not_title_case():
    """display_name() must return the formal name, not a prettified key."""
    reg = default_registry()
    for name, _ in reg.list_signals():
        label = reg.display_name(name)
        assert label == reg.get(name).formal_name, (
            f"signal {name!r}: display_name {label!r} != formal_name"
        )


def test_strategy_display_name_fallback_for_unknown_names():
    assert strategy_display_name("not_registered") == "Not Registered"


def test_signal_display_name_fallback_for_unknown_names():
    reg = default_registry()
    assert reg.display_name("not_registered") == "Not Registered"


def test_expected_formal_names():
    strategies = _strategy_display_names()
    assert strategies["momentum"] == "12-1 Month Cross-Sectional Momentum"
    assert strategies["trend"] == "200-Day Moving-Average Trend Filter"
    assert strategies["low_volatility"] == "Low Volatility (3-Month)"
    assert strategies["trending_value"] == "Trending Value (O'Shaughnessy)"
    assert strategies["value_momentum"] == "Value + Momentum Composite (AQR-Style)"
    assert strategies["quality_lowvol"] == "Quality + Low Volatility Composite"
    assert strategies["continuous_momentum"] == "Continuous Momentum (Frog-in-the-Pan)"
    assert (
        strategies["overnight_intraday"]
        == "Intraday Institutional Flow (Overnight Divergence)"
    )
    assert strategies["twin_momentum"] == "Twin Momentum (Price + Earnings Surprise)"

    signals = _signal_display_names()
    assert signals["momentum"] == "12-1 Month Cross-Sectional Momentum"
    assert signals["trend"] == "200-Day Moving-Average Trend Filter"
    assert signals["low_volatility"] == "Low Volatility (3-Month)"
    assert signals["trending_value"] == "Trending Value (O'Shaughnessy)"
    assert signals["continuous_momentum"] == "Continuous Momentum (Frog-in-the-Pan)"
    assert (
        signals["overnight_intraday"]
        == "Intraday Institutional Flow (Overnight Divergence)"
    )
    assert signals["twin_momentum"] == "Twin Momentum (Price + Earnings Surprise)"
    # Signals with no same-key strategy still get honest formal names.
    assert "Book-to-Market" in signals["value_composite"]
    assert "Gross Profit" in signals["profitability"]


def test_backtest_cli_unknown_strategy_lists_formal_names(capsys):
    from qrp.backtest.cli import main

    rc = main(["--strategy", "nope_not_real"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "12-1 Month Cross-Sectional Momentum [momentum]" in out
    assert "Twin Momentum (Price + Earnings Surprise) [twin_momentum]" in out


def test_backtest_cli_unknown_signal_lists_formal_names(capsys):
    from qrp.backtest.cli import main

    rc = main(["--signal", "nope_not_real"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "12-1 Month Cross-Sectional Momentum [momentum]" in out
    assert "Twin Momentum (Price + Earnings Surprise) [twin_momentum]" in out


def test_signals_cli_unknown_signal_lists_formal_names(capsys):
    from qrp.signals.cli import main

    rc = main(["--signals", "nope_not_real"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "12-1 Month Cross-Sectional Momentum [momentum]" in out


def test_strategy_help_text_shows_formal_names():
    from qrp.backtest.cli import parse_args

    # --help text is built at parse time; formal names must be in it.
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            parse_args(["--help"])
    except SystemExit:
        pass
    out = buf.getvalue()
    flat = " ".join(out.split())  # argparse wraps help text across lines
    assert "Trending Value (O'Shaughnessy) [trending_value]" in flat


# -- chronological order (signal list == strategy list sequence) --------------

# The one canonical order every picker shares: build order, matching the
# strategy list (original technicals, fundamental batch, research batch).
CANONICAL_SIGNAL_ORDER = [
    "momentum",
    "trend",
    "low_volatility",
    "trending_value",
    "value_composite",
    "profitability",
    "continuous_momentum",
    "overnight_intraday",
    "twin_momentum",
]

# Signal keys that are also strategy keys: their display names must be
# identical in both registries (the two building-block signals,
# value_composite and profitability, have no same-key strategy).
SHARED_SIGNAL_STRATEGY_KEYS = [
    "momentum",
    "trend",
    "low_volatility",
    "trending_value",
    "continuous_momentum",
    "overnight_intraday",
    "twin_momentum",
]


def test_signal_list_is_in_canonical_chronological_order():
    reg = default_registry()
    names = [name for name, _ in reg.list_signals()]
    assert names == CANONICAL_SIGNAL_ORDER


def test_shared_signal_names_match_strategy_names():
    reg = default_registry()
    for key in SHARED_SIGNAL_STRATEGY_KEYS:
        assert reg.display_name(key) == strategy_display_name(key), (
            f"{key!r}: signal label {reg.display_name(key)!r} != "
            f"strategy label {strategy_display_name(key)!r}"
        )


def test_signal_display_helper_uses_signal_formal_names():
    """H.signal_display_name resolves SIGNAL formal names (leg captions)."""
    import qrp.ui.helpers as H

    assert H.signal_display_name("momentum") == "12-1 Month Cross-Sectional Momentum"
    assert "Book-to-Market" in H.signal_display_name("value_composite")
    assert "Gross Profit" in H.signal_display_name("profitability")
