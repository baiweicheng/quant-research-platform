"""Named investment strategies (Phase 8). See qrp.strategies.registry."""

from qrp.strategies.registry import (
    STRATEGIES,
    Strategy,
    composite_scores,
    display_name,
    get_strategy,
    list_strategies,
    make_strategy_fn,
    strategy_signal_matrices,
    strategy_target_portfolio,
    strategy_target_weights,
)

__all__ = [
    "Strategy",
    "STRATEGIES",
    "get_strategy",
    "list_strategies",
    "display_name",
    "strategy_signal_matrices",
    "composite_scores",
    "strategy_target_weights",
    "strategy_target_portfolio",
    "make_strategy_fn",
]
