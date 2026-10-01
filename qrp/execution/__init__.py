"""Trade execution: broker interface + paper trading (Phase 5).

- broker.py: BrokerClient abstract base class (the seam a real broker plugs into later).
- paper.py: PaperBroker — simulated account, real math, zero real dollars.
- rebalance.py: turn a target portfolio into an order list (sells before buys).
- cli.py: python -m qrp.execution.cli --dry-run

Paper trading ONLY. There is no live-broker code in this repo.
"""
