"""Trade execution: broker interface + paper trading (Phase 5).

- broker.py: BrokerClient abstract base class (the simulation seam) plus the
  Broker order-lifecycle interface (AccountInfo/OrderInfo) that both the
  Alpaca adapter and the local paper broker speak.
- paper.py: PaperBroker — simulated account, real math, zero real dollars —
  plus PaperBrokerAdapter, which wraps it in the Broker interface.
- alpaca.py: AlpacaBroker — Alpaca PAPER account via its REST API (paper
  only; no live-trading code path is exposed).
- rebalance.py: turn a target portfolio into an order list (sells before buys).
- cli.py: python -m qrp.execution.cli --dry-run

Paper trading ONLY. There is no live-broker code path in this repo.
"""
