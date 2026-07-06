# polymarket-predictor

Live trading bot and research suite for Polymarket's 5-minute and 15-minute
Chainlink-settled crypto up/down markets (BTC/ETH/SOL/XRP/DOGE/BNB).

The production strategy is a **dollar-bar "price-to-beat" (PTB) contrarian
model**: calibrated gradient-boosted classifiers predict `P(close > PTB)` on
information-driven dollar bars and take the predicted side. The repo also
contains the full body of research that led there — directional, market-making,
LSTM, and stop-loss variants — most of which were tested and killed.

## Honest results

This is a research record, not a money printer. The rigorous, fee-aware,
bias-controlled backtests in here concluded that **the BTC 5m/15m book is
efficient**: directional signals, passive market-making, and the dollar-bar
model all net to ~zero or negative *after* the Polymarket crypto taker fee
(≈0.07·p·(1−p) per share). The only edge that survived scrutiny is cross-book
**arbitrage** (see the companion `polymarket-arb-scanner` repo), and that is
latency-gated. The code is published for the engineering and the methodology.

## Layout

- `live_trader/` — the live bot: order placement, redemption, risk, PTB
  features, decisioning, arb executor.
- `datastore/` — market-data access layer.
- `models/*.py` — model inference/definition code. **Trained artifacts
  (`*.joblib`) live in the `trading-models` repo.**
- `tools/`, `sti/` — utilities.
- `analyze_*.py`, `*_backtest.py`, `s5_*.py`, `mm_*.py`, `quant_*.py` —
  the backtest / analysis suite.
- `docs/` — design specs and plans for each strategy iteration.
- `tests/`, `test_*.py` — the test suite.
- Notebooks — baseline / indicator / oracle / signal demos.

## Companion repos

- [`trading-models`](https://github.com/vatupelage/trading-models) — trained model artifacts + training code
- [`market-data-captures`](https://github.com/vatupelage/market-data-captures) — datasets
- [`polymarket-arb-scanner`](https://github.com/vatupelage/polymarket-arb-scanner) — the one surviving edge
- [`latency-edge-lab`](https://github.com/vatupelage/latency-edge-lab) — CEX→CLOB latency study

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-live.txt
cp .env.example .env   # fill in your own Polymarket keys — never commit .env
```

Secrets (private key, API credentials, funder address) load from `.env`. No
credentials are committed to this repo.

---

## Support

If this work helped you, you can support me:

- **USDT** (ERC-20): `0xfba2de3360ae0d98ec44216191d143bc28676af5`
- **USDC** (ERC-20): `0xfba2de3360ae0d98ec44216191d143bc28676af5`
- **BTC**: `14EVe4ejvXrSS6s34AUBP9TMoSsuzShJ8o`

Have a vibe-coding project you'd like to collaborate on? Get in touch: **rawanaholdingslk@gmail.com**
