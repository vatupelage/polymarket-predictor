# Multi-bot bankroll allocation simulator

**Date:** 2026-06-18
**Status:** Approved, building

## Goal

Given **$200 shared capital** and three live-candidate dbmodel bots — `btc_5m`,
`bnb_15m`, `eth_15m` — output a **fixed dollar stake per bot** (maps directly to
`BOT_STAKE_USDC`) that **maximizes 30-day median bankroll growth** subject to a
hard risk cap: **P(drawdown > 50% of $200, i.e. bankroll ever below $100) < 5%**
over the horizon, evaluated under a stressed (decayed) edge.

This is a sizing decision before going live on a shared wallet. It is NOT a
go-live; it produces three numbers + the evidence behind them.

## Decisions locked in brainstorming

| Question | Decision |
|---|---|
| Objective | Max median growth s.t. low ruin |
| Risk limit | P(drawdown > 50%) < 5% over horizon |
| Edge haircut | Exact Polymarket fee per-trade + decay stress band |
| Sizing form | Fixed $ per bot, re-sized periodically |
| Horizon | 30 days (~10,800 combined trades) |
| Capital | $200 shared pool across all 3 bots |

## 1. Data & return model

Source `(won, ask)` pairs per bot:
- `btc_5m` -> live `trade_history.jsonl` (eu-west-1, real fills, n~1823)
- `bnb_15m` / `eth_15m` -> paper logs (all-markets box, n~84/86)

Net return **per $1 staked**, per trade:

```
gross = (1/ask - 1) if won else -1
fee   = 0.07 * (1 - ask)        # exact Polymarket crypto taker fee, entry only
net   = gross - fee
```

Fee formula derivation: Polymarket taker fee = `shares * 0.07 * p * (1-p)`.
For a fixed-$ stake, `shares = stake/ask`, so `fee/stake = 0.07*(1-ask)`.
Source: https://docs.polymarket.com/trading/fees (crypto feeRate = 0.07).

**Fee-model validation:** for `btc_5m`, compare predicted `net` to the actual
realized `pnl/stake` from the live log. Match -> trust `0.07*(1-ask)` for bnb/eth
(no live fills yet). Divergence -> report the empirical btc gap and use it.

## 2. Edge scenarios

- **Base:** net returns as above (paper edge minus fee).
- **Stressed:** halve each bot's residual mean edge via a parallel downward
  shift by `mu/2` (preserves variance, halves mean) to model IS->OOS decay.

The **recommended stakes must satisfy the ruin constraint under the stressed
case.** Base is reported for upside only.

## 3. Monte-Carlo engine

- Bankroll $200, 30-day horizon. Firing rates: btc~160/day, bnb/eth~100/day each
  -> ~360/day -> ~10,800 trades/path.
- Each trade: assign to a bot proportional to firing rate, bootstrap-draw a net
  return from that bot's empirical distribution, `pnl = stake_bot * draw`, update
  bankroll. Markov / i.i.d. resample (same assumption as
  `bankroll_quant_backtest.py`).
- ~50k paths. Track final bankroll and running minimum (max drawdown).
- Capital feasibility: skip a trade if bankroll < stake. Concurrent open capital
  is negligible vs $200 at these stakes; v1 assumes sequential settlement
  (documented simplification).

## 4. Optimization

1. Compute each bot's **Kelly fraction** from its stressed net-return
   distribution -> relative allocation weights.
2. Binary-search a single **scale multiplier** on that weight vector until
   `P(dd>50%) = 5%` under the stressed edge -> recommended 3 stakes.
3. **Comparison table:** Kelly-scaled (recommended) vs equal-$ vs
   edge-proportional, each with median final bankroll, P(dd>50%), drawdown
   P50/P90/P99, P(profit).

## 5. Deliverable & structure

- One script: `predictor/sim_bankroll_allocation.py`.
- Pure functions: `load_returns`, `validate_fee_model`, `kelly_fraction`,
  `simulate`, `optimize`, `main`.
- **TDD** on math kernels: fee formula, net-return calc, Kelly, ruin metric on a
  known distribution, seeded-sim determinism.
- Output: recommended stakes, metric tables, fee-validation result, caveats
  (small bnb/eth sample, edge uncertainty, regime risk, no live stop-loss).

## Known caveat to surface in results

After fees, `btc_5m`'s edge is ~+0.010/$1 — marginal. The optimizer will likely
allocate it a small stake and concentrate on bnb/eth. Worth weighing before
going live on all three.
