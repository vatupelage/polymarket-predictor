# Fee-Aware Capturability Check — Design

**Date:** 2026-06-21
**Status:** approved (brainstorm), pending plan
**Branch:** predictfun-port
**Depends on:** the latency-harness collector (Spec 1) and its captured data; see
`2026-06-21-latency-harness-collector-design.md`.

## 1. Objective and honest limits

Answer exactly one question: **does the captured Binance→Polymarket-CLOB lag translate
into a fill that is profitable after the Polymarket taker fee?**

This is a **kill test**, deliberately the simplest trigger that could possibly show edge.
The motivating finding (1h Ireland capture, exploratory): sign-aligned pooled
cross-correlation across 24 BTC markets shows Binance spot **leads** the CLOB reprice by
~0.3s (peak corr +0.224 at +0.3s). Latency is not the blocker from Ireland (reaction fits
inside the window); the open question is whether the mispricing exceeds the fee.

**It is directional, not the pre-registered acceptance gate.** One hour cannot satisfy the
Spec-1 gate (`D ≥ 7` capture-days **AND** `N ≥ 30` capturable fills, later-of). A clear
"net edge ≪ fee" kills the project cheaply. A "looks positive" result only justifies
accumulating more capture-days and, after that, the multi-region build (deferred Spec 2).

A complex fair-value model is **deferred to a later spec on purpose**: it is exactly where
look-ahead and overfitting manufacture false positives. If a dead-simple causal trigger
cannot find edge, a fancier one "finding" it would be the thing to distrust.

## 2. Data and scope

- **Input:** an existing capture directory (the 1h Ireland run to start; reruns unchanged
  on more days/regions as data accrues). BTC only, code symbol/region-agnostic.
- **Sources used:** `binance_bookticker` (spot mid), `pm_clob_book` (hittable ask + size +
  mid, per `asset_id`), `pm_oracle` (settlement reference for resolution).
- **Single clock:** every row is stamped on one box's `recv_wall_ns`; lead/lag and fills are
  computed entirely within that one clock, so no cross-region clock question arises.
- **No collector changes.** This is a read-only analysis module over captured Parquet.

## 3. Method (strictly causal)

All times are in the capture's Ireland `recv_wall_ns` clock.

1. **Trigger** at detection time `t` (a `binance_bookticker` event): a Binance spot move of
   magnitude `≥ θ` over a short trailing lookback `L` (default `L = 0.5s`), where the move is
   a **relative return in bps** (`|mid_t / mid_{t−L} − 1|`), not an absolute dollar move, so
   `θ` is scale-invariant across price levels. Detection at `t`
   already carries the ~105ms Binance→Ireland feed-arrival (it is `recv_wall`, not exchange
   time) — see §6 on reaction-time coordinates.
2. **Predicted side** `d`: the up-token if spot rose, the down-token if it fell. Up/down
   **polarity is determined per asset from price levels** (lag-independent: sign of
   `corr(spot_level, clob_mid_level)` over the asset's life), never assumed from token order.
3. **Arrival** `A = t + R`, where `R` = detection→order-land (default headline `R = 100ms`;
   see §6).
4. **Hittable ask:** the predicted-side token's real `best_ask`, forward-filled to `A`
   (last book update with `recv_wall ≤ A`). **Depth-capped** to the captured `best_ask_sz`
   and a fixed stake (a parameter; default $5, matching live dbmodel sizing). Never the mid;
   never a future ask.
5. **Fee:** **per-share** `fee = FEE_RATE × ask × (1 − ask)`, charged **on entry only** (see
   §5). `FEE_RATE = 0.07` is **verified** against the Polymarket docs for crypto markets.
6. **Valuation — two, with a strict hierarchy (§4):**
   - **Primary — hold-to-resolution:** value `∈ {1, 0}` from the oracle value vs the
     market strike at expiry, for fills whose entry preceded an expiry inside the capture.
     `net_edge = value − ask − fee`.
   - **Diagnostic — mark-to-reprice:** value = CLOB mid at `A + H` (sweep `H` across the
     lead decay, ~0.6s to a few s). Explains *why* (did we capture the lag; was it
     informative). `net_edge_mark = mark − ask − fee` (entry-only fee; the code asserts the
     valuation being costed so an entry-only fee is never applied to a notional "sale").
7. **Null control — random entry:** identical machinery (same fill count, depth-cap, fee,
   both valuations) on random entry times. The momentum trigger must **beat the random
   baseline by a margin**, not merely clear zero. Random clearing the fee under
   mark-to-reprice is evidence the *valuation* is leaky, not that signal is real.

## 4. Valuation hierarchy (the axis a kill test most easily lies on)

Marking a fill at the very reprice it exploits is near-tautological — buy-before-reprice,
mark-at-reprice books a near-guaranteed paper gain that is real only if the reprice
direction matches settlement. Therefore:

- **A positive verdict is gated on hold-to-resolution net of fee.** Mark-to-reprice cannot
  produce a positive verdict on its own.
- **Mark-to-reprice is diagnostic only:** lag-captured + resolution-loss ⇒ the reprice was
  uninformative (a mirage), and a real position would have lost.
- **Thin resolution N is itself the finding**, reported verbatim as "mechanism
  directionally encouraging, cannot be valued honestly in 1h — accumulate days." It is
  never papered over by a thicker mark-to-reprice sample; the readout leads with
  resolution N.

## 5. Fee (verified 2026-06-21)

The whole verdict scales with the fee, so it was confirmed against the source before the
plan, not inherited:

- **Verified formula (Polymarket docs, crypto markets):** `fee = C × 0.07 × p × (1 − p)`,
  C = shares, p = share price, rate `0.07` for crypto (BTC/ETH/SOL, all windows incl. 5/15
  min). Source: `https://docs.polymarket.com/trading/fees`.
  - **Per share:** `fee = 0.07 × p × (1 − p)` — symmetric, **peaks at p = 0.50 at 1.75%**
    ($1.75 / 100 shares), decaying toward 1¢/99¢. (Note: equivalently, fee **per dollar
    staked** = `0.07 × (1 − p)`, ≈3.5% at the money — a different denominator; the sim works
    in per-share units, so it uses `0.07 × p × (1 − p)`.) This corrects the earlier working
    prior `0.07 × (1 − ask)`, which was the per-stake rate misapplied per share.
  - The peak bar is therefore **1.75% per share at the money**, materially lower than the
    3.5% previously assumed — it makes the kill test less harsh, not more.
- **Maker/taker:** makers never pay; takers pay; charged once at match time.
- **Round-trip vs entry-only:** the primary (resolution) path is **entry-only** — winners
  resolve at $1.00 and losers at $0, and redemption is not a match, so a single entry fee is
  correct. Mark-to-reprice notionally "sells"; if a sale ever carries its own fee, that path
  is a ~2× round-trip. The code **asserts which valuation it is costing** so the two never
  cross.

## 6. Reaction-time coordinate (avoiding a double-count)

Detection in the sim occurs at the Binance event's `recv_wall` = `E + ~105ms` (feed-arrival
already included). So "reaction-from-detection" `R` and an "end-to-end-from-exchange"
number are the same instant from different origins: `135ms from exchange ≈ 30ms from
detection`. Adding a literal 135ms on top of detection would land at `E + 240ms`,
double-counting the 105ms and rigging the test to fail.

Everything stays in the one Ireland `recv_wall` clock with `R` = detection→order-land:

- **Headline cell: `R = 100ms`** (≈ `E + 205ms` end-to-end — already more conservative than
  the live ~135ms; covers JSON parse + signal compute + EIP-712 signing + POST +
  FOK/queueing slack).
- **Sweep:** `R ∈ {30 (network-only "best conceivable" bound), 60, 100 (headline), 150
  (stress)}`. The optimistic 30ms is reported only as a bound, never as the verdict.

This is strictly more pessimistic than live end-to-end without the double-count.

## 7. Parameter sweeps and outputs

- **Sweeps:** `θ` (move threshold), `R` (reaction, §6), `H` (mark horizon, diagnostic only).
- **Per cell, report:** `N` fills; **(primary)** resolution net-edge median + fraction
  positive + the count of resolved fills; **(diagnostic)** mark net-edge median; the
  **random-entry baseline** for the same cell; and **fills/hour ⇒ days-to-`N≥30`** per
  market.
- **Decision readout (mood, not gate):** resolution median net edge per fill placed beside
  `fee × K` with `K ≥ 3`, with `N_resolved` shown first. Plus the trigger-vs-random margin.

## 8. Honesty guards (summary)

- Causal split: trigger uses data `≤ t`; ask at `A`; value strictly after `A`.
- **Sign-align up/down tokens** from levels before any pooling (the trap that zeroed the
  naïve cross-correlation: 12 up + 12 down cancelled to ~0).
- Depth-cap fills to real captured `best_ask_sz`.
- Fee on every fill; assert which valuation it is costed against.
- Resolution-primary; mark-to-reprice diagnostic-only; thin `N_resolved` reported, not hidden.
- Random-entry null baseline that the trigger must beat by a margin.
- Report `N` per cell so thin cells are not over-read.

## 9. Structure

- A self-contained, read-only analysis module under `predictor/edgelab/` (working name
  `capture_sim.py`) plus a thin CLI/report, runnable against any capture directory.
- **Tests on synthetic fixtures** with a known injected lead → known profit/loss, a known
  no-signal series (random control must wash out), and a sign-flipped (down-token) fixture
  to prove polarity handling. Tests named `predictor/test_edgelab_capture_sim.py`, run from
  `predictor/`.
- Reuses `edgelab.schema` columns and, for strike/outcome resolution, the existing
  `edgelab.logger` token/market resolution path (Gamma).

## 10. Out of scope (later)

- Fair-value model and richer signals (overfitting/look-ahead risk) — later spec.
- Multi-region deploy + cross-region clock validation + PHC + S3 (Spec 2) — revisited only
  if the edge survives fees and feed-arrival shaving becomes the marginal lever.
- Live execution of any kind. This module never submits, signs, or holds a key.
