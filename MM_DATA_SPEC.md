# Market-Making Data Logging Spec — Polymarket BTC 5m

Goal: collect enough data to **backtest** whether providing liquidity (posting
resting limit orders) is profitable on the BTC up/down 5m markets — *before*
risking capital. We are no longer predicting direction (proven efficient); we
are testing whether we can earn the **spread** and/or **liquidity rewards**
faster than **adverse selection** + **inventory risk** eat them.

The three numbers the backtest must produce:
1. **Fill rate** — P(a resting quote at distance d from mid gets hit before window close).
2. **Adverse selection** — after a fill, how far does mid move *against* us (informed flow picking us off).
3. **Net edge** = captured spread + liquidity reward − adverse selection − fees − inventory mark-to-resolution.

If (1)+(rewards) − (2) − fees > 0, market-making is viable. Nothing else matters.

---

## What we have vs. what we throw away

`live_trader/polymarket.py` already calls `get_order_book(token_id)` but keeps
only `min(asks)` / `max(bids)` (polymarket.py:91-130). The full depth, the trade
tape, tick size, and the rewards config are all reachable through the v2 client
but never logged. This spec adds **read-only logging** — no new orders, no risk.

Client methods to use (all confirmed present on `ClobClient`):
- `get_order_book(token_id)` → `{bids:[{price,size}], asks:[{price,size}], timestamp, hash}`
- `get_market_trades_events(condition_id)` / `get_trades(...)` → executed prints (the tape)
- `get_tick_size(token_id)` → min price increment (defines the quotable grid)
- `get_spread(token_id)`, `get_midpoint(token_id)` → convenience (derivable from book)
- `get_raw_rewards_for_market(condition_id)` → the liquidity-mining reward schedule
- `fetch_resolution(slug)` → final mark (already implemented, polymarket.py:214)

---

## Log 1 — `mm_book.jsonl` (full depth time series)

One record per **snapshot**, per **window**, for **both** outcome tokens.
Cadence: every **2 seconds** from window open to close (~150 snapshots/window).
Both UP and DOWN tokens each snapshot (a fill on either side is inventory).

```json
{
  "ts": "2026-05-29T08:48:01.123",
  "slug": "btc-updown-5m-1780024500",
  "secs_to_close": 247,
  "token": "UP",                       // log a paired record for "DOWN" too
  "token_id": "...",
  "tick_size": 0.01,
  "bids": [[0.46, 1200.0], [0.45, 800.0], [0.44, 350.0]],   // [price, size], best-first, full depth
  "asks": [[0.48, 950.0], [0.49, 600.0], [0.50, 2000.0]],
  "book_hash": "...",                  // dedupe identical snapshots
  // derived (so the backtest doesn't recompute):
  "best_bid": 0.46, "best_ask": 0.48, "mid": 0.47, "spread": 0.02,
  "bid_depth_2c": 2000.0,              // total size within 2 cents of mid
  "ask_depth_2c": 1550.0,
  "microprice": 0.4712                 // size-weighted mid = (bid*ask_sz + ask*bid_sz)/(bid_sz+ask_sz)
}
```

Why each field:
- **full bids/asks** → spread, depth, book shape, and **queue position** (size resting
  at our intended price tells us how much fills ahead of us).
- **secs_to_close** → spread/fill behavior changes drastically near resolution; must bucket by it.
- **book_hash** → skip writing unchanged snapshots; keeps file size sane.
- **microprice** → better short-horizon fair value than mid; adverse-selection benchmark.

## Log 2 — `mm_tape.jsonl` (executed trades — THE critical one)

One record per executed trade on either token. Poll `get_market_trades_events`
every 2s; dedupe by trade id. **Without this, fill rate is unmeasurable.**

```json
{
  "ts": "2026-05-29T08:48:03.500",
  "slug": "btc-updown-5m-1780024500",
  "token": "UP", "token_id": "...",
  "trade_id": "...",                   // dedupe key
  "price": 0.48,
  "size": 320.0,
  "taker_side": "BUY",                 // BUY = someone lifted an ask (a resting ask got filled)
  "secs_to_close": 245
}
```

From the tape we reconstruct: at each price level, how much volume traded through
it = how much of a resting order at that level **would have filled**, and in what
direction the taker came (which tells us adverse selection on the next snapshot).

## Log 3 — `mm_sim_quotes.jsonl` (our hypothetical maker quotes)

We never send these — we *record what we would have posted*, then the backtest
checks the tape + book to see if/when they'd fill and what happened to mid after.
Write once per snapshot (every 2s), for a small grid of quote distances.

```json
{
  "ts": "2026-05-29T08:48:01.123",
  "slug": "btc-updown-5m-1780024500",
  "token": "UP",
  "mid_at_quote": 0.47,
  "secs_to_close": 247,
  "quotes": [                          // candidate resting orders at d cents off mid
    {"side": "BID", "price": 0.46, "dist_c": 1, "size_ahead": 1200.0},
    {"side": "BID", "price": 0.45, "dist_c": 2, "size_ahead": 2000.0},
    {"side": "ASK", "price": 0.48, "dist_c": 1, "size_ahead": 950.0},
    {"side": "ASK", "price": 0.49, "dist_c": 2, "size_ahead": 1550.0}
  ]
}
```

`size_ahead` = size already resting at that price (our queue position). The
backtest fills our sim order only after `size_ahead` units have traded through
that price on the tape (Log 2), then marks the post-fill mid (Log 1) for adverse
selection.

## Log 4 — `mm_rewards.jsonl` (liquidity-mining schedule, once per window)

```json
{
  "ts": "2026-05-29T08:45:00",
  "slug": "btc-updown-5m-1780024500",
  "condition_id": "...",
  "rewards": { /* raw get_raw_rewards_for_market payload */ },
  "min_size": 100.0,                   // min quote size to qualify (from payload)
  "max_spread": 0.03,                  // max distance from mid that still earns (from payload)
  "rate_per_day": 5.0                  // reward pool / token, if exposed
}
```

This decides whether MM is viable **even if spread capture nets zero** — if the
reward for two-sided quoting exceeds adverse selection, you get paid to provide
liquidity regardless of direction.

---

## Implementation notes

- **Read-only.** All four logs come from `get_*` calls only. No orders. Zero capital risk. Safe to run alongside or instead of the trader.
- **Reuse the shadow-log pattern.** `bot.py` already appends JSONL per poll
  (S5_SHADOW_LOG_PATH, bot.py:58). Add a `MMLogger` that, given the two token_ids
  for the active window, writes Logs 1–4 on a 2s timer until `secs_to_close <= 0`.
- **Rate limits.** 2s cadence × 2 tokens × (book + tape) = 4 calls / 2s. Batch with
  `get_order_books([...])` / `get_last_trades_prices([...])` (plural variants exist)
  to cut this to ~2 calls / 2s. Back off on 429.
- **Storage.** ~150 snapshots × 2 tokens × ~400 bytes ≈ 120 KB/window, ~35 MB/day
  at full window coverage. Use `book_hash` dedupe to roughly halve it. Rotate daily.
- **Clock.** Log wall-clock ts AND secs_to_close; the backtest keys off secs_to_close.

## Minimum collection before backtesting

- **≥ 300 windows** (~5–7 days of continuous logging) to estimate fill rate and
  adverse selection per (secs_to_close bucket × quote distance) cell with n≥30.
- Must span **multiple BTC regimes** (a trend day and a chop day at minimum) —
  adverse selection is regime-dependent and will be worst on trend days.

## The backtest this enables (quant level, same rigor as the S5 work)

1. **Fill model**: P(fill | dist_c, secs_to_close, size) from Logs 2+3, walk-forward.
2. **Adverse-selection model**: E[mid move | filled] over next N seconds, by regime.
3. **Reward credit**: per-window reward from Log 4, prorated by qualifying quote time.
4. **Net P&L sim**: post two-sided quotes each snapshot → fills (1) → spread captured
   − adverse selection (2) + rewards (3) − fees − final inventory marked at resolution.
5. **Significance**: bootstrap CI on per-window P&L, walk-forward train/test, regime
   buckets — refuse to confirm unless the holdout EV is positive with t>2 and the
   CI excludes zero (the bar S5 failed).
