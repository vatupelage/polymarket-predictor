# Spec 1 — Single-region capture collector (latency-edge measurement harness)

**Date:** 2026-06-21
**Status:** design approved, pending spec review
**Project:** Distributed latency-edge measurement harness for Polymarket Chainlink-resolved crypto markets
**This spec covers:** the first sub-project only — a single-region, record-only capture collector + its Parquet schema + a 1-hour sanity gate. Cross-region merge, clock-offset *validation between regions*, PHC, the fair-value model, the capturability simulation, and the acceptance-gate report are **explicitly later specs**.

---

## 1. Objective (scientific, not a trading bot)

Measure whether a *capturable* latency edge exists in the gap between exchange spot feeds
(Binance, Coinbase) and Polymarket CLOB repricing on short-duration (5m/15m) Chainlink-settled
crypto markets — and, across later specs, from which of three AWS regions it is reachable
(ap-northeast-1 Tokyo, eu-west-1 Ireland, us-east-1 N. Virginia).

This sub-project builds **only the data-capture foundation on one region**. It produces clean,
clock-stamped, restart-safe capture data and proves it with a 1-hour sanity check. It draws **no
edge conclusions** and contains **no analysis**.

## 2. Non-negotiable principles

1. **Record-only.** No order submission, no wallet, no signing, no private key. Execution is made
   *structurally impossible*: the collector package imports nothing from `live_trader`'s
   order/wallet path; the only Polygon endpoint it touches is a read-only RPC it *pings*. There is
   no code path that builds, signs, or sends a transaction.
2. **Clock discipline is the foundation.** Every captured event carries: wall-clock time, a
   monotonic timestamp, AND the current estimated clock offset/uncertainty. No downstream
   conclusion (in later specs) may be reported with an error bar smaller than the clock
   uncertainty. *For a single region there is one clock*, so this spec only requires per-event
   stamping from chrony; cross-region offset validation and PHC are deferred to Spec 2.
3. **Distributions, not point estimates.** Latency probes report p50/p90/p99, never just the mean.
4. **Pre-registered acceptance gates** (section 9) are fixed now, before any data exists, and are
   not moved post-hoc.

## 3. Scope boundaries

**In scope (Spec 1):**
- One region-agnostic collector daemon capturing all five feed families + latency probes from
  whatever box it runs on.
- Per-event clock-stamp envelope (wall + monotonic + chrony offset + error bound).
- Restart-safe, immutable, time-bucketed Parquet storage + a documented schema.
- Per-feed reconnect, sequence-gap accounting, and a `--dry-run` replay path.
- A 1-hour live-capture sanity gate.

**Out of scope (later specs):**
- Cross-region log merge and the clock-corrected unified timeline (Spec 2/3).
- PHC detection/fallback and cross-region clock-offset *validation* (Spec 2).
- Fair-value model, signal detection, capturability simulation, depth-constrained PnL, the
  per-region acceptance-gate report (Spec 3).
- Three-region systemd deploy (Spec 2) — Spec 1 develops/proves on one box.

## 4. Architecture

- **Language:** Python (extends the existing `predictor/edgelab/` harness rather than a Rust
  rewrite). Rationale: the binding constraint for this whole project is the Polygon-submit RTT
  (tens of ms), not feed-arrival scheduling jitter (sub-ms), so the GIL/asyncio jitter is not
  decision-relevant; reusing a working, restart-safe capture harness is the fastest honest path
  to first data. (If a later spec proves feed-arrival jitter *is* decision-relevant, a small Rust
  arrival-stamp binary can be added then — YAGNI until shown.)
- **Shape:** one asyncio process per region. Independent tasks per feed family + a probe task, all
  writing to a shared local writer, then (later spec) shipped to S3.
- **Develop/prove on:** the Ireland / all-markets box (`eu-west-1`, existing Python env + working
  edgelab capture). Later regions (Tokyo, Virginia) are a config-only roll.

### 4.1 Reuse vs. new

| Reused from `predictor/edgelab/` | New for this spec |
|---|---|
| Gamma poller (`resolve_tokens`, `fetch_outcome`) for window/token/strike discovery + rollover | Binance combined-WS collector (trade + bookTicker) |
| Per-horizon asyncio rollover pattern (`horizon_loop`) | Coinbase Exchange-WS collector (matches + ticker) |
| Restart-safe "immutable Parquet file, never rewrite" discipline | PM **RTDS oracle** WS collector (`wss://ws-live-data.polymarket.com`) |
| Parquet/Arrow writer scaffolding | PM **CLOB** collector at **raw event level** (not distilled to top-of-book — latency needs per-event arrival times) |
| `replay.py` offline-validation pattern | **Clock-stamp module** (chrony offset + error bound per event) |
| | **Latency-probe subsystem** (JSON-RPC ping + TLS-handshake RTT) |

## 5. Feeds captured

All tagged with the section-6 envelope. Symbol set lives in config; **Spec 1 runs BTC-only** while
the code stays symbol-agnostic (every row carries `symbol`; adding ETH/SOL is a config + Gamma-check,
not a rebuild).

| Feed | Endpoint | Streams |
|---|---|---|
| Binance combined WS | `wss://stream.binance.com:9443` | `<sym>usdt@trade`, `<sym>usdt@bookTicker` |
| Coinbase Exchange WS | `wss://ws-feed.exchange.coinbase.com` | `matches`, `ticker` for `<SYM>-USD` |
| PM oracle (RTDS) | `wss://ws-live-data.polymarket.com` | Chainlink price as surfaced |
| PM CLOB market | `wss://ws-subscriptions-clob.polymarket.com/ws/market` | `book` + `price_change` for current token ids (raw events) |
| Gamma poller | `https://gamma-api.polymarket.com` | discover active 5m/15m BTC markets: token ids, strike, window open/close; handle rollover |
| Latency probes | Polygon read-only RPC (config) + each feed host | JSON-RPC round-trip ping (lower bound on submit leg) + TLS-handshake RTT |

### 5.1 Latency probe (first-class data stream)

- **Submit-leg proxy:** continuously time lightweight read-only JSON-RPC calls (`eth_blockNumber` /
  `eth_gasPrice`) to the *exact* RPC endpoint a real submission would use, sampling the full
  distribution including p99. This is a deliberate **lower bound** on real land time (it omits
  mempool propagation and block inclusion) and MUST be labelled as such everywhere it surfaces.
- **TLS-handshake RTT:** measure connection-establishment RTT to the RPC host (and each feed host)
  as a cheaper, purely network-layer second probe, for network-vs-RPC attribution.
- Probes run on a fixed cadence (configurable, default ~1–5 s) as their own `source` rows; they are
  treated as a first-class feed, not an afterthought.

## 6. Per-event schema (uniform stamp envelope)

Every captured event — feed or probe — is one Parquet row:

```
region_id            str    # "eu-west-1"
source               str    # binance_trade | binance_bookticker | coinbase_match
                            # | coinbase_ticker | pm_oracle | pm_clob_book
                            # | pm_clob_price_change | probe_rpc | probe_tls | gap | session
symbol               str?   # BTC (null for non-symbol probes / session rows)
window_slug          str?   # PM window this event belongs to (null for CEX feeds / probes)
recv_wall_ns         i64    # CLOCK_REALTIME at ingest
recv_monotonic_ns    i64    # CLOCK_MONOTONIC at ingest (clean intra-host deltas)
clock_offset_ns      i64    # chrony estimated offset at ingest
clock_err_ns         i64    # chrony root dispersion (= the error bound, the caveat source)
local_ingest_seq     i64    # per-source monotonic counter (gap detection)
exch_seq             i64?   # exchange/feed-native sequence number when present
payload_json         str    # raw message, for forensic re-parse
# parsed convenience columns (nullable per source):
price                f64?
size                 f64?
side                 str?   # buy|sell|bid|ask
best_bid             f64?
best_ask             f64?
best_bid_sz          f64?
best_ask_sz          f64?
rtt_ns               i64?   # probe round-trip (probe_* rows)
```

`clock_err_ns` is carried on **every** row so no downstream conclusion can be stated without its
error bar. `payload_json` guarantees lossless re-parse if a parser bug is found later.

## 7. Storage & restart-safety

CEX/oracle/probe feeds are continuous (unlike edgelab's window-bounded PM capture), so files are
**time-bucketed**, not window-bucketed:

```
events/day=YYYY-MM-DD/source=<source>/<region>-<epoch_minute>.parquet
```

- Files are rotated each minute (or every N buffered events, whichever first) and are **immutable
  once closed** — a mid-flush restart loses at most the current in-memory buffer (a gap, never a
  dup or corrupt append), exactly edgelab's guarantee. Closed files are never rewritten.
- `EDGELAB_OUT`-style env override for the output root (server keeps data on `/mnt/data`).
- **Accounting:** per-source `local_ingest_seq` plus exchange-native `exch_seq` (where available)
  drive a sequence-gap detector; detected gaps are written as `source=gap` rows (start_seq,
  end_seq, count) so dropped-message volume is itself part of the dataset.
- A `sessions.jsonl` (mirroring edgelab's `windows.jsonl`) records collector start/stop and every
  per-feed reconnect with timestamps.

## 8. Robustness, dry-run, testing

- **Reconnect:** each feed task auto-reconnects with exponential backoff; reconnects are logged as
  `source=session` rows/lines.
- **Dry-run:** `--dry-run` replays a recorded raw-message sample file through the identical
  parse → stamp → write path, so the full pipeline is testable offline with no network (mirrors
  `edgelab/replay.py`).
- **TDD on the pure units:**
  - one message parser per feed (raw message → parsed columns), incl. malformed-message handling;
  - the clock-stamp envelope builder (given a fake chrony reading → correct fields);
  - the sequence-gap accountant (seq stream with holes → correct `gap` rows);
  - the Parquet rotator (buffer + time/N trigger → immutable file naming, no overwrite).

## 9. Pre-registered acceptance gates (recorded now; used in Spec 3, not moved later)

Declare "edge reachable from region R" (a **later** spec) ONLY if ALL hold:

- **Exposure:** `D ≥ 7` capture-days **AND** `N ≥ 30` capturable fills — whichever resolves
  **later** is binding. (Signal-exposure, not calendar time, is the real unit: a quiet low-vol week
  must not manufacture a confident null off near-zero N, and a wild 3 hours must not manufacture a
  confident yes off tiny N.)
- **Edge margin:** median net edge per fill `>` (fee + modeled slippage) by a factor `K ≥ 3`.
  Rationale specific to this harness: the submit-latency proxy is a deliberate lower bound, so the
  capturability sim *already over-counts* reachable fills; modeled-vs-realized slippage error cuts
  the same direction. K≥3 buys back that optimism. We are not seeking the smallest detectable edge;
  we are seeking one robust enough to survive the gap between the lower-bound proxy and real land
  time. A marginal edge that only clears at lower-bound latency is one that would be lost at real
  latency — calling it inconclusive is the correct error for this project.
- **Clock gate:** the `clock_err_ns` upper bound must be `<` the median signal-persistence window;
  otherwise the harness refuses to declare an edge.

No post-hoc threshold tuning.

## 10. Milestone-1 sanity gate (this spec's "done")

A 1-hour live BTC-only capture on the Ireland box, asserting:

1. All five feed sources produced rows (`binance_*`, `coinbase_*`, `pm_oracle`, `pm_clob_*`).
2. Per-source dropped-message rate (from `gap` rows) is below a documented threshold.
3. Every row carries a non-null, finite `clock_err_ns`.
4. Both probe sources (`probe_rpc`, `probe_tls`) have a populated RTT distribution (p50/p90/p99
   computable).
5. Gamma-discovery check: confirm BTC lists on **both** 5m and 15m with concurrent liquidity; if
   the 5m book is materially thinner, note it explicitly (thin-book signals fail the depth>0 filter
   downstream and must not inflate the signal count).

## 11. Deliverables

- Collector package under `predictor/edgelab/` (new modules: `collectors/` per feed, `clockstamp.py`,
  `probes.py`, `writer.py`, a `harness.py` entrypoint) + a `--dry-run` sample.
- Documented Parquet schema (this section 6) + storage layout (section 7) in the package README.
- Unit tests for every pure unit (section 8).
- A 1-hour sanity-check script/report producing the section-10 assertions.

## 12. Pitfalls handled explicitly

- **Public WS feeds may egress from a CDN/edge, not the engine region** — so we measure *actual*
  per-region arrival time and never assume Tokyo is fastest to Binance until the data says so (this
  bites in Spec 2's cross-region comparison; Spec 1 just records honestly).
- **"Saw it first" ≠ "could have taken it"** — the Polygon-submit leg is usually the binding
  constraint, which is why the probe subsystem is first-class and the submit proxy is a labelled
  lower bound.
- **Symbol-agnostic but BTC-only** — running BTC-only avoids tangling ETH/SOL thin-book /
  adverse-selection issues with first-capture schema bugs; the code carries `symbol` from day one.
