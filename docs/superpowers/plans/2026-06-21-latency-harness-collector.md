# Latency-Harness Single-Region Collector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a record-only, single-region capture collector that streams five Polymarket/CEX feed families plus first-class latency probes, stamps every event with a chrony-derived clock envelope, and writes restart-safe time-bucketed Parquet — proven by a 1-hour sanity check.

**Architecture:** Pure parse/stamp/account/write units (TDD'd in isolation) under `predictor/edgelab/`, wired together by async feed collectors and a harness entrypoint. Reuses edgelab's Gamma poller and restart-safe "immutable Parquet per file, never rewrite" discipline. A `build_rows(family, raw, ...)` pure function is the seam between the network loop and the tested units, which also powers `--dry-run` replay.

**Tech Stack:** Python 3.12, asyncio, `websockets`, `requests`, `pyarrow`/`pyarrow.parquet`, `pytest`. Runs from the `predictor/` directory (so `import edgelab.*` resolves).

## Global Constraints

Every task's requirements implicitly include these (copied from the spec):

- **Record-only.** No imports from `live_trader` order/wallet/signing paths; no private key is ever loaded; no code path builds, signs, or sends a transaction. The only Polygon endpoint touched is a read-only RPC that is *pinged*.
- **`clock_err_ns` on every row.** Every emitted row (feed, probe, gap, session) carries a non-null, finite `clock_err_ns`. No exceptions.
- **Probes report distributions.** Latency probe output must support p50/p90/p99 (store raw `rtt_ns` per sample); the RPC ping is a labelled **lower bound** on the submit leg.
- **BTC-only, symbol-agnostic code.** Symbol set lives in config; every row carries a `symbol` field. Spec 1 runs BTC-only; adding ETH/SOL must be a config + Gamma-check, not a rebuild.
- **Restart-safe storage.** Parquet files are immutable once closed and never rewritten; a mid-flush restart loses at most the current buffer (a gap, never a dup/corruption). Compression `zstd`.
- **Output root via env override.** Output directory honors `EDGELAB_OUT` (data lives on `/mnt/data` on the server).
- **Tests discoverable by the existing glob.** Test files are named `test_edgelab_<module>.py` at the `predictor/` root so `pytest test_edgelab_*.py` runs them; modules import as `from edgelab.<module> import ...`.
- **No new heavy deps.** Only the libraries already used by edgelab (`websockets`, `requests`, `pyarrow`, `pyyaml`) plus the stdlib.

---

## File Structure

New modules (all under `predictor/edgelab/`):

- `schema.py` — source constants, the row column list, the pyarrow schema. The downstream contract.
- `clockstamp.py` — pure `chronyc tracking` parser + `ClockStamper` (builds the per-event envelope, owns per-source seq counters).
- `parse.py` — one pure parser per feed family (`parse_binance`, `parse_coinbase`, `parse_pm_oracle`, `parse_pm_clob`); each returns `list[dict]` of parsed columns + `source`.
- `seqgap.py` — `SeqGapTracker` (per-source exchange-sequence hole detection → `gap` rows).
- `writer.py` — `RotatingParquetWriter` (per-source buffering, minute/N rotation, immutable non-overwriting file naming).
- `probes.py` — JSON-RPC ping payload/parse helpers + `probe_rpc` / `probe_tls` timing coroutines.
- `collect.py` — `build_rows(family, raw, ctx)` pure seam + the async feed collectors (connect/subscribe/recv/reconnect/session rows).
- `windows.py` — Gamma-backed current-BTC-window tracker (reuses `logger.resolve_tokens`) producing the `token_index` for CLOB tagging + subscription.
- `harness.py` — config load, wiring, `asyncio.gather`, `--dry-run` replay entrypoint.
- `sanity_check.py` — reads captured Parquet and asserts the Milestone-1 gate.

Tests at `predictor/` root: `test_edgelab_schema.py`, `test_edgelab_clockstamp.py`, `test_edgelab_parse.py`, `test_edgelab_seqgap.py`, `test_edgelab_writer.py`, `test_edgelab_probes.py`, `test_edgelab_collect.py`.

Config: extend `predictor/edgelab/config.yaml` with a `harness:` section.

---

## Task 1: Schema & source constants

**Files:**
- Create: `predictor/edgelab/schema.py`
- Test: `predictor/test_edgelab_schema.py`

**Interfaces:**
- Produces: `SOURCES` (frozenset of source names), `COLUMNS` (ordered list[str] of all row keys), `ARROW_SCHEMA` (`pyarrow.Schema`), `empty_row() -> dict` (all columns present, set to `None`).

- [ ] **Step 1: Write the failing test**

```python
# predictor/test_edgelab_schema.py
from edgelab import schema

def test_sources_cover_all_families():
    expected = {
        "binance_trade", "binance_bookticker",
        "coinbase_match", "coinbase_ticker",
        "pm_oracle", "pm_clob_book", "pm_clob_price_change",
        "probe_rpc", "probe_tls", "gap", "session",
    }
    assert expected <= schema.SOURCES

def test_empty_row_has_every_column_and_envelope():
    row = schema.empty_row()
    assert set(row) == set(schema.COLUMNS)
    for k in ("region_id", "source", "symbol", "window_slug",
              "recv_wall_ns", "recv_monotonic_ns", "clock_offset_ns",
              "clock_err_ns", "local_ingest_seq", "exch_seq",
              "payload_json", "price", "size", "side",
              "best_bid", "best_ask", "best_bid_sz", "best_ask_sz", "rtt_ns"):
        assert k in row and row[k] is None

def test_arrow_schema_matches_columns():
    assert [f.name for f in schema.ARROW_SCHEMA] == schema.COLUMNS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd predictor && python3 -m pytest test_edgelab_schema.py -v`
Expected: FAIL with `ModuleNotFoundError`/`AttributeError` (schema module/attrs not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# predictor/edgelab/schema.py
"""Row contract for the latency-harness collector: source names, the ordered
column list, and the pyarrow schema every Parquet file conforms to."""
import pyarrow as pa

SOURCES = frozenset({
    "binance_trade", "binance_bookticker",
    "coinbase_match", "coinbase_ticker",
    "pm_oracle", "pm_clob_book", "pm_clob_price_change",
    "probe_rpc", "probe_tls", "gap", "session",
})

# Ordered: envelope first, then parsed convenience columns.
COLUMNS = [
    "region_id", "source", "symbol", "window_slug",
    "recv_wall_ns", "recv_monotonic_ns", "clock_offset_ns", "clock_err_ns",
    "local_ingest_seq", "exch_seq", "payload_json",
    "price", "size", "side",
    "best_bid", "best_ask", "best_bid_sz", "best_ask_sz", "rtt_ns",
]

_TYPES = {
    "region_id": pa.string(), "source": pa.string(), "symbol": pa.string(),
    "window_slug": pa.string(),
    "recv_wall_ns": pa.int64(), "recv_monotonic_ns": pa.int64(),
    "clock_offset_ns": pa.int64(), "clock_err_ns": pa.int64(),
    "local_ingest_seq": pa.int64(), "exch_seq": pa.int64(),
    "payload_json": pa.string(),
    "price": pa.float64(), "size": pa.float64(), "side": pa.string(),
    "best_bid": pa.float64(), "best_ask": pa.float64(),
    "best_bid_sz": pa.float64(), "best_ask_sz": pa.float64(),
    "rtt_ns": pa.int64(),
}

ARROW_SCHEMA = pa.schema([(c, _TYPES[c]) for c in COLUMNS])


def empty_row() -> dict:
    """A row dict with every column present and None — callers fill what they have."""
    return {c: None for c in COLUMNS}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd predictor && python3 -m pytest test_edgelab_schema.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add predictor/edgelab/schema.py predictor/test_edgelab_schema.py
git commit -m "feat(latency-harness): row schema + source constants"
```

---

## Task 2: Clock-stamp envelope

**Files:**
- Create: `predictor/edgelab/clockstamp.py`
- Test: `predictor/test_edgelab_clockstamp.py`

**Interfaces:**
- Consumes: `schema.empty_row`.
- Produces:
  - `parse_chrony_tracking(text: str) -> tuple[int, int]` — returns `(offset_ns, err_ns)` where `offset_ns` is signed (`Last offset`) and `err_ns = round((root_dispersion + root_delay/2) * 1e9)` (NTP maximum-error bound).
  - `class ClockStamper(region_id: str, clock_reader=parse-backed callable)`: method `stamp(source: str, **parsed) -> dict` returns a full `schema` row with `region_id`, `source`, monotonic+wall ns, cached `(offset_ns, err_ns)`, and a per-source incrementing `local_ingest_seq`; unknown `**parsed` keys that are not columns raise `KeyError`.

- [ ] **Step 1: Write the failing test**

```python
# predictor/test_edgelab_clockstamp.py
from edgelab import clockstamp, schema

SAMPLE = """Reference ID    : C0248F88 (time.example)
Stratum         : 3
System time     : 0.000012345 seconds slow of NTP time
Last offset     : -0.000004567 seconds
RMS offset      : 0.000010000 seconds
Root delay      : 0.000500000 seconds
Root dispersion : 0.000250000 seconds
Leap status     : Normal
"""

def test_parse_chrony_offset_and_error_bound():
    offset_ns, err_ns = clockstamp.parse_chrony_tracking(SAMPLE)
    assert offset_ns == -4567               # -0.000004567 s -> ns
    # err = dispersion + delay/2 = 0.000250000 + 0.000250000 = 0.000500000 s
    assert err_ns == 500000

def test_stamp_builds_full_row_with_envelope_and_seq():
    cs = clockstamp.ClockStamper("eu-west-1", reader=lambda: (-4567, 500000))
    r1 = cs.stamp("binance_trade", symbol="BTC", price=64000.0)
    assert set(r1) == set(schema.COLUMNS)
    assert r1["region_id"] == "eu-west-1"
    assert r1["source"] == "binance_trade"
    assert r1["symbol"] == "BTC"
    assert r1["price"] == 64000.0
    assert r1["clock_offset_ns"] == -4567
    assert r1["clock_err_ns"] == 500000
    assert r1["recv_wall_ns"] > 0 and r1["recv_monotonic_ns"] > 0
    assert r1["local_ingest_seq"] == 0
    r2 = cs.stamp("binance_trade", symbol="BTC")
    assert r2["local_ingest_seq"] == 1          # per-source counter advances
    r3 = cs.stamp("coinbase_match", symbol="BTC")
    assert r3["local_ingest_seq"] == 0          # independent per source

def test_stamp_rejects_unknown_column():
    cs = clockstamp.ClockStamper("eu-west-1", reader=lambda: (0, 1))
    try:
        cs.stamp("binance_trade", not_a_column=1)
        assert False, "expected KeyError"
    except KeyError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd predictor && python3 -m pytest test_edgelab_clockstamp.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# predictor/edgelab/clockstamp.py
"""Per-event clock envelope. `parse_chrony_tracking` is pure (text -> ns);
ClockStamper caches a chrony reading and stamps each event with wall+monotonic
time, the offset/error bound, and a per-source ingest sequence."""
import re
import subprocess
import time

from edgelab.schema import empty_row, COLUMNS

_COLSET = set(COLUMNS)


def parse_chrony_tracking(text: str) -> tuple[int, int]:
    """`chronyc tracking` text -> (offset_ns signed, err_ns).
    err = root_dispersion + root_delay/2  (NTP maximum-error bound)."""
    def grab(label: str) -> float:
        m = re.search(rf"{label}\s*:\s*([-+]?[0-9.]+)", text)
        if not m:
            raise ValueError(f"chrony field not found: {label}")
        return float(m.group(1))
    offset = grab("Last offset")
    dispersion = grab("Root dispersion")
    delay = grab("Root delay")
    offset_ns = round(offset * 1e9)
    err_ns = round((dispersion + delay / 2.0) * 1e9)
    return offset_ns, err_ns


def read_chrony() -> tuple[int, int]:
    """Live reading via `chronyc tracking`; returns (0, very-large) if chrony is
    unavailable so the error bound is honestly huge rather than silently small."""
    try:
        out = subprocess.run(["chronyc", "tracking"], capture_output=True,
                             text=True, timeout=2).stdout
        return parse_chrony_tracking(out)
    except Exception:
        return 0, 10**12  # 1 second: unknown clock => refuse to claim precision


class ClockStamper:
    def __init__(self, region_id: str, reader=read_chrony, refresh_s: float = 5.0):
        self.region_id = region_id
        self._reader = reader
        self._refresh_s = refresh_s
        self._cache = reader()
        self._cache_mono = time.monotonic()
        self._seq: dict[str, int] = {}

    def _clock(self) -> tuple[int, int]:
        if time.monotonic() - self._cache_mono >= self._refresh_s:
            self._cache = self._reader()
            self._cache_mono = time.monotonic()
        return self._cache

    def stamp(self, source: str, **parsed) -> dict:
        bad = set(parsed) - _COLSET
        if bad:
            raise KeyError(f"unknown columns: {sorted(bad)}")
        offset_ns, err_ns = self._clock()
        seq = self._seq.get(source, 0)
        self._seq[source] = seq + 1
        row = empty_row()
        row.update(parsed)
        row["region_id"] = self.region_id
        row["source"] = source
        row["recv_wall_ns"] = time.time_ns()
        row["recv_monotonic_ns"] = time.monotonic_ns()
        row["clock_offset_ns"] = offset_ns
        row["clock_err_ns"] = err_ns
        row["local_ingest_seq"] = seq
        return row
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd predictor && python3 -m pytest test_edgelab_clockstamp.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add predictor/edgelab/clockstamp.py predictor/test_edgelab_clockstamp.py
git commit -m "feat(latency-harness): chrony clock-stamp envelope"
```

---

## Task 3: Feed parsers

**Files:**
- Create: `predictor/edgelab/parse.py`
- Test: `predictor/test_edgelab_parse.py`

**Interfaces:**
- Produces four pure functions, each `(msg: dict) -> list[dict]` where each returned dict has a `source` key plus a subset of parsed columns (`price`, `size`, `side`, `best_bid`, `best_ask`, `best_bid_sz`, `best_ask_sz`, `exch_seq`, and for PM clob an `asset_id` lookup key); unrecognized/ignored messages return `[]`:
  - `parse_binance(msg)` — combined-stream envelope `{"stream","data"}`; `bookTicker` → `binance_bookticker`, `trade` → `binance_trade`.
  - `parse_coinbase(msg)` — `match`/`last_match` → `coinbase_match`; `ticker` → `coinbase_ticker`.
  - `parse_pm_oracle(msg)` — `topic=="crypto_prices"` update → `pm_oracle` (carries `symbol_raw` like `btcusdt`).
  - `parse_pm_clob(msg)` — `event_type=="book"` → `pm_clob_book` (top-of-book); `event_type=="price_change"` → one `pm_clob_price_change` row per change. Each carries `asset_id`.

Real sample messages (captured live 2026-06-21) are embedded in the tests below.

- [ ] **Step 1: Write the failing test**

```python
# predictor/test_edgelab_parse.py
from edgelab import parse

def test_binance_bookticker():
    msg = {"stream": "btcusdt@bookTicker", "data": {
        "u": 96099642589, "s": "BTCUSDT",
        "b": "64059.99000000", "B": "0.98934000",
        "a": "64060.00000000", "A": "3.26974000"}}
    rows = parse.parse_binance(msg)
    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == "binance_bookticker"
    assert r["best_bid"] == 64059.99 and r["best_ask"] == 64060.0
    assert r["best_bid_sz"] == 0.98934 and r["best_ask_sz"] == 3.26974
    assert r["exch_seq"] == 96099642589

def test_binance_trade():
    msg = {"stream": "btcusdt@trade", "data": {
        "e": "trade", "E": 1782030435506, "s": "BTCUSDT", "t": 6427631900,
        "p": "64060.00000000", "q": "0.00033000", "T": 1782030435506,
        "m": False, "M": True}}
    rows = parse.parse_binance(msg)
    assert rows[0]["source"] == "binance_trade"
    assert rows[0]["price"] == 64060.0 and rows[0]["size"] == 0.00033
    assert rows[0]["side"] == "buy"          # m=False => buyer is taker
    assert rows[0]["exch_seq"] == 6427631900

def test_coinbase_match():
    msg = {"type": "last_match", "trade_id": 1040839665, "side": "buy",
           "size": "0.00000015", "price": "63983.99", "product_id": "BTC-USD",
           "sequence": 130993420811, "time": "2026-06-21T08:27:16.364726Z"}
    rows = parse.parse_coinbase(msg)
    assert rows[0]["source"] == "coinbase_match"
    assert rows[0]["price"] == 63983.99 and rows[0]["side"] == "buy"
    assert rows[0]["exch_seq"] == 130993420811

def test_coinbase_ticker():
    msg = {"type": "ticker", "sequence": 130993420900, "product_id": "BTC-USD",
           "price": "63984.00", "best_bid": "63983.99", "best_ask": "63984.01",
           "best_bid_size": "1.5", "best_ask_size": "2.0",
           "time": "2026-06-21T08:27:17.0Z"}
    rows = parse.parse_coinbase(msg)
    assert rows[0]["source"] == "coinbase_ticker"
    assert rows[0]["best_bid"] == 63983.99 and rows[0]["best_ask"] == 63984.01
    assert rows[0]["best_bid_sz"] == 1.5 and rows[0]["best_ask_sz"] == 2.0

def test_pm_oracle_update():
    msg = {"connection_id": "x", "payload": {
        "full_accuracy_value": "64000.12000000", "symbol": "btcusdt",
        "timestamp": 1782030467000, "value": 64000.12},
        "timestamp": 1782030467151, "topic": "crypto_prices", "type": "update"}
    rows = parse.parse_pm_oracle(msg)
    assert rows[0]["source"] == "pm_oracle"
    assert rows[0]["price"] == 64000.12
    assert rows[0]["symbol_raw"] == "btcusdt"

def test_pm_clob_book_top_of_book():
    msg = {"event_type": "book", "asset_id": "TKN_UP",
           "bids": [{"price": "0.48", "size": "100"}, {"price": "0.47", "size": "50"}],
           "asks": [{"price": "0.52", "size": "200"}, {"price": "0.53", "size": "80"}]}
    rows = parse.parse_pm_clob(msg)
    assert len(rows) == 1 and rows[0]["source"] == "pm_clob_book"
    assert rows[0]["asset_id"] == "TKN_UP"
    assert rows[0]["best_bid"] == 0.48 and rows[0]["best_ask"] == 0.52
    assert rows[0]["best_bid_sz"] == 100.0 and rows[0]["best_ask_sz"] == 200.0

def test_pm_clob_price_change_rows():
    msg = {"event_type": "price_change", "asset_id": "TKN_UP",
           "price_changes": [
               {"price": "0.49", "size": "10", "side": "buy"},
               {"price": "0.51", "size": "20", "side": "sell"}]}
    rows = parse.parse_pm_clob(msg)
    assert len(rows) == 2
    assert all(r["source"] == "pm_clob_price_change" for r in rows)
    assert rows[0]["price"] == 0.49 and rows[0]["side"] == "buy"
    assert rows[1]["price"] == 0.51 and rows[1]["side"] == "sell"

def test_unknown_messages_ignored():
    assert parse.parse_binance({"stream": "x", "data": {"e": "depthUpdate"}}) == []
    assert parse.parse_coinbase({"type": "subscriptions"}) == []
    assert parse.parse_pm_oracle({"topic": "other"}) == []
    assert parse.parse_pm_clob({"event_type": "tick_size_change"}) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd predictor && python3 -m pytest test_edgelab_parse.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# predictor/edgelab/parse.py
"""Pure per-feed parsers: raw WS message dict -> list of parsed-column dicts.
Each row carries a `source`; the caller adds the clock envelope + payload_json.
Unrecognized messages return []. Sample shapes verified live 2026-06-21."""


def _f(x):
    return None if x is None else float(x)


def parse_binance(msg: dict) -> list[dict]:
    d = msg.get("data") or {}
    et = d.get("e")
    if "@bookTicker" in (msg.get("stream") or "") or ("b" in d and "a" in d and et is None):
        return [{"source": "binance_bookticker",
                 "best_bid": _f(d.get("b")), "best_ask": _f(d.get("a")),
                 "best_bid_sz": _f(d.get("B")), "best_ask_sz": _f(d.get("A")),
                 "exch_seq": d.get("u")}]
    if et == "trade":
        return [{"source": "binance_trade",
                 "price": _f(d.get("p")), "size": _f(d.get("q")),
                 # m=True => buyer is market maker => taker SOLD
                 "side": "sell" if d.get("m") else "buy",
                 "exch_seq": d.get("t")}]
    return []


def parse_coinbase(msg: dict) -> list[dict]:
    t = msg.get("type")
    if t in ("match", "last_match"):
        return [{"source": "coinbase_match",
                 "price": _f(msg.get("price")), "size": _f(msg.get("size")),
                 "side": msg.get("side"), "exch_seq": msg.get("sequence")}]
    if t == "ticker":
        return [{"source": "coinbase_ticker",
                 "price": _f(msg.get("price")),
                 "best_bid": _f(msg.get("best_bid")), "best_ask": _f(msg.get("best_ask")),
                 "best_bid_sz": _f(msg.get("best_bid_size")),
                 "best_ask_sz": _f(msg.get("best_ask_size")),
                 "exch_seq": msg.get("sequence")}]
    return []


def parse_pm_oracle(msg: dict) -> list[dict]:
    if msg.get("topic") != "crypto_prices" or msg.get("type") != "update":
        return []
    p = msg.get("payload") or {}
    val = p.get("value")
    if val is None:
        val = p.get("full_accuracy_value")
    return [{"source": "pm_oracle", "price": _f(val),
             "symbol_raw": p.get("symbol")}]


def parse_pm_clob(msg: dict) -> list[dict]:
    et = msg.get("event_type")
    if et == "book":
        bids = msg.get("bids") or []
        asks = msg.get("asks") or []
        tb = bids[0] if bids else {}
        ta = asks[0] if asks else {}
        return [{"source": "pm_clob_book", "asset_id": msg.get("asset_id"),
                 "best_bid": _f(tb.get("price")), "best_ask": _f(ta.get("price")),
                 "best_bid_sz": _f(tb.get("size")), "best_ask_sz": _f(ta.get("size"))}]
    if et == "price_change":
        out = []
        for ch in msg.get("price_changes") or []:
            out.append({"source": "pm_clob_price_change",
                        "asset_id": msg.get("asset_id"),
                        "price": _f(ch.get("price")), "size": _f(ch.get("size")),
                        "side": ch.get("side")})
        return out
    return []
```

> **Note for implementer:** `symbol_raw` and `asset_id` are *parser-local* keys, NOT schema columns. The caller (Task 7 `build_rows`) maps `symbol_raw`/`asset_id` to the `symbol`/`window_slug` schema columns and drops the helper keys before stamping. `ClockStamper.stamp` rejects non-columns, so they must not be passed through.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd predictor && python3 -m pytest test_edgelab_parse.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add predictor/edgelab/parse.py predictor/test_edgelab_parse.py
git commit -m "feat(latency-harness): pure per-feed parsers"
```

---

## Task 4: Sequence-gap accountant

**Files:**
- Create: `predictor/edgelab/seqgap.py`
- Test: `predictor/test_edgelab_seqgap.py`

**Interfaces:**
- Produces: `class SeqGapTracker` with `check(source: str, seq: int | None) -> dict | None`. Returns a gap descriptor `{"source": source, "gap_start": int, "gap_end": int, "count": int}` when the incoming `seq` skips ahead of `last+1` for that source; otherwise `None`. `seq is None`, a first-seen source, equal, or out-of-order (`seq <= last`) returns `None` (only forward holes count). State is per source.

- [ ] **Step 1: Write the failing test**

```python
# predictor/test_edgelab_seqgap.py
from edgelab.seqgap import SeqGapTracker

def test_no_gap_on_contiguous():
    t = SeqGapTracker()
    assert t.check("binance_trade", 10) is None     # first seen
    assert t.check("binance_trade", 11) is None
    assert t.check("binance_trade", 12) is None

def test_detects_forward_hole():
    t = SeqGapTracker()
    t.check("coinbase_match", 100)
    g = t.check("coinbase_match", 104)
    assert g == {"source": "coinbase_match", "gap_start": 101,
                 "gap_end": 103, "count": 3}

def test_none_and_out_of_order_and_dupes_ignored():
    t = SeqGapTracker()
    t.check("s", 5)
    assert t.check("s", None) is None
    assert t.check("s", 5) is None          # dup
    assert t.check("s", 3) is None          # out of order
    assert t.check("s", 6) is None          # resumes contiguous from last max(5)

def test_sources_independent():
    t = SeqGapTracker()
    t.check("a", 1); t.check("b", 100)
    assert t.check("a", 2) is None
    assert t.check("b", 102)["count"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd predictor && python3 -m pytest test_edgelab_seqgap.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# predictor/edgelab/seqgap.py
"""Per-source exchange-sequence hole detection. Only forward holes count as
dropped messages; duplicates / out-of-order / missing seq are ignored."""


class SeqGapTracker:
    def __init__(self):
        self._last: dict[str, int] = {}

    def check(self, source: str, seq):
        if seq is None:
            return None
        last = self._last.get(source)
        if last is None:
            self._last[source] = seq
            return None
        if seq <= last:
            return None
        if seq > last + 1:
            gap = {"source": source, "gap_start": last + 1,
                   "gap_end": seq - 1, "count": seq - 1 - last}
            self._last[source] = seq
            return gap
        self._last[source] = seq
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd predictor && python3 -m pytest test_edgelab_seqgap.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add predictor/edgelab/seqgap.py predictor/test_edgelab_seqgap.py
git commit -m "feat(latency-harness): per-source sequence-gap accountant"
```

---

## Task 5: Rotating Parquet writer

**Files:**
- Create: `predictor/edgelab/writer.py`
- Test: `predictor/test_edgelab_writer.py`

**Interfaces:**
- Consumes: `schema.ARROW_SCHEMA`, `schema.COLUMNS`.
- Produces: `class RotatingParquetWriter(out_dir: str, rotate_secs: int = 60, rotate_n: int = 5000, clock=time.time)`:
  - `write(row: dict)` — buffers the row by `(source, minute_bucket)`; when a buffer's minute bucket changes or it reaches `rotate_n`, the previous buffer is flushed to an immutable file.
  - `flush_all()` — flushes every non-empty buffer (called on shutdown).
  - File path: `<out_dir>/events/day=YYYY-MM-DD/source=<source>/<epoch_minute>.parquet`, written via `pyarrow.parquet.write_table(..., compression="zstd")`. If the target path already exists, append `-1`, `-2`, … so a closed file is **never** overwritten.
  - Rows are coerced to `ARROW_SCHEMA` (extra parser-local keys must already be stripped by the caller; `write` keeps only `COLUMNS`).

- [ ] **Step 1: Write the failing test**

```python
# predictor/test_edgelab_writer.py
import glob, os
import pyarrow.parquet as pq
from edgelab.writer import RotatingParquetWriter
from edgelab import schema

def _row(source, wall_min):
    r = schema.empty_row()
    r["source"] = source
    r["region_id"] = "eu-west-1"
    r["recv_wall_ns"] = wall_min * 60 * 1_000_000_000
    r["clock_err_ns"] = 1000
    return r

def test_rotation_on_minute_change_writes_immutable_file(tmp_path):
    clock = {"t": 60.0}  # epoch_minute 1
    w = RotatingParquetWriter(str(tmp_path), clock=lambda: clock["t"])
    w.write(_row("binance_trade", 1))
    w.write(_row("binance_trade", 1))
    clock["t"] = 125.0   # epoch_minute 2 -> triggers flush of minute 1 buffer
    w.write(_row("binance_trade", 2))
    files = glob.glob(str(tmp_path / "events/day=*/source=binance_trade/*.parquet"))
    assert len(files) == 1
    tbl = pq.read_table(files[0])
    assert tbl.num_rows == 2
    assert tbl.schema.names == schema.COLUMNS

def test_flush_all_drains_open_buffers(tmp_path):
    w = RotatingParquetWriter(str(tmp_path), clock=lambda: 60.0)
    w.write(_row("coinbase_match", 1))
    w.flush_all()
    files = glob.glob(str(tmp_path / "events/day=*/source=coinbase_match/*.parquet"))
    assert len(files) == 1 and pq.read_table(files[0]).num_rows == 1

def test_never_overwrites_existing_file(tmp_path):
    w = RotatingParquetWriter(str(tmp_path), clock=lambda: 60.0)
    w.write(_row("pm_oracle", 1)); w.flush_all()
    w2 = RotatingParquetWriter(str(tmp_path), clock=lambda: 60.0)
    w2.write(_row("pm_oracle", 1)); w2.flush_all()
    files = glob.glob(str(tmp_path / "events/day=*/source=pm_oracle/*.parquet"))
    assert len(files) == 2          # second got a -1 suffix, original intact

def test_rotate_n_triggers_flush(tmp_path):
    w = RotatingParquetWriter(str(tmp_path), rotate_n=3, clock=lambda: 60.0)
    for _ in range(7):
        w.write(_row("probe_rpc", 1))
    files = glob.glob(str(tmp_path / "events/day=*/source=probe_rpc/*.parquet"))
    assert len(files) == 2          # 3 + 3 flushed, 1 still buffered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd predictor && python3 -m pytest test_edgelab_writer.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# predictor/edgelab/writer.py
"""Restart-safe, immutable, time-bucketed Parquet writer. One file per
(source, minute) buffer, flushed on minute-change / N-rows / shutdown; a closed
file is never overwritten (suffixes -1, -2, ...). Mirrors edgelab.logger's
'one immutable file per closed unit' guarantee."""
import os
import time
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from edgelab.schema import ARROW_SCHEMA, COLUMNS


class RotatingParquetWriter:
    def __init__(self, out_dir: str, rotate_secs: int = 60, rotate_n: int = 5000,
                 clock=time.time):
        self.out_dir = out_dir
        self.rotate_n = rotate_n
        self._clock = clock
        # key: source -> {"minute": int, "rows": list[dict]}
        self._buffers: dict[str, dict] = {}

    @staticmethod
    def _minute(row: dict, fallback_clock) -> int:
        wall_ns = row.get("recv_wall_ns")
        secs = (wall_ns / 1e9) if wall_ns else fallback_clock()
        return int(secs // 60)

    def write(self, row: dict) -> None:
        source = row["source"]
        minute = self._minute(row, self._clock)
        buf = self._buffers.get(source)
        if buf is not None and buf["minute"] != minute:
            self._flush(source)
            buf = None
        if buf is None:
            buf = {"minute": minute, "rows": []}
            self._buffers[source] = buf
        buf["rows"].append({c: row.get(c) for c in COLUMNS})
        if len(buf["rows"]) >= self.rotate_n:
            self._flush(source)

    def _flush(self, source: str) -> None:
        buf = self._buffers.pop(source, None)
        if not buf or not buf["rows"]:
            return
        epoch_minute = buf["minute"]
        day = datetime.fromtimestamp(epoch_minute * 60, tz=timezone.utc).strftime("%Y-%m-%d")
        part = os.path.join(self.out_dir, "events", f"day={day}", f"source={source}")
        os.makedirs(part, exist_ok=True)
        path = os.path.join(part, f"{epoch_minute}.parquet")
        n = 1
        while os.path.exists(path):
            path = os.path.join(part, f"{epoch_minute}-{n}.parquet")
            n += 1
        tbl = pa.Table.from_pylist(buf["rows"], schema=ARROW_SCHEMA)
        pq.write_table(tbl, path, compression="zstd")

    def flush_all(self) -> None:
        for source in list(self._buffers):
            self._flush(source)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd predictor && python3 -m pytest test_edgelab_writer.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add predictor/edgelab/writer.py predictor/test_edgelab_writer.py
git commit -m "feat(latency-harness): rotating immutable parquet writer"
```

---

## Task 6: Latency probes

**Files:**
- Create: `predictor/edgelab/probes.py`
- Test: `predictor/test_edgelab_probes.py`

**Interfaces:**
- Produces:
  - `rpc_payload() -> dict` — `{"jsonrpc":"2.0","id":1,"method":"eth_blockNumber","params":[]}`.
  - `parse_rpc_block(resp: dict) -> int` — returns the block number from `resp["result"]` (hex); raises `ValueError` if `resp` carries an `error` or no result.
  - `async timed_call(async_fn) -> tuple[int, object]` — `(rtt_ns, result)` measured with `time.monotonic_ns()` around `await async_fn()`.
  - `async probe_rpc(url, caller=None, timeout=5.0) -> int | None` — times one JSON-RPC `eth_blockNumber` round-trip (a **lower bound** on the real submit leg; documented as such). `None` on failure.
  - `async probe_tls(host, port=443, timeout=5.0) -> int | None` — times one TCP+TLS handshake. `None` on failure.

- [ ] **Step 1: Write the failing test**

```python
# predictor/test_edgelab_probes.py
import asyncio
from edgelab import probes

def test_rpc_payload():
    p = probes.rpc_payload()
    assert p["method"] == "eth_blockNumber" and p["jsonrpc"] == "2.0"

def test_parse_rpc_block_hex():
    assert probes.parse_rpc_block({"jsonrpc": "2.0", "id": 1, "result": "0x10"}) == 16

def test_parse_rpc_block_error_raises():
    for bad in ({"error": {"code": -1, "message": "x"}}, {"id": 1}):
        try:
            probes.parse_rpc_block(bad)
            assert False, "expected ValueError"
        except ValueError:
            pass

def test_timed_call_measures_elapsed():
    async def fake():
        await asyncio.sleep(0.01)
        return "ok"
    rtt_ns, res = asyncio.run(probes.timed_call(fake))
    assert res == "ok"
    assert rtt_ns >= 8_000_000          # ~10ms, allow scheduling slack

def test_probe_rpc_uses_injected_caller():
    async def run():
        return await probes.probe_rpc("http://unused", caller=lambda: 12345)
    rtt = asyncio.run(run())
    assert isinstance(rtt, int) and rtt >= 0

def test_probe_rpc_returns_none_on_failure():
    def boom():
        raise RuntimeError("down")
    rtt = asyncio.run(probes.probe_rpc("http://unused", caller=boom))
    assert rtt is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd predictor && python3 -m pytest test_edgelab_probes.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# predictor/edgelab/probes.py
"""Latency probes (first-class data stream). The RPC ping is a deliberate LOWER
BOUND on the Polygon submit leg: it times a read-only JSON-RPC round-trip and
omits mempool propagation + block inclusion, so true land time is always worse.
Never interpret probe_rpc as a real submission latency."""
import asyncio
import time


def rpc_payload() -> dict:
    return {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}


def parse_rpc_block(resp: dict) -> int:
    if not isinstance(resp, dict) or "error" in resp or "result" not in resp:
        raise ValueError(f"bad rpc response: {resp}")
    return int(resp["result"], 16)


async def timed_call(async_fn):
    t0 = time.monotonic_ns()
    res = await async_fn()
    return time.monotonic_ns() - t0, res


async def probe_rpc(url, caller=None, timeout: float = 5.0):
    """Time one eth_blockNumber round-trip. `caller` (sync, returns block int) is
    injectable for tests; default uses requests in a thread executor."""
    if caller is None:
        def caller():
            import requests
            r = requests.post(url, json=rpc_payload(), timeout=timeout)
            return parse_rpc_block(r.json())

    async def run():
        return await asyncio.get_event_loop().run_in_executor(None, caller)
    try:
        rtt_ns, _ = await asyncio.wait_for(timed_call(run), timeout=timeout + 1.0)
        return rtt_ns
    except Exception:
        return None


async def probe_tls(host, port: int = 443, timeout: float = 5.0):
    """Time one TCP+TLS handshake to host:port (network-layer lower bound)."""
    import ssl
    sslctx = ssl.create_default_context()

    async def run():
        reader, writer = await asyncio.open_connection(
            host, port, ssl=sslctx, server_hostname=host)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    try:
        rtt_ns, _ = await asyncio.wait_for(timed_call(run), timeout=timeout)
        return rtt_ns
    except Exception:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd predictor && python3 -m pytest test_edgelab_probes.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add predictor/edgelab/probes.py predictor/test_edgelab_probes.py
git commit -m "feat(latency-harness): rpc + tls latency probes (labelled lower bound)"
```

---

## Task 7: `build_rows` — the parse→tag→gap→stamp seam

**Files:**
- Create: `predictor/edgelab/collect.py` (this task adds only `BuildCtx` + `build_rows`; async collectors come in Task 8)
- Test: `predictor/test_edgelab_collect.py`

**Interfaces:**
- Consumes: `parse.*`, `clockstamp.ClockStamper`, `seqgap.SeqGapTracker`.
- Produces:
  - `ORACLE_SYMBOL: dict[str, str]` — maps oracle `symbol` (`"btcusdt"`) to canonical (`"BTC"`).
  - `@dataclass BuildCtx(stamper, gaps, symbol: str, symbols: set[str], token_index: dict[str, tuple[str, str]])` — `token_index` maps a CLOB `asset_id` to `(symbol, window_slug)`.
  - `build_rows(family: str, raw, ctx: BuildCtx) -> list[dict]` where `family in {"binance","coinbase","pm_oracle","pm_clob"}`. Parses, resolves `symbol`/`window_slug`, strips parser-local keys (`symbol_raw`, `asset_id`), runs gap accounting on `exch_seq` (emitting a `gap` row before the event row when a forward hole is found), and returns fully-stamped schema rows. Oracle rows for symbols not in `ctx.symbols` are dropped; CLOB events whose `asset_id` is not in `token_index` are dropped.

- [ ] **Step 1: Write the failing test**

```python
# predictor/test_edgelab_collect.py
import json
from edgelab.collect import build_rows, BuildCtx
from edgelab.clockstamp import ClockStamper
from edgelab.seqgap import SeqGapTracker

def _ctx(token_index=None):
    return BuildCtx(stamper=ClockStamper("eu-west-1", reader=lambda: (0, 1)),
                    gaps=SeqGapTracker(), symbol="BTC", symbols={"BTC"},
                    token_index=token_index or {})

def test_binance_event_row_is_fully_stamped():
    ctx = _ctx()
    raw = {"stream": "btcusdt@bookTicker",
           "data": {"u": 10, "s": "BTCUSDT", "b": "1", "B": "2", "a": "3", "A": "4"}}
    rows = build_rows("binance", raw, ctx)
    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == "binance_bookticker" and r["symbol"] == "BTC"
    assert r["region_id"] == "eu-west-1" and r["clock_err_ns"] == 1
    assert r["best_bid"] == 1.0 and r["best_ask"] == 3.0
    assert json.loads(r["payload_json"])["data"]["u"] == 10
    assert "symbol_raw" not in r and "asset_id" not in r

def test_gap_row_emitted_before_event_on_hole():
    ctx = _ctx()
    base = {"stream": "btcusdt@bookTicker",
            "data": {"s": "BTCUSDT", "b": "1", "B": "2", "a": "3", "A": "4"}}
    build_rows("binance", {**base, "data": {**base["data"], "u": 10}}, ctx)
    rows = build_rows("binance", {**base, "data": {**base["data"], "u": 14}}, ctx)
    assert [r["source"] for r in rows] == ["gap", "binance_bookticker"]
    assert json.loads(rows[0]["payload_json"])["count"] == 3

def test_oracle_filters_to_configured_symbols():
    ctx = _ctx()
    btc = {"topic": "crypto_prices", "type": "update",
           "payload": {"value": 64000.0, "symbol": "btcusdt"}}
    eth = {"topic": "crypto_prices", "type": "update",
           "payload": {"value": 1700.0, "symbol": "ethusdt"}}
    assert len(build_rows("pm_oracle", btc, ctx)) == 1
    assert build_rows("pm_oracle", eth, ctx) == []     # ETH not configured

def test_pm_clob_tagged_by_token_index_else_dropped():
    ctx = _ctx(token_index={"TKN_UP": ("BTC", "btc-updown-5m-100")})
    known = {"event_type": "book", "asset_id": "TKN_UP",
             "bids": [{"price": "0.48", "size": "100"}],
             "asks": [{"price": "0.52", "size": "200"}]}
    unknown = {"event_type": "book", "asset_id": "NOPE",
               "bids": [], "asks": []}
    rows = build_rows("pm_clob", known, ctx)
    assert rows[0]["window_slug"] == "btc-updown-5m-100" and rows[0]["best_bid"] == 0.48
    assert build_rows("pm_clob", unknown, ctx) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd predictor && python3 -m pytest test_edgelab_collect.py -v`
Expected: FAIL (module/symbols not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# predictor/edgelab/collect.py
"""The seam between the network loop and the tested units. `build_rows` turns one
raw feed message into fully-stamped schema rows (incl. gap rows). Async feed
collectors (Task 8) and --dry-run both go through build_rows."""
import json
from dataclasses import dataclass

from edgelab import parse
from edgelab.clockstamp import ClockStamper
from edgelab.seqgap import SeqGapTracker

_PARSERS = {
    "binance": parse.parse_binance,
    "coinbase": parse.parse_coinbase,
    "pm_oracle": parse.parse_pm_oracle,
    "pm_clob": parse.parse_pm_clob,
}

ORACLE_SYMBOL = {
    "btcusdt": "BTC", "ethusdt": "ETH", "solusdt": "SOL",
    "xrpusdt": "XRP", "dogeusdt": "DOGE", "bnbusdt": "BNB",
}


@dataclass
class BuildCtx:
    stamper: ClockStamper
    gaps: SeqGapTracker
    symbol: str
    symbols: set
    token_index: dict   # asset_id -> (symbol, window_slug)


def build_rows(family: str, raw, ctx: BuildCtx) -> list:
    msg = json.loads(raw) if isinstance(raw, str) else raw
    raw_str = raw if isinstance(raw, str) else json.dumps(msg, separators=(",", ":"))
    rows = []
    for p in _PARSERS[family](msg):
        p = dict(p)
        source = p.pop("source")
        symbol, window_slug = ctx.symbol, None
        if family == "pm_oracle":
            sym = ORACLE_SYMBOL.get(p.pop("symbol_raw", None))
            if sym is None or sym not in ctx.symbols:
                continue
            symbol = sym
        elif family == "pm_clob":
            sw = ctx.token_index.get(p.pop("asset_id", None))
            if sw is None:
                continue
            symbol, window_slug = sw
        seq = p.get("exch_seq")
        gap = ctx.gaps.check(source, seq) if seq is not None else None
        if gap:
            rows.append(ctx.stamper.stamp(
                "gap", symbol=symbol, window_slug=window_slug,
                exch_seq=gap["gap_end"],
                payload_json=json.dumps(gap, separators=(",", ":"))))
        rows.append(ctx.stamper.stamp(
            source, symbol=symbol, window_slug=window_slug,
            payload_json=raw_str, **p))
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd predictor && python3 -m pytest test_edgelab_collect.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add predictor/edgelab/collect.py predictor/test_edgelab_collect.py
git commit -m "feat(latency-harness): build_rows parse->tag->gap->stamp seam"
```

---

## Task 8: Async collectors, window tracker, harness + `--dry-run`

**Files:**
- Modify: `predictor/edgelab/collect.py` (append async feed collectors + probe loop)
- Create: `predictor/edgelab/windows.py`
- Create: `predictor/edgelab/harness.py`
- Modify: `predictor/edgelab/config.yaml` (add `harness:` section)
- Create: `predictor/edgelab/sample_dry_run.jsonl` (replay fixture)
- Test: append to `predictor/test_edgelab_collect.py` (a `--dry-run` integration test)

**Interfaces:**
- Consumes: `build_rows`, `BuildCtx`, `RotatingParquetWriter`, `probes.*`, `logger.resolve_tokens`.
- Produces:
  - `windows.current_windows(symbol="BTC", horizons=(("5m",300),("15m",900))) -> list[dict]` with keys `slug, horizon, up_token, down_token, symbol, close_ts` (skips windows Gamma can't yet resolve).
  - `collect.run_feed(name, url, sub, family, ctx, writer, stop, log)` — async; connect/subscribe/recv→build_rows→write, reconnect with backoff, `session` rows on connect/disconnect.
  - `collect.run_pm_clob(ctx, writer, stop, log, horizons, refresh_s=300)` — async; rebuilds `token_index` + CLOB subscription per refresh cycle.
  - `collect.run_probes(rpc_url, tls_hosts, ctx, writer, stop, period_s)` — async probe loop.
  - `harness.dry_run(sample_path, out_dir, region_id="test", symbols=("BTC",)) -> dict` and `harness.main()`.

- [ ] **Step 1: Write the failing integration test**

```python
# append to predictor/test_edgelab_collect.py
import glob
import pyarrow.parquet as pq
from edgelab import harness

def test_dry_run_writes_stamped_parquet(tmp_path):
    sample = tmp_path / "sample.jsonl"
    lines = [
        {"token_index": {"TKN_UP": ["BTC", "btc-updown-5m-100"]}},
        {"family": "binance", "raw": {"stream": "btcusdt@trade",
            "data": {"e": "trade", "s": "BTCUSDT", "t": 1, "p": "64000",
                     "q": "0.5", "m": False}}},
        {"family": "coinbase", "raw": {"type": "match", "price": "63999",
            "size": "0.1", "side": "sell", "sequence": 5, "product_id": "BTC-USD"}},
        {"family": "pm_oracle", "raw": {"topic": "crypto_prices", "type": "update",
            "payload": {"value": 64000.1, "symbol": "btcusdt"}}},
        {"family": "pm_clob", "raw": {"event_type": "book", "asset_id": "TKN_UP",
            "bids": [{"price": "0.48", "size": "100"}],
            "asks": [{"price": "0.52", "size": "200"}]}},
    ]
    sample.write_text("\n".join(json.dumps(x) for x in lines))
    out = tmp_path / "data"
    summary = harness.dry_run(str(sample), str(out))
    assert summary["rows"] == 4
    for src in ("binance_trade", "coinbase_match", "pm_oracle", "pm_clob_book"):
        files = glob.glob(str(out / f"events/day=*/source={src}/*.parquet"))
        assert files, f"no parquet for {src}"
        tbl = pq.read_table(files[0])
        assert tbl.num_rows >= 1
        col = tbl.column("clock_err_ns").to_pylist()
        assert all(v is not None for v in col)        # clock_err on every row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd predictor && python3 -m pytest test_edgelab_collect.py::test_dry_run_writes_stamped_parquet -v`
Expected: FAIL (`harness` not defined).

- [ ] **Step 3a: Append async collectors to `collect.py`**

```python
# append to predictor/edgelab/collect.py
import asyncio
import time

try:
    import websockets
except Exception:
    websockets = None

from edgelab import windows
from edgelab.probes import probe_rpc, probe_tls

CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def _session_row(ctx, feed, event, **extra):
    return ctx.stamper.stamp("session", payload_json=json.dumps(
        {"feed": feed, "event": event, **extra}, separators=(",", ":")))


async def _pump(ws, family, ctx, writer):
    raw = await asyncio.wait_for(ws.recv(), timeout=30)
    msgs = json.loads(raw)
    for m in (msgs if isinstance(msgs, list) else [msgs]):
        for row in build_rows(family, m, ctx):
            writer.write(row)


async def run_feed(name, url, sub, family, ctx, writer, stop, log):
    backoff = 1
    while not stop.is_set():
        try:
            async with websockets.connect(url, ping_interval=10, close_timeout=3,
                                          compression=None, max_size=2 ** 21) as ws:
                if sub is not None:
                    await ws.send(json.dumps(sub))
                writer.write(_session_row(ctx, name, "connect"))
                backoff = 1
                while not stop.is_set():
                    await _pump(ws, family, ctx, writer)
        except Exception as e:
            writer.write(_session_row(ctx, name, "disconnect",
                                      err=f"{type(e).__name__}:{e}"))
            log(f"{name} reconnect in {min(backoff,30)}s: {type(e).__name__}")
            await asyncio.sleep(min(backoff, 30))
            backoff *= 2


async def run_pm_clob(ctx, writer, stop, log, horizons, refresh_s=300):
    while not stop.is_set():
        wins = windows.current_windows(ctx.symbol, horizons)
        ctx.token_index.clear()
        tokens = []
        for w in wins:
            ctx.token_index[w["up_token"]] = (w["symbol"], w["slug"])
            ctx.token_index[w["down_token"]] = (w["symbol"], w["slug"])
            tokens += [w["up_token"], w["down_token"]]
        if not tokens:
            await asyncio.sleep(5)
            continue
        try:
            async with websockets.connect(CLOB_WS, ping_interval=10, close_timeout=3,
                                          compression=None, max_size=2 ** 21) as ws:
                await ws.send(json.dumps({"type": "market", "assets_ids": tokens}))
                writer.write(_session_row(ctx, "pm_clob", "connect", n_tokens=len(tokens)))
                t0 = time.time()
                while not stop.is_set() and time.time() - t0 < refresh_s:
                    await _pump(ws, "pm_clob", ctx, writer)
        except Exception as e:
            writer.write(_session_row(ctx, "pm_clob", "disconnect",
                                      err=f"{type(e).__name__}:{e}"))
            log(f"pm_clob reconnect: {type(e).__name__}")
            await asyncio.sleep(3)


async def run_probes(rpc_url, tls_hosts, ctx, writer, stop, period_s=2.0):
    while not stop.is_set():
        rtt = await probe_rpc(rpc_url)
        if rtt is not None:
            writer.write(ctx.stamper.stamp("probe_rpc", rtt_ns=rtt))
        for host, port in tls_hosts:
            t = await probe_tls(host, int(port))
            if t is not None:
                writer.write(ctx.stamper.stamp("probe_tls", rtt_ns=t, symbol=None))
        await asyncio.sleep(period_s)
```

- [ ] **Step 3b: Create `windows.py`**

```python
# predictor/edgelab/windows.py
"""Gamma-backed current-window discovery for the CLOB collector. Reuses
edgelab.logger.resolve_tokens (keyless Gamma)."""
import time

from edgelab.logger import resolve_tokens


def current_windows(symbol: str = "BTC",
                    horizons=(("5m", 300), ("15m", 900))) -> list:
    sym = symbol.lower()
    now = int(time.time())
    out = []
    for hz, period in horizons:
        start = (now // period) * period
        slug = f"{sym}-updown-{hz}-{start}"
        toks = resolve_tokens(slug)
        if toks is None:
            continue
        up, down = toks
        out.append({"slug": slug, "horizon": hz, "up_token": up,
                    "down_token": down, "symbol": symbol,
                    "close_ts": start + period})
    return out
```

- [ ] **Step 3c: Create `harness.py`**

```python
# predictor/edgelab/harness.py
"""Single-region capture harness entrypoint. Record-only: imports nothing from
live_trader; the only Polygon endpoint touched is a read-only RPC ping."""
import argparse
import asyncio
import json
import os

import yaml

from edgelab.clockstamp import ClockStamper
from edgelab.seqgap import SeqGapTracker
from edgelab.writer import RotatingParquetWriter
from edgelab import collect

_CFG = os.path.join(os.path.dirname(__file__), "config.yaml")
BINANCE_BASE = "wss://stream.binance.com:9443/stream?streams="
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
ORACLE_WS = "wss://ws-live-data.polymarket.com"
ORACLE_SUB = {"action": "subscribe",
              "subscriptions": [{"topic": "crypto_prices", "type": "update"}]}


def _make_ctx(region_id, symbols):
    return collect.BuildCtx(
        stamper=ClockStamper(region_id), gaps=SeqGapTracker(),
        symbol=list(symbols)[0], symbols=set(symbols), token_index={})


def dry_run(sample_path, out_dir, region_id="test", symbols=("BTC",)) -> dict:
    ctx = collect.BuildCtx(
        stamper=ClockStamper(region_id, reader=lambda: (0, 1)),
        gaps=SeqGapTracker(), symbol=symbols[0], symbols=set(symbols),
        token_index={})
    writer = RotatingParquetWriter(out_dir)
    n = 0
    with open(sample_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "token_index" in obj:
                ctx.token_index = {k: tuple(v) for k, v in obj["token_index"].items()}
                continue
            for row in collect.build_rows(obj["family"], obj["raw"], ctx):
                writer.write(row)
                n += 1
    writer.flush_all()
    return {"rows": n}


async def run_all(cfg, out_dir):
    region_id = cfg["region_id"]
    symbols = cfg["symbols"]
    horizons = tuple((k, int(v)) for k, v in cfg["horizons"].items())
    ctx = _make_ctx(region_id, symbols)
    writer = RotatingParquetWriter(out_dir)
    stop = asyncio.Event()
    log = lambda m: print(m, flush=True)
    sym = symbols[0].lower()
    streams = f"{sym}usdt@trade/{sym}usdt@bookTicker"
    cb_sub = {"type": "subscribe",
              "product_ids": [f"{symbols[0]}-USD"], "channels": ["matches", "ticker"]}
    tasks = [
        collect.run_feed("binance", BINANCE_BASE + streams, None, "binance", ctx, writer, stop, log),
        collect.run_feed("coinbase", COINBASE_WS, cb_sub, "coinbase", ctx, writer, stop, log),
        collect.run_feed("pm_oracle", ORACLE_WS, ORACLE_SUB, "pm_oracle", ctx, writer, stop, log),
        collect.run_pm_clob(ctx, writer, stop, log, horizons),
        collect.run_probes(cfg["rpc_url"], cfg["tls_hosts"], ctx, writer, stop,
                           float(cfg.get("probe_period_s", 2.0))),
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        writer.flush_all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", dest="dry", default=None,
                    help="replay a sample JSONL through build_rows (no network)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(_CFG))["harness"]
    out = os.environ.get("EDGELAB_OUT", args.out or cfg["out_dir"])
    if args.dry:
        print(dry_run(args.dry, out))
        return
    asyncio.run(run_all(cfg, out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3d: Add the `harness:` section to `predictor/edgelab/config.yaml`**

```yaml
harness:
  region_id: eu-west-1          # set per box: ap-northeast-1 / us-east-1 elsewhere
  symbols: [BTC]                # symbol-agnostic code; BTC-only for Spec 1
  horizons: {5m: 300, 15m: 900}
  out_dir: /tmp/latency_harness_data   # override on server via EDGELAB_OUT=/mnt/data/...
  rpc_url: "https://polygon-rpc.com"   # SET to the exact RPC you'd submit through
  probe_period_s: 2.0
  tls_hosts:
    - ["polygon-rpc.com", 443]
    - ["stream.binance.com", 9443]
    - ["ws-feed.exchange.coinbase.com", 443]
    - ["ws-subscriptions-clob.polymarket.com", 443]
```

- [ ] **Step 3e: Create the replay fixture `predictor/edgelab/sample_dry_run.jsonl`**

```
{"token_index": {"TKN_UP": ["BTC", "btc-updown-5m-100"], "TKN_DN": ["BTC", "btc-updown-5m-100"]}}
{"family": "binance", "raw": {"stream": "btcusdt@bookTicker", "data": {"u": 10, "s": "BTCUSDT", "b": "64059.99", "B": "0.98", "a": "64060.00", "A": "3.26"}}}
{"family": "binance", "raw": {"stream": "btcusdt@trade", "data": {"e": "trade", "s": "BTCUSDT", "t": 1, "p": "64060.0", "q": "0.0003", "m": false}}}
{"family": "coinbase", "raw": {"type": "match", "price": "63983.99", "size": "0.0001", "side": "buy", "sequence": 5, "product_id": "BTC-USD"}}
{"family": "pm_oracle", "raw": {"topic": "crypto_prices", "type": "update", "payload": {"value": 64000.12, "symbol": "btcusdt"}}}
{"family": "pm_clob", "raw": {"event_type": "book", "asset_id": "TKN_UP", "bids": [{"price": "0.48", "size": "100"}], "asks": [{"price": "0.52", "size": "200"}]}}
```

- [ ] **Step 4: Run the full test suite to verify it passes**

Run: `cd predictor && python3 -m pytest test_edgelab_collect.py test_edgelab_*.py -v`
Expected: PASS (all collect tests incl. the dry-run integration test).

- [ ] **Step 5: Commit**

```bash
git add predictor/edgelab/collect.py predictor/edgelab/windows.py \
        predictor/edgelab/harness.py predictor/edgelab/config.yaml \
        predictor/edgelab/sample_dry_run.jsonl predictor/test_edgelab_collect.py
git commit -m "feat(latency-harness): async collectors + window tracker + harness/--dry-run"
```

---

## Task 9: Milestone-1 sanity check

**Files:**
- Create: `predictor/edgelab/sanity_check.py`
- Test: `predictor/test_edgelab_sanity.py`

**Interfaces:**
- Consumes: captured Parquet (via `pyarrow`), `schema.COLUMNS`.
- Produces:
  - `summarize(data_dir: str) -> dict` — `{"total_rows", "by_source": {src: n}, "clock_err_nulls": int, "probe": {"probe_rpc": {p50,p90,p99,n}, "probe_tls": {...}}}` (percentiles via `numpy`).
  - `check_gate(summary: dict) -> tuple[bool, list[str]]` — asserts §10 gate: ≥1 row for a `binance_*`, a `coinbase_*`, `pm_oracle`, a `pm_clob_*`; `clock_err_nulls == 0`; both probe distributions populated (`n >= 1` and percentiles not None). Returns `(ok, reasons_for_failure)`.
  - `main()` — `python3 -m edgelab.sanity_check <data_dir>` prints the report + PASS/FAIL.

- [ ] **Step 1: Write the failing test**

```python
# predictor/test_edgelab_sanity.py
from edgelab.writer import RotatingParquetWriter
from edgelab import schema, sanity_check

def _row(source, rtt=None):
    r = schema.empty_row()
    r["source"] = source
    r["region_id"] = "eu-west-1"
    r["recv_wall_ns"] = 60 * 1_000_000_000
    r["clock_err_ns"] = 1000
    r["rtt_ns"] = rtt
    return r

def _populate(out_dir):
    w = RotatingParquetWriter(out_dir, clock=lambda: 60.0)
    for s in ("binance_trade", "coinbase_match", "pm_oracle", "pm_clob_book"):
        w.write(_row(s))
    for v in (1_000_000, 2_000_000, 3_000_000):
        w.write(_row("probe_rpc", rtt=v))
        w.write(_row("probe_tls", rtt=v))
    w.flush_all()

def test_summarize_and_gate_pass(tmp_path):
    _populate(str(tmp_path))
    s = sanity_check.summarize(str(tmp_path))
    assert s["clock_err_nulls"] == 0
    assert s["by_source"]["binance_trade"] == 1
    assert s["probe"]["probe_rpc"]["n"] == 3
    assert s["probe"]["probe_rpc"]["p50"] == 2_000_000
    ok, reasons = sanity_check.check_gate(s)
    assert ok, reasons

def test_gate_fails_when_a_family_missing(tmp_path):
    w = RotatingParquetWriter(str(tmp_path), clock=lambda: 60.0)
    w.write(_row("binance_trade"))           # only one family
    w.write(_row("probe_rpc", rtt=1)); w.write(_row("probe_tls", rtt=1))
    w.flush_all()
    ok, reasons = sanity_check.check_gate(sanity_check.summarize(str(tmp_path)))
    assert not ok
    assert any("coinbase" in r for r in reasons)

def test_gate_fails_on_clock_err_null(tmp_path):
    w = RotatingParquetWriter(str(tmp_path), clock=lambda: 60.0)
    for s in ("binance_trade", "coinbase_match", "pm_oracle", "pm_clob_book"):
        w.write(_row(s))
    bad = _row("binance_trade"); bad["clock_err_ns"] = None
    w.write(bad)
    w.write(_row("probe_rpc", rtt=1)); w.write(_row("probe_tls", rtt=1))
    w.flush_all()
    ok, reasons = sanity_check.check_gate(sanity_check.summarize(str(tmp_path)))
    assert not ok and any("clock_err" in r for r in reasons)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd predictor && python3 -m pytest test_edgelab_sanity.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# predictor/edgelab/sanity_check.py
"""Milestone-1 sanity gate: read the captured Parquet and assert all five feed
families produced rows, every row carries clock_err_ns, and both probe RTT
distributions are populated. NOT an edge test — just 'is the capture healthy'."""
import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq


def _percentiles(vals):
    if not vals:
        return {"p50": None, "p90": None, "p99": None, "n": 0}
    a = np.array(vals, dtype="float64")
    return {"p50": int(np.percentile(a, 50)),
            "p90": int(np.percentile(a, 90)),
            "p99": int(np.percentile(a, 99)), "n": len(vals)}


def summarize(data_dir: str) -> dict:
    files = glob.glob(os.path.join(data_dir, "events", "day=*", "source=*", "*.parquet"))
    by_source, clock_nulls = {}, 0
    rtts = {"probe_rpc": [], "probe_tls": []}
    total = 0
    for path in files:
        # partitioning=None: files sit under hive-style day=/source= dirs AND
        # carry their own `source` column; without this pyarrow infers `source`
        # from the path and clashes types (ArrowTypeError). Read in-file cols only.
        tbl = pq.read_table(path, partitioning=None)
        total += tbl.num_rows
        srcs = tbl.column("source").to_pylist()
        errs = tbl.column("clock_err_ns").to_pylist()
        rtt_col = tbl.column("rtt_ns").to_pylist()
        for s, e, r in zip(srcs, errs, rtt_col):
            by_source[s] = by_source.get(s, 0) + 1
            if e is None:
                clock_nulls += 1
            if s in rtts and r is not None:
                rtts[s].append(r)
    return {"total_rows": total, "by_source": by_source,
            "clock_err_nulls": clock_nulls,
            "probe": {k: _percentiles(v) for k, v in rtts.items()}}


def check_gate(summary: dict) -> tuple:
    reasons = []
    bs = summary["by_source"]
    for family, prefix in (("binance", "binance_"), ("coinbase", "coinbase_"),
                           ("pm_oracle", "pm_oracle"), ("pm_clob", "pm_clob_")):
        if not any(s.startswith(prefix) and n > 0 for s, n in bs.items()):
            reasons.append(f"no rows for feed family: {family}")
    if summary["clock_err_nulls"] != 0:
        reasons.append(f"clock_err_ns null on {summary['clock_err_nulls']} rows")
    for probe in ("probe_rpc", "probe_tls"):
        p = summary["probe"].get(probe, {})
        if not p.get("n") or p.get("p50") is None:
            reasons.append(f"{probe} distribution empty")
    return (len(reasons) == 0, reasons)


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EDGELAB_OUT", ".")
    s = summarize(data_dir)
    ok, reasons = check_gate(s)
    print(f"total_rows={s['total_rows']}")
    for src, n in sorted(s["by_source"].items()):
        print(f"  {src}: {n}")
    print(f"clock_err_nulls={s['clock_err_nulls']}")
    for probe, p in s["probe"].items():
        print(f"  {probe}: {p}")
    print("GATE:", "PASS" if ok else "FAIL")
    for r in reasons:
        print("  -", r)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd predictor && python3 -m pytest test_edgelab_sanity.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add predictor/edgelab/sanity_check.py predictor/test_edgelab_sanity.py
git commit -m "feat(latency-harness): milestone-1 capture sanity gate"
```

---

## Final manual step (not a code task): the 1-hour live capture

After Task 9, run the actual Milestone-1 gate on the Ireland box:

```bash
# on the all-markets / eu-west-1 box, from the predictor/ dir
EDGELAB_OUT=/mnt/data/latency_harness python3 -m edgelab.harness &   # ~1 hour
# then:
python3 -m edgelab.sanity_check /mnt/data/latency_harness
```

Confirm GATE: PASS, then manually verify the §10 Gamma-liquidity item: that BTC lists on **both** 5m and 15m with concurrent depth, and note in `edgelab/REPORT.md` if the 5m book is materially thinner than 15m (thin-book signals fail depth>0 downstream and must not inflate signal count). This closes Spec 1.

---

## Self-Review

**Spec coverage** (each spec section → task):
- §4 Python/extends edgelab, §4.1 reuse → Tasks 1–9 build under `edgelab/`, reuse `resolve_tokens` (Task 8 windows).
- §5 five feeds → parsers (Task 3) + collectors (Task 8: binance/coinbase/oracle feeds + `run_pm_clob`).
- §5.1 probes (RPC lower-bound + TLS) → Task 6 + `run_probes` (Task 8).
- §6 schema (envelope on every row) → Task 1 + stamping (Task 2) applied in `build_rows` (Task 7).
- §7 storage/restart-safety (immutable, never-rewrite, time-bucketed, gap rows) → Task 5 writer + Task 4 gap accountant + Task 7 gap-row emission.
- §8 reconnect/session rows + `--dry-run` + TDD units → Task 8 collectors + `dry_run`; every unit has a test task.
- §9 pre-registered gates → recorded in the spec; **not** code in Spec 1 (analysis is a later spec) — correct, no task needed.
- §10 Milestone-1 sanity gate → Task 9 + final manual capture step.
- §12 pitfalls → honest per-region recording (Task 8), submit-leg lower-bound labelled (Task 6 docstring), BTC-only symbol-agnostic (config + `ORACLE_SYMBOL` filter, Task 7).
- Record-only principle → harness imports nothing from `live_trader`; only a read-only RPC ping (Task 6/8) — verified in Task 8 module header.

**Placeholder scan:** none — every step has runnable code/commands. The one config value a human must set (`rpc_url` = the real submit endpoint) is explicitly flagged in the YAML comment, not left as a silent TODO.

**Type consistency:** `build_rows(family, raw, ctx)`, `BuildCtx(stamper, gaps, symbol, symbols, token_index)`, `ClockStamper.stamp(source, **parsed)`, `SeqGapTracker.check(source, seq)`, `RotatingParquetWriter(out_dir, rotate_secs, rotate_n, clock)` / `.write` / `.flush_all`, `probe_rpc(url, caller, timeout)`, `current_windows(symbol, horizons)`, `summarize`/`check_gate` — all names/signatures match across the tasks that consume them. Parser-local keys (`symbol_raw`, `asset_id`) are explicitly stripped in `build_rows` before stamping, consistent with `ClockStamper.stamp` rejecting non-columns (Task 2/3/7).

