# Fee-Aware Capturability Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only simulator that answers whether the captured Binance→Polymarket-CLOB lag produces a fill that is profitable after the verified Polymarket crypto taker fee.

**Architecture:** Three focused modules under `predictor/edgelab/`: `capsim.py` (pure simulation math — triggers, polarity, fee, fills, net edge), `capsim_io.py` (load captured Parquet into series + resolve market outcomes via Gamma), `capsim_run.py` (sweep θ/R/H, random-entry null control, report, CLI). No collector or execution code is touched. Resolution-primary verdict; mark-to-reprice is diagnostic only.

**Tech Stack:** Python 3.12, numpy, pyarrow (already deps of edgelab); pytest. Reuses `edgelab.logger` for Gamma market lookups.

## Global Constraints

- **Read-only / record-only.** No changes to collector, writer, or any order/wallet/signing path. The module never submits, signs, holds a key, or mutates capture data.
- **Single clock.** All times are the capture's `recv_wall_ns` (one box). No cross-region clock logic.
- **Fee (verified 2026-06-21, docs.polymarket.com/trading/fees):** per-share `fee = 0.07 × p × (1 − p)`, charged **entry-only** (redemption is not a match). Never apply a second fee on a mark-to-reprice "sale".
- **Resolution-primary.** A positive verdict is gated on hold-to-resolution net of fee. Mark-to-reprice is diagnostic only and can never produce a positive verdict. Thin resolution N is reported, never papered over.
- **Sign-align from levels.** Up/down token polarity is the sign of `corr(spot_level, clob_mid_level)` over the asset's life — never assumed from token order. (This is the trap that zeroed the naïve pooled cross-correlation: 12 up + 12 down cancelled.)
- **Pessimistic reaction coordinate.** `R` = detection→order-land in `recv_wall` (detection already carries the ~105ms feed-arrival). Headline `R = 0.100s`; sweep `R ∈ {0.030, 0.060, 0.100, 0.150}`. The 0.030 cell is a bound, never the verdict.
- **Trigger move** = relative return in bps over lookback `L` (default `L = 0.5s`), with a cooldown (default = `L`) so one move is not counted as many fills.
- **Depth-cap** every fill to the captured `best_ask_sz` and a fixed stake (default $5).
- **Random-entry null control** uses identical machinery; the momentum trigger must beat it by a margin.
- **Tests** are named `predictor/test_edgelab_capsim*.py` and run **from `predictor/`** (e.g. `python3 -m pytest test_edgelab_capsim.py`). Git commands run from the repo root with `predictor/` path prefixes.
- Use `python3` (no `python` on PATH).

---

### Task 1: `capsim.py` — bps returns + momentum & random triggers

**Files:**
- Create: `predictor/edgelab/capsim.py`
- Test: `predictor/test_edgelab_capsim.py`

**Interfaces:**
- Produces:
  - `bps_returns(t: np.ndarray, mid: np.ndarray, lookback_s: float) -> np.ndarray` — for each i, `(mid[i]/mid[j] - 1) * 1e4` where j is the last index with `t[j] <= t[i] - lookback_s`; `np.nan` if none. `t` ascending seconds, `mid` float.
  - `momentum_triggers(t, mid, theta_bps: float, lookback_s: float, cooldown_s: float) -> list[tuple[float, int]]` — `(t_i, dir)` for each event whose `|bps_return| >= theta_bps`, `dir = +1/-1` by sign, suppressing any trigger within `cooldown_s` of the previously emitted one.
  - `random_triggers(t0: float, t1: float, n: int, seed: int) -> list[tuple[float, int]]` — `n` entries with uniform `t ∈ [t0, t1)` (sorted) and `dir ∈ {-1,+1}` from a seeded `np.random.default_rng`.

- [ ] **Step 1: Write the failing test**

```python
# predictor/test_edgelab_capsim.py
import numpy as np
from edgelab import capsim

def test_bps_returns_basic():
    t = np.array([0.0, 0.4, 0.8, 1.2])
    mid = np.array([100.0, 100.0, 101.0, 101.0])  # +1% = +100 bps by t=0.8
    r = capsim.bps_returns(t, mid, lookback_s=0.5)
    assert np.isnan(r[0])                       # nothing 0.5s before t=0
    assert abs(r[2] - 100.0) < 1e-6             # 101/100-1 over [0.3,0.8] -> 100bps
    assert abs(r[3] - 100.0) < 1e-6

def test_momentum_triggers_threshold_and_cooldown():
    t = np.array([0.0, 0.4, 0.8, 1.2, 5.0])
    mid = np.array([100.0, 100.0, 101.0, 102.0, 100.0])
    trg = capsim.momentum_triggers(t, mid, theta_bps=50.0, lookback_s=0.5, cooldown_s=0.5)
    # first cross at t=0.8 (+100bps); t=1.2 within cooldown -> suppressed; t=5.0 is a drop
    assert trg[0][0] == 0.8 and trg[0][1] == +1
    assert all(abs(a[0] - 0.8) > 0.49 for a in trg[1:])   # no re-fire inside cooldown
    assert any(d == -1 for _, d in trg)                   # the drop fires a -1

def test_random_triggers_deterministic_and_bounded():
    a = capsim.random_triggers(10.0, 20.0, n=5, seed=7)
    b = capsim.random_triggers(10.0, 20.0, n=5, seed=7)
    assert a == b and len(a) == 5
    assert all(10.0 <= ti < 20.0 and di in (-1, 1) for ti, di in a)
    assert a == sorted(a)
```

- [ ] **Step 2: Run test to verify it fails**

From `predictor/`: `python3 -m pytest test_edgelab_capsim.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'edgelab.capsim'`).

- [ ] **Step 3: Write minimal implementation**

```python
# predictor/edgelab/capsim.py
"""Pure simulation math for the fee-aware capturability kill test. No I/O.
All times are seconds in the capture's single recv_wall clock."""
import numpy as np

FEE_RATE = 0.07  # verified crypto taker rate, docs.polymarket.com/trading/fees


def bps_returns(t, mid, lookback_s):
    t = np.asarray(t, float); mid = np.asarray(mid, float)
    j = np.searchsorted(t, t - lookback_s, side="right") - 1
    out = np.full(len(t), np.nan)
    ok = j >= 0
    out[ok] = (mid[ok] / mid[j[ok]] - 1.0) * 1e4
    return out


def momentum_triggers(t, mid, theta_bps, lookback_s, cooldown_s):
    t = np.asarray(t, float)
    r = bps_returns(t, mid, lookback_s)
    out = []
    last = -np.inf
    for i in range(len(t)):
        if np.isnan(r[i]) or abs(r[i]) < theta_bps:
            continue
        if t[i] - last < cooldown_s:
            continue
        out.append((float(t[i]), 1 if r[i] > 0 else -1))
        last = t[i]
    return out


def random_triggers(t0, t1, n, seed):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.uniform(t0, t1, size=n))
    ds = rng.choice((-1, 1), size=n)
    return [(float(a), int(b)) for a, b in zip(ts, ds)]
```

- [ ] **Step 4: Run test to verify it passes**

From `predictor/`: `python3 -m pytest test_edgelab_capsim.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

From repo root:
```bash
git add predictor/edgelab/capsim.py predictor/test_edgelab_capsim.py
git commit -m "feat(capsim): bps returns + momentum/random triggers"
```

---

### Task 2: `capsim.py` — polarity from price levels

**Files:**
- Modify: `predictor/edgelab/capsim.py`
- Test: `predictor/test_edgelab_capsim.py`

**Interfaces:**
- Produces: `polarity_from_levels(ta, lvla, tb, lvlb, dt=0.1) -> int` — forward-fill both level series onto a common `dt`-second grid over their overlap, return `+1` if `corr(spot, clob) > 0` (up-token), `-1` if `< 0` (down-token), `0` if degenerate (no overlap, zero variance, or NaN corr).

- [ ] **Step 1: Write the failing test**

```python
def test_polarity_up_and_down_token():
    ta = np.arange(0, 10, 0.1); spot = 100 + np.sin(ta)        # oscillating spot
    up = 0.5 + 0.01 * np.sin(ta)                                # moves WITH spot
    dn = 0.5 - 0.01 * np.sin(ta)                                # moves AGAINST spot
    assert capsim.polarity_from_levels(ta, spot, ta, up) == +1
    assert capsim.polarity_from_levels(ta, spot, ta, dn) == -1

def test_polarity_degenerate_returns_zero():
    ta = np.arange(0, 5, 0.1); spot = 100 + np.sin(ta)
    flat = np.full_like(ta, 0.5)                                # no variance
    assert capsim.polarity_from_levels(ta, spot, ta, flat) == 0
    assert capsim.polarity_from_levels(ta, spot, np.array([100.0]), np.array([0.5])) == 0
```

- [ ] **Step 2: Run test to verify it fails**

From `predictor/`: `python3 -m pytest test_edgelab_capsim.py::test_polarity_up_and_down_token -q`
Expected: FAIL (`AttributeError: module 'edgelab.capsim' has no attribute 'polarity_from_levels'`).

- [ ] **Step 3: Write minimal implementation**

```python
def _ffill(et, ev, grid):
    et = np.asarray(et, float); ev = np.asarray(ev, float)
    idx = np.searchsorted(et, grid, side="right") - 1
    idx[idx < 0] = 0
    return ev[idx]


def polarity_from_levels(ta, lvla, tb, lvlb, dt=0.1):
    ta = np.asarray(ta, float); tb = np.asarray(tb, float)
    if len(ta) < 2 or len(tb) < 2:
        return 0
    t0 = max(ta[0], tb[0]); t1 = min(ta[-1], tb[-1])
    if t1 - t0 < dt:
        return 0
    g = np.arange(t0, t1, dt)
    a = _ffill(ta, lvla, g); b = _ffill(tb, lvlb, g)
    if a.std() == 0 or b.std() == 0:
        return 0
    c = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(c) or c == 0:
        return 0
    return 1 if c > 0 else -1
```

- [ ] **Step 4: Run test to verify it passes**

From `predictor/`: `python3 -m pytest test_edgelab_capsim.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

From repo root:
```bash
git add predictor/edgelab/capsim.py predictor/test_edgelab_capsim.py
git commit -m "feat(capsim): per-asset polarity from price levels"
```

---

### Task 3: `capsim.py` — fee, hittable ask (depth-capped), mark value

**Files:**
- Modify: `predictor/edgelab/capsim.py`
- Test: `predictor/test_edgelab_capsim.py`

**Interfaces:**
- Produces:
  - `fee_per_share(price: float, rate=FEE_RATE) -> float` = `rate * price * (1 - price)`.
  - `scalar_ffill(et, ev, q: float) -> float` — value of `ev` at the last `et <= q`; `np.nan` if none or that value is NaN.
  - `hittable_ask(ta, ask, ask_sz, arrival: float, stake: float) -> tuple[float, float]` — `(ask_px, shares)`; `ask_px = scalar_ffill(ta, ask, arrival)`, size = `scalar_ffill(ta, ask_sz, arrival)`; `shares = min(stake/ask_px, size)` when `ask_px` finite and `> 0`, else `(nan, 0.0)`.
  - `mark_value(tm, mid, arrival: float, H: float) -> float` = `scalar_ffill(tm, mid, arrival + H)`.

- [ ] **Step 1: Write the failing test**

```python
def test_fee_peaks_at_half():
    assert abs(capsim.fee_per_share(0.5) - 0.0175) < 1e-9     # 1.75% at the money
    assert capsim.fee_per_share(0.1) < capsim.fee_per_share(0.5)
    assert abs(capsim.fee_per_share(0.1) - capsim.fee_per_share(0.9)) < 1e-9  # symmetric

def test_hittable_ask_depth_caps():
    ta = np.array([0.0, 1.0, 2.0])
    ask = np.array([0.50, 0.51, 0.52])
    sz  = np.array([100.0, 3.0, 100.0])
    # arrival 1.5 -> last<=1.5 is index1: ask 0.51, size 3 shares; stake $5 wants 9.8 shares
    px, sh = capsim.hittable_ask(ta, ask, sz, arrival=1.5, stake=5.0)
    assert abs(px - 0.51) < 1e-9 and abs(sh - 3.0) < 1e-9     # capped by size
    # deep size -> capped by stake
    px2, sh2 = capsim.hittable_ask(ta, ask, np.array([100.,100.,100.]), 1.5, 5.0)
    assert abs(sh2 - 5.0/0.51) < 1e-6

def test_hittable_ask_no_book_returns_nan():
    ta = np.array([2.0]); ask = np.array([0.5]); sz = np.array([10.0])
    px, sh = capsim.hittable_ask(ta, ask, sz, arrival=1.0, stake=5.0)  # arrival before any book
    assert np.isnan(px) and sh == 0.0

def test_mark_value_uses_arrival_plus_H():
    tm = np.array([0.0, 1.0, 2.0]); mid = np.array([0.50, 0.55, 0.60])
    assert abs(capsim.mark_value(tm, mid, arrival=1.0, H=1.0) - 0.60) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

From `predictor/`: `python3 -m pytest test_edgelab_capsim.py::test_fee_peaks_at_half -q`
Expected: FAIL (`AttributeError: ... 'fee_per_share'`).

- [ ] **Step 3: Write minimal implementation**

```python
def fee_per_share(price, rate=FEE_RATE):
    return rate * price * (1.0 - price)


def scalar_ffill(et, ev, q):
    et = np.asarray(et, float); ev = np.asarray(ev, float)
    i = int(np.searchsorted(et, q, side="right")) - 1
    if i < 0:
        return np.nan
    return float(ev[i])


def hittable_ask(ta, ask, ask_sz, arrival, stake):
    px = scalar_ffill(ta, ask, arrival)
    if not np.isfinite(px) or px <= 0:
        return (np.nan, 0.0)
    size = scalar_ffill(ta, ask_sz, arrival)
    if not np.isfinite(size) or size <= 0:
        return (np.nan, 0.0)
    shares = min(stake / px, size)
    return (px, shares)


def mark_value(tm, mid, arrival, H):
    return scalar_ffill(tm, mid, arrival + H)
```

- [ ] **Step 4: Run test to verify it passes**

From `predictor/`: `python3 -m pytest test_edgelab_capsim.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

From repo root:
```bash
git add predictor/edgelab/capsim.py predictor/test_edgelab_capsim.py
git commit -m "feat(capsim): fee, depth-capped hittable ask, mark value"
```

---

### Task 4: `capsim.py` — per-share net edge (entry-only fee, both valuations)

**Files:**
- Modify: `predictor/edgelab/capsim.py`
- Test: `predictor/test_edgelab_capsim.py`

**Interfaces:**
- Produces: `net_edge_per_share(value: float, ask: float, rate=FEE_RATE) -> float` = `value - ask - fee_per_share(ask, rate)`. `value` is the fill's settlement (`1.0`/`0.0` for resolution) or its mark mid (diagnostic). The fee is **entry-only on `ask`** — this single function is used for both valuations and never adds a sale fee.

- [ ] **Step 1: Write the failing test**

```python
def test_net_edge_entry_only_fee():
    # win at $1 bought at 0.5: edge = 1 - 0.5 - 0.0175 = 0.4825
    assert abs(capsim.net_edge_per_share(1.0, 0.5) - 0.4825) < 1e-9
    # loss at $0 bought at 0.5: edge = -0.5 - 0.0175
    assert abs(capsim.net_edge_per_share(0.0, 0.5) - (-0.5175)) < 1e-9
    # mark valuation uses the SAME entry-only fee (no second fee on the notional sale)
    assert abs(capsim.net_edge_per_share(0.55, 0.50) - (0.05 - 0.0175)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

From `predictor/`: `python3 -m pytest test_edgelab_capsim.py::test_net_edge_entry_only_fee -q`
Expected: FAIL (`AttributeError: ... 'net_edge_per_share'`).

- [ ] **Step 3: Write minimal implementation**

```python
def net_edge_per_share(value, ask, rate=FEE_RATE):
    # entry-only fee on the buy; used identically for resolution and mark valuations
    return value - ask - fee_per_share(ask, rate)
```

- [ ] **Step 4: Run test to verify it passes**

From `predictor/`: `python3 -m pytest test_edgelab_capsim.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

From repo root:
```bash
git add predictor/edgelab/capsim.py predictor/test_edgelab_capsim.py
git commit -m "feat(capsim): per-share net edge with entry-only fee"
```

---

### Task 5: `capsim_io.py` — load a capture directory into series

**Files:**
- Create: `predictor/edgelab/capsim_io.py`
- Test: `predictor/test_edgelab_capsim_io.py`

**Interfaces:**
- Produces:
  - `AssetSeries` (a dict with keys `t, ask, ask_sz, mid` as np arrays, ascending `t` seconds).
  - `CaptureData` (a dict with keys `bt, bmid` (np arrays), `assets: dict[str, AssetSeries]`, `ot, oval` (oracle arrays, may be empty)).
  - `load_capture(data_dir: str) -> CaptureData` — globs `events/day=*/source=<src>/*.parquet`, reads with `partitioning=None`, builds: binance mid from `binance_bookticker` (`(best_bid+best_ask)/2`), per-asset clob series from `pm_clob_book` keyed by `asset_id` (parsed from `payload_json`), `ask=best_ask`, `ask_sz=best_ask_sz`, `mid=(best_bid+best_ask)/2` (NaN where a side is None), oracle from `pm_oracle` (`recv_wall_ns`, `price`). All `t` are `recv_wall_ns / 1e9`, sorted ascending.

- [ ] **Step 1: Write the failing test**

```python
# predictor/test_edgelab_capsim_io.py
import json, numpy as np
from edgelab import schema, capsim_io
from edgelab.writer import RotatingParquetWriter

def _row(src, wall_ns, **kw):
    r = schema.empty_row(); r["source"] = src; r["region_id"] = "eu-west-1"
    r["recv_wall_ns"] = wall_ns; r["clock_err_ns"] = 1
    r.update(kw); return r

def test_load_capture_builds_series(tmp_path):
    w = RotatingParquetWriter(str(tmp_path), clock=lambda: 60.0)
    w.write(_row("binance_bookticker", 1_000_000_000, best_bid=100.0, best_ask=100.02))
    w.write(_row("binance_bookticker", 2_000_000_000, best_bid=101.0, best_ask=101.02))
    pj = lambda a: json.dumps({"asset_id": a})
    w.write(_row("pm_clob_book", 1_500_000_000, best_bid=0.50, best_ask=0.52,
                 best_ask_sz=7.0, payload_json=pj("AAA")))
    w.write(_row("pm_clob_book", 1_800_000_000, best_bid=None, best_ask=None,
                 best_ask_sz=None, payload_json=pj("AAA")))   # one-sided -> mid NaN
    w.write(_row("pm_oracle", 1_200_000_000, price=100.0))
    w.flush_all()

    d = capsim_io.load_capture(str(tmp_path))
    assert list(d["bt"]) == [1.0, 2.0]
    assert abs(d["bmid"][1] - 101.01) < 1e-6
    a = d["assets"]["AAA"]
    assert abs(a["ask"][0] - 0.52) < 1e-9 and abs(a["ask_sz"][0] - 7.0) < 1e-9
    assert abs(a["mid"][0] - 0.51) < 1e-9 and np.isnan(a["mid"][1])
    assert list(d["ot"]) == [1.2] and d["oval"][0] == 100.0
```

- [ ] **Step 2: Run test to verify it fails**

From `predictor/`: `python3 -m pytest test_edgelab_capsim_io.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'edgelab.capsim_io'`).

- [ ] **Step 3: Write minimal implementation**

```python
# predictor/edgelab/capsim_io.py
"""Load a capture directory into numpy series for the capturability sim, and
resolve market outcomes. Read-only; never mutates capture data."""
import glob
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _read(data_dir, src, cols):
    fs = sorted(glob.glob(os.path.join(data_dir, "events", "day=*",
                                       f"source={src}", "*.parquet")))
    if not fs:
        return {c: np.array([]) for c in cols}
    t = pa.concat_tables([pq.read_table(f, partitioning=None, columns=cols) for f in fs])
    return {c: np.array(t.column(c).to_pylist()) for c in cols}


def _nan(arr):
    return np.array([np.nan if x is None else float(x) for x in arr], dtype=float)


def load_capture(data_dir):
    b = _read(data_dir, "binance_bookticker", ["recv_wall_ns", "best_bid", "best_ask"])
    bt = np.asarray(b["recv_wall_ns"], float) / 1e9
    bmid = (_nan(b["best_bid"]) + _nan(b["best_ask"])) / 2.0
    o = np.argsort(bt); bt, bmid = bt[o], bmid[o]

    c = _read(data_dir, "pm_clob_book",
              ["recv_wall_ns", "best_bid", "best_ask", "best_ask_sz", "payload_json"])
    assets = {}
    if len(c["recv_wall_ns"]):
        aid = np.array([json.loads(p)["asset_id"] for p in c["payload_json"]])
        ct = np.asarray(c["recv_wall_ns"], float) / 1e9
        ask = _nan(c["best_ask"]); asz = _nan(c["best_ask_sz"])
        mid = (_nan(c["best_bid"]) + _nan(c["best_ask"])) / 2.0
        for a in set(aid):
            m = aid == a
            idx = np.argsort(ct[m])
            assets[a] = {"t": ct[m][idx], "ask": ask[m][idx],
                         "ask_sz": asz[m][idx], "mid": mid[m][idx]}

    orc = _read(data_dir, "pm_oracle", ["recv_wall_ns", "price"])
    ot = np.asarray(orc["recv_wall_ns"], float) / 1e9
    oval = _nan(orc["price"])
    oo = np.argsort(ot); ot, oval = ot[oo], oval[oo]
    return {"bt": bt, "bmid": bmid, "assets": assets, "ot": ot, "oval": oval}
```

- [ ] **Step 4: Run test to verify it passes**

From `predictor/`: `python3 -m pytest test_edgelab_capsim_io.py -q`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

From repo root:
```bash
git add predictor/edgelab/capsim_io.py predictor/test_edgelab_capsim_io.py
git commit -m "feat(capsim_io): load capture directory into numpy series"
```

---

### Task 6: `capsim_io.py` — resolve market outcomes (winning token)

**Files:**
- Modify: `predictor/edgelab/capsim_io.py`
- Test: `predictor/test_edgelab_capsim_io.py`

**Interfaces:**
- Produces: `resolve_outcomes(asset_ids: list[str], fetcher=None) -> dict[str, int]` — for each `asset_id` whose market has resolved, `1` if that token is the winning outcome else `0`; **unresolved/unknown asset_ids are omitted** (so callers can count `n_resolved`). `fetcher(asset_id) -> int | None` is injectable for tests (`1` win, `0` lose, `None` unresolved). The default fetcher queries Gamma for the asset's market and reads the winning token; any error → `None` (omit), never a guess.

- [ ] **Step 1: Write the failing test**

```python
def test_resolve_outcomes_uses_fetcher_and_omits_unknown():
    calls = {"AAA": 1, "BBB": 0, "CCC": None}
    out = capsim_io.resolve_outcomes(["AAA", "BBB", "CCC"], fetcher=lambda a: calls[a])
    assert out == {"AAA": 1, "BBB": 0}        # CCC unresolved -> omitted

def test_resolve_outcomes_default_fetch_error_omits(monkeypatch):
    # a fetcher that raises must be swallowed into omission, never a guess
    def boom(a): raise RuntimeError("gamma down")
    assert capsim_io.resolve_outcomes(["AAA"], fetcher=boom) == {}
```

- [ ] **Step 2: Run test to verify it fails**

From `predictor/`: `python3 -m pytest test_edgelab_capsim_io.py::test_resolve_outcomes_uses_fetcher_and_omits_unknown -q`
Expected: FAIL (`AttributeError: ... 'resolve_outcomes'`).

- [ ] **Step 3: Write minimal implementation**

```python
def _gamma_outcome(asset_id):
    """Query Gamma for the asset's market; return 1 if this token won, 0 if it
    lost, None if unresolved/unknown. Network errors -> None (never a guess)."""
    import requests
    url = "https://gamma-api.polymarket.com/markets"
    r = requests.get(url, params={"clob_token_ids": asset_id}, timeout=10)
    r.raise_for_status()
    mkts = r.json()
    if not mkts:
        return None
    m = mkts[0]
    if not m.get("closed") or m.get("umaResolutionStatus") not in (None, "resolved") \
            and not m.get("closed"):
        return None
    prices = m.get("outcomePrices")
    toks = m.get("clobTokenIds")
    if isinstance(prices, str):
        prices = json.loads(prices)
    if isinstance(toks, str):
        toks = json.loads(toks)
    if not prices or not toks or asset_id not in toks:
        return None
    win = [i for i, p in enumerate(prices) if float(p) >= 0.99]
    if len(win) != 1:
        return None                       # not cleanly resolved yet
    return 1 if toks[win[0]] == asset_id else 0


def resolve_outcomes(asset_ids, fetcher=None):
    fetcher = fetcher or _gamma_outcome
    out = {}
    for a in asset_ids:
        try:
            v = fetcher(a)
        except Exception:
            v = None
        if v is not None:
            out[a] = int(v)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

From `predictor/`: `python3 -m pytest test_edgelab_capsim_io.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

From repo root:
```bash
git add predictor/edgelab/capsim_io.py predictor/test_edgelab_capsim_io.py
git commit -m "feat(capsim_io): resolve market outcomes via Gamma (injectable)"
```

---

### Task 7: `capsim_run.py` — assemble fills for one (θ, R) cell

**Files:**
- Create: `predictor/edgelab/capsim_run.py`
- Test: `predictor/test_edgelab_capsim_run.py`

**Interfaces:**
- Consumes: `capsim` (Task 1–4), `capsim_io.CaptureData` shape (Task 5).
- Produces:
  - `asset_polarity(data, ntm=(0.05, 0.95), min_pts=50) -> dict[str, int]` — `polarity_from_levels(data.bt, data.bmid, a.t, a.mid)` for each asset with ≥ `min_pts` two-sided NTM mid points; assets returning polarity 0 are omitted.
  - `assemble_fills(data, polarity, triggers, R, stake, H, outcomes) -> list[dict]` — for each `(t, d)` trigger and each asset whose `polarity[a] == d`, simulate a buy at `arrival = t + R`: compute `(ask, shares)` via `capsim.hittable_ask`; skip if `shares == 0` or NaN ask. Each fill dict: `{asset, t, arrival, dir, ask, shares, mark_net, res_net}` where `mark_net = capsim.net_edge_per_share(capsim.mark_value(a.t, a.mid, arrival, H), ask)` (NaN if mark mid NaN) and `res_net = capsim.net_edge_per_share(outcomes[asset], ask)` if `asset in outcomes` else `None`.

- [ ] **Step 1: Write the failing test**

```python
# predictor/test_edgelab_capsim_run.py
import numpy as np
from edgelab import capsim_run

def _data():
    bt = np.arange(0.0, 10.0, 0.1)
    bmid = 100.0 + (bt >= 5.0) * 1.0                      # +1% step up at t=5
    up = {"t": bt.copy(), "ask": np.full_like(bt, 0.50),
          "ask_sz": np.full_like(bt, 100.0),
          "mid": 0.50 + (bt >= 5.3) * 0.05}              # clob reprices up at 5.3 (laggy)
    return {"bt": bt, "bmid": bmid, "assets": {"UP": up}, "ot": np.array([]),
            "oval": np.array([])}

def test_asset_polarity_detects_up():
    d = _data()
    assert capsim_run.asset_polarity(d) == {"UP": +1}

def test_assemble_fills_buys_predicted_side_before_reprice():
    d = _data()
    pol = {"UP": +1}
    trg = [(5.05, +1)]                                    # detected the step up
    outcomes = {"UP": 1}                                  # market resolved up (won)
    fills = capsim_run.assemble_fills(d, pol, trg, R=0.1, stake=5.0, H=1.0,
                                      outcomes=outcomes)
    assert len(fills) == 1
    f = fills[0]
    assert f["asset"] == "UP" and abs(f["ask"] - 0.50) < 1e-9
    # bought at 0.50 before reprice; mark at arrival+H=6.15 -> mid 0.55
    assert abs(f["mark_net"] - (0.55 - 0.50 - 0.0175)) < 1e-9
    # resolution: won -> 1 - 0.50 - 0.0175
    assert abs(f["res_net"] - (1.0 - 0.50 - 0.0175)) < 1e-9

def test_assemble_fills_skips_wrong_polarity_and_no_book():
    d = _data()
    fills = capsim_run.assemble_fills(d, {"UP": +1}, [(5.05, -1)], R=0.1, stake=5.0,
                                      H=1.0, outcomes={})
    assert fills == []                                    # trigger dir -1 != polarity +1
```

- [ ] **Step 2: Run test to verify it fails**

From `predictor/`: `python3 -m pytest test_edgelab_capsim_run.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'edgelab.capsim_run'`).

- [ ] **Step 3: Write minimal implementation**

```python
# predictor/edgelab/capsim_run.py
"""Orchestrate the capturability sim: polarity, fills per (theta,R) cell, the
sweep with a random-entry null control, and the report/CLI. Read-only."""
import numpy as np

from edgelab import capsim


def asset_polarity(data, ntm=(0.05, 0.95), min_pts=50):
    out = {}
    for a, s in data["assets"].items():
        mid = s["mid"]
        ok = np.isfinite(mid) & (mid > ntm[0]) & (mid < ntm[1])
        if ok.sum() < min_pts:
            continue
        p = capsim.polarity_from_levels(data["bt"], data["bmid"], s["t"][ok], mid[ok])
        if p != 0:
            out[a] = p
    return out


def assemble_fills(data, polarity, triggers, R, stake, H, outcomes):
    fills = []
    for t, d in triggers:
        arrival = t + R
        for a, p in polarity.items():
            if p != d:
                continue
            s = data["assets"][a]
            ask, shares = capsim.hittable_ask(s["t"], s["ask"], s["ask_sz"], arrival, stake)
            if not np.isfinite(ask) or shares <= 0:
                continue
            mark = capsim.mark_value(s["t"], s["mid"], arrival, H)
            mark_net = capsim.net_edge_per_share(mark, ask) if np.isfinite(mark) else np.nan
            res_net = (capsim.net_edge_per_share(float(outcomes[a]), ask)
                       if a in outcomes else None)
            fills.append({"asset": a, "t": t, "arrival": arrival, "dir": d,
                          "ask": ask, "shares": shares,
                          "mark_net": mark_net, "res_net": res_net})
    return fills
```

- [ ] **Step 4: Run test to verify it passes**

From `predictor/`: `python3 -m pytest test_edgelab_capsim_run.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

From repo root:
```bash
git add predictor/edgelab/capsim_run.py predictor/test_edgelab_capsim_run.py
git commit -m "feat(capsim_run): polarity + fill assembly for one cell"
```

---

### Task 8: `capsim_run.py` — sweep, random control, report

**Files:**
- Modify: `predictor/edgelab/capsim_run.py`
- Test: `predictor/test_edgelab_capsim_run.py`

**Interfaces:**
- Produces:
  - `summarize(fills) -> dict` — `{n, n_resolved, res_median, res_frac_pos, mark_median, median_ask}`; medians over the non-None/non-NaN subset; `None`/`nan` when empty.
  - `days_to_N(n_fills, span_s, N=30) -> float` — `inf` if `n_fills == 0`, else `(N / (n_fills / (span_s/86400)))` capped sensibly (fills/day → days to N).
  - `run_sweep(data, polarity, outcomes, thetas, Rs, Hs, stake, lookback_s, cooldown_s, seed) -> dict` — for each `(theta, R, H)`: momentum fills via `capsim.momentum_triggers` + `assemble_fills`; a **random-entry control** with the same fill count using `capsim.random_triggers`; `summarize` for both; plus `fee_bar = capsim.fee_per_share(median_ask) * 3` (K=3) and `days_to_N`. Returns `{"cells": [...], "span_s": ...}`.

- [ ] **Step 1: Write the failing test**

```python
def test_summarize_and_days_to_N():
    fills = [{"ask": 0.5, "mark_net": 0.03, "res_net": 0.48},
             {"ask": 0.5, "mark_net": -0.01, "res_net": -0.52},
             {"ask": 0.5, "mark_net": 0.02, "res_net": None}]
    s = capsim_run.summarize(fills)
    assert s["n"] == 3 and s["n_resolved"] == 2
    assert abs(s["res_median"] - (-0.02)) < 1e-9          # median of [0.48,-0.52]
    assert abs(s["res_frac_pos"] - 0.5) < 1e-9
    assert capsim_run.days_to_N(0, 3600) == float("inf")
    assert abs(capsim_run.days_to_N(30, 86400) - 30.0) < 1e-6   # 30/day -> 30 days... 

def test_run_sweep_has_momentum_and_random_cells():
    import numpy as np
    bt = np.arange(0.0, 60.0, 0.1)
    bmid = 100.0 + np.floor(bt / 10.0)                    # periodic up-steps
    up = {"t": bt.copy(), "ask": np.full_like(bt, 0.5),
          "ask_sz": np.full_like(bt, 100.0), "mid": 0.5 + 0.01*np.floor(bt/10.0)}
    data = {"bt": bt, "bmid": bmid, "assets": {"UP": up}, "ot": np.array([]),
            "oval": np.array([])}
    pol = {"UP": +1}
    rep = capsim_run.run_sweep(data, pol, outcomes={"UP": 1}, thetas=[50.0], Rs=[0.1],
                               Hs=[1.0], stake=5.0, lookback_s=0.5, cooldown_s=0.5, seed=1)
    c = rep["cells"][0]
    assert c["theta"] == 50.0 and c["R"] == 0.1
    assert "momentum" in c and "random" in c and "fee_bar" in c and "days_to_N" in c
    assert c["momentum"]["n"] >= 1
```

Note: fix the `days_to_N` assertion to the intended semantics — `days_to_N(30, 86400)` means 30 fills in 1 day = 30/day, so days-to-30 = 1.0. Use `assert abs(capsim_run.days_to_N(30, 86400) - 1.0) < 1e-6`.

- [ ] **Step 2: Run test to verify it fails**

From `predictor/`: `python3 -m pytest test_edgelab_capsim_run.py::test_run_sweep_has_momentum_and_random_cells -q`
Expected: FAIL (`AttributeError: ... 'run_sweep'`).

- [ ] **Step 3: Write minimal implementation**

```python
def summarize(fills):
    if not fills:
        return {"n": 0, "n_resolved": 0, "res_median": None, "res_frac_pos": None,
                "mark_median": None, "median_ask": None}
    res = [f["res_net"] for f in fills if f["res_net"] is not None]
    mark = [f["mark_net"] for f in fills if f["mark_net"] is not None
            and np.isfinite(f["mark_net"])]
    asks = [f["ask"] for f in fills]
    return {"n": len(fills), "n_resolved": len(res),
            "res_median": float(np.median(res)) if res else None,
            "res_frac_pos": float(np.mean([r > 0 for r in res])) if res else None,
            "mark_median": float(np.median(mark)) if mark else None,
            "median_ask": float(np.median(asks))}


def days_to_N(n_fills, span_s, N=30):
    if n_fills <= 0 or span_s <= 0:
        return float("inf")
    per_day = n_fills / (span_s / 86400.0)
    return N / per_day


def run_sweep(data, polarity, outcomes, thetas, Rs, Hs, stake,
              lookback_s, cooldown_s, seed):
    bt = data["bt"]
    span_s = float(bt[-1] - bt[0]) if len(bt) > 1 else 0.0
    cells = []
    for theta in thetas:
        trg = capsim.momentum_triggers(bt, data["bmid"], theta, lookback_s, cooldown_s)
        for R in Rs:
            for H in Hs:
                mf = assemble_fills(data, polarity, trg, R, stake, H, outcomes)
                rnd = capsim.random_triggers(bt[0], bt[-1], max(len(trg), 1), seed)
                rf = assemble_fills(data, polarity, rnd, R, stake, H, outcomes)
                ms = summarize(mf); rs = summarize(rf)
                bar = (capsim.fee_per_share(ms["median_ask"]) * 3.0
                       if ms["median_ask"] is not None else None)
                cells.append({"theta": theta, "R": R, "H": H,
                              "momentum": ms, "random": rs, "fee_bar": bar,
                              "days_to_N": days_to_N(ms["n"], span_s)})
    return {"cells": cells, "span_s": span_s}
```

- [ ] **Step 4: Run test to verify it passes** (apply the `days_to_N` assertion fix from Step 1)

From `predictor/`: `python3 -m pytest test_edgelab_capsim_run.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

From repo root:
```bash
git add predictor/edgelab/capsim_run.py predictor/test_edgelab_capsim_run.py
git commit -m "feat(capsim_run): sweep with random-entry control + report"
```

---

### Task 9: CLI + run against the real Ireland capture

**Files:**
- Modify: `predictor/edgelab/capsim_run.py` (add `main`)
- Test: `predictor/test_edgelab_capsim_run.py` (CLI smoke via `print_report`)

**Interfaces:**
- Produces:
  - `print_report(report) -> None` — prints, per cell, a one-line readout leading with `n_resolved`, then momentum res_median (primary), the `fee_bar` (fee×3), the momentum vs random mark_median (leak check), and `days_to_N`.
  - `main()` — `argv[1]` = capture dir (default `EDGELAB_OUT` or `.`); loads data, computes polarity, resolves outcomes (Gamma), runs `run_sweep` with the global-constraint sweeps (`thetas` default `[20, 50, 100]` bps, `Rs=[0.030, 0.060, 0.100, 0.150]`, `Hs=[0.6, 1.0, 2.0]`, stake `5.0`, `lookback_s=0.5`, `cooldown_s=0.5`, `seed=1`), prints the report.

- [ ] **Step 1: Write the failing test**

```python
def test_print_report_leads_with_resolution(capsys):
    report = {"span_s": 3600.0, "cells": [{
        "theta": 50.0, "R": 0.1, "H": 1.0, "fee_bar": 0.0525, "days_to_N": 2.5,
        "momentum": {"n": 12, "n_resolved": 4, "res_median": -0.03,
                     "res_frac_pos": 0.5, "mark_median": 0.02, "median_ask": 0.5},
        "random": {"n": 12, "n_resolved": 4, "res_median": -0.04,
                   "res_frac_pos": 0.5, "mark_median": 0.018, "median_ask": 0.5}}]}
    capsim_run.print_report(report)
    out = capsys.readouterr().out
    assert "n_resolved=4" in out          # leads with the honest sample size
    assert "res_median" in out and "fee_bar" in out and "days_to_N" in out
```

- [ ] **Step 2: Run test to verify it fails**

From `predictor/`: `python3 -m pytest test_edgelab_capsim_run.py::test_print_report_leads_with_resolution -q`
Expected: FAIL (`AttributeError: ... 'print_report'`).

- [ ] **Step 3: Write minimal implementation**

```python
def print_report(report):
    print(f"span_s={report['span_s']:.0f}")
    for c in report["cells"]:
        m = c["momentum"]; r = c["random"]
        print(f"theta={c['theta']:g}bps R={c['R']*1000:.0f}ms H={c['H']:g}s | "
              f"n_resolved={m['n_resolved']} (N={m['n']}) | "
              f"res_median={_fmt(m['res_median'])} vs fee_bar={_fmt(c['fee_bar'])} | "
              f"mark mom={_fmt(m['mark_median'])} rnd={_fmt(r['mark_median'])} | "
              f"days_to_N={c['days_to_N']:.1f}")


def _fmt(x):
    return "n/a" if x is None else f"{x:+.4f}"


def main():
    import os
    import sys
    from edgelab import capsim_io
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EDGELAB_OUT", ".")
    data = capsim_io.load_capture(data_dir)
    polarity = asset_polarity(data)
    outcomes = capsim_io.resolve_outcomes(list(polarity.keys()))
    report = run_sweep(data, polarity, outcomes, thetas=[20.0, 50.0, 100.0],
                       Rs=[0.030, 0.060, 0.100, 0.150], Hs=[0.6, 1.0, 2.0], stake=5.0,
                       lookback_s=0.5, cooldown_s=0.5, seed=1)
    print(f"assets_with_polarity={len(polarity)} resolved={len(outcomes)}")
    print_report(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

From `predictor/`: `python3 -m pytest test_edgelab_capsim_run.py -q`
Expected: PASS (6 tests). Then run the full module suite:
`python3 -m pytest test_edgelab_capsim.py test_edgelab_capsim_io.py test_edgelab_capsim_run.py -q` → all green.

- [ ] **Step 5: Commit**

From repo root:
```bash
git add predictor/edgelab/capsim_run.py predictor/test_edgelab_capsim_run.py
git commit -m "feat(capsim_run): CLI + resolution-led report"
```

- [ ] **Step 6: Run against the real Ireland capture (the actual answer)**

Sync the three new modules + tests to the Ireland box and run on the captured hour:
```bash
rsync -az -e "ssh -i $HOME/.ssh/ireland.pem -o BatchMode=yes" \
  predictor/edgelab/capsim.py predictor/edgelab/capsim_io.py predictor/edgelab/capsim_run.py \
  ubuntu@<server-host>:~/btcpredictor/predictor/edgelab/
ssh -i ~/.ssh/ireland.pem -o BatchMode=yes \
  ubuntu@<server-host> \
  'cd ~/btcpredictor/predictor && OUT=$(cut -d= -f2 .last_latency_run) && \
   ~/btcpredictor/.venv/bin/python -m edgelab.capsim_run $OUT'
```
Expected: a per-cell readout. **Interpretation (resolution-primary):** the verdict is the momentum `res_median` vs `fee_bar`, *gated on `n_resolved`*. If `n_resolved` is tiny (likely in 1h), the honest finding is "mechanism diagnostic only, accumulate days"; record it and do not let a thick `mark_median` override. If momentum `mark_median` ≈ random `mark_median`, the mark valuation is leaky (signal not real). Record the numbers in `.superpowers/sdd/progress.md` and the project memory.

---

## Self-Review

**1. Spec coverage:** §1 objective → Task 9 readout; §2 data → Task 5; §3 trigger → Task 1, polarity → Task 2, arrival/hittable ask → Task 3, fee → Task 3/4, valuation both → Task 4/7, random control → Task 1/8; §4 resolution-primary → Task 7 (`res_net`) + Task 9 readout gating on `n_resolved`; §5 fee verified → Task 1 constant + Task 3 formula; §6 reaction coordinate → Task 8/9 `Rs` sweep with 0.100 headline; §7 sweeps + days-to-N → Task 8/9; §8 honesty guards → causal arrival (Task 3/7), sign-align (Task 2), depth-cap (Task 3), entry-only fee (Task 4), random control (Task 8); §9 structure → 3 modules + tests; §10 out-of-scope respected (no fair-value, no multi-region, no execution). All covered.

**2. Placeholder scan:** No TBD/TODO; every code step is complete. The Task 8 test carries a corrected-assertion note (intentional, explicit value given).

**3. Type consistency:** `net_edge_per_share`, `hittable_ask`, `mark_value`, `polarity_from_levels`, `momentum_triggers`, `random_triggers`, `fee_per_share`, `scalar_ffill`, `load_capture`, `resolve_outcomes`, `asset_polarity`, `assemble_fills`, `summarize`, `days_to_N`, `run_sweep`, `print_report`, `main` — names and signatures are used identically across tasks. CaptureData dict keys (`bt, bmid, assets, ot, oval`) and AssetSeries keys (`t, ask, ask_sz, mid`) are consistent between Task 5 producer and Task 7 consumer.
