# bnb_15m dollar-bar threshold recalibration + retrain — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the bnb_15m dbmodel on a liquidity-appropriate dollar-bar threshold (~$24k vs the current $125k), and deploy it **only if it beats the incumbent out-of-sample.**

**Architecture:** Reuse the existing offline pipeline (`tools/fetch_aggtrades.py` → `train/calibrate_threshold.py` → `train/train_db_model.py`) to fetch BNB trades, pick a ~20-25s-per-bar threshold, and train a t2 bundle. Add one net-new harness (`train/validate_threshold.py`) that scores t2 vs the incumbent on a purged out-of-sample slice and gates deployment on a CI-backed prediction-quality improvement. Add a `tools/cadence_audit.py` so threshold miscalibration is caught going forward.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn (IsotonicRegression), xgboost, lightgbm, joblib. Binance public data dumps. systemd `dblive@.service` on the all-markets box.

## Global Constraints

- 15m model env: `DBM_WINDOW_S=900`, `DBM_MONITOR_START_S=840` (1-min-in entry) — set for every train/validate run.
- `VOL_WINDOW = 10` bars; feature set is fixed: `['drift_pct','secs_to_close','duration','ret','log_ret','volatility','mean_price','rvol']` (do not add/remove features).
- Taker fee: `fee_per_$1 = 0.07 * (1 - ask)`; per-share `0.07 * p * (1-p)`. Use verbatim.
- Target bar cadence: **median bar duration ~20-25s** (matches btc/eth).
- Incumbent bundle (control, never overwrite): `models/db_ptb_bnb_15m_t1.joblib` (threshold $125,000).
- New bundle: `models/db_ptb_bnb_15m_t2.joblib`.
- No look-ahead: train slice strictly precedes test slice, with a ≥1-window embargo at the split.
- bnb_15m is **paused live** for the duration; do not restart it until the validation gate passes.
- Work on a branch; commit after each task. Do not push or deploy without explicit approval.

---

### Task 1: Fetch BNB aggTrades for train + OOS span

**Files:**
- Create: `data/bnb_aggtrades.parquet` (output artifact, git-ignored)

**Interfaces:**
- Produces: `data/bnb_aggtrades.parquet` with columns `ts` (ms int64), `price` (f32), `qty` (f32).

- [ ] **Step 1: Fetch the available monthly dumps**

Binance monthly dumps lag (current month not published until next month). Pull the largest available contiguous span ending at the most recent published month.

Run:
```bash
cd /home/vidura/btcpredictor/predictor
python tools/fetch_aggtrades.py BNBUSDT 2026-02 2026-05 data/bnb_aggtrades.parquet
```
Expected: prints each month download; months that 404 are skipped; final parquet written. Confirm with:
```bash
python -c "import pandas as pd; d=pd.read_parquet('data/bnb_aggtrades.parquet'); print(len(d), d['ts'].min(), d['ts'].max())"
```
Expected: tens of millions of rows spanning ~Feb–May 2026.

- [ ] **Step 2: Commit the fetch command (not the data)**

```bash
echo "data/*.parquet" >> .gitignore
git add .gitignore
git commit -m "chore: gitignore aggtrade parquet data"
```

---

### Task 2: Make the threshold grid CLI-overridable, then pick the BNB threshold

`train/calibrate_threshold.py` hardcodes a BTC-scale grid (1e6…5e7) — far too high for BNB. Make the grid an optional CLI arg, then run it to find the ~20-25s threshold.

**Files:**
- Modify: `train/calibrate_threshold.py`
- Test: `tests/test_calibrate_threshold.py`

**Interfaces:**
- Consumes: `live_trader.dollar_bars.DollarBarBuilder`.
- Produces: `median_duration(trades, threshold) -> (median_seconds|None, n_bars)` (unchanged signature, now exercised by a test); a CLI that accepts an optional comma-separated grid as `argv[2]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibrate_threshold.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
from calibrate_threshold import median_duration

def test_median_duration_lower_threshold_shorter_bars():
    # synthetic trades: $1000 of volume per second for 200s
    trades = [(i*1000, 100.0, 10.0) for i in range(200)]  # ts ms, price, qty -> $1000/trade @1/s
    med_lo, n_lo = median_duration(trades, 2000)   # 2 trades/bar -> ~1-2s
    med_hi, n_hi = median_duration(trades, 20000)  # 20 trades/bar -> ~19-20s
    assert n_lo > n_hi
    assert med_hi > med_lo
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_calibrate_threshold.py -v`
Expected: FAIL on import (`train` not importable) or assertion — confirm it fails before editing.

- [ ] **Step 3: Make the grid CLI-overridable**

In `train/calibrate_threshold.py`, replace the hardcoded grid loop in `main()` with:
```python
def main():
    df = pd.read_parquet(sys.argv[1])
    grid = ([float(x) for x in sys.argv[2].split(",")]
            if len(sys.argv) > 2
            else [1e6, 2e6, 5e6, 1e7, 2e7, 5e7])
    cutoff = df["ts"].max() - 24 * 3600 * 1000
    day = df[df["ts"] >= cutoff]
    trades = list(zip(day["ts"].values, day["price"].values, day["qty"].values))
    print(f"calibrating on {len(trades):,} trades (last 24h)")
    for thr in grid:
        med, n = median_duration(trades, thr)
        print(f"  threshold=${thr:>12,.0f}  median_dur={med}  bars={n}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_calibrate_threshold.py -v`
Expected: PASS.

- [ ] **Step 5: Pick the BNB threshold**

Run:
```bash
DBM_WINDOW_S=900 DBM_MONITOR_START_S=840 \
python train/calibrate_threshold.py data/bnb_aggtrades.parquet 10000,15000,20000,25000,30000,40000,60000
```
Expected: a table of median bar durations. **Record the threshold whose `median_dur` is closest to 20-25s** (anticipated ~$24k). Call it `THR` for later tasks.

- [ ] **Step 6: Commit**

```bash
git add train/calibrate_threshold.py tests/test_calibrate_threshold.py
git commit -m "feat(train): CLI-overridable threshold grid for calibrate_threshold"
```

---

### Task 3: Expose a reusable window-reconstruction function from the trainer

The validation harness must reconstruct OOS windows the *same way* the trainer does. Extract that logic into an importable function with a characterization test proving behavior is unchanged.

**Files:**
- Modify: `train/train_db_model.py`
- Test: `tests/test_reconstruct_windows.py`

**Interfaces:**
- Produces: `reconstruct_windows(df, threshold, window_s, monitor_start_s) -> (X: np.ndarray[n,8], y: np.ndarray[n], ws_ids: list[int])` where `X` columns are in `FEATURE_NAMES` order, `y` is 1 if window close > PTB else 0.

- [ ] **Step 1: Read the trainer's existing window loop**

Run: `sed -n '55,200p' train/train_db_model.py`
Identify the block that, given bars + the trade frame, builds per-window feature rows `X` and labels `y` at the decision instant. This block becomes the body of `reconstruct_windows`.

- [ ] **Step 2: Write the characterization test**

```python
# tests/test_reconstruct_windows.py
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
from train_db_model import reconstruct_windows
from live_trader.db_features import FEATURE_NAMES

def test_reconstruct_windows_shapes_and_labels():
    # 3 full 900s windows of synthetic 1/sec trades, gently trending up
    rows=[]
    for s in range(2700):
        rows.append((s*1000, 100.0 + s*0.001, 50.0))   # ts ms, price, qty ($5000/s)
    df = pd.DataFrame(rows, columns=["ts","price","qty"])
    X, y, ws = reconstruct_windows(df, threshold=20000, window_s=900, monitor_start_s=840)
    assert X.shape[1] == len(FEATURE_NAMES)
    assert X.shape[0] == len(y) == len(ws)
    assert set(np.unique(y)).issubset({0,1})
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_reconstruct_windows.py -v`
Expected: FAIL with `ImportError: cannot import name 'reconstruct_windows'`.

- [ ] **Step 4: Extract the function**

In `train/train_db_model.py`, wrap the identified window-building block in:
```python
def reconstruct_windows(df, threshold, window_s, monitor_start_s):
    """Build (X, y, ws_ids) at the decision instant for every full window.
    X columns follow FEATURE_NAMES order; y=1 if close>PTB. Pure: no I/O."""
    # <existing bar-build + per-window feature/label loop, returning np arrays>
    ...
```
Then have the trainer's `main()` call `reconstruct_windows(df, threshold, WINDOW_S, MONITOR_START_S)` instead of the inline block. Keep all numbers identical — this is a pure refactor.

- [ ] **Step 5: Run tests to verify pass + no regression**

Run: `python -m pytest tests/test_reconstruct_windows.py -v`
Expected: PASS.
Then retrain the *incumbent* threshold on a small slice to confirm the refactor didn't change behavior:
```bash
DBM_WINDOW_S=900 DBM_MONITOR_START_S=840 \
python train/train_db_model.py data/bnb_aggtrades.parquet 125000 /tmp/sanity.joblib | tail -20
```
Expected: runs cleanly, prints metrics (no crash). Spot-check accuracy/log-loss are in the same ballpark as the incumbent bundle's meta.

- [ ] **Step 6: Commit**

```bash
git add train/train_db_model.py tests/test_reconstruct_windows.py
git commit -m "refactor(train): extract reconstruct_windows() for reuse"
```

---

### Task 4: Train the t2 model on the train slice

**Files:**
- Create: `models/db_ptb_bnb_15m_t2.joblib`

**Interfaces:**
- Consumes: `data/bnb_aggtrades.parquet`, `THR` from Task 2.
- Produces: `models/db_ptb_bnb_15m_t2.joblib` (keys: `model, calibrator, feature_names, threshold_usd, window_s, monitor_start_s, meta`).

- [ ] **Step 1: Carve a train-only parquet (chronological, leave OOS out)**

```bash
python - <<'PY'
import pandas as pd
d = pd.read_parquet("data/bnb_aggtrades.parquet").sort_values("ts")
cut = d["ts"].quantile(0.70)          # first 70% = train, last 30% = OOS
d[d["ts"] <= cut].to_parquet("data/bnb_train.parquet")
print("train rows:", (d["ts"]<=cut).sum(), "cutoff ms:", int(cut))
PY
```
Expected: prints train row count + cutoff timestamp (save the cutoff for Task 6).

- [ ] **Step 2: Train t2 at the chosen threshold**

Replace `THR` with the value chosen in Task 2 (e.g. 24000):
```bash
DBM_WINDOW_S=900 DBM_MONITOR_START_S=840 \
python train/train_db_model.py data/bnb_train.parquet THR models/db_ptb_bnb_15m_t2.joblib
```
Expected: prints train metrics + saves the bundle.

- [ ] **Step 3: Verify the bundle's cadence is right**

```bash
python -c "import joblib; o=joblib.load('models/db_ptb_bnb_15m_t2.joblib'); print(o['threshold_usd'], o['window_s'], o['monitor_start_s'], o['meta'])"
```
Expected: `threshold_usd == THR`, `window_s 900`, `monitor_start_s 840`.

- [ ] **Step 4: Commit**

```bash
git add models/db_ptb_bnb_15m_t2.joblib
git commit -m "feat(model): train bnb_15m t2 at recalibrated threshold"
```

---

### Task 5: Build the OOS head-to-head validation harness

Score t2 (at THR) vs incumbent (at $125k) on the held-out OOS slice. Primary gate = prediction quality (log-loss/Brier/accuracy) with a paired bootstrap CI on the difference. Pure metric functions are unit-tested.

**Files:**
- Create: `train/validate_threshold.py`
- Test: `tests/test_validate_metrics.py`

**Interfaces:**
- Consumes: `reconstruct_windows` (Task 3); `live_trader.db_model` loader for a bundle's `predict_proba`.
- Produces: `bootstrap_diff_ci(a, b, metric, n=10000, seed=0) -> (mean_diff, lo, hi)` and `eval_bundle(bundle, X, y) -> {'logloss','brier','acc'}`; a CLI printing the head-to-head table + a PASS/FAIL verdict.

- [ ] **Step 1: Write failing tests for the pure metrics**

```python
# tests/test_validate_metrics.py
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
from validate_threshold import bootstrap_diff_ci

def test_bootstrap_diff_ci_detects_real_improvement():
    rng = np.random.default_rng(0)
    # model A errors smaller than B -> mean_diff (A-B) negative, CI excludes 0
    a = rng.normal(0.30, 0.02, 400)   # per-window log-loss of A
    b = rng.normal(0.45, 0.02, 400)   # per-window log-loss of B
    md, lo, hi = bootstrap_diff_ci(a, b, n=2000, seed=1)
    assert md < 0 and hi < 0          # A strictly better, CI clears zero

def test_bootstrap_diff_ci_ties_straddle_zero():
    rng = np.random.default_rng(2)
    a = rng.normal(0.40, 0.05, 400); b = rng.normal(0.40, 0.05, 400)
    md, lo, hi = bootstrap_diff_ci(a, b, n=2000, seed=3)
    assert lo < 0 < hi                # indistinguishable -> CI straddles 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_validate_metrics.py -v`
Expected: FAIL with `ImportError: cannot import name 'bootstrap_diff_ci'`.

- [ ] **Step 3: Implement the harness**

```python
# train/validate_threshold.py
"""OOS head-to-head: t2 (recalibrated threshold) vs incumbent ($125k).
Primary gate = per-window log-loss improvement with a paired bootstrap CI<0.
    python train/validate_threshold.py data/bnb_aggtrades.parquet <CUTOFF_MS> \
        models/db_ptb_bnb_15m_t2.joblib <THR> models/db_ptb_bnb_15m_t1.joblib 125000
"""
import os, sys
import numpy as np, pandas as pd, joblib
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.train_db_model import reconstruct_windows  # noqa: E402

def _proba(bundle, X):
    raw = bundle["model"].predict_proba(X)[:, 1]
    cal = bundle["calibrator"]
    return cal.predict(raw) if cal is not None else raw

def eval_bundle(bundle, X, y):
    p = np.clip(_proba(bundle, X), 1e-6, 1-1e-6)
    return {"logloss": log_loss(y, p), "brier": brier_score_loss(y, p),
            "acc": accuracy_score(y, (p >= 0.5).astype(int))}

def _per_window_logloss(bundle, X, y):
    p = np.clip(_proba(bundle, X), 1e-6, 1-1e-6)
    return -(y*np.log(p) + (1-y)*np.log(1-p))   # per-window loss vector

def bootstrap_diff_ci(a, b, n=10000, seed=0):
    """Paired bootstrap on (a-b). Returns (mean_diff, lo95, hi95)."""
    a = np.asarray(a); b = np.asarray(b); d = a - b
    rng = np.random.default_rng(seed); m = len(d)
    means = np.array([d[rng.integers(0, m, m)].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def main():
    pq, cutoff = sys.argv[1], int(sys.argv[2])
    b2, thr2 = joblib.load(sys.argv[3]), float(sys.argv[4])
    b1, thr1 = joblib.load(sys.argv[5]), float(sys.argv[6])
    df = pd.read_parquet(pq).sort_values("ts")
    oos = df[df["ts"] > cutoff]                 # purged: strictly after train cutoff
    W = int(os.environ.get("DBM_WINDOW_S", "900")); M = int(os.environ.get("DBM_MONITOR_START_S", "840"))
    X2, y2, _ = reconstruct_windows(oos, thr2, W, M)
    X1, y1, _ = reconstruct_windows(oos, thr1, W, M)
    m2, m1 = eval_bundle(b2, X2, y2), eval_bundle(b1, X1, y1)
    print(f"  t2   (${thr2:,.0f}): {m2}")
    print(f"  inc  (${thr1:,.0f}): {m1}")
    # Compare per-window log-loss on the OVERLAPPING set of windows (align by count)
    n = min(len(y2), len(y1))
    ll2 = _per_window_logloss(b2, X2[:n], y2[:n]); ll1 = _per_window_logloss(b1, X1[:n], y1[:n])
    md, lo, hi = bootstrap_diff_ci(ll2, ll1)
    verdict = "PASS — t2 better (CI<0)" if hi < 0 else "FAIL — not a CI-backed improvement"
    print(f"  per-window logloss diff (t2-inc): mean={md:+.4f} CI=[{lo:+.4f},{hi:+.4f}] -> {verdict}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_validate_metrics.py -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add train/validate_threshold.py tests/test_validate_metrics.py
git commit -m "feat(train): OOS head-to-head threshold validation harness"
```

---

### Task 6: Run the validation gate and decide

**Files:** none (decision step; produces a recorded result)

- [ ] **Step 1: Run the head-to-head on the OOS slice**

Use `THR` from Task 2 and the cutoff ms from Task 4 Step 1:
```bash
DBM_WINDOW_S=900 DBM_MONITOR_START_S=840 \
python train/validate_threshold.py data/bnb_aggtrades.parquet <CUTOFF_MS> \
    models/db_ptb_bnb_15m_t2.joblib THR models/db_ptb_bnb_15m_t1.joblib 125000
```
Expected: prints t2 vs incumbent log-loss/Brier/acc + the bootstrap verdict line.

- [ ] **Step 2: Apply the gate**

- **PASS** (`hi < 0`, t2's per-window log-loss CI-better): proceed to Task 7 then deploy (Task 8).
- **FAIL** (CI straddles 0 or t2 worse): **stop. Keep the incumbent.** Record the result in the spec file under a "## Validation result" heading and conclude the threshold was not the bottleneck. Do not deploy.

- [ ] **Step 3: Record the outcome in the spec**

Append the printed numbers + verdict to `docs/superpowers/specs/2026-06-18-bnb-dollarbar-threshold-retrain-design.md` and commit:
```bash
git add docs/superpowers/specs/2026-06-18-bnb-dollarbar-threshold-retrain-design.md
git commit -m "docs(bnb-retrain): record OOS validation result"
```

---

### Task 7: Cadence-audit tool (recurrence prevention) — do regardless of gate outcome

**Files:**
- Create: `tools/cadence_audit.py`
- Test: `tests/test_cadence_audit.py`

**Interfaces:**
- Produces: `sec_per_bar(quote_vol_24h_usd, threshold_usd) -> float` and `audit(threshold, vol24h, target_lo=18, target_hi=30) -> (sec_per_bar, ok: bool)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cadence_audit.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from cadence_audit import sec_per_bar, audit

def test_sec_per_bar_basic():
    # $86,400 over 24h = $1/sec; $20 threshold -> 20s/bar
    assert abs(sec_per_bar(86400, 20) - 20.0) < 1e-6

def test_audit_flags_too_slow():
    spb, ok = audit(threshold=125000, vol24h=0.10e9)   # bnb today ~105s
    assert spb > 30 and ok is False

def test_audit_passes_in_band():
    spb, ok = audit(threshold=24000, vol24h=0.10e9)     # ~20s
    assert 18 <= spb <= 30 and ok is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cadence_audit.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# tools/cadence_audit.py
"""Flag dbot dollar-bar thresholds whose live bar cadence drifts out of the
~20-25s target band. Pulls Binance 24h quote volume per symbol.
    python tools/cadence_audit.py
"""
import sys, requests

BOTS = [("btc_5m","BTCUSDT",250000),("eth_15m","ETHUSDT",150000),("bnb_15m","BNBUSDT",125000)]

def sec_per_bar(quote_vol_24h_usd, threshold_usd):
    return threshold_usd / (quote_vol_24h_usd / 86400.0)

def audit(threshold, vol24h, target_lo=18, target_hi=30):
    spb = sec_per_bar(vol24h, threshold)
    return spb, (target_lo <= spb <= target_hi)

def main():
    for lbl, pair, thr in BOTS:
        qv = float(requests.get("https://api.binance.com/api/v3/ticker/24hr",
                                params={"symbol": pair}, timeout=15).json()["quoteVolume"])
        spb, ok = audit(thr, qv)
        print(f"  {lbl:8s} thr=${thr:,} vol=${qv/1e9:.2f}B -> {spb:5.0f}s/bar  {'OK' if ok else 'RETRAIN'}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + the tool**

Run: `python -m pytest tests/test_cadence_audit.py -v` → PASS.
Run: `python tools/cadence_audit.py` → expect btc/eth `OK`, bnb `RETRAIN` (at $125k).

- [ ] **Step 5: Commit**

```bash
git add tools/cadence_audit.py tests/test_cadence_audit.py
git commit -m "feat(tools): dollar-bar cadence audit to catch threshold drift"
```

---

### Task 8: Deploy t2 (CONDITIONAL — only if Task 6 gate PASSED)

**Files:**
- Modify (on all-markets box): `live_env/bnb_15m.env`

- [ ] **Step 1: Copy t2 bundle to the box**

```bash
scp -i ~/.ssh/all_markets.pem models/db_ptb_bnb_15m_t2.joblib \
  ubuntu@<all-markets-host>:/home/ubuntu/btcpredictor/predictor/models/
```

- [ ] **Step 2: Point bnb at t2 and dry-run smoke test**

On the box: set `BOT_DBMODEL_PATH=models/db_ptb_bnb_15m_t2.joblib` in `live_env/bnb_15m.env`, then start in DRY-RUN (main `.env` `BOT_DRY_RUN=true` temporarily, or a one-off run) and confirm **dollar bars warm up in <~3 min** (not ~17) and a fire logs with `duration` ~20s.

- [ ] **Step 3: Go live + keep incumbent as A/B control**

Restart `dblive@bnb_15m` live at $5. Re-arm the hourly snapshot. Watch the first few fires + redeemer. Record t2 vs t1 live edge over the following days before trusting it.

- [ ] **Step 4: Commit deploy notes**

Update `multibot_live_deployment` memory + commit any local config changes.

---

## Self-Review

- **Spec coverage:** threshold rule (Task 2), data+features (Tasks 1,4), train (Task 4), OOS validation gate w/ CI (Tasks 5,6), recurrence prevention (Task 7), conditional deploy (Task 8). ✓
- **Placeholders:** `THR`, `<CUTOFF_MS>`, `<all-markets-host>` are explicitly defined as values to fill from prior task output — not vague TODOs. ✓
- **Type consistency:** `reconstruct_windows(df, threshold, window_s, monitor_start_s) -> (X, y, ws_ids)` used identically in Tasks 3 and 5; `bootstrap_diff_ci`/`eval_bundle`/`sec_per_bar`/`audit` signatures match between tests and impl. ✓
