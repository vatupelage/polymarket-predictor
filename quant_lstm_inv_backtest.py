"""
Quant backtest of the LSTM-inv-contra gate on the latest trade_history.jsonl.

Principles enforced (per the transcripts on Markov + backtest pitfalls):
- Each trade is treated as a Markov-independent observation; no path-dependent
  features (no "after N losses", no streaks, no running PnL conditioning).
- Conditional distributions analysed separately from the unconditional pool.
- Look-ahead bias avoided: chronological 50/50 IS/OOS split, plus walk-forward
  CV with expanding-window train sets.
- Win rate -> Wilson 95% CI. Mean per-$1 PnL -> bootstrap 95% CI (10k resamples).
- Profile A universe: entry_price in [0.50, 0.75). LSTM-inv gate only acts on
  this universe in production.
"""
import json
import math
import random
from datetime import datetime
from collections import defaultdict

random.seed(42)

PATH = "trade_history.jsonl"

def load():
    out = []
    with open(PATH) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                t = json.loads(line)
            except: continue
            # Filter to rich-schema, profile-A, executed trades
            if t.get("stake_usdc") in (None, 0, 0.0): continue
            if t.get("pnl") is None: continue
            ep = t.get("entry_price")
            if ep is None or not (0.50 <= ep < 0.75): continue
            if "lstm_prob" not in t: continue
            try:
                dt = datetime.fromisoformat(t["ts"])
            except: continue
            t["_dt"] = dt
            t["_per_dollar"] = t["pnl"] / t["stake_usdc"]
            t["_aligns"] = (
                (t["direction"] == "UP" and t["lstm_prob"] >= 0.5) or
                (t["direction"] == "DOWN" and t["lstm_prob"] < 0.5)
            )
            out.append(t)
    out.sort(key=lambda x: x["_dt"])
    return out

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    margin = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    return (max(0, centre - margin), min(1, centre + margin))

def bootstrap_mean_ci(values, n_boot=10000, alpha=0.05):
    if not values: return (0.0, 0.0)
    n = len(values)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += values[random.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return (lo, hi)

def summary(label, trades):
    n = len(trades)
    if n == 0:
        print(f"  {label:<35} n=0")
        return
    wins = sum(1 for t in trades if t["won"])
    pnl = sum(t["pnl"] for t in trades)
    stake = sum(t["stake_usdc"] for t in trades)
    per_dollar = [t["_per_dollar"] for t in trades]
    mean_pd = sum(per_dollar) / n
    w_lo, w_hi = wilson_ci(wins, n)
    pd_lo, pd_hi = bootstrap_mean_ci(per_dollar)
    edge_marker = " EDGE" if pd_lo > 0 else (" LOSS" if pd_hi < 0 else "")
    print(f"  {label:<35} n={n:>3}  W={wins:>3}/{n} ({100*wins/n:>4.1f}%, "
          f"CI[{100*w_lo:>4.1f},{100*w_hi:>4.1f}])  "
          f"PnL=${pnl:>+7.2f} stake=${stake:>6.0f}  "
          f"E[$1]=${mean_pd:>+.3f} CI[${pd_lo:>+.3f},${pd_hi:>+.3f}]{edge_marker}")

# ---------- LOAD ----------
trades = load()
print(f"Total Profile-A trades with lstm_prob: {len(trades)}")
print(f"Date range: {trades[0]['_dt'].date()} -> {trades[-1]['_dt'].date()}")
print()

# ============================================================
# SECTION 1: UNCONDITIONAL POOL (BASELINE)
# ============================================================
print("=" * 78)
print("1. UNCONDITIONAL POOL (BASELINE — no gate applied)")
print("=" * 78)
summary("All Profile-A trades", trades)
print()

# ============================================================
# SECTION 2: CONDITIONAL SPLIT BY LSTM-INV-CONTRA GATE
# ============================================================
print("=" * 78)
print("2. CONDITIONAL SPLIT: aligns-with-LSTM vs disagrees-with-LSTM")
print("=" * 78)
aligns = [t for t in trades if t["_aligns"]]
disagrees = [t for t in trades if not t["_aligns"]]
summary("AGREES with LSTM (GATE SKIPS)", aligns)
summary("DISAGREES with LSTM (gate TAKES)", disagrees)
print()
print("  Interpretation: the gate skips 'AGREES'. The bot keeps 'DISAGREES'.")
print("  An edge exists iff DISAGREES has CI(E[$1]) strictly above 0.")
print()

# ============================================================
# SECTION 3: OUT-OF-SAMPLE (chronological 50/50 split)
# ============================================================
print("=" * 78)
print("3. OUT-OF-SAMPLE VALIDATION (chronological 50/50 split)")
print("=" * 78)
mid = len(trades) // 2
is_set = trades[:mid]
oos_set = trades[mid:]
print(f"  IS:  {is_set[0]['_dt'].date()} -> {is_set[-1]['_dt'].date()}  (n={len(is_set)})")
print(f"  OOS: {oos_set[0]['_dt'].date()} -> {oos_set[-1]['_dt'].date()}  (n={len(oos_set)})")
print()

for label, dataset in [("IN-SAMPLE", is_set), ("OUT-OF-SAMPLE", oos_set)]:
    print(f"  --- {label} ---")
    a = [t for t in dataset if t["_aligns"]]
    d = [t for t in dataset if not t["_aligns"]]
    summary("All trades", dataset)
    summary("Agrees w/ LSTM (skipped)", a)
    summary("Disagrees w/ LSTM (kept)", d)
    print()

# ============================================================
# SECTION 4: WALK-FORWARD CV (expanding window, 4 folds)
# ============================================================
print("=" * 78)
print("4. WALK-FORWARD CV (expanding training window, 4 folds)")
print("=" * 78)
print("  Each fold: train = data[:cutoff], test = next fold of trades.")
print("  We report the OOS edge of the DISAGREES bucket in each fold.")
print()

n = len(trades)
fold_size = n // 5   # 1 warmup + 4 test folds
print(f"  {'Fold':<6} {'Train n':<8} {'Test n':<8} {'Disagree n':<11} {'Disagree W%':<13} {'E[$1]':<10} {'CI':<22}")
print("  " + "-" * 75)
for fold in range(4):
    train_end = fold_size * (fold + 1)
    test_end = fold_size * (fold + 2)
    test = trades[train_end:test_end]
    d = [t for t in test if not t["_aligns"]]
    if len(d) < 5:
        print(f"  {fold+1:<6} {train_end:<8} {len(test):<8} (too few disagree samples)")
        continue
    wins = sum(1 for t in d if t["won"])
    pd = [t["_per_dollar"] for t in d]
    mpd = sum(pd) / len(pd)
    pd_lo, pd_hi = bootstrap_mean_ci(pd, n_boot=2000)
    print(f"  {fold+1:<6} {train_end:<8} {len(test):<8} {len(d):<11} "
          f"{100*wins/len(d):>5.1f}%       ${mpd:>+.3f}   [${pd_lo:>+.3f},${pd_hi:>+.3f}]")
print()

# ============================================================
# SECTION 5: CONDITIONAL DRILLDOWN ON THE DISAGREES BUCKET
# ============================================================
print("=" * 78)
print("5. CONDITIONAL DRILLDOWN — where does the DISAGREES edge concentrate?")
print("=" * 78)
print("  (Markov: no path-dependent features. Just trade-level conditions.)")
print()

# By direction
print("  Direction:")
for d_label in ["UP", "DOWN"]:
    bucket = [t for t in disagrees if t["direction"] == d_label]
    summary(f"DISAGREES & {d_label}", bucket)
print()

# By entry-price band
print("  Entry price band:")
for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75)]:
    bucket = [t for t in disagrees if lo <= t["entry_price"] < hi]
    summary(f"DISAGREES & entry [{lo:.2f},{hi:.2f})", bucket)
print()

# By hour bucket (local time)
print("  Hour bucket (local):")
for lo, hi in [(0, 6), (6, 12), (12, 18), (18, 24)]:
    bucket = [t for t in disagrees if lo <= t["_dt"].hour < hi]
    summary(f"DISAGREES & hour [{lo:02d},{hi:02d})", bucket)
print()

# By confidence band
print("  Confidence band:")
for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 100)]:
    bucket = [t for t in disagrees if lo <= t["confidence"] < hi]
    summary(f"DISAGREES & conf [{lo},{hi})", bucket)
print()

# ============================================================
# SECTION 6: GATE IMPACT — counterfactual on the full pool
# ============================================================
print("=" * 78)
print("6. GATE IMPACT — what happens if we apply the gate to the full pool?")
print("=" * 78)
total_pnl = sum(t["pnl"] for t in trades)
total_stake = sum(t["stake_usdc"] for t in trades)
gated_pnl = sum(t["pnl"] for t in disagrees)
gated_stake = sum(t["stake_usdc"] for t in disagrees)
skipped_pnl = sum(t["pnl"] for t in aligns)
skipped_stake = sum(t["stake_usdc"] for t in aligns)

print(f"  Without gate (all trades):")
print(f"    n={len(trades)}  stake=${total_stake:.0f}  PnL=${total_pnl:+.2f}  ROI={100*total_pnl/total_stake:+.2f}%")
print(f"  With gate (skip aligns):")
print(f"    n={len(disagrees)}  stake=${gated_stake:.0f}  PnL=${gated_pnl:+.2f}  ROI={100*gated_pnl/gated_stake:+.2f}%")
print(f"  PnL we 'avoided' by skipping aligns: ${skipped_pnl:+.2f}  (negative = good — gate saved money)")
print()

# Daily trade-rate after gate
days = (trades[-1]["_dt"] - trades[0]["_dt"]).days or 1
print(f"  Trade rate after gate: {len(disagrees)/days:.1f} trades/day over {days} days")
print()

# ============================================================
# SECTION 7: STABILITY — recent-window OOS only
# ============================================================
print("=" * 78)
print("7. RECENT-DAYS STABILITY (last 14 days vs prior period)")
print("=" * 78)
cutoff = trades[-1]["_dt"].replace(hour=0, minute=0, second=0, microsecond=0)
from datetime import timedelta
recent_start = cutoff - timedelta(days=14)
recent = [t for t in trades if t["_dt"] >= recent_start]
prior = [t for t in trades if t["_dt"] < recent_start]

print(f"  Prior ({prior[0]['_dt'].date()} -> {recent_start.date()}, n={len(prior)}):")
summary("    All", prior)
summary("    DISAGREES (kept)", [t for t in prior if not t["_aligns"]])
summary("    AGREES (skipped)", [t for t in prior if t["_aligns"]])
print()
print(f"  Recent ({recent_start.date()} -> {trades[-1]['_dt'].date()}, n={len(recent)}):")
summary("    All", recent)
summary("    DISAGREES (kept)", [t for t in recent if not t["_aligns"]])
summary("    AGREES (skipped)", [t for t in recent if t["_aligns"]])
print()

# ============================================================
# SECTION 8: VERDICT
# ============================================================
print("=" * 78)
print("8. VERDICT")
print("=" * 78)
d_n = len(disagrees)
d_w = sum(1 for t in disagrees if t["won"])
d_pd = [t["_per_dollar"] for t in disagrees]
d_mean = sum(d_pd) / len(d_pd) if d_pd else 0
d_lo, d_hi = bootstrap_mean_ci(d_pd) if d_pd else (0, 0)
a_n = len(aligns)
a_w = sum(1 for t in aligns if t["won"])
a_pd = [t["_per_dollar"] for t in aligns]
a_mean = sum(a_pd) / len(a_pd) if a_pd else 0
a_lo, a_hi = bootstrap_mean_ci(a_pd) if a_pd else (0, 0)

print(f"  Gate-keep bucket (DISAGREES): n={d_n}, W={100*d_w/d_n:.1f}%, E[$1]=${d_mean:+.3f} CI[${d_lo:+.3f},${d_hi:+.3f}]")
print(f"  Gate-skip bucket (AGREES):    n={a_n}, W={100*a_w/a_n:.1f}%, E[$1]=${a_mean:+.3f} CI[${a_lo:+.3f},${a_hi:+.3f}]")
print()
if d_lo > 0 and a_hi < d_lo:
    print("  -> Gate has a clean separation: KEEP bucket is profitable AND SKIP bucket is")
    print("     statistically worse. Gate should remain enabled.")
elif d_lo > 0:
    print("  -> Gate KEEP bucket is profitable but SKIP bucket isn't clearly bad —")
    print("     gate is helpful but not as crisp as before. Worth keeping.")
elif d_mean > a_mean:
    print("  -> KEEP is better than SKIP in point estimate, but CIs overlap.")
    print("     Weak evidence — defensible to keep, defensible to disable.")
else:
    print("  -> SIGNAL HAS INVERTED OR DECAYED. Re-examine.")
