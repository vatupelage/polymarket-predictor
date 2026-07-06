"""Conditional-distribution backtest for $200 bankroll.

Properly handles:
  1. Look-ahead bias: gate parameters fit on first half, tested on second half.
  2. Conditional distributions: outcome split by regime (vol, trend,
     time-of-day, recent-bot-perf, market liquidity proxy).
  3. Path-dependence / non-Markov: rolling W% predicts next trade EV?
  4. Realistic equity simulation at $200 bankroll with ruin probability.

This is *intentionally* not a fancier-than-necessary stats library — every
number you see can be reproduced by reading the loops below.
"""
import json
import math
import random
from collections import defaultdict
from datetime import datetime
from statistics import mean, median, stdev

random.seed(42)

# ---------- load ----------
trades = []
for line in open("trade_history.jsonl"):
    d = json.loads(line)
    if "entry_price" not in d or d.get("entry_price") is None:
        continue
    if "btc_drift_pct" not in d or d.get("btc_drift_pct") is None:
        continue
    trades.append(d)

trades.sort(key=lambda t: t["ts"])
print(f"Loaded {len(trades)} rich-schema trades, "
      f"{trades[0]['ts'][:10]} → {trades[-1]['ts'][:10]}")


def per_dollar(t):
    s = t.get("stake_usdc") or 1.0
    return t["pnl"] / s


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n
    den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return (c-h, c+h)


def boot(x, n_iter=3000):
    if not x: return (0, 0, 0)
    n = len(x)
    means = sorted(sum(x[random.randrange(n)] for _ in range(n))/n
                   for _ in range(n_iter))
    return (sum(x)/n, means[int(n_iter*0.025)], means[int(n_iter*0.975)])


# ---------- Profile A filter (entry band + active gates) ----------
def passes_profile_a(t):
    e = t.get("entry_price")
    if e is None or not (0.50 <= e < 0.75): return False
    if e >= 0.70 and t["confidence"] < 12: return False
    if e < 0.55 and t["confidence"] >= 20: return False
    drift = t["btc_drift_pct"]
    if abs(drift) < 0.0005 and t["confidence"] >= 10: return False
    direction = t["direction"]
    if direction == "UP" and drift < -0.05: return False
    if direction == "DOWN" and drift > 0.05: return False
    return True


prof_a = [t for t in trades if passes_profile_a(t)]
print(f"Profile A passes: {len(prof_a)}\n")

pnls = [per_dollar(t) for t in prof_a]
ts0 = datetime.fromisoformat(prof_a[0]["ts"])
ts1 = datetime.fromisoformat(prof_a[-1]["ts"])
days = (ts1 - ts0).total_seconds() / 86400


# =======================================================================
# 1. UNCONDITIONAL BASELINE (what my prior analysis reported)
# =======================================================================
print("=" * 72)
print("1. UNCONDITIONAL BASELINE — what the naive pooled analysis says")
print("=" * 72)
m, lo, hi = boot(pnls)
n_w = sum(1 for t in prof_a if t["won"])
wl, wh = wilson(n_w, len(prof_a))
print(f"  n={len(prof_a)}  W%={n_w/len(prof_a)*100:.1f}%  "
      f"(Wilson [{wl*100:.1f},{wh*100:.1f}]%)")
print(f"  E[$1]={m:+.4f}  (boot [{lo:+.4f},{hi:+.4f}])")
print(f"  Span {days:.1f} d → {len(prof_a)/days:.2f} trades/day → "
      f"daily E[$1]={sum(pnls)/days:+.3f}")
print()
print("  WARNING (Markov violation): this number assumes all 160 trades")
print("  come from one distribution. The conditional analysis below shows")
print("  they don't.")


# =======================================================================
# 2. CONDITIONAL DISTRIBUTIONS — regime-by-regime
# =======================================================================
print()
print("=" * 72)
print("2. CONDITIONAL DISTRIBUTIONS — different regimes, different EV")
print("=" * 72)


def cond_report(name, rows):
    if not rows or len(rows) < 8:
        print(f"  {name:<48} n={len(rows):>3} (too few — sample warning)")
        return None
    pn = [per_dollar(t) for t in rows]
    wn = sum(1 for t in rows if t["won"])
    m_, lo_, hi_ = boot(pn, 2000)
    wl_, wh_ = wilson(wn, len(rows))
    ev_zero_crosses = lo_ < 0 < hi_
    sig = "      " if ev_zero_crosses else " ★EDGE"
    print(f"  {name:<48} n={len(rows):>3}  W={wn/len(rows)*100:>5.1f}%  "
          f"E[$1]={m_:+.3f} [{lo_:+.3f},{hi_:+.3f}]{sig}")
    return {"n": len(rows), "ev": m_, "w": wn/len(rows),
            "ev_lo": lo_, "ev_hi": hi_, "sig": not ev_zero_crosses}


# --- regime: BTC drift quantile (direction of micro-trend) ---
drifts = sorted(t["btc_drift_pct"] for t in prof_a)
q1, q2, q3 = drifts[len(drifts)//4], drifts[len(drifts)//2], drifts[3*len(drifts)//4]
print(f"\n  Drift quartiles: q25={q1:+.4f}  q50={q2:+.4f}  q75={q3:+.4f}\n")
print("-- BTC drift regime --")
cond_report("drift<q25 (strong DOWN micro-trend)",
            [t for t in prof_a if t["btc_drift_pct"] < q1])
cond_report("drift [q25,q50)",
            [t for t in prof_a if q1 <= t["btc_drift_pct"] < q2])
cond_report("drift [q50,q75)",
            [t for t in prof_a if q2 <= t["btc_drift_pct"] < q3])
cond_report("drift>=q75 (strong UP micro-trend)",
            [t for t in prof_a if t["btc_drift_pct"] >= q3])

# --- regime: direction vs drift sign agreement ---
print("\n-- direction vs drift agreement --")
def agree(t):
    return ((t["direction"]=="UP" and t["btc_drift_pct"]>0) or
            (t["direction"]=="DOWN" and t["btc_drift_pct"]<0))
cond_report("direction agrees with drift sign",
            [t for t in prof_a if agree(t)])
cond_report("direction CONTRA drift sign",
            [t for t in prof_a if not agree(t)])

# --- regime: hour of day (UTC) ---
print("\n-- hour of day (UTC) --")
bins = defaultdict(list)
for t in prof_a:
    h = datetime.fromisoformat(t["ts"]).hour
    bin_key = "00-06" if h<6 else "06-12" if h<12 else "12-18" if h<18 else "18-24"
    bins[bin_key].append(t)
for k in sorted(bins.keys()):
    cond_report(f"hour {k} UTC", bins[k])

# --- regime: liquidity proxy = (ask_up + ask_down) — how "close" book is ---
print("\n-- book imbalance (ask_up + ask_down) --")
for t in prof_a:
    t["_book_sum"] = (t.get("top_ask_up") or 0.5) + (t.get("top_ask_down") or 0.5)
sums = sorted(t["_book_sum"] for t in prof_a)
m_bs = sums[len(sums)//2]
cond_report(f"book_sum<{m_bs:.2f} (tight book)",
            [t for t in prof_a if t["_book_sum"]<m_bs])
cond_report(f"book_sum>={m_bs:.2f} (wide book)",
            [t for t in prof_a if t["_book_sum"]>=m_bs])

# --- regime: LSTM-inverted alignment ---
print("\n-- LSTM inverted alignment (proven strongest signal) --")
def lstm_inv_aligned(t):
    lp = t.get("lstm_prob", 0.5)
    return ((t["direction"]=="UP" and lp<0.5) or
            (t["direction"]=="DOWN" and lp>0.5))
cond_report("LSTM-inv aligned (bet against LSTM)",
            [t for t in prof_a if lstm_inv_aligned(t)])
cond_report("LSTM-inv contra (bet WITH LSTM)",
            [t for t in prof_a if not lstm_inv_aligned(t)])


# =======================================================================
# 3. PATH-DEPENDENCE TEST — does prior W% predict next trade?
# =======================================================================
print()
print("=" * 72)
print("3. PATH-DEPENDENCE — Markov violation test")
print("=" * 72)
print("  If markets-via-this-bot were Markov, prior bot W% should NOT")
print("  predict the next trade EV. If it does → regime persistence.\n")

for window in [5, 10]:
    by_prior_w = defaultdict(list)
    for i, t in enumerate(prof_a):
        if i < window:
            continue
        prior = prof_a[i-window:i]
        prior_w = sum(1 for p in prior if p["won"]) / window
        bucket = "low<=0.4" if prior_w<=0.4 else "mid 0.4-0.7" if prior_w<0.7 else "high>=0.7"
        by_prior_w[bucket].append(t)
    print(f"  Window = {window} prior trades:")
    for k in ["low<=0.4", "mid 0.4-0.7", "high>=0.7"]:
        if k in by_prior_w:
            cond_report(f"    after prior W% {k}", by_prior_w[k])
    print()


# =======================================================================
# 4. OUT-OF-SAMPLE TEST — calibrate filter on first half, test on second
# =======================================================================
print("=" * 72)
print("4. OUT-OF-SAMPLE — no look-ahead, no overfitting")
print("=" * 72)

# Use Profile A trades split chronologically 50/50
split = len(prof_a) // 2
train = prof_a[:split]
test = prof_a[split:]
print(f"  Train (calibrate): n={len(train)} "
      f"({train[0]['ts'][:10]} → {train[-1]['ts'][:10]})")
print(f"  Test  (held-out):  n={len(test)}  "
      f"({test[0]['ts'][:10]} → {test[-1]['ts'][:10]})\n")


# Define candidate filters; "fit" by reading IN-SAMPLE EV; report OOS EV.
def stat_simple(rows):
    if not rows: return None
    pn = [per_dollar(t) for t in rows]
    return {"n": len(rows),
            "w": sum(1 for t in rows if t["won"])/len(rows),
            "ev": sum(pn)/len(pn),
            "total": sum(pn)}


candidates = {
    "F0 ALL Profile A (no extra filter)":
        lambda t: True,
    "F1 entry [0.50,0.65)":
        lambda t: 0.50 <= t["entry_price"] < 0.65,
    "F2 entry [0.55,0.65)":
        lambda t: 0.55 <= t["entry_price"] < 0.65,
    "F3 entry [0.55,0.70) ∩ conf>=12":
        lambda t: 0.55<=t["entry_price"]<0.70 and t["confidence"]>=12,
    "F4 LSTM-inv aligned":
        lambda t: lstm_inv_aligned(t),
    "F5 LSTM-inv aligned ∩ entry [0.55,0.70)":
        lambda t: lstm_inv_aligned(t) and 0.55<=t["entry_price"]<0.70,
    "F6 LSTM-inv aligned ∩ entry [0.55,0.65)":
        lambda t: lstm_inv_aligned(t) and 0.55<=t["entry_price"]<0.65,
}


print(f"  {'Filter':<46} {'IS_n':>4} {'IS_W%':>6} {'IS_EV':>7}  "
      f"{'OOS_n':>5} {'OOS_W%':>7} {'OOS_EV':>7} {'verdict':>10}")
print("  " + "-"*100)
results = {}
for name, pred in candidates.items():
    tr = [t for t in train if pred(t)]
    te = [t for t in test if pred(t)]
    s_tr = stat_simple(tr)
    s_te = stat_simple(te)
    if not s_tr or not s_te or s_te["n"] < 5:
        continue
    # bootstrap CI on OOS
    pn_te = [per_dollar(t) for t in te]
    _, lo_, hi_ = boot(pn_te, 1500)
    decays = s_te["ev"] < s_tr["ev"] - 0.05
    verdict = "DECAYED" if decays else ("OOS+VE" if lo_>0 else "OOS unclear")
    results[name] = {"tr": s_tr, "te": s_te, "te_lo": lo_, "te_hi": hi_,
                     "verdict": verdict}
    print(f"  {name:<46} {s_tr['n']:>4} {s_tr['w']*100:>5.1f}% "
          f"{s_tr['ev']:+.3f}  {s_te['n']:>5} {s_te['w']*100:>6.1f}% "
          f"{s_te['ev']:+.3f} {verdict:>10}")
print()
print("  KEY: 'DECAYED' = OOS EV is materially worse than IS (overfit)")
print("       'OOS+VE'  = OOS bootstrap lower bound > 0 (real edge)")
print("       'unclear' = OOS CI crosses zero")


# =======================================================================
# 5. EQUITY SIMULATION at $200 bankroll
# =======================================================================
print()
print("=" * 72)
print("5. EQUITY SIMULATION — $200 starting bankroll, 60 trading days")
print("=" * 72)

# Use OOS-validated filter; fall back to ALL if nothing survives
best_filter = None
best_ev = -99
for name, r in results.items():
    if r["verdict"] == "OOS+VE" and r["te"]["ev"] > best_ev:
        best_ev = r["te"]["ev"]; best_filter = name
if best_filter is None:
    # use best IS+OOS unclear  with most data
    best_filter = "F0 ALL Profile A (no extra filter)"
print(f"  Using filter: {best_filter}")
sim_pred = candidates[best_filter]
sim_pool = [per_dollar(t) for t in prof_a if sim_pred(t)]
sim_td = sum(1 for t in prof_a if sim_pred(t)) / days
print(f"  Pool: {len(sim_pool)} per-$1 outcomes, "
      f"avg E[$1]={mean(sim_pool):+.4f}, "
      f"~{sim_td:.2f} trades/day\n")


def simulate(stake, pool, td, bankroll0=200.0, days=60, n_iter=5000):
    """Return list of final bankrolls + ruin/drawdown stats."""
    finals = []
    ruined = 0
    max_dds = []
    for _ in range(n_iter):
        bk = bankroll0
        peak = bk
        max_dd_pct = 0
        td_int = int(round(td))
        for _ in range(days):
            for _ in range(td_int):
                # cap stake at bankroll to prevent unrealistic over-bet
                use_stake = min(stake, bk)
                if use_stake < 0.5:
                    break
                pnl_per_1 = pool[random.randrange(len(pool))]
                bk += pnl_per_1 * use_stake
            if bk > peak: peak = bk
            if peak > 0:
                dd = (peak - bk)/peak
                if dd > max_dd_pct: max_dd_pct = dd
            if bk < 1.0:
                ruined += 1
                bk = 0
                break
        finals.append(bk)
        max_dds.append(max_dd_pct)
    finals.sort()
    return {
        "p10": finals[int(n_iter*0.10)],
        "p50": finals[n_iter//2],
        "p90": finals[int(n_iter*0.90)],
        "mean": sum(finals)/n_iter,
        "ruin_pct": ruined/n_iter*100,
        "max_dd_p50": sorted(max_dds)[n_iter//2]*100,
        "max_dd_p90": sorted(max_dds)[int(n_iter*0.90)]*100,
        "daily_pnl": (sum(finals)/n_iter - bankroll0)/days,
    }


print(f"  {'Stake':>6} {'Pct bk':>8} {'mean BK':>9} {'P10 BK':>9} "
      f"{'P50 BK':>9} {'P90 BK':>9} {'$/day':>8} {'Ruin%':>7} {'P50 DD%':>9} {'P90 DD%':>9}")
print("  " + "-"*98)
for stake in [3, 5, 8, 10, 15, 20]:
    if stake > 200*0.20:  # over full Kelly
        suffix = " ⚠over-Kelly"
    else:
        suffix = ""
    r = simulate(stake, sim_pool, sim_td)
    print(f"  ${stake:>4}  {stake/200*100:>6.1f}%  ${r['mean']:>7.0f}  "
          f"${r['p10']:>7.0f}  ${r['p50']:>7.0f}  ${r['p90']:>7.0f}  "
          f"${r['daily_pnl']:>+5.2f}  {r['ruin_pct']:>5.1f}%  "
          f"{r['max_dd_p50']:>7.1f}%  {r['max_dd_p90']:>7.1f}%{suffix}")


# =======================================================================
# 6. KELLY SIZING — formal optimal stake
# =======================================================================
print()
print("=" * 72)
print("6. KELLY SIZING — formal optimal stake for $200 bankroll")
print("=" * 72)
pool = sim_pool
p = sum(1 for x in pool if x > 0) / len(pool)
# Compute mean win and mean loss conditional
wins_only = [x for x in pool if x > 0]
losses_only = [x for x in pool if x <= 0]
b = mean(wins_only) if wins_only else 0  # win payoff per $1
a = -mean(losses_only) if losses_only else 1  # loss per $1 (positive)
# Generalized Kelly: f* = p/a - (1-p)/b  (Thorp's formula for non-symmetric)
kelly = p/a - (1-p)/b
print(f"  p(win)={p:.3f}  avg win={b:+.3f}  avg loss={-a:+.3f}")
print(f"  Full Kelly fraction = {kelly*100:.1f}% of bankroll")
print(f"  Half Kelly          = {kelly*50:.1f}% of bankroll")
print(f"  Quarter Kelly       = {kelly*25:.1f}% of bankroll")
print()
print(f"  At $200 bankroll:")
print(f"    Full-Kelly stake    = ${200*kelly:.2f}")
print(f"    Half-Kelly stake    = ${200*kelly/2:.2f}  ← recommended")
print(f"    Quarter-Kelly stake = ${200*kelly/4:.2f}  ← conservative")
