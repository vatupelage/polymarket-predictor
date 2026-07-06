"""Quant-level backtest of Profile A and sub-strategies for $120/day target."""
import json
import math
import random
from collections import defaultdict
from statistics import mean, stdev

random.seed(42)

# ---------- load ----------
trades = []
for line in open("trade_history.jsonl"):
    d = json.loads(line)
    if "entry_price" not in d or "btc_drift_pct" not in d:
        continue
    trades.append(d)

print(f"Loaded {len(trades)} rich-schema trades (since 2026-04-30)")

# ---------- Profile A filter ----------
# entry [0.50, 0.75), all other gates as currently enabled
# Sketch the gate stack by emulating the bot's filters from the data.
def passes_profile_a(t):
    e = t.get("entry_price")
    if e is None or not (0.50 <= e < 0.75):
        return False
    # high_entry_low_conf: skip entry>=0.70 AND conf<12
    if e >= 0.70 and t["confidence"] < 12:
        return False
    # mid_price_high_conf: skip entry<0.55 AND conf>=20
    if e < 0.55 and t["confidence"] >= 20:
        return False
    # drift_noise: skip |drift|<0.0005% AND conf>=10
    if abs(t["btc_drift_pct"]) < 0.0005 and t["confidence"] >= 10:
        return False
    # contra_book ish: skip if chosen ask<0.40 AND conf<7  (contra to book)
    # The bot's actual ask for chosen side is entry_price; if entry<0.40 → too cheap
    # but we already require entry>=0.50 so this is dead.
    # contra_drift: skip if direction conflicts with strong drift
    drift = t["btc_drift_pct"]
    direction = t["direction"]
    if direction == "UP" and drift < -0.05:
        return False
    if direction == "DOWN" and drift > 0.05:
        return False
    return True

prof_a = [t for t in trades if passes_profile_a(t)]
prof_a.sort(key=lambda t: t["ts"])
print(f"Profile A filter passes: {len(prof_a)} / {len(trades)} trades")

# ---------- per-$1 PnL ----------
def per_dollar(t):
    stake = t.get("stake_usdc") or 1.0
    return t["pnl"] / stake

pnls = [per_dollar(t) for t in prof_a]
wins = [1 if t["won"] else 0 for t in prof_a]

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (center-half, center+half)

def bootstrap_ci(x, n_iter=5000, alpha=0.05):
    if not x: return (0, 0, 0)
    means = []
    n = len(x)
    for _ in range(n_iter):
        s = [x[random.randrange(n)] for _ in range(n)]
        means.append(sum(s)/n)
    means.sort()
    lo = means[int(n_iter*alpha/2)]
    hi = means[int(n_iter*(1-alpha/2))]
    return (sum(x)/n, lo, hi)

print()
print("=" * 70)
print("PROFILE A — UNIFIED ENTRY BAND")
print("=" * 70)
print(f"  n trades:        {len(prof_a)}")
print(f"  Wins:            {sum(wins)} ({sum(wins)/len(prof_a)*100:.1f}%)")
w_lo, w_hi = wilson_ci(sum(wins), len(prof_a))
print(f"  Wilson 95% CI:   [{w_lo*100:.1f}%, {w_hi*100:.1f}%]")
m, lo, hi = bootstrap_ci(pnls)
print(f"  E[$1] / trade:   {m:+.4f}  (boot 95% CI [{lo:+.4f}, {hi:+.4f}])")
print(f"  Total $1:        {sum(pnls):+.2f}")

# days span
from datetime import datetime
ts0 = datetime.fromisoformat(prof_a[0]["ts"])
ts1 = datetime.fromisoformat(prof_a[-1]["ts"])
days = max(1.0, (ts1 - ts0).total_seconds() / 86400)
print(f"  Span:            {days:.1f} days ({ts0.date()} → {ts1.date()})")
print(f"  Trades/day:      {len(prof_a)/days:.2f}")
print(f"  Daily E[$1]:     {sum(pnls)/days:+.3f}")

# ---------- find $120/day stake at Profile A level ----------
daily_ev_per_1 = sum(pnls)/days
trades_per_day = len(prof_a)/days
print()
print(f"  → $120/day requires stake = $120 / {daily_ev_per_1:.3f}  = ${120/daily_ev_per_1:.0f} per trade")
print(f"    at {trades_per_day:.1f} trades/day")

# ---------- segment hunt: find higher-EV sub-buckets ----------
print()
print("=" * 70)
print("SUB-BUCKET HUNT — highest E[$1] per trade for FEWER trades/day")
print("=" * 70)

def stat(rows):
    if not rows:
        return None
    pn = [per_dollar(t) for t in rows]
    wn = sum(1 for t in rows if t["won"])
    return {
        "n": len(rows),
        "w_pct": wn/len(rows)*100,
        "ev": sum(pn)/len(rows),
        "total": sum(pn),
        "td": len(rows)/days,
    }

def show(name, rows):
    s = stat(rows)
    if not s or s["n"] < 5:
        print(f"  {name:<40} n={s['n'] if s else 0:>3} (too few)")
        return
    m_, lo_, hi_ = bootstrap_ci([per_dollar(t) for t in rows], n_iter=3000)
    print(f"  {name:<40} n={s['n']:>3}  W={s['w_pct']:>5.1f}%  "
          f"E[$1]={s['ev']:+.3f}  CI[{lo_:+.3f},{hi_:+.3f}]  "
          f"{s['td']:.2f}/day")

# direction split
show("ALL", prof_a)
show("UP only", [t for t in prof_a if t["direction"]=="UP"])
show("DOWN only", [t for t in prof_a if t["direction"]=="DOWN"])
print()

# entry buckets
for lo, hi in [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75)]:
    show(f"entry [{lo:.2f},{hi:.2f})", [t for t in prof_a if lo<=t["entry_price"]<hi])
print()

# conf buckets
for lo, hi in [(0,5),(5,7),(7,10),(10,15),(15,20),(20,100)]:
    show(f"conf [{lo:>2},{hi:>3})", [t for t in prof_a if lo<=t["confidence"]<hi])
print()

# drift buckets
for lo, hi, name in [(-99,-0.05,"drift<-0.05"),(-0.05,-0.01,"drift[-0.05,-0.01)"),
                     (-0.01,0.01,"drift[-0.01,0.01)"),(0.01,0.05,"drift[0.01,0.05)"),
                     (0.05,99,"drift>=0.05")]:
    show(name, [t for t in prof_a if lo<=t["btc_drift_pct"]<hi])
print()

# PTB alignment: direction matches sign of (ptb - live_price)/live_price
def ptb_aligned(t):
    dp = t.get("ptb_distance_pct", 0)
    return (t["direction"]=="UP" and dp>0) or (t["direction"]=="DOWN" and dp<0)
show("PTB aligned (direction = drift sign)", [t for t in prof_a if ptb_aligned(t)])
show("PTB contra (direction ≠ drift sign)", [t for t in prof_a if not ptb_aligned(t)])
print()

# crowd alignment: direction matches crowd_prob
def crowd_aligned(t):
    cp = t.get("crowd_prob", 0.5)
    return (t["direction"]=="UP" and cp>0.5) or (t["direction"]=="DOWN" and cp<0.5)
show("Crowd aligned", [t for t in prof_a if crowd_aligned(t)])
show("Crowd contra", [t for t in prof_a if not crowd_aligned(t)])
print()

# orderbook alignment
def ob_aligned(t):
    op = t.get("orderbook_prob", 0.5)
    return (t["direction"]=="UP" and op>0.5) or (t["direction"]=="DOWN" and op<0.5)
show("Orderbook aligned", [t for t in prof_a if ob_aligned(t)])
show("Orderbook contra", [t for t in prof_a if not ob_aligned(t)])
print()

# LSTM inverted (bot treats LSTM as anti-signal)
def lstm_inverted_aligned(t):
    lp = t.get("lstm_prob", 0.5)
    return (t["direction"]=="UP" and lp<0.5) or (t["direction"]=="DOWN" and lp>0.5)
show("LSTM inverted aligned", [t for t in prof_a if lstm_inverted_aligned(t)])
show("LSTM inverted contra", [t for t in prof_a if not lstm_inverted_aligned(t)])
print()

# all 3 secondary signals align: PTB + Crowd + LSTM-inv
def triple_aligned(t):
    return ptb_aligned(t) and crowd_aligned(t) and lstm_inverted_aligned(t)
show("Triple aligned (PTB+Crowd+LSTMinv)", [t for t in prof_a if triple_aligned(t)])
show("NOT triple aligned", [t for t in prof_a if not triple_aligned(t)])
print()

# entry × direction
for d in ["UP","DOWN"]:
    for lo, hi in [(0.50,0.60),(0.60,0.70),(0.70,0.75)]:
        show(f"{d} entry[{lo:.2f},{hi:.2f})",
             [t for t in prof_a if t["direction"]==d and lo<=t["entry_price"]<hi])
print()

# ---------- SELECTIVE STRATEGY HUNT ----------
# Find strategies with high W% and decent EV that filter to <=3 trades/day
print()
print("=" * 70)
print("SELECTIVE FILTERS — targeting <=3 trades/day with high W% and EV")
print("=" * 70)

def show_strat(name, rows):
    if not rows:
        print(f"  {name:<50}: n=0")
        return None
    pn = [per_dollar(t) for t in rows]
    wn = sum(1 for t in rows if t["won"])
    n = len(rows)
    td = n/days
    ev = sum(pn)/n
    total = sum(pn)
    daily = total/days
    m_, lo_, hi_ = bootstrap_ci(pn, n_iter=3000)
    w_lo, w_hi = wilson_ci(wn, n)
    stake_for_120 = 120/daily if daily > 0 else float('inf')
    print(f"  {name:<50}")
    print(f"    n={n:>3} W={wn/n*100:>5.1f}% (CI [{w_lo*100:.0f},{w_hi*100:.0f}]%)  "
          f"E[$1]={ev:+.3f} CI[{lo_:+.3f},{hi_:+.3f}]")
    print(f"    {td:.2f} trades/day · daily E[$1]={daily:+.3f} · "
          f"stake for $120/day = ${stake_for_120:.0f}")
    return {"n": n, "w": wn/n, "ev": ev, "td": td, "daily": daily, "stake_120": stake_for_120}

# S1: entry [0.55, 0.70) + conf >= 7 + PTB aligned
s1 = [t for t in prof_a if 0.55<=t["entry_price"]<0.70 and t["confidence"]>=7 and ptb_aligned(t)]
show_strat("S1: entry[.55,.70) ∩ conf>=7 ∩ PTB-aligned", s1)

# S2: entry [0.55, 0.70) + conf >= 12 (high conf)
s2 = [t for t in prof_a if 0.55<=t["entry_price"]<0.70 and t["confidence"]>=12]
show_strat("S2: entry[.55,.70) ∩ conf>=12", s2)

# S3: triple-aligned + conf>=7
s3 = [t for t in prof_a if triple_aligned(t) and t["confidence"]>=7]
show_strat("S3: triple-aligned ∩ conf>=7", s3)

# S4: high conf only
s4 = [t for t in prof_a if t["confidence"]>=15]
show_strat("S4: conf>=15", s4)

# S5: triple-aligned alone
s5 = [t for t in prof_a if triple_aligned(t)]
show_strat("S5: triple-aligned (any conf)", s5)

# S6: PTB aligned + drift >=0.02 in same direction
def strong_drift_aligned(t):
    return (t["direction"]=="UP" and t["btc_drift_pct"]>=0.02) or (t["direction"]=="DOWN" and t["btc_drift_pct"]<=-0.02)
s6 = [t for t in prof_a if 0.55<=t["entry_price"]<0.70 and strong_drift_aligned(t)]
show_strat("S6: entry[.55,.70) ∩ strong-drift-aligned(|d|>=0.02)", s6)

# S7: entry [0.60, 0.70) + conf >= 7
s7 = [t for t in prof_a if 0.60<=t["entry_price"]<0.70 and t["confidence"]>=7]
show_strat("S7: entry[.60,.70) ∩ conf>=7", s7)

# S8: entry [0.55, 0.65) (the "fat middle")
s8 = [t for t in prof_a if 0.55<=t["entry_price"]<0.65]
show_strat("S8: entry[.55,.65) (fat middle)", s8)

# ---------- walk-forward CV on Profile A ----------
print()
print("=" * 70)
print("WALK-FORWARD CV (Profile A)")
print("=" * 70)
n = len(prof_a)
folds = 4
chunk = n // folds
for i in range(folds):
    start = i * chunk
    end = (i+1)*chunk if i < folds-1 else n
    fold = prof_a[start:end]
    fp = [per_dollar(t) for t in fold]
    if not fold:
        continue
    wn = sum(1 for t in fold if t["won"])
    print(f"  Fold {i+1} ({fold[0]['ts'][:10]} → {fold[-1]['ts'][:10]}): "
          f"n={len(fold)} W={wn/len(fold)*100:.0f}% E[$1]={sum(fp)/len(fp):+.3f}")

# ---------- Monte Carlo: $120/day at various stakes ----------
print()
print("=" * 70)
print("MONTE CARLO — daily PnL distribution at $120 target stakes")
print("=" * 70)

def mc_one_day(per_1_pnls, trades_per_day, stake, n_iter=10000):
    """Sample one day's PnL by drawing trades_per_day samples from observed per-$1 outcomes."""
    pool = per_1_pnls
    td_int = int(round(trades_per_day))
    daily = []
    for _ in range(n_iter):
        s = sum(pool[random.randrange(len(pool))] for _ in range(td_int))
        daily.append(s * stake)
    daily.sort()
    return daily

print(f"\n  PROFILE A ({trades_per_day:.1f} trades/day, E[$1]={daily_ev_per_1/trades_per_day:+.3f}/trade)")
for stake in [50, 100, 150, 200]:
    d = mc_one_day(pnls, trades_per_day, stake)
    p10 = d[1000]
    p50 = d[5000]
    p90 = d[9000]
    prob_pos = sum(1 for x in d if x>0)/len(d)*100
    prob_120 = sum(1 for x in d if x>=120)/len(d)*100
    print(f"    stake ${stake:>3}: P10={p10:+7.0f}  P50={p50:+7.0f}  P90={p90:+7.0f}  "
          f"P(>0)={prob_pos:.0f}%  P(>=120)={prob_120:.0f}%")

# Sub-strategy MC
for name, sset in [("S7 entry[.60,.70) ∩ conf>=7", s7)]:
    if not sset: continue
    sp = [per_dollar(t) for t in sset]
    td = len(sset)/days
    ev_per_trade = sum(sp)/len(sset)
    daily_ev = ev_per_trade * td
    print(f"\n  {name} ({td:.1f}/day, E[$1]={ev_per_trade:+.3f}/trade)")
    for stake in [50, 100, 150, 200, 300]:
        d = mc_one_day(sp, td, stake)
        p10 = d[1000]; p50 = d[5000]; p90 = d[9000]
        prob_120 = sum(1 for x in d if x>=120)/len(d)*100
        print(f"    stake ${stake:>3}: P10={p10:+7.0f}  P50={p50:+7.0f}  P90={p90:+7.0f}  "
              f"P(>=120)={prob_120:.0f}%")

# Kelly + bankroll required
print()
print("=" * 70)
print("KELLY + REQUIRED BANKROLL")
print("=" * 70)
# binary bet: p win, payoff b (odds-to-1)
# avg payoff per dollar: roughly (1/entry - 1)
avg_entry = mean(t["entry_price"] for t in prof_a)
avg_b = (1/avg_entry) - 1
p_win = sum(wins)/len(wins)
kelly = (p_win*avg_b - (1-p_win))/avg_b
print(f"  p(win)={p_win:.3f}  avg entry={avg_entry:.3f}  avg payoff b={avg_b:.3f}")
print(f"  Full Kelly fraction = {kelly*100:.1f}% of bankroll")
print(f"  Half Kelly = {kelly*50:.1f}% of bankroll")
for stake_target in [100, 150, 200]:
    full_bk = stake_target/kelly
    half_bk = stake_target/(kelly/2)
    print(f"  Stake ${stake_target}: full-Kelly bankroll ${full_bk:.0f}, "
          f"half-Kelly ${half_bk:.0f}")
PY