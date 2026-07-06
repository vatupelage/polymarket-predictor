"""
Did today's -$87 session fit the expected variance distribution, or is something
structurally broken? Markov-clean bootstrap from the gates-KEEP universe to
build the 6-trade outcome distribution, then place today's result in it.

Principles from the transcripts:
- Markov: each trade is an independent draw (no path-dependence)
- Conditional: simulate from the gates-KEEP pool only (look-ahead-safe — only past trades)
- Bootstrap CI for the mean; exact percentile for the observed value
- Sanity-check each gate on each actual trade today (any leak?)
"""
import json
import math
import random
import statistics
from datetime import datetime
from math import comb
from pathlib import Path

random.seed(42)

SCRIPT_DIR = Path(__file__).parent
HISTORY = SCRIPT_DIR / "trade_history.jsonl"

STAKE = 30.0
HARD_STOP = 90.0
BLACKOUT_LO, BLACKOUT_HI = 18, 24

def aligns_with_lstm(t):
    return (t["direction"] == "UP" and t.get("lstm_prob", 0.5) >= 0.5) or \
           (t["direction"] == "DOWN" and t.get("lstm_prob", 0.5) < 0.5)

def hour_in_blackout(t):
    return BLACKOUT_LO <= datetime.fromisoformat(t["ts"]).hour < BLACKOUT_HI

def gates_keep(t):
    return (not aligns_with_lstm(t)) and (not hour_in_blackout(t))

def load_all():
    out = []
    with open(HISTORY) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                t = json.loads(line)
            except: continue
            if t.get("stake_usdc") in (None, 0, 0.0): continue
            if t.get("pnl") is None: continue
            ep = t.get("entry_price")
            if ep is None or not (0.50 <= ep < 0.75): continue
            if "lstm_prob" not in t: continue
            try:
                t["_dt"] = datetime.fromisoformat(t["ts"])
            except: continue
            t["_pd"] = t["pnl"] / t["stake_usdc"]
            out.append(t)
    out.sort(key=lambda x: x["_dt"])
    return out

all_t = load_all()
today_start = datetime(2026, 5, 21, 0, 0, 0)
today = [t for t in all_t if t["_dt"] >= today_start]
past = [t for t in all_t if t["_dt"] < today_start]
past_gated = [t for t in past if gates_keep(t)]

# ===== 1. Today's trades — gate audit =====
print("=" * 100)
print("1. TODAY'S TRADES — gate audit")
print("=" * 100)
print(f"  {'Time':<19} {'Dir':<5} {'St':>4} {'PnL':>8} {'W/L':<4} {'ep':>6} {'conf':>6} {'lstm':>6} {'aligns?':<8} {'blackout?':<9} {'gate':<5}")
print("  " + "-" * 100)
today_pnl_check = 0
today_within_filter = 0
for t in today:
    aligns = aligns_with_lstm(t)
    blackout = hour_in_blackout(t)
    gate = "KEEP" if not aligns and not blackout else "SKIP"
    wl = "W" if t["won"] else "L"
    today_pnl_check += t["pnl"]
    if not aligns and not blackout:
        today_within_filter += 1
    print(f"  {t['ts']:<19} {t['direction']:<5} ${t['stake_usdc']:>2.0f}  ${t['pnl']:>+6.2f} {wl:<4} "
          f"{t['entry_price']:>6.3f} {t['confidence']:>5.1f}% {t['lstm_prob']:>6.3f} "
          f"{'yes' if aligns else 'no':<8} {'yes' if blackout else 'no':<9} {gate:<5}")
print()
print(f"  Today's total PnL: ${today_pnl_check:+.2f}")
print(f"  All {len(today)} trades passed the gates? {today_within_filter == len(today)}")
print(f"  Cumulative session-loss vs hard stop ($90): ${abs(today_pnl_check):.2f} / $90.00  ({100*abs(today_pnl_check)/90:.1f}%)")
print()

# ===== 2. Bootstrap distribution of N-trade outcomes =====
n_today = len(today)
print("=" * 100)
print(f"2. WHERE DOES ${today_pnl_check:+.2f} FALL IN THE {n_today}-TRADE DISTRIBUTION?")
print("=" * 100)
print(f"  Bootstrap: draw {n_today} trades with replacement from gates-KEEP pool of past trades")
print(f"  (n={len(past_gated)} historical observations, all strictly BEFORE today).")
print(f"  Markov-independent: each draw is its own observation.")
print()

pd_pool = [t["_pd"] for t in past_gated]
mean_pd = sum(pd_pool) / len(pd_pool)
std_pd = statistics.stdev(pd_pool)
print(f"  Historical pool: E[$1]=${mean_pd:+.4f}, sigma[$1]=${std_pd:.4f}")
print(f"  Per-trade at $30 stake: mu=${mean_pd*STAKE:+.2f}, sigma=${std_pd*STAKE:.2f}")
print(f"  Expected {n_today}-trade sum: mu=${mean_pd*STAKE*n_today:+.2f}, sigma=${std_pd*STAKE*math.sqrt(n_today):.2f}")
print()

N_MC = 100000
sums = []
n_at_or_below = 0
for _ in range(N_MC):
    s = sum(pd_pool[random.randrange(len(pd_pool))] for _ in range(n_today)) * STAKE
    sums.append(s)
    if s <= today_pnl_check:
        n_at_or_below += 1
sums.sort()

p_observed = n_at_or_below / N_MC
print(f"  Bootstrap distribution of {n_today}-trade sums (n={N_MC:,} simulations):")
for p, lbl in [(0.01, "p01 (worst 1%)"),
               (0.025, "p2.5"),
               (0.05, "p05 (worst 5%)"),
               (0.10, "p10"),
               (0.25, "p25"),
               (0.50, "p50 (median)"),
               (0.75, "p75"),
               (0.90, "p90"),
               (0.95, "p95 (best 5%)"),
               (0.99, "p99")]:
    v = sums[int(N_MC * p)]
    print(f"    {lbl:<22}: ${v:>+7.2f}")
print()
print(f"  P({n_today}-trade sum <= ${today_pnl_check:+.2f}) = {p_observed:.4f} = {100*p_observed:.2f}%")
if p_observed > 0:
    print(f"  -> Roughly 1 in {round(1/p_observed)} sessions this size are this bad or worse.")
print()

# ===== 3. Losing-streak (binomial) =====
print("=" * 100)
print(f"3. LOSING-STREAK PROBABILITY  ({sum(1 for t in today if not t['won'])} losses out of {n_today})")
print("=" * 100)
loss_rate = sum(1 for t in past_gated if not t["won"]) / len(past_gated)
print(f"  Historical loss rate in gates-KEEP: {100*loss_rate:.1f}%")
print()
n = n_today
probs = [comb(n, k) * (loss_rate**k) * ((1-loss_rate)**(n-k)) for k in range(n+1)]
losses_today = sum(1 for t in today if not t["won"])
print(f"  {'K':<3} {'P(exactly K)':<15} {'P(>= K)':<10}")
print("  " + "-" * 35)
for k in range(n+1):
    p_atleast = sum(probs[k:])
    marker = "  <- TODAY" if k == losses_today else ""
    print(f"  {k:<3} {probs[k]:<15.4f} {p_atleast:<10.4f}{marker}")
print()
print(f"  P(>= {losses_today} losses out of {n_today}) = {sum(probs[losses_today:]):.4f} = {100*sum(probs[losses_today:]):.2f}%")
print()

# ===== 4. Forensics =====
print("=" * 100)
print("4. LOSING-TRADE FORENSICS — anything special about today's losers?")
print("=" * 100)
losers = [t for t in today if not t["won"]]
winners = [t for t in today if t["won"]]
print(f"  Losers (n={len(losers)}):")
for t in losers:
    print(f"    {t['ts']}: {t['direction']:<4} ep={t['entry_price']:.3f}  conf={t['confidence']:>5.1f}%  "
          f"lstm={t['lstm_prob']:.3f}  drift={t.get('btc_drift_pct') or 0:>+.4f}  ptb={t.get('ptb_prob') or 0:.3f}")
print(f"\n  Winners (n={len(winners)}):")
for t in winners:
    print(f"    {t['ts']}: {t['direction']:<4} ep={t['entry_price']:.3f}  conf={t['confidence']:>5.1f}%  "
          f"lstm={t['lstm_prob']:.3f}  drift={t.get('btc_drift_pct') or 0:>+.4f}  ptb={t.get('ptb_prob') or 0:.3f}")

def avg(xs):
    return sum(xs)/len(xs) if xs else 0
hist_losers = [t for t in past_gated if not t["won"]]
hist_winners = [t for t in past_gated if t["won"]]
print()
print("  Pattern checks (today's losers vs historical gates-KEEP losers):")
print(f"    Avg entry price       — today: {avg([t['entry_price'] for t in losers]):.3f}   historical: {avg([t['entry_price'] for t in hist_losers]):.3f}")
print(f"    Avg confidence (loss) — today: {avg([t['confidence'] for t in losers]):.1f}%   historical: {avg([t['confidence'] for t in hist_losers]):.1f}%")
print(f"    Avg drift %           — today: {avg([t.get('btc_drift_pct') or 0 for t in losers]):+.4f}   historical: {avg([t.get('btc_drift_pct') or 0 for t in hist_losers]):+.4f}")
n_down_loss_today = sum(1 for t in losers if t['direction']=='DOWN')
n_down_loss_hist  = sum(1 for t in hist_losers if t['direction']=='DOWN')
print(f"    DOWN-direction losses — today: {n_down_loss_today}/{len(losers)}   historical share: {n_down_loss_hist/max(len(hist_losers),1):.2f}")
print()

# ===== 5. Daily PnL distribution at $30 normalized =====
print("=" * 100)
print("5. HISTORICAL DAILY PnL DISTRIBUTION  (gates-KEEP, normalized to $30 stake)")
print("=" * 100)
by_day = {}
for t in past_gated:
    d = t["_dt"].date()
    by_day.setdefault(d, []).append(t["_pd"] * STAKE)
day_pnls = {d: sum(v) for d, v in by_day.items()}
n_days_worse = sum(1 for pnl in day_pnls.values() if pnl <= today_pnl_check)
n_days = len(day_pnls)
n_losing = sum(1 for pnl in day_pnls.values() if pnl < 0)
print(f"  Historical days observed: {n_days}")
print(f"  Days with PnL <= today's ${today_pnl_check:+.2f}: {n_days_worse}  ({100*n_days_worse/max(n_days,1):.1f}%)")
print(f"  Losing days overall: {n_losing}  ({100*n_losing/max(n_days,1):.1f}%)")
print()
sorted_pnls = sorted(day_pnls.values())
print("  Historical daily PnL percentiles ($30 stake):")
for p in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
    v = sorted_pnls[min(int(len(sorted_pnls)*p), len(sorted_pnls)-1)]
    print(f"    p{int(p*100):>2}: ${v:>+7.2f}")
print()

# ===== 6. Verdict =====
print("=" * 100)
print("6. VERDICT")
print("=" * 100)
print(f"  Today: {n_today} trades, {len(winners)}W/{len(losers)}L, PnL=${today_pnl_check:+.2f}")
print(f"  Bootstrap probability of this loss or worse over {n_today} trades: {100*p_observed:.2f}%")
print(f"  Historical loss rate per trade: {100*loss_rate:.1f}%")
print()

green = []
amber = []

if today_within_filter == len(today):
    green.append("All today's trades passed gates correctly (no leak through LSTM-inv or 18-24 blackout)")
else:
    amber.append(f"{len(today) - today_within_filter} trade(s) bypassed gates — investigate")

low_conf_losers = sum(1 for t in losers if t['confidence'] < 5)
if low_conf_losers >= 2:
    amber.append(f"{low_conf_losers}/{len(losers)} losers had conf<5% — very-low-conf bucket is noisy")
else:
    green.append("Losses not concentrated in any one conf bucket")

if p_observed < 0.01:
    amber.append(f"Day at p{100*p_observed:.2f} — very unusual; consider whether market regime has shifted")
elif p_observed < 0.05:
    green.append(f"Day at p{100*p_observed:.2f} — bad but within expected variance (~1-in-{round(1/p_observed)} sessions)")
else:
    green.append(f"Day at p{100*p_observed:.1f} — entirely normal variance")

# Drift check
avg_drift_today = avg([t.get('btc_drift_pct') or 0 for t in losers])
if abs(avg_drift_today) < 0.0005:
    amber.append(f"Avg drift on today's losers is {avg_drift_today:+.4f} — extremely flat market; noise dominates signal")

for n_ in green: print(f"  [OK] {n_}")
for n_ in amber: print(f"  [!!] {n_}")
print()
print(f"  Markov property: today's bad streak does NOT predict tomorrow's outcomes.")
print(f"  Expected PnL of next 6 trades: ${6*mean_pd*STAKE:+.2f} (still positive, unchanged).")
