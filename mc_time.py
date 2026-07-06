"""Time-aware Monte Carlo: same $100 → +$40 target, but per-hour WR
and tradable-window strategies. Each path advances 10 wall-clock min/step
(matches the bot's 5-min loop with skip-windows interleaved). Win rates per
hour bucket come from full history (large sample); win-payoffs come from
the post-filter $1-stake era (representative of current bot economics).
"""
import json
import numpy as np
from datetime import datetime

NUM_PATHS = 20000
MAX_STEPS = 600        # 100 wall-clock hours per path (>4 days)
MIN_PER_STEP = 10
TARGET = 40.0
BANKROLL = 100.0

trades = [json.loads(l) for l in open("predictor/trade_history.jsonl")]
for t in trades:
    t["dt"] = datetime.fromisoformat(t["ts"])
trades = sorted([t for t in trades if abs(t.get("pnl") or 0) >= 0.01], key=lambda t: t["dt"])

# Per-hour-bucket WR from full history (larger sample = stable estimate)
BUCKETS = [
    ("night",     0,  6),
    ("morning",   6, 12),
    ("afternoon",12, 18),
    ("evening",  18, 24),
]

def bucket_of(h):
    for name, lo, hi in BUCKETS:
        if lo <= h < hi:
            return name
    return "evening"

bucket_stats = {}
for name, lo, hi in BUCKETS:
    lst = [t for t in trades if lo <= t["dt"].hour < hi]
    w = sum(1 for t in lst if t["won"])
    bucket_stats[name] = {"wr": w/len(lst) if lst else 0.5, "n": len(lst), "lo": lo, "hi": hi}

# Per-hour WR array indexed 0..23
hour_wr = np.zeros(24)
for name, bs in bucket_stats.items():
    for h in range(bs["lo"], bs["hi"]):
        hour_wr[h] = bs["wr"]

# Win payoffs from post-filter $1-stake era (representative of current bot)
post = [t for t in trades if "entry_price" in t and t.get("stake_usdc") == 1.0]
pf_wins = [t for t in post if t["won"]]
pf_payoffs = np.array([t["pnl"] / float(t["stake_usdc"]) for t in pf_wins]) if pf_wins else np.array([0.5])

# Strategies: which hours of day the bot is allowed to trade
STRATEGIES = {
    "24/7 (all hours)":           set(range(24)),
    "skip afternoon (12-18)":     set(range(24)) - set(range(12, 18)),
    "evening+night (18-06)":      set(range(18, 24)) | set(range(0, 6)),
    "evening only (18-24)":       set(range(18, 24)),
    "afternoon only (12-18)":     set(range(12, 18)),
}

def simulate(allowed_hours, stake, start_hour=0, paths=NUM_PATHS, max_steps=MAX_STEPS, seed=42):
    rng = np.random.default_rng(seed)
    bk = np.full(paths, BANKROLL, dtype=float)
    peak = np.full(paths, BANKROLL, dtype=float)
    max_dd = np.zeros(paths)
    success = np.zeros(paths, dtype=bool)
    bust = np.zeros(paths, dtype=bool)
    trades_fired = np.zeros(paths, dtype=int)
    steps_used = np.full(paths, max_steps, dtype=int)
    minute = np.full(paths, start_hour * 60, dtype=int)

    is_tradable = np.zeros(24, dtype=bool)
    for h in allowed_hours:
        is_tradable[h] = True

    for n in range(max_steps):
        active = ~(success | bust)
        if not active.any():
            break
        cur_hour = (minute // 60) % 24
        tradable = is_tradable[cur_hour] & active

        if tradable.any():
            new_bust = tradable & (bk < stake)
            bust |= new_bust
            steps_used[new_bust] = n
            tradable &= ~new_bust

            if tradable.any():
                wr = hour_wr[cur_hour]
                u = rng.random(paths)
                won = (u < wr)
                p_idx = rng.integers(0, len(pf_payoffs), size=paths)
                delta = np.where(won, stake * pf_payoffs[p_idx], -stake)
                bk[tradable] += delta[tradable]
                trades_fired[tradable] += 1
                peak[tradable] = np.maximum(peak[tradable], bk[tradable])
                max_dd = np.maximum(max_dd, peak - bk)
                new_succ = tradable & ((bk - BANKROLL) >= TARGET)
                success |= new_succ
                steps_used[new_succ] = n + 1

        minute += MIN_PER_STEP

    hours_elapsed = steps_used * MIN_PER_STEP / 60.0
    return {
        "p_success": float(success.mean()),
        "p_bust": float(bust.mean()),
        "med_trades": int(np.median(trades_fired[success])) if success.any() else None,
        "med_hours": float(np.median(hours_elapsed[success])) if success.any() else None,
        "p75_hours": float(np.percentile(hours_elapsed[success], 75)) if success.any() else None,
        "med_dd": float(np.median(max_dd)),
        "p95_dd": float(np.percentile(max_dd, 95)),
        "expected_final": float(bk.mean()),
    }

print("="*108)
print("TIME-AWARE MONTE CARLO — $100 → +$40, 5-min bot, per-hour WR")
print("="*108)
print(f"per-hour bucket stats (from {len(trades)} full-history trades):")
for name, bs in bucket_stats.items():
    print(f"  {name:>10} ({bs['lo']:02d}-{bs['hi']:02d}): WR={bs['wr']*100:5.1f}%  n={bs['n']}")
print(f"win-payoff dist (post-filter $1 era, n={len(pf_wins)}): mean={pf_payoffs.mean():.3f}x  median={np.median(pf_payoffs):.3f}x")
print(f"paths/cell={NUM_PATHS}, max_steps={MAX_STEPS} ({MAX_STEPS*MIN_PER_STEP/60:.0f} wall-clock hours)")

for sname, allowed in STRATEGIES.items():
    print(f"\n--- STRATEGY: {sname}  (tradable hours={sorted(allowed)}) ---")
    print(f"  {'stake':>5} {'P(succ)':>8} {'P(bust)':>8} {'med_trd':>8} {'med_hrs':>8} {'p75_hrs':>8} {'medDD':>7} {'p95DD':>7} {'E[final]':>9}")
    for stake in [1, 2, 3, 5, 7, 10, 15, 20]:
        r = simulate(allowed, stake)
        med_t = r['med_trades'] if r['med_trades'] is not None else "—"
        med_h = f"{r['med_hours']:.1f}" if r['med_hours'] is not None else "—"
        p75_h = f"{r['p75_hours']:.1f}" if r['p75_hours'] is not None else "—"
        print(f"  ${stake:>3} {r['p_success']*100:>7.1f}% {r['p_bust']*100:>7.1f}% {str(med_t):>8} {med_h:>8} {p75_h:>8} ${r['med_dd']:>5.1f} ${r['p95_dd']:>5.1f} ${r['expected_final']:>7.2f}")

# Side-by-side at a couple of recommended stakes
print("\n" + "="*108)
print("HEAD-TO-HEAD AT $5 STAKE — strategy comparison")
print("="*108)
print(f"  {'strategy':<32} {'P(succ)':>8} {'P(bust)':>8} {'med_trd':>8} {'med_hrs':>8} {'p95DD':>7}")
for sname, allowed in STRATEGIES.items():
    r = simulate(allowed, 5)
    med_t = r['med_trades'] if r['med_trades'] is not None else "—"
    med_h = f"{r['med_hours']:.1f}" if r['med_hours'] is not None else "—"
    print(f"  {sname:<32} {r['p_success']*100:>7.1f}% {r['p_bust']*100:>7.1f}% {str(med_t):>8} {med_h:>8} ${r['p95_dd']:>5.1f}")

print(f"\n  same at $10 stake:")
print(f"  {'strategy':<32} {'P(succ)':>8} {'P(bust)':>8} {'med_trd':>8} {'med_hrs':>8} {'p95DD':>7}")
for sname, allowed in STRATEGIES.items():
    r = simulate(allowed, 10)
    med_t = r['med_trades'] if r['med_trades'] is not None else "—"
    med_h = f"{r['med_hours']:.1f}" if r['med_hours'] is not None else "—"
    print(f"  {sname:<32} {r['p_success']*100:>7.1f}% {r['p_bust']*100:>7.1f}% {str(med_t):>8} {med_h:>8} ${r['p95_dd']:>5.1f}")
