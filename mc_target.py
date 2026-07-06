"""Monte Carlo: $100 → +$40 with 5-min bot, vectorized.

Empirical payoffs come from the recovered (won, payoff_ratio) pairs
in trade_history.jsonl. We sweep stake sizes per regime, simulating
20k paths each, tracking P(success), P(bust), trades-to-target,
max drawdown.
"""
import json
import numpy as np
from datetime import datetime

NUM_PATHS = 20000
MAX_TRADES = 200
TARGET = 40.0
BANKROLL = 100.0

trades = [json.loads(l) for l in open("predictor/trade_history.jsonl")]
for t in trades:
    t["dt"] = datetime.fromisoformat(t["ts"])
trades.sort(key=lambda t: t["dt"])
trades = [t for t in trades if abs(t["pnl"]) >= 0.01]

# Legacy payoffs (old trades — use nearest-loss stake imputation)
wins = [t for t in trades if t["won"]]
losses = [t for t in trades if not t["won"]]
win_payoffs = []
for t in wins:
    if t.get("stake_usdc"):
        s = float(t["stake_usdc"])
        win_payoffs.append(t["pnl"] / s)
        continue
    nearest = min(losses, key=lambda l: abs((l["dt"]-t["dt"]).total_seconds()), default=None)
    if nearest and abs((nearest["dt"]-t["dt"]).total_seconds()) < 6*3600:
        s = abs(nearest["pnl"])
        if s > 0:
            win_payoffs.append(t["pnl"]/s)
win_payoffs = np.array(win_payoffs)

# Post-filter payoffs (rich-schema trades only, $1 stake era)
post_filter = [t for t in trades if "entry_price" in t and t.get("stake_usdc") == 1.0]
pf_wins = [t for t in post_filter if t["won"]]
pf_losses = [t for t in post_filter if not t["won"]]
pf_payoffs = np.array([t["pnl"] / float(t["stake_usdc"]) for t in pf_wins]) if pf_wins else win_payoffs
pf_wr = len(pf_wins) / len(post_filter) if post_filter else 0
print(f"post-filter: n={len(post_filter)} W={len(pf_wins)} L={len(pf_losses)} "
      f"WR={pf_wr*100:.1f}% mean_payoff={pf_payoffs.mean():.3f}x")

def kelly(wr, payoffs):
    b = float(np.mean(payoffs))
    return (wr*b - (1-wr)) / b

def simulate_vec(wr, payoffs, stake, paths=NUM_PATHS, max_trades=MAX_TRADES, seed=42):
    rng = np.random.default_rng(seed)
    # pre-generate all randomness
    u = rng.random((paths, max_trades))
    w_idx = rng.integers(0, len(payoffs), size=(paths, max_trades))
    won = u < wr
    # delta per trade per path
    delta = np.where(won, stake * payoffs[w_idx], -stake)
    # walk forward but stop on bust or success
    bk = np.full(paths, BANKROLL, dtype=float)
    peak = np.full(paths, BANKROLL, dtype=float)
    max_dd = np.zeros(paths)
    success = np.zeros(paths, dtype=bool)
    bust = np.zeros(paths, dtype=bool)
    trades_used = np.full(paths, max_trades, dtype=int)

    for n in range(max_trades):
        active = ~(success | bust)
        if not active.any():
            break
        # bust check: bankroll < stake
        new_bust = active & (bk < stake)
        bust |= new_bust
        trades_used[new_bust] = n
        active &= ~new_bust
        if not active.any():
            break
        bk[active] += delta[active, n]
        peak[active] = np.maximum(peak[active], bk[active])
        dd = peak - bk
        max_dd = np.maximum(max_dd, dd)
        new_success = active & ((bk - BANKROLL) >= TARGET)
        success |= new_success
        trades_used[new_success] = n + 1

    return {
        "p_success": float(success.mean()),
        "p_bust": float(bust.mean()),
        "median_n": int(np.median(trades_used[success])) if success.any() else None,
        "p25_n": int(np.percentile(trades_used[success], 25)) if success.any() else None,
        "p75_n": int(np.percentile(trades_used[success], 75)) if success.any() else None,
        "p90_n": int(np.percentile(trades_used[success], 90)) if success.any() else None,
        "median_dd": float(np.median(max_dd)),
        "p95_dd": float(np.percentile(max_dd, 95)),
        "expected_final": float(bk.mean()),
    }

REGIMES = {
    "post_filter (NEW, n=21, $1 stake)": {"wr": pf_wr, "payoffs": pf_payoffs},
    "post_filter_conservative (60% WR)": {"wr": 0.60, "payoffs": pf_payoffs},
    "post_filter_stress (55% WR)":       {"wr": 0.55, "payoffs": pf_payoffs},
    "all (212 trades, full history)":    {"wr": 0.643, "payoffs": win_payoffs},
    "stress_50pct (worst plausible)":    {"wr": 0.50, "payoffs": pf_payoffs},
}

print("="*108)
print("MONTE CARLO — $100 bankroll → +$40 target, 5-min Polymarket bot")
print("="*108)
print(f"empirical win-payoff: mean={win_payoffs.mean():.3f}x  median={np.median(win_payoffs):.3f}x  n={len(win_payoffs)}")
print(f"loss payoff: -1.000x always")
print(f"paths/cell={NUM_PATHS}, max_trades={MAX_TRADES}")

for regime_name, regime in REGIMES.items():
    wr, payoffs = regime["wr"], regime["payoffs"]
    avg_b = float(np.mean(payoffs))
    ev = wr*avg_b - (1-wr)
    f_kelly = kelly(wr, payoffs)
    print(f"\n--- {regime_name}  WR={wr*100:.1f}%  EV/trade={ev*100:+.2f}%  Kelly={f_kelly*100:.1f}% (~${BANKROLL*f_kelly:.2f}) ---")
    print(f"  {'stake':>6} {'P(succ)':>8} {'P(bust)':>8} {'med_n':>6} {'p25':>5} {'p75':>5} {'p90':>5} {'medDD':>7} {'p95DD':>7} {'E[final]':>10}")
    for stake in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]:
        r = simulate_vec(wr, payoffs, stake)
        med = r['median_n'] if r['median_n'] is not None else "—"
        p25 = r['p25_n'] if r['p25_n'] is not None else "—"
        p75 = r['p75_n'] if r['p75_n'] is not None else "—"
        p90 = r['p90_n'] if r['p90_n'] is not None else "—"
        print(f"  ${stake:>4} {r['p_success']*100:>7.1f}% {r['p_bust']*100:>7.1f}% {str(med):>6} {str(p25):>5} {str(p75):>5} {str(p90):>5} ${r['median_dd']:>5.1f} ${r['p95_dd']:>5.1f} ${r['expected_final']:>8.2f}")

# fractional/dynamic stake sizing — half-Kelly with floor and ceiling
print("\n" + "="*108)
print("DYNAMIC SIZING — half-Kelly of CURRENT bankroll, floor $5, cap $25")
print("="*108)

def simulate_kelly(wr, payoffs, kelly_frac, paths=NUM_PATHS, max_trades=MAX_TRADES, floor=5, cap=25, seed=42):
    rng = np.random.default_rng(seed)
    f_kelly = kelly(wr, payoffs)
    f_used = max(0.0, f_kelly * kelly_frac)

    success = np.zeros(paths, dtype=bool)
    bust = np.zeros(paths, dtype=bool)
    trades_used = np.full(paths, max_trades, dtype=int)
    max_dd = np.zeros(paths)
    bk = np.full(paths, BANKROLL, dtype=float)
    peak = np.full(paths, BANKROLL, dtype=float)

    u = rng.random((paths, max_trades))
    widx = rng.integers(0, len(payoffs), size=(paths, max_trades))
    won = u < wr

    for n in range(max_trades):
        active = ~(success | bust)
        if not active.any(): break
        stake = np.clip(bk * f_used, floor, cap)
        new_bust = active & (bk < floor)
        bust |= new_bust
        trades_used[new_bust] = n
        active &= ~new_bust
        if not active.any(): break
        delta = np.where(won[:, n], stake * payoffs[widx[:, n]], -stake)
        bk[active] += delta[active]
        peak[active] = np.maximum(peak[active], bk[active])
        max_dd = np.maximum(max_dd, peak - bk)
        new_succ = active & ((bk - BANKROLL) >= TARGET)
        success |= new_succ
        trades_used[new_succ] = n + 1

    return {
        "p_success": float(success.mean()),
        "p_bust": float(bust.mean()),
        "median_n": int(np.median(trades_used[success])) if success.any() else None,
        "median_dd": float(np.median(max_dd)),
        "p95_dd": float(np.percentile(max_dd, 95)),
        "f_kelly": f_kelly,
    }

for regime_name, regime in REGIMES.items():
    wr, payoffs = regime["wr"], regime["payoffs"]
    print(f"\n  {regime_name}")
    for frac in [0.25, 0.5, 1.0]:
        r = simulate_kelly(wr, payoffs, frac)
        med = r['median_n'] if r['median_n'] is not None else "—"
        print(f"    {frac:.2f}x Kelly  P(succ)={r['p_success']*100:5.1f}%  P(bust)={r['p_bust']*100:5.1f}%  med_n={med}  medDD=${r['median_dd']:.1f}  p95DD=${r['p95_dd']:.1f}")
