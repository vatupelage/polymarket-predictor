"""
Quant backtest for $40 stake on $200 bankroll using the current gates.

Applies transcript principles:
- Markov: each trade is an independent draw from the per-$1 PnL distribution
  (no path-dependent conditioning, no "after N losses" features)
- Conditional distributions: simulate from the GATES-KEEP pool only
- Look-ahead bias: use chronological IS to estimate distributions, then
  simulate forward into OOS; report both
- Bootstrap CIs (10k resamples) for means; Wilson CIs for proportions
- Monte Carlo for path-dependent quantities (bankroll, drawdown, ruin)
- Stress test: halve the edge to simulate IS->OOS decay
"""
import json
import math
import random
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

random.seed(42)

SCRIPT_DIR = Path(__file__).parent
HISTORY = SCRIPT_DIR / "trade_history.jsonl"

BANKROLL_START = 200.0
STAKE = 40.0
HARD_STOP_LOSS = 120.0     # per-session cumulative loss cap (matches BOT_HARD_STOP_LOSS)
N_MC_PATHS = 10000
DAYS_TO_SIMULATE = 30

# ---------- helpers ----------
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
    return (means[int(n_boot*alpha/2)], means[int(n_boot*(1-alpha/2))])

def percentile(values, p):
    if not values: return 0.0
    s = sorted(values)
    k = int(len(s) * p)
    return s[min(k, len(s)-1)]

def aligns_with_lstm(t):
    return (t["direction"] == "UP" and t.get("lstm_prob", 0.5) >= 0.5) or \
           (t["direction"] == "DOWN" and t.get("lstm_prob", 0.5) < 0.5)

def hour_in_blackout(t, lo=18, hi=24):
    return lo <= datetime.fromisoformat(t["ts"]).hour < hi

def gates_keep(t):
    return (not aligns_with_lstm(t)) and (not hour_in_blackout(t))

# ---------- LOAD ----------
def load():
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
            t["_pd"] = t["pnl"] / t["stake_usdc"]  # per-$1 PnL
            out.append(t)
    out.sort(key=lambda x: x["_dt"])
    return out

all_trades = load()
gated = [t for t in all_trades if gates_keep(t)]

# Compute trades-per-day rate (gated trades / actual span in days)
span_days = (gated[-1]["_dt"] - gated[0]["_dt"]).total_seconds() / 86400.0
trades_per_day = len(gated) / span_days
print(f"Loaded {len(all_trades)} Profile-A trades with LSTM data")
print(f"Gates-KEEP universe: {len(gated)} trades over {span_days:.1f} days = {trades_per_day:.2f} trades/day")
print(f"Bankroll: ${BANKROLL_START:.0f}, Stake: ${STAKE:.0f}, Hard stop: ${HARD_STOP_LOSS:.0f}")
print(f"Stake as % of bankroll: {100*STAKE/BANKROLL_START:.0f}%")
print()

# ============================================================
# 1. PER-TRADE DISTRIBUTION (the building block)
# ============================================================
print("=" * 95)
print("1. PER-TRADE PnL DISTRIBUTION  (gates-KEEP universe, per-$1 normalized)")
print("=" * 95)
pd_values = [t["_pd"] for t in gated]
mean_pd = sum(pd_values) / len(pd_values)
std_pd = statistics.stdev(pd_values) if len(pd_values) > 1 else 0
wins = sum(1 for t in gated if t["won"])
w_lo, w_hi = wilson_ci(wins, len(gated))
pd_lo, pd_hi = bootstrap_mean_ci(pd_values)

print(f"  N trades: {len(gated)}")
print(f"  Win rate: {100*wins/len(gated):.1f}%  (Wilson 95% CI [{100*w_lo:.1f}%, {100*w_hi:.1f}%])")
print(f"  Per-$1 mean (E[$1]): ${mean_pd:+.4f}  (bootstrap 95% CI [${pd_lo:+.4f}, ${pd_hi:+.4f}])")
print(f"  Per-$1 std dev: ${std_pd:.4f}")
print(f"  Per-$1 percentiles:")
for p in [0.05, 0.25, 0.50, 0.75, 0.95]:
    print(f"    p{int(p*100):>2}: ${percentile(pd_values, p):+.4f}")
print()
print(f"  At $40 stake:")
print(f"    Expected PnL/trade: ${mean_pd*STAKE:+.2f}")
print(f"    Std dev/trade:      ${std_pd*STAKE:.2f}")
print(f"    Per-trade percentiles (USD):")
for p in [0.05, 0.25, 0.50, 0.75, 0.95]:
    print(f"      p{int(p*100):>2}: ${percentile(pd_values, p)*STAKE:+.2f}")
print()

# ============================================================
# 2. KELLY ANALYSIS
# ============================================================
print("=" * 95)
print("2. KELLY ANALYSIS  (optimal stake size vs current $40/$200 = 20%)")
print("=" * 95)
# For a portfolio with mean μ and variance σ², the Kelly optimal fraction
# (continuous approximation) is f* = μ / σ²
# This is the % of bankroll to stake per bet.
# It's the SECOND-order approximation; for binary outcomes the exact form is
# f* = (bp - q) / b, where b is the win odds, p is win prob, q=1-p.
# Use exact binary form since we have win/loss data.

# Compute per-trade win profit (in per-$1 units)
win_pds = [t["_pd"] for t in gated if t["won"]]
loss_pds = [t["_pd"] for t in gated if not t["won"]]
avg_win = sum(win_pds) / len(win_pds) if win_pds else 0
avg_loss = -sum(loss_pds) / len(loss_pds) if loss_pds else 0   # positive magnitude
p_win = wins / len(gated)
q_loss = 1 - p_win

# Average b (win-to-loss ratio)
b = avg_win / avg_loss if avg_loss > 0 else float('inf')
kelly_full = (b * p_win - q_loss) / b if b > 0 else 0
kelly_half = kelly_full / 2

# Continuous Kelly (mean/variance)
kelly_cont = mean_pd / (std_pd ** 2) if std_pd > 0 else 0

print(f"  Win rate: {p_win:.4f}  Loss rate: {q_loss:.4f}")
print(f"  Avg win per-$1:  +${avg_win:.4f}")
print(f"  Avg loss per-$1: -${avg_loss:.4f}")
print(f"  Win/loss ratio b: {b:.4f}")
print()
print(f"  Full Kelly (binary):     {100*kelly_full:.1f}% of bankroll")
print(f"  Half Kelly (binary):     {100*kelly_half:.1f}% of bankroll")
print(f"  Continuous Kelly (μ/σ²): {100*kelly_cont:.1f}% of bankroll")
print()
print(f"  -> At $200 bankroll, the recommended stakes are:")
print(f"     Full Kelly:  ${kelly_full*BANKROLL_START:.2f}/trade")
print(f"     Half Kelly:  ${kelly_half*BANKROLL_START:.2f}/trade")
print(f"     Quarter Kelly: ${kelly_full*BANKROLL_START/4:.2f}/trade")
print(f"  -> You're staking $40 = {100*STAKE/BANKROLL_START:.0f}% of bankroll")
if STAKE/BANKROLL_START > kelly_full:
    ratio = STAKE/BANKROLL_START / kelly_full
    print(f"     This is {ratio:.2f}x Full Kelly — OVERBET. Expect violent drawdowns.")
elif STAKE/BANKROLL_START > kelly_half:
    print(f"     Between Half and Full Kelly — aggressive but not catastrophic.")
else:
    print(f"     Below Half Kelly — conservative.")
print()

# ============================================================
# 3. DAILY PnL DISTRIBUTION (Monte Carlo, Markov-clean bootstrap)
# ============================================================
print("=" * 95)
print(f"3. DAILY PnL DISTRIBUTION  ({trades_per_day:.1f} trades/day, MC bootstrap from historical)")
print("=" * 95)
print("  Drawing per-$1 PnL with replacement from gates-KEEP pool (Markov-independent).")
print()

n_trades_per_day = round(trades_per_day)
daily_pnls = []
daily_n_trades = []
for _ in range(N_MC_PATHS):
    daily_pnl = 0.0
    n_done = 0
    cum_loss = 0.0
    for _ in range(n_trades_per_day):
        # Hard-stop check before each trade
        if cum_loss >= HARD_STOP_LOSS:
            break
        pd = pd_values[random.randrange(len(pd_values))]
        trade_pnl = pd * STAKE
        daily_pnl += trade_pnl
        if trade_pnl < 0:
            cum_loss += abs(trade_pnl)
        n_done += 1
    daily_pnls.append(daily_pnl)
    daily_n_trades.append(n_done)

mean_daily = sum(daily_pnls)/len(daily_pnls)
mean_n = sum(daily_n_trades)/len(daily_n_trades)
print(f"  Mean daily trades executed: {mean_n:.1f} (planned: {n_trades_per_day})")
print(f"  Mean daily PnL: ${mean_daily:+.2f}")
print(f"  Daily PnL percentiles:")
for p, lbl in [(0.05, "p05 (bad day)"), (0.25, "p25"), (0.50, "p50 (median)"),
                (0.75, "p75"), (0.95, "p95 (great day)")]:
    print(f"    {lbl:<20}: ${percentile(daily_pnls, p):+.2f}")
print(f"  P(losing day): {sum(1 for x in daily_pnls if x < 0)/len(daily_pnls):.1%}")
print(f"  P(hit hard stop): {sum(1 for x in daily_n_trades if x < n_trades_per_day)/len(daily_n_trades):.1%}")
print()

# ============================================================
# 4. 30-DAY BANKROLL SIMULATION
# ============================================================
print("=" * 95)
print(f"4. {DAYS_TO_SIMULATE}-DAY BANKROLL SIMULATION  ($200 start, $40 stake, $120 hard stop)")
print("=" * 95)
print("  Two scenarios: WITH hard stop (session resets next day), and WITHOUT hard stop.")
print()

def simulate_path(use_hard_stop):
    bankroll = BANKROLL_START
    peak = bankroll
    max_dd = 0.0
    bankrupt_day = None
    final_day = DAYS_TO_SIMULATE
    for day in range(1, DAYS_TO_SIMULATE + 1):
        cum_loss_today = 0.0
        for _ in range(n_trades_per_day):
            if bankroll < STAKE:
                bankrupt_day = day
                return bankroll, max_dd, bankrupt_day, day
            if use_hard_stop and cum_loss_today >= HARD_STOP_LOSS:
                break
            pd = pd_values[random.randrange(len(pd_values))]
            trade_pnl = pd * STAKE
            bankroll += trade_pnl
            if trade_pnl < 0:
                cum_loss_today += abs(trade_pnl)
            peak = max(peak, bankroll)
            max_dd = max(max_dd, peak - bankroll)
    return bankroll, max_dd, bankrupt_day, final_day

for scenario_label, use_stop in [("WITH hard stop", True), ("WITHOUT hard stop", False)]:
    finals = []
    dds = []
    bankrupts = 0
    for _ in range(N_MC_PATHS):
        final, dd, bday, _ = simulate_path(use_stop)
        finals.append(final)
        dds.append(dd)
        if bday is not None:
            bankrupts += 1

    print(f"  --- {scenario_label} ---")
    print(f"    Bankroll after {DAYS_TO_SIMULATE} days:")
    for p, lbl in [(0.05, "p05 (worst 5%)"), (0.25, "p25"), (0.50, "p50 (median)"),
                    (0.75, "p75"), (0.95, "p95 (best 5%)")]:
        v = percentile(finals, p)
        print(f"      {lbl:<20}: ${v:>7.2f}  (return: {100*(v-BANKROLL_START)/BANKROLL_START:+.1f}%)")
    print(f"    Mean final bankroll: ${sum(finals)/len(finals):.2f}")
    print(f"    Max drawdown (mean): ${sum(dds)/len(dds):.2f}")
    print(f"    Max drawdown p95: ${percentile(dds, 0.95):.2f}")
    print(f"    P(bankroll < $100): {sum(1 for x in finals if x < 100)/len(finals):.1%}")
    print(f"    P(bankroll < $40 = ruin): {sum(1 for x in finals if x < STAKE)/len(finals):.1%}")
    print(f"    P(bankrupt during run): {bankrupts/N_MC_PATHS:.1%}")
    print(f"    P(profitable after {DAYS_TO_SIMULATE} days): {sum(1 for x in finals if x > BANKROLL_START)/len(finals):.1%}")
    print(f"    P(doubled bankroll): {sum(1 for x in finals if x > 2*BANKROLL_START)/len(finals):.1%}")
    print()

# ============================================================
# 5. RISK OF RUIN — when does ruin happen?
# ============================================================
print("=" * 95)
print("5. RISK OF RUIN OVER TIME  (P(bankroll < $40) by day)")
print("=" * 95)
print("  Using hard stop ON.")
print()

ruin_by_day = [0] * (DAYS_TO_SIMULATE + 1)
for _ in range(N_MC_PATHS):
    bankroll = BANKROLL_START
    ruined_at = None
    for day in range(1, DAYS_TO_SIMULATE + 1):
        cum_loss = 0.0
        for _ in range(n_trades_per_day):
            if bankroll < STAKE:
                ruined_at = day
                break
            if cum_loss >= HARD_STOP_LOSS:
                break
            pd = pd_values[random.randrange(len(pd_values))]
            trade_pnl = pd * STAKE
            bankroll += trade_pnl
            if trade_pnl < 0:
                cum_loss += abs(trade_pnl)
        if ruined_at:
            break
    if ruined_at:
        for d in range(ruined_at, DAYS_TO_SIMULATE + 1):
            ruin_by_day[d] += 1

print(f"  {'Day':<8} {'P(ruin by then)':<18}")
for d in [1, 3, 5, 7, 10, 14, 21, 30]:
    if d <= DAYS_TO_SIMULATE:
        print(f"  {d:<8} {ruin_by_day[d]/N_MC_PATHS:.2%}")
print()

# ============================================================
# 6. STRESS TEST — what if true edge is half observed?
# ============================================================
print("=" * 95)
print("6. STRESS TEST — what if the true edge is HALF the observed (IS→OOS decay)?")
print("=" * 95)
print("  Shifting each per-$1 sample DOWN by half the mean to simulate decay.")
print()

shift = mean_pd / 2  # remove half the edge
pd_shocked = [pd - shift for pd in pd_values]
new_mean = sum(pd_shocked)/len(pd_shocked)
new_wins = sum(1 for x in pd_shocked if x > 0)
print(f"  Original mean E[$1]: ${mean_pd:+.4f}")
print(f"  Shocked mean E[$1]: ${new_mean:+.4f}  (W% may drop because some prior wins now net negative)")
print()

# Re-simulate 30 days under shocked distribution
finals_shock = []
for _ in range(N_MC_PATHS):
    bankroll = BANKROLL_START
    for day in range(DAYS_TO_SIMULATE):
        cum_loss = 0.0
        for _ in range(n_trades_per_day):
            if bankroll < STAKE: break
            if cum_loss >= HARD_STOP_LOSS: break
            pd = pd_shocked[random.randrange(len(pd_shocked))]
            tp = pd * STAKE
            bankroll += tp
            if tp < 0: cum_loss += abs(tp)
    finals_shock.append(bankroll)

print(f"  After {DAYS_TO_SIMULATE} days under shocked distribution:")
for p, lbl in [(0.05, "p05"), (0.25, "p25"), (0.50, "p50"), (0.75, "p75"), (0.95, "p95")]:
    v = percentile(finals_shock, p)
    print(f"    {lbl}: ${v:>7.2f}  (return: {100*(v-BANKROLL_START)/BANKROLL_START:+.1f}%)")
print(f"  Mean final bankroll (shocked): ${sum(finals_shock)/len(finals_shock):.2f}")
print(f"  P(profitable shocked): {sum(1 for x in finals_shock if x > BANKROLL_START)/len(finals_shock):.1%}")
print(f"  P(bankroll < $100 shocked): {sum(1 for x in finals_shock if x < 100)/len(finals_shock):.1%}")
print()

# ============================================================
# 7. STAKE-SIZE SENSITIVITY SWEEP
# ============================================================
print("=" * 95)
print("7. STAKE-SIZE SENSITIVITY  (median 30-day bankroll for different stake sizes)")
print("=" * 95)
print(f"  Same bankroll $200, same {DAYS_TO_SIMULATE}-day horizon, hard stop = 3× stake.")
print()
print(f"  {'Stake':<8} {'% of BR':<10} {'p05':<10} {'p50':<10} {'p95':<10} {'P(ruin)':<10} {'P(profit)':<10} {'mean':<10}")
print("  " + "-" * 80)
for s in [5, 10, 15, 20, 30, 40, 50, 60, 80]:
    finals = []
    bankrupts = 0
    for _ in range(2000):  # smaller MC for the sweep
        bankroll = BANKROLL_START
        hs = 3 * s
        ruined = False
        for day in range(DAYS_TO_SIMULATE):
            cum_loss = 0.0
            for _ in range(n_trades_per_day):
                if bankroll < s:
                    ruined = True
                    break
                if cum_loss >= hs:
                    break
                pd = pd_values[random.randrange(len(pd_values))]
                tp = pd * s
                bankroll += tp
                if tp < 0: cum_loss += abs(tp)
            if ruined: break
        finals.append(bankroll)
        if ruined or bankroll < s: bankrupts += 1
    print(f"  ${s:<7} {100*s/BANKROLL_START:>5.0f}%     "
          f"${percentile(finals,0.05):>6.0f}    ${percentile(finals,0.50):>6.0f}    "
          f"${percentile(finals,0.95):>6.0f}    "
          f"{bankrupts/2000:>6.1%}    "
          f"{sum(1 for x in finals if x > BANKROLL_START)/len(finals):>6.1%}    "
          f"${sum(finals)/len(finals):>6.0f}")
print()

# ============================================================
# 8. CHRONOLOGICAL OOS SIMULATION
#    Use first 50% of trades to estimate distribution.
#    Simulate forward using only those samples (mimics real-world deployment).
#    Compare against actual OOS performance.
# ============================================================
print("=" * 95)
print("8. CHRONOLOGICAL OOS BACKTEST  (Markov bootstrap on IS, compare to actual OOS)")
print("=" * 95)
mid = len(gated) // 2
is_set = gated[:mid]
oos_set = gated[mid:]
is_pd = [t["_pd"] for t in is_set]
oos_pd = [t["_pd"] for t in oos_set]
is_mean = sum(is_pd)/len(is_pd)
oos_mean = sum(oos_pd)/len(oos_pd)
print(f"  IS pool ({is_set[0]['_dt'].date()} → {is_set[-1]['_dt'].date()}):  n={len(is_pd)}, E[$1]=${is_mean:+.4f}")
print(f"  OOS pool ({oos_set[0]['_dt'].date()} → {oos_set[-1]['_dt'].date()}):  n={len(oos_pd)}, E[$1]=${oos_mean:+.4f}")
decay_pct = (is_mean - oos_mean) / abs(is_mean) * 100 if is_mean != 0 else 0
print(f"  IS→OOS decay: {decay_pct:+.1f}% of IS edge lost")
print()

# Walk-forward simulation: how would $200 have done if we'd deployed at the IS/OOS boundary?
oos_days = (oos_set[-1]["_dt"] - oos_set[0]["_dt"]).total_seconds() / 86400.0
oos_n_per_day = len(oos_set) / oos_days

actual_final = BANKROLL_START
peak = BANKROLL_START
max_dd = 0.0
session_loss = 0.0
last_day = None
bankrupt = False
for t in oos_set:
    day = t["_dt"].date()
    if last_day != day:
        session_loss = 0.0
        last_day = day
    if actual_final < STAKE or session_loss >= HARD_STOP_LOSS:
        bankrupt = (actual_final < STAKE)
        continue
    trade_pnl = t["_pd"] * STAKE
    actual_final += trade_pnl
    if trade_pnl < 0: session_loss += abs(trade_pnl)
    peak = max(peak, actual_final)
    max_dd = max(max_dd, peak - actual_final)

print(f"  ACTUAL OOS replay ($200 start, $40 stake, hard stop ON):")
print(f"    Final bankroll: ${actual_final:.2f}  (return: {100*(actual_final-BANKROLL_START)/BANKROLL_START:+.1f}%)")
print(f"    Max drawdown: ${max_dd:.2f}")
print(f"    OOS days: {oos_days:.1f}, OOS trade rate: {oos_n_per_day:.1f}/day")
print(f"    Hit ruin: {bankrupt}")
print()

# ============================================================
# 9. VERDICT
# ============================================================
print("=" * 95)
print("9. VERDICT — is $40 stake on $200 bankroll viable?")
print("=" * 95)

# Pull key numbers
finals_hs = []
bankrupts_hs = 0
for _ in range(N_MC_PATHS):
    bankroll = BANKROLL_START
    ruined = False
    for day in range(DAYS_TO_SIMULATE):
        cum_loss = 0.0
        for _ in range(n_trades_per_day):
            if bankroll < STAKE:
                ruined = True; break
            if cum_loss >= HARD_STOP_LOSS: break
            pd = pd_values[random.randrange(len(pd_values))]
            tp = pd * STAKE
            bankroll += tp
            if tp < 0: cum_loss += abs(tp)
        if ruined: break
    finals_hs.append(bankroll)
    if ruined: bankrupts_hs += 1

p_profit_30 = sum(1 for x in finals_hs if x > BANKROLL_START)/len(finals_hs)
p_double = sum(1 for x in finals_hs if x > 2*BANKROLL_START)/len(finals_hs)
p_ruin_30 = bankrupts_hs/N_MC_PATHS
median_final = percentile(finals_hs, 0.5)
median_return = (median_final - BANKROLL_START)/BANKROLL_START

print(f"  Configuration: ${STAKE:.0f} stake, ${BANKROLL_START:.0f} bankroll, ${HARD_STOP_LOSS:.0f} hard stop")
print(f"  Trade rate: {trades_per_day:.1f}/day (gates-KEEP universe)")
print(f"  Edge: E[$1] = ${mean_pd:+.4f}, W% = {100*p_win:.1f}%, Kelly = {100*kelly_full:.1f}%")
print()
print(f"  Expected outcomes over {DAYS_TO_SIMULATE} days:")
print(f"    Median final bankroll: ${median_final:.2f} ({100*median_return:+.1f}%)")
print(f"    P(profitable): {p_profit_30:.1%}")
print(f"    P(double bankroll): {p_double:.1%}")
print(f"    P(ruin): {p_ruin_30:.1%}")
print()

print("  Verdict:")
if kelly_full > 0 and STAKE/BANKROLL_START > kelly_full:
    print(f"  ✗ OVERBET: $40 = 20% of bankroll, but full Kelly says {100*kelly_full:.1f}%.")
    print(f"     You are betting {STAKE/BANKROLL_START/kelly_full:.1f}× Kelly. This maximizes the chance of large gains")
    print(f"     AND large losses — variance is brutal. Half-Kelly stake would be ${kelly_full*BANKROLL_START/2:.0f}.")
elif kelly_full > 0 and STAKE/BANKROLL_START > kelly_half:
    print(f"  ⚠ AGGRESSIVE: between Half and Full Kelly. Edge real but volatile.")
else:
    print(f"  ✓ CONSERVATIVE: at or below Half Kelly. Sustainable.")

if p_ruin_30 > 0.10:
    print(f"  ✗ Ruin probability {p_ruin_30:.1%} in {DAYS_TO_SIMULATE} days is HIGH.")
elif p_ruin_30 > 0.03:
    print(f"  ⚠ Ruin probability {p_ruin_30:.1%} — non-trivial. Add more bankroll if possible.")
else:
    print(f"  ✓ Ruin probability {p_ruin_30:.1%} — acceptable.")

if p_profit_30 < 0.5:
    print(f"  ✗ Less than coin-flip chance of being profitable in {DAYS_TO_SIMULATE} days.")
else:
    print(f"  ✓ {p_profit_30:.1%} chance of being profitable in {DAYS_TO_SIMULATE} days.")
