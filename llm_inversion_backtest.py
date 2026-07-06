"""
Hypothesis: within the gates-KEEP universe, INVERT the LLM's TAKE/SKIP signal.
The "gates KEEP & LLM SKIP" cell showed +$0.394/$1 (n=39) in the post-hoc
agreement matrix — but that was discovered by inspecting 4 cells after seeing
all the data. Classic data-snooping risk.

Applies the transcripts' guidance:
- Markov: each trade is an independent observation; no path-dependent features
- Conditional dists: we're testing a conditional (LLM SKIP | gates KEEP), not pooling
- Look-ahead bias: chronological IS/OOS split; walk-forward CV
- Multiple comparisons: Bonferroni-correct (we picked 1 of 4 cells)
- Permutation test: shuffle LLM labels within gates-KEEP to estimate H0 distribution
- Bootstrap CI: 10k resamples on per-$1 PnL
"""
import json
import math
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

random.seed(42)

SCRIPT_DIR = Path(__file__).parent
CACHE = SCRIPT_DIR / "llm_filter_cache.jsonl"
HISTORY = SCRIPT_DIR / "trade_history.jsonl"

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
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return (lo, hi)

def aligns_with_lstm(t):
    return (t["direction"] == "UP" and t.get("lstm_prob", 0.5) >= 0.5) or \
           (t["direction"] == "DOWN" and t.get("lstm_prob", 0.5) < 0.5)

def hour_in_blackout(t, lo=18, hi=24):
    return lo <= datetime.fromisoformat(t["ts"]).hour < hi

def gates_keep(t):
    return (not aligns_with_lstm(t)) and (not hour_in_blackout(t))

def summary(label, trades, ci_alpha=0.05):
    n = len(trades)
    if n == 0:
        print(f"  {label:<46} n=0")
        return None
    wins = sum(1 for t in trades if t["won"])
    pnl = sum(t["pnl"] for t in trades)
    stake = sum(t["stake_usdc"] for t in trades)
    pd = [t["pnl"]/t["stake_usdc"] for t in trades]
    mpd = sum(pd) / n
    z = 1.96 if ci_alpha == 0.05 else (2.5 if abs(ci_alpha - 0.0125) < 1e-6 else 1.96)
    w_lo, w_hi = wilson_ci(wins, n, z=z)
    pd_lo, pd_hi = bootstrap_mean_ci(pd, alpha=ci_alpha)
    edge = " EDGE" if pd_lo > 0 else (" LOSS" if pd_hi < 0 else "")
    pct = int(100*(1-ci_alpha))
    print(f"  {label:<46} n={n:>3}  W={wins:>3}/{n} ({100*wins/n:>4.1f}%, CI{pct}%[{100*w_lo:>4.1f},{100*w_hi:>4.1f}])  "
          f"PnL=${pnl:>+8.2f}  E[$1]=${mpd:>+.3f} CI{pct}%[${pd_lo:>+.3f},${pd_hi:>+.3f}]{edge}")
    return {"n": n, "w": wins, "pnl": pnl, "stake": stake, "mpd": mpd, "pd_lo": pd_lo, "pd_hi": pd_hi}

# ---------- load ----------
def load():
    history = {}
    with open(HISTORY) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                t = json.loads(line)
                history[t["ts"]] = t
            except: continue
    trades = []
    with open(CACHE) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                t = history.get(rec["ts"])
                if t is None: continue
                t = dict(t)
                t["llm_decision"] = rec["decision"]
                t["llm_confidence"] = rec["confidence"]
                t["_dt"] = datetime.fromisoformat(t["ts"])
                trades.append(t)
            except: continue
    trades.sort(key=lambda x: x["_dt"])
    return trades

trades = load()
print(f"Loaded {len(trades)} trades with LLM decisions")
print(f"Date range: {trades[0]['_dt'].date()} -> {trades[-1]['_dt'].date()}")
print()

# ============================================================
# 1. The full agreement matrix (already known but recap)
# ============================================================
print("=" * 100)
print("1. AGREEMENT MATRIX RECAP  (the 4 cells we're choosing among)")
print("=" * 100)
keep = [t for t in trades if gates_keep(t)]
skip_gate = [t for t in trades if not gates_keep(t)]

keep_llm_take = [t for t in keep if t["llm_decision"] == "TAKE"]
keep_llm_skip = [t for t in keep if t["llm_decision"] == "SKIP"]
skip_llm_take = [t for t in skip_gate if t["llm_decision"] == "TAKE"]
skip_llm_skip = [t for t in skip_gate if t["llm_decision"] == "SKIP"]

summary("Gates KEEP & LLM TAKE  (current 'stacked')", keep_llm_take)
summary("Gates KEEP & LLM SKIP  (INVERSION HYPOTHESIS)", keep_llm_skip)
summary("Gates SKIP & LLM TAKE", skip_llm_take)
summary("Gates SKIP & LLM SKIP", skip_llm_skip)
print()
summary("Gates KEEP (everything)", keep)
print()

# ============================================================
# 2. Bonferroni-corrected CI
#    We picked 1 of 4 cells. At alpha=0.05/4 = 0.0125 → z=2.5
# ============================================================
print("=" * 100)
print("2. BONFERRONI CORRECTION  (we cherry-picked 1 of 4 cells)")
print("=" * 100)
print("  Using alpha=0.0125 (= 0.05/4) → 98.75% CI instead of 95%")
print()
summary("Inversion bucket — 95% CI",  keep_llm_skip, ci_alpha=0.05)
summary("Inversion bucket — 98.75% CI", keep_llm_skip, ci_alpha=0.0125)
print()

# ============================================================
# 3. Chronological IS/OOS split
#    Discover the hypothesis on IS only, test on OOS.
# ============================================================
print("=" * 100)
print("3. CHRONOLOGICAL IS / OOS SPLIT  (50/50)")
print("=" * 100)
mid = len(trades) // 2
is_set = trades[:mid]
oos_set = trades[mid:]
print(f"  IS:  {is_set[0]['_dt'].date()} -> {is_set[-1]['_dt'].date()}  n={len(is_set)}")
print(f"  OOS: {oos_set[0]['_dt'].date()} -> {oos_set[-1]['_dt'].date()}  n={len(oos_set)}")
print()

for label, ds in [("IN-SAMPLE", is_set), ("OUT-OF-SAMPLE", oos_set)]:
    print(f"  --- {label} ---")
    k = [t for t in ds if gates_keep(t)]
    klt = [t for t in k if t["llm_decision"] == "TAKE"]
    kls = [t for t in k if t["llm_decision"] == "SKIP"]
    summary("Gates KEEP (baseline)", k)
    summary("Stacked (gates+LLM TAKE)", klt)
    summary("INVERSION (gates KEEP & LLM SKIP)", kls)
    print()

# ============================================================
# 4. Walk-forward CV (4 folds, expanding window)
# ============================================================
print("=" * 100)
print("4. WALK-FORWARD CV  (4 disjoint test folds, ordered in time)")
print("=" * 100)
print(f"  {'Fold':<5} {'period':<22} {'n_gate_keep':<13} {'n_invert':<10} {'invert W%':<11} {'invert E[$1]':<14} {'95% CI':<24}")
print("  " + "-" * 95)
fold_size = len(trades) // 4
fold_results = []
for fold in range(4):
    a = fold * fold_size
    b = (fold + 1) * fold_size if fold < 3 else len(trades)
    chunk = trades[a:b]
    k = [t for t in chunk if gates_keep(t)]
    kls = [t for t in k if t["llm_decision"] == "SKIP"]
    if len(kls) >= 3:
        pd = [t["pnl"]/t["stake_usdc"] for t in kls]
        mpd = sum(pd)/len(pd)
        wins = sum(1 for t in kls if t["won"])
        pd_lo, pd_hi = bootstrap_mean_ci(pd, n_boot=3000)
        period = f"{chunk[0]['_dt'].date()}→{chunk[-1]['_dt'].date()}"
        fold_results.append({"n": len(kls), "mpd": mpd, "wins": wins, "lo": pd_lo, "hi": pd_hi, "period": period})
        print(f"  {fold+1:<5} {period:<22} {len(k):<13} {len(kls):<10} {100*wins/len(kls):>5.1f}%      ${mpd:>+.3f}        [${pd_lo:>+.3f},${pd_hi:>+.3f}]")
    else:
        print(f"  {fold+1:<5} {chunk[0]['_dt'].date()}→{chunk[-1]['_dt'].date()} (n_invert < 3)")
print()
n_positive_folds = sum(1 for f in fold_results if f["mpd"] > 0)
n_edge_folds = sum(1 for f in fold_results if f["lo"] > 0)
print(f"  Folds with positive mean: {n_positive_folds}/{len(fold_results)}")
print(f"  Folds with CI strictly above zero: {n_edge_folds}/{len(fold_results)}")
print()

# ============================================================
# 5. Permutation test
#    Under H0: LLM label is independent of trade outcome within gates-KEEP.
#    Shuffle LLM TAKE/SKIP labels within the gates-KEEP universe many times
#    and measure how often the "SKIP" bucket beats the observed +$0.394.
# ============================================================
print("=" * 100)
print("5. PERMUTATION TEST  (does the inversion edge survive label-shuffle?)")
print("=" * 100)
print("  H0: LLM TAKE/SKIP labels are independent of trade outcome within gates-KEEP.")
print("  Procedure: shuffle the LLM labels 10,000 times; recompute the inversion-")
print("  bucket E[$1] each time. Count how often shuffled >= observed.")
print()

observed_mpd = sum(t["pnl"]/t["stake_usdc"] for t in keep_llm_skip) / max(len(keep_llm_skip), 1)
n_skip_in_keep = len(keep_llm_skip)
keep_pd = [t["pnl"]/t["stake_usdc"] for t in keep]
n_perm = 10000
n_beat = 0
for _ in range(n_perm):
    # Sample n_skip_in_keep trades uniformly at random from gates-KEEP universe
    # (this is what would happen if LLM SKIP was independent of outcome)
    idxs = random.sample(range(len(keep_pd)), n_skip_in_keep)
    shuffled_mpd = sum(keep_pd[i] for i in idxs) / n_skip_in_keep
    if shuffled_mpd >= observed_mpd:
        n_beat += 1
p_value = n_beat / n_perm
print(f"  Observed inversion E[$1]: ${observed_mpd:+.3f}  (n={n_skip_in_keep} of {len(keep)} gate-KEEP trades)")
print(f"  Shuffled >= observed: {n_beat}/{n_perm} = p-value {p_value:.4f}")
print(f"  Interpretation: under random LLM labeling, {100*p_value:.2f}% of permutations match or beat the observed edge.")
print(f"  -> {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05  "
      f"|  {'SIGNIFICANT' if p_value < 0.0125 else 'NOT SIGNIFICANT'} at Bonferroni-corrected alpha=0.0125")
print()

# ============================================================
# 6. Monthly counterfactual equity (sequential PnL, no compounding)
# ============================================================
print("=" * 100)
print("6. SEQUENTIAL EQUITY CURVE  (cumulative PnL, $40 unit stake)")
print("=" * 100)
def equity_curve(label, ts_pnls):
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for _, p in ts_pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return cum, max_dd

# Use actual PnL (preserves stake variance from the live runs)
strategies = {
    "Raw (no gates)": [(t["ts"], t["pnl"]) for t in trades],
    "Gates only": [(t["ts"], t["pnl"]) for t in keep],
    "Stacked (gates+LLM TAKE)": [(t["ts"], t["pnl"]) for t in keep_llm_take],
    "INVERSION (gates+LLM SKIP)": [(t["ts"], t["pnl"]) for t in keep_llm_skip],
}
print(f"  {'Strategy':<32} {'n':<6} {'final PnL':<14} {'max DD':<12} {'PnL/trade':<12}")
print("  " + "-" * 80)
for name, seq in strategies.items():
    if not seq:
        print(f"  {name:<32} (empty)")
        continue
    cum, dd = equity_curve(name, seq)
    print(f"  {name:<32} {len(seq):<6} ${cum:>+8.2f}     ${dd:>6.2f}      ${cum/len(seq):>+6.2f}")
print()

# ============================================================
# 7. NORMALIZED equity (per-$1 cumulative, removes stake variance)
# ============================================================
print("=" * 100)
print("7. NORMALIZED EQUITY  (sum of per-$1 PnL — removes stake-size variance)")
print("=" * 100)
print(f"  {'Strategy':<32} {'n':<6} {'sum per-$1':<14} {'avg per-$1':<12}")
print("  " + "-" * 70)
for name, ds in [("Raw (no gates)", trades), ("Gates only", keep),
                 ("Stacked (gates+LLM TAKE)", keep_llm_take),
                 ("INVERSION (gates+LLM SKIP)", keep_llm_skip)]:
    if not ds:
        print(f"  {name:<32} (empty)")
        continue
    pd_sum = sum(t["pnl"]/t["stake_usdc"] for t in ds)
    print(f"  {name:<32} {len(ds):<6} {pd_sum:>+8.3f}     {pd_sum/len(ds):>+6.3f}")
print()

# ============================================================
# 8. CONDITIONAL DRILLDOWN — where does the inversion edge concentrate?
# ============================================================
print("=" * 100)
print("8. CONDITIONAL DRILLDOWN — where does the INVERSION edge live?")
print("=" * 100)
print()

print("  By direction:")
for d in ["UP", "DOWN"]:
    summary(f"INVERSION & {d}", [t for t in keep_llm_skip if t["direction"] == d])
print()

print("  By entry-price band:")
for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75)]:
    summary(f"INVERSION & entry [{lo:.2f},{hi:.2f})",
            [t for t in keep_llm_skip if lo <= t["entry_price"] < hi])
print()

print("  By LLM confidence band:")
for lo, hi in [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]:
    summary(f"INVERSION & llm-conf [{lo:.1f},{hi:.1f})",
            [t for t in keep_llm_skip if lo <= t["llm_confidence"] < hi])
print()

# ============================================================
# 9. VERDICT
# ============================================================
print("=" * 100)
print("9. VERDICT")
print("=" * 100)

# OOS inversion bucket
oos_keep = [t for t in oos_set if gates_keep(t)]
oos_inv = [t for t in oos_keep if t["llm_decision"] == "SKIP"]
if oos_inv:
    pd_oos = [t["pnl"]/t["stake_usdc"] for t in oos_inv]
    oos_mpd = sum(pd_oos)/len(pd_oos)
    oos_lo, oos_hi = bootstrap_mean_ci(pd_oos)
else:
    oos_mpd, oos_lo, oos_hi = 0, 0, 0

print(f"  Full-sample inversion bucket:")
print(f"    n={len(keep_llm_skip)}, W%={100*sum(1 for t in keep_llm_skip if t['won'])/len(keep_llm_skip):.1f}%, "
      f"E[$1]=${observed_mpd:+.3f}")
print(f"  Out-of-sample inversion bucket:")
if oos_inv:
    print(f"    n={len(oos_inv)}, W%={100*sum(1 for t in oos_inv if t['won'])/len(oos_inv):.1f}%, "
          f"E[$1]=${oos_mpd:+.3f}, 95% CI[${oos_lo:+.3f},${oos_hi:+.3f}]")
print(f"  Permutation p-value: {p_value:.4f}")
print(f"  Walk-forward positive folds: {n_positive_folds}/{len(fold_results)}")
print(f"  Walk-forward edge (CI>0) folds: {n_edge_folds}/{len(fold_results)}")
print()

# Decision logic
green_lights = 0
red_lights = 0
notes = []

if p_value < 0.05:
    green_lights += 1
    notes.append("✓ Permutation test significant at 5%")
else:
    red_lights += 1
    notes.append("✗ Permutation test NOT significant — edge could be chance")

if p_value < 0.0125:
    green_lights += 1
    notes.append("✓ Permutation test survives Bonferroni correction")
else:
    notes.append("- Permutation test does NOT survive Bonferroni correction")

if oos_inv and oos_lo > 0:
    green_lights += 1
    notes.append(f"✓ OOS edge: CI strictly above zero")
elif oos_inv and oos_mpd > 0:
    notes.append(f"- OOS edge positive (${oos_mpd:+.3f}) but CI crosses zero")
else:
    red_lights += 1
    notes.append("✗ OOS edge negative or zero")

if n_edge_folds >= 2:
    green_lights += 1
    notes.append(f"✓ {n_edge_folds}/{len(fold_results)} walk-forward folds had CI above zero")
elif n_positive_folds >= 3:
    notes.append(f"- {n_positive_folds}/{len(fold_results)} folds positive but CIs cross zero")
else:
    red_lights += 1
    notes.append(f"✗ Only {n_positive_folds}/{len(fold_results)} folds positive")

for note in notes:
    print(f"  {note}")
print()
print(f"  Green lights: {green_lights} / Red lights: {red_lights}")
if green_lights >= 3 and red_lights == 0:
    print("  → DEPLOY: hypothesis is robust across all tests.")
elif green_lights >= 2 and red_lights <= 1:
    print("  → CAUTIOUSLY DEPLOY at reduced size; collect more data.")
else:
    print("  → DO NOT DEPLOY: hypothesis is likely cherry-picked / overfit.")
