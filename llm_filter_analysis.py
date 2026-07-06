"""
Analyze llm_filter_cache.jsonl: compare LLM filter against current gates,
no-gates baseline, and stacked. Wilson + bootstrap CIs throughout.
"""
import json
import math
import random
from pathlib import Path
from datetime import datetime

random.seed(42)

SCRIPT_DIR = Path(__file__).parent
CACHE = SCRIPT_DIR / "llm_filter_cache.jsonl"
HISTORY = SCRIPT_DIR / "trade_history.jsonl"

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

def load_cache():
    out = []
    with open(CACHE) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                out.append(rec)
            except: continue
    return out

def load_history_by_ts():
    by_ts = {}
    with open(HISTORY) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                t = json.loads(line)
                by_ts[t["ts"]] = t
            except: continue
    return by_ts

def aligns_with_lstm(t):
    return (t["direction"] == "UP" and t.get("lstm_prob", 0.5) >= 0.5) or \
           (t["direction"] == "DOWN" and t.get("lstm_prob", 0.5) < 0.5)

def hour_in_blackout(t, blackout_lo=18, blackout_hi=24):
    h = datetime.fromisoformat(t["ts"]).hour
    return blackout_lo <= h < blackout_hi

def summary(label, trades):
    n = len(trades)
    if n == 0:
        print(f"  {label:<40} n=0")
        return None
    wins = sum(1 for t in trades if t["won"])
    pnl = sum(t["pnl"] for t in trades)
    stake = sum(t["stake_usdc"] for t in trades)
    per_dollar = [t["pnl"]/t["stake_usdc"] for t in trades]
    mean_pd = sum(per_dollar) / n
    w_lo, w_hi = wilson_ci(wins, n)
    pd_lo, pd_hi = bootstrap_mean_ci(per_dollar)
    edge = " EDGE" if pd_lo > 0 else (" LOSS" if pd_hi < 0 else "")
    print(f"  {label:<40} n={n:>3}  W={wins:>3}/{n} ({100*wins/n:>4.1f}%, CI[{100*w_lo:>4.1f},{100*w_hi:>4.1f}])  "
          f"PnL=${pnl:>+8.2f}  stake=${stake:>6.0f}  E[$1]=${mean_pd:>+.3f} CI[${pd_lo:>+.3f},${pd_hi:>+.3f}]{edge}")
    return {"n": n, "w": wins, "pnl": pnl, "stake": stake, "mean_pd": mean_pd, "pd_lo": pd_lo, "pd_hi": pd_hi}

def main():
    cache = load_cache()
    history = load_history_by_ts()

    # Hydrate each cache record with full history fields needed for gate logic
    trades = []
    for rec in cache:
        t = history.get(rec["ts"])
        if t is None: continue
        merged = dict(t)
        merged["llm_decision"] = rec["decision"]
        merged["llm_confidence"] = rec["confidence"]
        merged["llm_reasoning"] = rec.get("reasoning", "")
        trades.append(merged)

    trades.sort(key=lambda x: x["ts"])
    n_take = sum(1 for t in trades if t["llm_decision"] == "TAKE")
    n_skip = sum(1 for t in trades if t["llm_decision"] == "SKIP")
    n_err  = sum(1 for t in trades if t["llm_decision"] == "PARSE_ERROR")

    print(f"Loaded {len(cache)} LLM decisions, hydrated {len(trades)} trades")
    print(f"LLM: TAKE={n_take}  SKIP={n_skip}  PARSE_ERROR={n_err}")
    print(f"Date range: {trades[0]['ts'][:10]} -> {trades[-1]['ts'][:10]}")
    print()

    # ============================================================
    # 1. STRATEGY COMPARISON
    # ============================================================
    print("=" * 95)
    print("1. STRATEGY COMPARISON  (all 187 Profile-A trades)")
    print("=" * 95)

    # A. Raw — take everything
    summary("A. RAW (no gates)", trades)

    # B. Current gates: skip if aligns with LSTM OR in 18-24 hour blackout
    gated = [t for t in trades if not aligns_with_lstm(t) and not hour_in_blackout(t)]
    summary("B. CURRENT GATES (lstm-inv + 18-24)", gated)

    # C. LLM-only: take only when LLM says TAKE
    llm_take = [t for t in trades if t["llm_decision"] == "TAKE"]
    summary("C. LLM-ONLY (decision==TAKE)", llm_take)

    # D. Stacked: current gates AND LLM TAKE
    stacked = [t for t in trades if not aligns_with_lstm(t) and not hour_in_blackout(t) and t["llm_decision"] == "TAKE"]
    summary("D. STACKED (gates + LLM)", stacked)

    print()
    print("=" * 95)
    print("2. CONFIDENCE THRESHOLD SWEEP — LLM-ONLY")
    print("=" * 95)
    print("  Only TAKE trades where LLM confidence >= threshold")
    print()
    for thresh in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
        bucket = [t for t in trades if t["llm_decision"] == "TAKE" and t["llm_confidence"] >= thresh]
        summary(f"  conf >= {thresh}", bucket)

    print()
    print("=" * 95)
    print("3. AGREEMENT MATRIX — LLM vs CURRENT GATES")
    print("=" * 95)
    print("  Both decisions made independently. Each trade falls into one of 4 cells.")
    print()

    for gate_keep in [True, False]:
        for llm_take_flag in [True, False]:
            bucket = []
            for t in trades:
                gate_decision = (not aligns_with_lstm(t)) and (not hour_in_blackout(t))
                llm_decision = (t["llm_decision"] == "TAKE")
                if gate_decision == gate_keep and llm_decision == llm_take_flag:
                    bucket.append(t)
            gate_str = "KEEP" if gate_keep else "SKIP"
            llm_str = "TAKE" if llm_take_flag else "SKIP"
            label = f"  gates={gate_str}, llm={llm_str}"
            summary(label, bucket)

    print()
    print("=" * 95)
    print("4. WHERE THEY DISAGREE — interesting cases")
    print("=" * 95)

    # Gate says skip, LLM says take — does LLM rescue good trades?
    gate_skip_llm_take = [t for t in trades
                          if (aligns_with_lstm(t) or hour_in_blackout(t))
                          and t["llm_decision"] == "TAKE"]
    print(f"\n  Gate SKIPs but LLM says TAKE (n={len(gate_skip_llm_take)}):")
    if gate_skip_llm_take:
        summary("    outcomes", gate_skip_llm_take)

    # Gate says keep, LLM says skip — does LLM avoid losses?
    gate_keep_llm_skip = [t for t in trades
                          if (not aligns_with_lstm(t)) and (not hour_in_blackout(t))
                          and t["llm_decision"] == "SKIP"]
    print(f"\n  Gate would TAKE but LLM says SKIP (n={len(gate_keep_llm_skip)}):")
    if gate_keep_llm_skip:
        summary("    outcomes", gate_keep_llm_skip)

    print()
    print("=" * 95)
    print("5. OUT-OF-SAMPLE 50/50 — does the LLM filter hold up?")
    print("=" * 95)
    mid = len(trades) // 2
    is_set = trades[:mid]
    oos_set = trades[mid:]
    print(f"  IS:  {is_set[0]['ts'][:10]} -> {is_set[-1]['ts'][:10]}  (n={len(is_set)})")
    print(f"  OOS: {oos_set[0]['ts'][:10]} -> {oos_set[-1]['ts'][:10]}  (n={len(oos_set)})")
    print()
    for label, dataset in [("IN-SAMPLE", is_set), ("OUT-OF-SAMPLE", oos_set)]:
        print(f"  --- {label} ---")
        summary("Raw", dataset)
        summary("Current gates", [t for t in dataset if not aligns_with_lstm(t) and not hour_in_blackout(t)])
        summary("LLM-only", [t for t in dataset if t["llm_decision"] == "TAKE"])
        summary("Stacked", [t for t in dataset
                            if not aligns_with_lstm(t) and not hour_in_blackout(t)
                            and t["llm_decision"] == "TAKE"])
        print()

    print()
    print("=" * 95)
    print("6. LLM REASONING SAMPLES — wins it skipped and losses it took")
    print("=" * 95)
    # Wins the LLM said SKIP to (LLM was wrong: it skipped a winner)
    llm_missed_wins = [t for t in trades if t["llm_decision"] == "SKIP" and t["won"]]
    # Losses the LLM said TAKE on (LLM was wrong: it took a loser)
    llm_took_losses = [t for t in trades if t["llm_decision"] == "TAKE" and not t["won"]]

    print(f"\n  LLM-skipped winners (n={len(llm_missed_wins)}):  sample of 3:")
    for t in llm_missed_wins[:3]:
        print(f"    {t['ts']} {t['direction']} ep={t['entry_price']:.3f} conf={t['confidence']:.1f}% pnl=${t['pnl']:+.2f}")
        print(f"      LLM (c={t['llm_confidence']:.2f}): {t['llm_reasoning'][:200]}")

    print(f"\n  LLM-taken losers (n={len(llm_took_losses)}):  sample of 3:")
    for t in llm_took_losses[:3]:
        print(f"    {t['ts']} {t['direction']} ep={t['entry_price']:.3f} conf={t['confidence']:.1f}% pnl=${t['pnl']:+.2f}")
        print(f"      LLM (c={t['llm_confidence']:.2f}): {t['llm_reasoning'][:200]}")

    print()
    print("=" * 95)
    print("7. VERDICT — flip BOT_USE_LLM=true?")
    print("=" * 95)
    raw = summary("Raw (no gates)", trades)
    gates = summary("Current gates", gated)
    llm = summary("LLM-only", llm_take)
    stk = summary("Stacked (gates+LLM)", stacked)
    print()

    best = max([("Raw", raw), ("Gates", gates), ("LLM", llm), ("Stacked", stk)],
               key=lambda x: x[1]["pnl"] if x[1] else -1e9)
    print(f"  Best by total PnL: {best[0]} (${best[1]['pnl']:+.2f} on n={best[1]['n']})")
    best_pd = max([("Raw", raw), ("Gates", gates), ("LLM", llm), ("Stacked", stk)],
                  key=lambda x: x[1]["mean_pd"] if x[1] else -1e9)
    print(f"  Best by E[$1]:     {best_pd[0]} (${best_pd[1]['mean_pd']:+.3f})")

if __name__ == "__main__":
    main()
