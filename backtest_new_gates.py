"""Replay the new gate cascade against historical trade + skip records.

Measures: under the new rules, which trades get TAKEN, which get SKIPPED?
PnL is taken from actual `pnl` (for past trades) or `would_have_pnl`
(for past skips), so we get a clean counterfactual.

Only considers rich-schema records (those with signal fields populated).
Stake assumption: $2 DOWN, $1 UP — but PnL is computed at the actual stake
present in each record (preserves the existing dollar magnitudes).
"""
import json
from collections import defaultdict, Counter

# new-gate config (mirrors .env)
UP_MIN_ASK = 0.50
UP_MAX_ASK = 0.75
EXPENSIVE_FILL = 0.85
STRONG_PTB = 0.60
STRONG_DRIFT = 0.05
REQUIRE_PTB_UP = True
REQUIRE_DRIFT_POS_UP = True
CROWD_INDEC_BAND = 0.05  # |crowd-0.5| < 0.05

UP_STAKE = 1.0
DN_STAKE = 2.0

# pre-ask gates
HIGH_CONF = 1.0       # min confidence
MAX_CONF = 0          # 0 = off
NOISE_BAND = 0.01     # min_drift_pct
# book_vanished is execution failure — not a decision gate. Keep as skipped.

def load_rich(path):
    out = []
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        # need at least these fields to backtest
        ask_up = r.get("top_ask_up")
        ask_dn = r.get("top_ask_down")
        if ask_up is None or ask_dn is None:
            continue
        out.append(r)
    return out

trades = load_rich("predictor/trade_history.jsonl")
skips  = load_rich("predictor/skip_history.jsonl")

def evaluate_gates(r):
    """Return (skip_reason or None, our_ask).

    None reason = trade would be taken under new rules.
    """
    direction = r["direction"]
    ask_up = r.get("top_ask_up")
    ask_dn = r.get("top_ask_down")
    our_ask = ask_up if direction == "UP" else ask_dn
    confidence = r.get("confidence") or 0
    drift_pct = r.get("ptb_distance_pct") if r.get("ptb_distance_pct") is not None else r.get("btc_drift_pct")
    ptb_up = r.get("ptb_prob")
    crowd_up = r.get("crowd_prob")

    # Pre-ask gate 0a: conf floor
    if confidence < HIGH_CONF:
        return ("conf_too_low", our_ask)
    # Pre-ask gate 0b: conf cap (off when MAX_CONF=0)
    if MAX_CONF > 0 and confidence > MAX_CONF:
        return ("conf_too_high", our_ask)
    # Pre-ask gate 0c: contra_drift
    if drift_pct is not None:
        if (direction == "UP" and drift_pct < -NOISE_BAND) or (direction == "DOWN" and drift_pct > NOISE_BAND):
            return ("contra_drift", our_ask)

    # Gate 1: contra_book (any dir)
    if our_ask is not None and our_ask < 0.40:
        return ("contra_book", our_ask)
    # Gate 2: overconfident_contra
    if our_ask is not None and our_ask < 0.50 and confidence > 12.0:
        return ("overconfident_contra", our_ask)

    if direction == "UP":
        # Gate 3: ask floor
        if our_ask is not None and our_ask < UP_MIN_ASK:
            return ("up_ask_too_low", our_ask)
        # Gate 4: drift not negative
        if REQUIRE_DRIFT_POS_UP and drift_pct is not None and drift_pct < 0:
            return ("up_drift_negative", our_ask)
        # Gate 5: PTB support
        if REQUIRE_PTB_UP and ptb_up is not None and ptb_up < 0.50:
            return ("up_no_ptb_support", our_ask)
        # Gate 6: too expensive (with PTB+drift override)
        if our_ask is not None and our_ask > UP_MAX_ASK:
            strong_ptb = ptb_up is not None and ptb_up >= STRONG_PTB
            strong_drift = drift_pct is not None and drift_pct >= STRONG_DRIFT
            if not (strong_ptb and strong_drift):
                return ("up_too_expensive", our_ask)

    # Gate 7: crowd indecision (both dirs)
    if (crowd_up is not None and abs(crowd_up - 0.5) < CROWD_INDEC_BAND
            and our_ask is not None and our_ask <= 0.50):
        return ("crowd_indecision_contra", our_ask)

    # Gate 8: expensive fill (both dirs, with override)
    if our_ask is not None and our_ask > EXPENSIVE_FILL:
        if direction == "UP":
            strong_ptb_d = ptb_up is not None and ptb_up >= STRONG_PTB
            strong_drift_d = drift_pct is not None and drift_pct >= STRONG_DRIFT
        else:
            strong_ptb_d = ptb_up is not None and (1 - ptb_up) >= STRONG_PTB
            strong_drift_d = drift_pct is not None and drift_pct <= -STRONG_DRIFT
        if not (strong_ptb_d and strong_drift_d):
            return ("expensive_fill", our_ask)

    return (None, our_ask)


def _infer_stake(r, won, pnl):
    """Recover the stake at trade-time even when stake_usdc field is None.

    Loss: |pnl| ≈ stake. Win: pnl = stake * (1/ask - 1) → stake = pnl / (1/ask - 1).
    """
    explicit = r.get("stake_usdc")
    if explicit:
        return float(explicit)
    if won is False and pnl is not None:
        return abs(float(pnl))
    if won is True and pnl is not None:
        direction = r["direction"]
        ask = r.get("top_ask_up") if direction == "UP" else r.get("top_ask_down")
        if ask and 0 < ask < 1:
            denom = (1.0/ask - 1.0)
            if denom > 0:
                return float(pnl) / denom
    return 1.0  # last-ditch fallback

def get_outcome(r, is_skip):
    direction = r["direction"]
    if is_skip:
        won = r.get("would_have_won")
        pnl = r.get("would_have_pnl")
    else:
        won = r.get("won")
        pnl = r.get("pnl")
    if pnl is None or won is None:
        return (None, None)
    old_stake = _infer_stake(r, won, pnl)
    if old_stake <= 0:
        return (None, None)
    new_stake = UP_STAKE if direction == "UP" else DN_STAKE
    return (won, pnl * (new_stake / old_stake))


# Tally
results = {
    "old_taken_now_taken": [],   # trades we took, would still take → keep PnL
    "old_taken_now_skip":  [],   # trades we took, would now skip → "saved/lost" trade
    "old_skip_now_taken":  [],   # skips that would now be taken → "found" trade
    "old_skip_now_skip":   [],   # skips that would still be skipped → keep skip
}
new_skip_reasons = Counter()
old_skip_reasons_now_taken = Counter()

for r in trades:
    if not r.get("won") and not r.get("pnl"):
        continue
    new_reason, our_ask = evaluate_gates(r)
    won, pnl = get_outcome(r, is_skip=False)
    if pnl is None:
        continue
    if new_reason is None:
        results["old_taken_now_taken"].append((r, won, pnl))
    else:
        results["old_taken_now_skip"].append((r, won, pnl, new_reason))
        new_skip_reasons[new_reason] += 1

for r in skips:
    won, pnl = get_outcome(r, is_skip=True)
    if pnl is None:
        continue
    old_reason = r.get("skip_reason") or r.get("reason")
    # book_vanished and position_open are execution / state issues, not decision gates.
    # New rules can't recover them — keep them skipped.
    if old_reason in ("book_vanished", "position_open"):
        results["old_skip_now_skip"].append((r, won, pnl, old_reason))
        continue
    new_reason, our_ask = evaluate_gates(r)
    if new_reason is None:
        results["old_skip_now_taken"].append((r, won, pnl))
        old_skip_reasons_now_taken[old_reason] += 1
    else:
        results["old_skip_now_skip"].append((r, won, pnl, new_reason))


def summarize(name, lst, has_reason=False):
    if not lst:
        print(f"  {name}: 0"); return 0.0
    n = len(lst)
    wins = sum(1 for t in lst if t[1])
    pnl = sum(t[2] for t in lst)
    print(f"  {name}: n={n} W={wins} ({wins/n*100:.1f}%) total_pnl=${pnl:+.2f} avg=${pnl/n:+.3f}")
    return pnl

print("="*108)
print("BACKTEST: new gates vs old, on rich-schema records")
print("="*108)
print(f"input: {len(trades)} trades + {len(skips)} skips with signal fields\n")

print("--- OLD: TAKEN trades ---")
old_taken_pnl = sum(t["pnl"] or 0 for t in trades if (t.get("pnl") or 0))
old_taken_n = sum(1 for t in trades if (t.get("pnl") or 0))
print(f"  n={old_taken_n}  total_pnl=${old_taken_pnl:+.2f} (at original stakes)\n")

print("--- NEW: trades we took, NEW rules would have KEPT ---")
keep_pnl = summarize("kept", results["old_taken_now_taken"])

print("\n--- NEW: trades we took, NEW rules would have SKIPPED ---")
skip_pnl = summarize("would-skip", results["old_taken_now_skip"])
print("  break by new gate:")
for reason, count in new_skip_reasons.most_common():
    sub = [(r,w,p) for r,w,p,rs in results["old_taken_now_skip"] if rs == reason]
    if sub: summarize(f"    {reason}", sub)

print("\n--- NEW: trades we SKIPPED, NEW rules would have TAKEN (recovered) ---")
recover_pnl = summarize("recovered", results["old_skip_now_taken"])
print("  these were originally skipped for:")
for reason, count in old_skip_reasons_now_taken.most_common():
    sub = [(r,w,p) for r,w,p in results["old_skip_now_taken"] if (r.get("skip_reason") or r.get("reason"))==reason]
    if sub: summarize(f"    was {reason}", sub)

# Direction breakdown for kept + recovered (the actual portfolio under new rules)
print("\n--- NEW PORTFOLIO (kept + recovered) by direction ---")
for d in ["UP", "DOWN"]:
    kept = [(r,w,p) for r,w,p in results["old_taken_now_taken"] if r["direction"]==d]
    rec  = [(r,w,p) for r,w,p in results["old_skip_now_taken"] if r["direction"]==d]
    combined = kept + rec
    summarize(f"  {d:>4}", combined)

print("\n--- NET COMPARISON ---")
print(f"  OLD (rich-schema only) total: ${sum(t[2] for t in results['old_taken_now_taken']) + sum(t[2] for t in results['old_taken_now_skip']):+.2f}")
print(f"  NEW (kept + recovered) total: ${keep_pnl + recover_pnl:+.2f}")
print(f"  delta from blocking bad takes:  ${-skip_pnl:+.2f}  (positive = avoided losses)")
print(f"  delta from recovering skips:    ${recover_pnl:+.2f}  (positive = found wins)")
print(f"  net improvement vs old:         ${(keep_pnl + recover_pnl) - (sum(t[2] for t in results['old_taken_now_taken']) + sum(t[2] for t in results['old_taken_now_skip'])):+.2f}")
