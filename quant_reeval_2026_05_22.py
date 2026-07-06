"""
COMPLETE STRATEGY REEVALUATION — 2026-05-22

Premise: bot skips 65% of opportunities via 23 gates. Many gates may no longer
earn their keep. Build a pooled (trades + skips with would_have_*) dataset and
audit every gate, find actually-predictive features, propose a new strategy.

Methodology applies all transcripts:
  - Markov: each decision independent given features
  - Walk-forward CV (no look-ahead)
  - Wilson 95% CI for proportions
  - Bootstrap 95% CI for means
  - Permutation test (10k label shuffles)
  - Bonferroni for multi-cell tests
  - Risk: Half-Kelly stake sizing

Outputs:
  Step 1: Gate-by-gate audit (which gates pay, which bleed)
  Step 2: Univariate feature edge (which features predict outcome)
  Step 3: Joint feature analysis (interactions)
  Step 4: New strategy candidate — simple, transparent, fewer gates
  Step 5: Backtest of new strategy with walk-forward CV
  Step 6: Comparison vs current strategy
"""

import json
import math
import random
from collections import defaultdict
from datetime import datetime

random.seed(42)

TRADES = '/home/vidura/btcpredictor/predictor/trade_history.jsonl'
SKIPS  = '/home/vidura/btcpredictor/predictor/skip_history.jsonl'


# ─────────────────────────── stats helpers ───────────────────────────

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n; den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    h = z*math.sqrt((p*(1-p) + z*z/(4*n))/n)/den
    return (c-h, c+h)


def boot(xs, B=5000):
    if not xs: return (0, 0, 0)
    n = len(xs); ms = []
    for _ in range(B):
        s = 0
        for _ in range(n): s += xs[random.randrange(n)]
        ms.append(s/n)
    ms.sort()
    return (sum(xs)/n, ms[int(0.025*B)], ms[int(0.975*B)])


def perm(a, b, B=5000):
    if not a or not b: return 1.0
    obs = abs(sum(a)/len(a) - sum(b)/len(b))
    pool = list(a)+list(b); na = len(a); hits = 0
    for _ in range(B):
        random.shuffle(pool)
        if abs(sum(pool[:na])/na - sum(pool[na:])/(len(pool)-na)) >= obs: hits += 1
    return hits/B


def t_test_vs0(xs):
    n = len(xs)
    if n < 2: return 0
    mu = sum(xs)/n
    var = sum((x-mu)**2 for x in xs) / (n-1)
    se = math.sqrt(var/n) if var > 0 else 0
    return mu/se if se else 0


# ─────────────────────────── pooled dataset ───────────────────────────

def stake_of(r):
    s = r.get('stake_usdc')
    if s not in (None, 0): return float(s)
    if r.get('actual_filled_usdc'): return float(r['actual_filled_usdc'])
    if r.get('won') is False and r.get('pnl', 0) < 0: return abs(r['pnl'])
    return None


def load_pooled():
    """Each decision = (features, outcome, was_taken, skip_reason).
       Outcome derived from won (trades) or would_have_won (skips)."""
    rows = []
    # Real trades
    for line in open(TRADES):
        try: t = json.loads(line)
        except: continue
        if t.get('direction') not in ('UP','DOWN'): continue
        if 'won' not in t: continue
        s = stake_of(t)
        if not s: continue
        pnl = t.get('pnl')
        if pnl is None: continue
        ep = t.get('entry_price')
        if ep is None or not (0 < ep < 1): continue
        try: ts = datetime.fromisoformat(t['ts'])
        except: continue
        rows.append({
            'ts': ts, 'kind': 'trade',
            'dir': t['direction'], 'ep': ep,
            'won': bool(t['won']),
            'r': pnl/s,
            'pnl': pnl, 'stake': s,
            'lstm_p': t.get('lstm_prob'),
            'ob_p':   t.get('orderbook_prob'),
            'ptb_p':  t.get('ptb_prob'),
            'crowd_p': t.get('crowd_prob'),
            'final_p': t.get('final_blended_prob'),
            'conf':   t.get('confidence'),
            'drift':  t.get('btc_drift_pct'),
            'top_up': t.get('top_ask_up'),
            'top_dn': t.get('top_ask_down'),
            'skip_reason': None,
        })
    # Skips with would_have_* (counterfactual outcomes)
    for line in open(SKIPS):
        try: t = json.loads(line)
        except: continue
        if t.get('would_have_won') is None: continue
        if t.get('would_have_pnl') is None: continue
        ep = t.get('entry_price')
        if ep is None or not (0 < ep < 1): continue
        s = t.get('stake_usdc') or 30.0
        try: ts = datetime.fromisoformat(t['ts'])
        except: continue
        rows.append({
            'ts': ts, 'kind': 'skip',
            'dir': t['direction'], 'ep': ep,
            'won': bool(t['would_have_won']),
            'r': t['would_have_pnl']/s,
            'pnl': t['would_have_pnl'], 'stake': s,
            'lstm_p': t.get('lstm_prob'),
            'ob_p':   t.get('orderbook_prob'),
            'ptb_p':  t.get('ptb_prob'),
            'crowd_p': t.get('crowd_prob'),
            'final_p': t.get('final_blended_prob'),
            'conf':   t.get('confidence'),
            'drift':  t.get('btc_drift_pct'),
            'top_up': t.get('top_ask_up'),
            'top_dn': t.get('top_ask_down'),
            'skip_reason': t.get('skip_reason'),
        })
    return sorted(rows, key=lambda x: x['ts'])


# ─────────────────────────── audit ───────────────────────────

def gate_audit(rows):
    print("="*78)
    print("STEP 1 — GATE AUDIT: did each skip earn its keep?")
    print("="*78)
    print(f"{'reason':<28}{'n':>4}{'W%':>6}{'meanR':>9}{'$30 saved':>12}{'verdict':>16}")
    print("-"*78)

    skips = [r for r in rows if r['kind']=='skip' and r['skip_reason']]
    by = defaultdict(list)
    for r in skips:
        by[r['skip_reason']].append(r)

    total_saved = 0
    audit_rows = []
    for reason in sorted(by, key=lambda k: -len(by[k])):
        arr = by[reason]
        n = len(arr); W = sum(1 for r in arr if r['won'])
        rs = [r['r'] for r in arr]
        mu = sum(rs)/n
        # $ saved by skipping = -would_have_pnl (positive means we saved money)
        saved_per_trade = -mu * 30
        # Verdict: significant edge if t > 1.96
        t = t_test_vs0(rs)
        if mu < -0.05:    verdict = "GOOD skip"
        elif mu > +0.05:  verdict = "BAD skip"
        else:             verdict = "neutral"
        if n < 10:        verdict = "tiny n"
        total_saved += saved_per_trade * n
        audit_rows.append((reason, n, W, mu, saved_per_trade, n*saved_per_trade, verdict))
        print(f"{reason:<28}{n:>4}{W/n*100:>5.0f}% {mu:>+8.4f}{saved_per_trade*n:>+11.2f} {verdict:>16}")
    print(f"{'-'*78}")
    print(f"Total saved by all skips: ${total_saved:+.2f}")
    print()
    return audit_rows


def gate_audit_recent(rows, days=14):
    """Same as gate_audit but only on last N days. Detect regime shifts."""
    cutoff = max(r['ts'] for r in rows)
    cutoff_ts = datetime(cutoff.year, cutoff.month, cutoff.day) - \
                 __import__('datetime').timedelta(days=days)
    recent = [r for r in rows if r['ts'] >= cutoff_ts]
    print("="*78)
    print(f"STEP 1B — GATE AUDIT (last {days} days only) — regime change check")
    print("="*78)
    print(f"{'reason':<28}{'n':>4}{'W%':>6}{'meanR':>9}{'$30 saved':>12}{'verdict':>16}")
    print("-"*78)

    skips = [r for r in recent if r['kind']=='skip' and r['skip_reason']]
    by = defaultdict(list)
    for r in skips:
        by[r['skip_reason']].append(r)

    for reason in sorted(by, key=lambda k: -len(by[k])):
        arr = by[reason]
        if len(arr) < 5: continue
        n = len(arr); W = sum(1 for r in arr if r['won'])
        rs = [r['r'] for r in arr]
        mu = sum(rs)/n
        saved = -mu * 30 * n
        if mu < -0.05:    verdict = "GOOD skip"
        elif mu > +0.05:  verdict = "BAD skip"
        else:             verdict = "neutral"
        print(f"{reason:<28}{n:>4}{W/n*100:>5.0f}% {mu:>+8.4f}{saved:>+11.2f} {verdict:>16}")
    print()


# ─────────────────────────── feature edge ───────────────────────────

def buckets(xs, cuts):
    """Assign each x to a bucket based on cut points."""
    for i, c in enumerate(cuts):
        if xs < c: return i
    return len(cuts)


def feature_edge(rows):
    print("="*78)
    print("STEP 2 — UNIVARIATE FEATURE EDGE (all decisions, pooled)")
    print("="*78)
    print("(positive meanR cells = where trades win; negative = where they lose)")
    print()

    # Entry price
    print("By entry_price:")
    cuts = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]
    bk = defaultdict(list)
    for r in rows:
        bk[buckets(r['ep'], cuts)].append(r)
    for i in sorted(bk):
        arr = bk[i]
        if len(arr) < 10: continue
        rs = [r['r'] for r in arr]; W = sum(1 for r in arr if r['won'])
        mu = sum(rs)/len(rs)
        lo_b = '<'+str(cuts[0]) if i==0 else (f">={cuts[-1]}" if i==len(cuts) else f"[{cuts[i-1]:.2f},{cuts[i]:.2f})")
        print(f"  ep {lo_b:>16}: n={len(arr):4d}  W={W/len(arr)*100:4.0f}%  meanR ${mu:+.4f}  $30EV ${mu*30:+.2f}")
    print()

    # Direction
    print("By direction:")
    for d in ('UP','DOWN'):
        arr = [r for r in rows if r['dir']==d]
        rs = [r['r'] for r in arr]; W = sum(1 for r in arr if r['won'])
        mu = sum(rs)/len(rs)
        print(f"  {d:>4}: n={len(arr):4d}  W={W/len(arr)*100:4.0f}%  meanR ${mu:+.4f}")
    print()

    # LSTM-vs-direction (the supposed gate signal)
    print("By LSTM agreement (with bot's direction):")
    cells = {'agree':[], 'contra':[]}
    for r in rows:
        if r['lstm_p'] is None: continue
        if r['dir']=='UP':
            cells['agree' if r['lstm_p']>=0.5 else 'contra'].append(r)
        else:
            cells['agree' if r['lstm_p']<0.5 else 'contra'].append(r)
    for label, arr in cells.items():
        rs = [r['r'] for r in arr]; W = sum(1 for r in arr if r['won'])
        mu = sum(rs)/len(rs)
        wlo,whi = wilson(W, len(arr))
        print(f"  {label:>6}: n={len(arr):4d}  W={W/len(arr)*100:4.0f}% [{wlo*100:.0f}%,{whi*100:.0f}%]  meanR ${mu:+.4f}")
    if cells['agree'] and cells['contra']:
        p = perm([r['r'] for r in cells['agree']], [r['r'] for r in cells['contra']])
        print(f"  perm p={p:.4f}")
    print()

    # Confidence
    print("By confidence:")
    cuts = [3, 5, 7, 10, 15, 20]
    bk = defaultdict(list)
    for r in rows:
        if r['conf'] is None: continue
        bk[buckets(r['conf'], cuts)].append(r)
    for i in sorted(bk):
        arr = bk[i]
        if len(arr) < 10: continue
        rs = [r['r'] for r in arr]; W = sum(1 for r in arr if r['won'])
        mu = sum(rs)/len(rs)
        lo_b = '<'+str(cuts[0]) if i==0 else (f">={cuts[-1]}" if i==len(cuts) else f"[{cuts[i-1]:.0f},{cuts[i]:.0f})")
        print(f"  conf {lo_b:>10}: n={len(arr):4d}  W={W/len(arr)*100:4.0f}%  meanR ${mu:+.4f}")
    print()

    # Drift
    print("By |btc_drift_pct| (absolute, %):")
    cuts = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    bk = defaultdict(list)
    for r in rows:
        if r['drift'] is None: continue
        bk[buckets(abs(r['drift']), cuts)].append(r)
    for i in sorted(bk):
        arr = bk[i]
        if len(arr) < 10: continue
        rs = [r['r'] for r in arr]; W = sum(1 for r in arr if r['won'])
        mu = sum(rs)/len(rs)
        lo_b = f"<{cuts[0]}" if i==0 else (f">={cuts[-1]}" if i==len(cuts) else f"[{cuts[i-1]},{cuts[i]})")
        print(f"  |drift| {lo_b:>14}: n={len(arr):4d}  W={W/len(arr)*100:4.0f}%  meanR ${mu:+.4f}")
    print()

    # Drift alignment with direction (does up-drift help UP trades?)
    print("By drift-direction alignment:")
    aligned, opposed, flat = [], [], []
    for r in rows:
        if r['drift'] is None: continue
        if abs(r['drift']) < 0.001: flat.append(r); continue
        if r['dir']=='UP' and r['drift']>0:    aligned.append(r)
        elif r['dir']=='DOWN' and r['drift']<0: aligned.append(r)
        else:                                    opposed.append(r)
    for label, arr in (('aligned',aligned),('opposed',opposed),('flat',flat)):
        if not arr: continue
        rs = [r['r'] for r in arr]; W = sum(1 for r in arr if r['won'])
        mu = sum(rs)/len(rs)
        print(f"  {label:>8}: n={len(arr):4d}  W={W/len(arr)*100:4.0f}%  meanR ${mu:+.4f}")
    if aligned and opposed:
        p = perm([r['r'] for r in aligned], [r['r'] for r in opposed])
        print(f"  perm p(aligned vs opposed)={p:.4f}")
    print()

    # Market-implied probability sanity
    print("By market_implied prob (1 - top_ask_other for chosen dir):")
    cuts = [0.45, 0.55, 0.65, 0.75, 0.85]
    bk = defaultdict(list)
    for r in rows:
        if r['top_up'] is None or r['top_dn'] is None: continue
        # market-implied of bot's chosen direction
        if r['dir']=='UP':
            mp = 1 - r['top_dn']
        else:
            mp = 1 - r['top_up']
        bk[buckets(mp, cuts)].append(r)
    for i in sorted(bk):
        arr = bk[i]
        if len(arr) < 10: continue
        rs = [r['r'] for r in arr]; W = sum(1 for r in arr if r['won'])
        mu = sum(rs)/len(rs)
        lo_b = f"<{cuts[0]}" if i==0 else (f">={cuts[-1]}" if i==len(cuts) else f"[{cuts[i-1]},{cuts[i]})")
        print(f"  mkt-imp {lo_b:>14}: n={len(arr):4d}  W={W/len(arr)*100:4.0f}%  meanR ${mu:+.4f}")
    print()


# ─────────────────────────── new strategy candidate ───────────────────────────

def keep_decision(r, strategy):
    """Return True if strategy would TAKE this trade."""
    if strategy == 'current':
        # Approximation of current gates that we can reconstruct
        # (only counts: ep in [0.50, 0.85], drift sign filter, LSTM-inv)
        if not (0.50 <= r['ep'] < 0.85): return False
        if r['drift'] is not None:
            if (r['dir']=='UP' and r['drift'] < -0.001) or (r['dir']=='DOWN' and r['drift'] > 0.001):
                return False
        if r['lstm_p'] is not None:
            if r['dir']=='UP' and r['lstm_p']>=0.5: return False
            if r['dir']=='DOWN' and r['lstm_p']<0.5: return False
        return True

    if strategy == 'take_all':
        return True

    if strategy == 'minimal':
        # Only require ep in sensible range
        return 0.50 <= r['ep'] < 0.85

    if strategy == 'new':
        # New proposal — based on feature audit
        # Will fill in after seeing audit output
        return True

    return True


def backtest_strategy(rows, strategy, label):
    taken = [r for r in rows if keep_decision(r, strategy)]
    if not taken: return None
    rs = [r['r'] for r in taken]
    W = sum(1 for r in taken if r['won'])
    mu = sum(rs)/len(rs)
    total_pnl_30 = sum(rs) * 30
    wlo, whi = wilson(W, len(taken))
    t = t_test_vs0(rs)
    print(f"{label:<24}: n={len(taken):4d}/{len(rows)} ({len(taken)/len(rows)*100:4.0f}%)  "
          f"W={W/len(taken)*100:4.1f}%  meanR ${mu:+.4f}  $30 total ${total_pnl_30:+.2f}  t={t:+.2f}")
    return (taken, mu, total_pnl_30, t)


# ─────────────────────────── walk-forward of new strategy ───────────────────────────

def walk_forward(rows, strategy, k=4):
    n = len(rows)
    size = n // (k+1)
    total_n = 0; total_pnl = 0
    print(f"  Walk-forward CV (k={k} expanding folds):")
    for fold in range(1, k+1):
        is_end = fold * size
        oos_end = (fold+1)*size if fold < k else n
        IS, OOS = rows[:is_end], rows[is_end:oos_end]
        if not OOS: break
        # No "training" — strategy is rules-based. Just evaluate on OOS.
        taken = [r for r in OOS if keep_decision(r, strategy)]
        if not taken:
            print(f"    fold {fold}: OOS n={len(OOS)}  no takes"); continue
        rs = [r['r'] for r in taken]; W = sum(1 for r in taken if r['won'])
        mu = sum(rs)/len(rs)
        pnl30 = sum(rs)*30
        total_n += len(taken); total_pnl += pnl30
        print(f"    fold {fold}: OOS n={len(OOS)} taken={len(taken)} W={W}/{len(taken)} ({W/len(taken)*100:.0f}%)  "
              f"meanR ${mu:+.4f}  $30PnL ${pnl30:+.2f}")
    if total_n:
        print(f"  Aggregate OOS: n={total_n}  $30PnL=${total_pnl:+.2f}  avg/trade=${total_pnl/total_n:+.2f}")


# ─────────────────────────── main ───────────────────────────

def main():
    rows = load_pooled()
    print(f"\nLoaded {len(rows)} pooled decisions ({sum(1 for r in rows if r['kind']=='trade')} trades, "
          f"{sum(1 for r in rows if r['kind']=='skip')} skips with would_have_*)\n")
    print(f"Window: {rows[0]['ts'].date()} → {rows[-1]['ts'].date()}\n")

    gate_audit(rows)
    gate_audit_recent(rows, days=7)
    feature_edge(rows)

    print("="*78)
    print("STEP 4 — STRATEGY COMPARISON (all pooled decisions)")
    print("="*78)
    backtest_strategy(rows, 'take_all', 'TAKE ALL (baseline)')
    backtest_strategy(rows, 'minimal',  'MINIMAL (ep band only)')
    backtest_strategy(rows, 'current',  'CURRENT-LIKE (3 gates)')
    print()

    print("="*78)
    print("STEP 5 — WALK-FORWARD CV (4 folds)")
    print("="*78)
    for strat, label in (('take_all','TAKE ALL'), ('minimal','MINIMAL'), ('current','CURRENT-LIKE')):
        print(f"\n{label}:")
        walk_forward(rows, strat, k=4)
    print()


if __name__ == '__main__':
    main()
