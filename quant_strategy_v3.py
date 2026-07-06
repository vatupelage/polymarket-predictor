"""
STRATEGY V3 — Design + Walk-Forward CV

Based on quant_reeval_2026_05_22.py audit findings:

GATES TO KEEP (good skips, meanR << 0):
  - entry_out_of_band      (saved +$254, n=137)
  - crowd_indecision_contra (saved +$293, n=22)
  - high_entry_low_conf     (saved +$167, n=24)
  - up_too_expensive        (saved +$125, n=60)
  - conf_too_low            (saved +$87,  n=15)
  - contra_book             (saved +$65,  n=57)

GATES TO KILL (bad skips, meanR >> 0 — gate cost money):
  - mid_price_high_conf  (cost $263, n=19)
  - hour_blackout        (cost $164, n=19)
  - contra_drift         (cost $167, n=123)
  - lstm_inv_contra      (cost $77,  n=90)  borderline; killed for v3 test

V3 ENTRY RULES:
  1. ep ∈ [0.50, 0.65)   (sweet-spot zone)
  2. |btc_drift_pct| < 0.01  (calm-BTC zone)
  3. Respect all GOOD/PHYSICAL gates

We approximate by: if the original record was skipped by a BAD gate, V3 takes it.
                   if original was skipped by a GOOD/PHYSICAL gate, V3 still skips.
                   if original was a real trade, V3 evaluates v3 rules.
"""

import json
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta

random.seed(42)

TRADES = '/home/vidura/btcpredictor/predictor/trade_history.jsonl'
SKIPS  = '/home/vidura/btcpredictor/predictor/skip_history.jsonl'

GOOD_GATES = {'entry_out_of_band', 'crowd_indecision_contra', 'high_entry_low_conf',
              'up_too_expensive', 'conf_too_low', 'contra_book', 'up_ask_too_low',
              'overconfident_contra', 'drift_noise', 'up_conf_too_low'}
BAD_GATES  = {'mid_price_high_conf', 'hour_blackout', 'contra_drift', 'lstm_inv_contra',
              'expensive_fill'}
PHYSICAL_GATES = {'position_open', 'book_vanished', 'ask_moved_against', 'direction_filter',
                  'up_drift_negative', 'up_no_ptb_support', 'negative_kelly'}


# ─────────────────────────── stats ───────────────────────────

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


def t_test(xs):
    n = len(xs)
    if n < 2: return 0
    mu = sum(xs)/n
    var = sum((x-mu)**2 for x in xs)/(n-1)
    se = math.sqrt(var/n) if var > 0 else 0
    return mu/se if se else 0


# ─────────────────────────── data ───────────────────────────

def stake_of(r):
    s = r.get('stake_usdc')
    if s not in (None, 0): return float(s)
    if r.get('actual_filled_usdc'): return float(r['actual_filled_usdc'])
    if r.get('won') is False and r.get('pnl', 0) < 0: return abs(r['pnl'])
    return None


def load_pooled():
    rows = []
    for line in open(TRADES):
        try: t = json.loads(line)
        except: continue
        if t.get('direction') not in ('UP','DOWN'): continue
        if 'won' not in t: continue
        s = stake_of(t)
        if not s: continue
        if t.get('pnl') is None: continue
        ep = t.get('entry_price')
        if ep is None or not (0 < ep < 1): continue
        try: ts = datetime.fromisoformat(t['ts'])
        except: continue
        rows.append({
            'ts': ts, 'kind': 'trade', 'skip_reason': None,
            'dir': t['direction'], 'ep': ep,
            'won': bool(t['won']), 'r': t['pnl']/s,
            'pnl': t['pnl'], 'stake': s,
            'lstm_p': t.get('lstm_prob'),
            'conf': t.get('confidence'),
            'drift': t.get('btc_drift_pct'),
            'top_up': t.get('top_ask_up'),
            'top_dn': t.get('top_ask_down'),
        })
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
            'ts': ts, 'kind': 'skip', 'skip_reason': t.get('skip_reason'),
            'dir': t['direction'], 'ep': ep,
            'won': bool(t['would_have_won']),
            'r': t['would_have_pnl']/s,
            'pnl': t['would_have_pnl'], 'stake': s,
            'lstm_p': t.get('lstm_prob'),
            'conf': t.get('confidence'),
            'drift': t.get('btc_drift_pct'),
            'top_up': t.get('top_ask_up'),
            'top_dn': t.get('top_ask_down'),
        })
    return sorted(rows, key=lambda x: x['ts'])


# ─────────────────────────── decision rules ───────────────────────────

def take_all(r):
    return True


def current_takes(r):
    """The bot today: any skip_reason means skip."""
    return r['skip_reason'] is None


def v3_takes(r):
    """V3 rule. Respect GOOD/PHYSICAL gates; apply new ep + drift filter."""
    if r['skip_reason'] in GOOD_GATES or r['skip_reason'] in PHYSICAL_GATES:
        return False
    if not (0.50 <= r['ep'] < 0.65):
        return False
    if r['drift'] is not None and abs(r['drift']) >= 0.01:
        return False
    return True


def v3_strict_takes(r):
    """V3 + avoid the conf [5,7) dead zone."""
    if not v3_takes(r): return False
    if r['conf'] is not None and 5 <= r['conf'] < 7:
        return False
    return True


# ─────────────────────────── evaluation ───────────────────────────

def evaluate(rows, fn, label):
    taken = [r for r in rows if fn(r)]
    if not taken:
        print(f"{label:<22}: 0 takes"); return None
    rs = [r['r'] for r in taken]
    W = sum(1 for r in taken if r['won'])
    mu, lo, hi = boot(rs)
    wlo, whi = wilson(W, len(taken))
    t = t_test(rs)
    pnl30 = sum(rs) * 30
    print(f"{label:<22}: n={len(taken):4d}/{len(rows)} ({len(taken)/len(rows)*100:4.0f}%)  "
          f"W={W/len(taken)*100:4.1f}% [{wlo*100:.0f}%,{whi*100:.0f}%]  "
          f"meanR ${mu:+.4f} [{lo:+.4f},{hi:+.4f}]  "
          f"$30 ${pnl30:+8.2f}  t={t:+.2f}")
    return {'taken': taken, 'n': len(taken), 'mu': mu, 'lo': lo, 'hi': hi,
            'pnl30': pnl30, 'W': W, 't': t}


def walk_forward(rows, fn, label, k=4):
    print(f"\n{label} — walk-forward (k={k} expanding folds):")
    n = len(rows); size = n // (k+1)
    total_n = 0; total_pnl = 0
    for fold in range(1, k+1):
        is_end = fold*size
        oos_end = (fold+1)*size if fold < k else n
        OOS = rows[is_end:oos_end]
        if not OOS: break
        taken = [r for r in OOS if fn(r)]
        if not taken:
            print(f"  fold {fold}: OOS n={len(OOS)} no takes"); continue
        rs = [r['r'] for r in taken]; W = sum(1 for r in taken if r['won'])
        mu = sum(rs)/len(rs); pnl30 = sum(rs)*30
        total_n += len(taken); total_pnl += pnl30
        print(f"  fold {fold}: OOS={len(OOS)} taken={len(taken)} "
              f"W={W}/{len(taken)} ({W/len(taken)*100:.0f}%)  "
              f"meanR ${mu:+.4f}  $30PnL ${pnl30:+.2f}")
    if total_n:
        print(f"  → OOS total: n={total_n}  $30PnL=${total_pnl:+.2f}  "
              f"avg/trade=${total_pnl/total_n:+.2f}")
    return total_pnl, total_n


def recent_check(rows, fn, label, days=7):
    cutoff = max(r['ts'] for r in rows)
    cutoff_ts = cutoff.replace(hour=0, minute=0, second=0) - timedelta(days=days)
    arr = [r for r in rows if r['ts'] >= cutoff_ts]
    taken = [r for r in arr if fn(r)]
    if not taken:
        print(f"  {label:<22}: 0 takes in last {days} days"); return
    rs = [r['r'] for r in taken]; W = sum(1 for r in taken if r['won'])
    mu = sum(rs)/len(rs); pnl30 = sum(rs)*30
    print(f"  {label:<22}: n={len(taken):3d}/{len(arr)}  W={W}/{len(taken)} "
          f"({W/len(taken)*100:.0f}%)  meanR ${mu:+.4f}  $30PnL ${pnl30:+.2f}")


def kelly_calc(taken, bankroll=200):
    rs = [r['r'] for r in taken]
    if not rs: return
    n = len(rs); mu = sum(rs)/n
    e2 = sum(r*r for r in rs)/n
    f_full = mu/e2 if e2 > 0 else 0
    print(f"  E[r]=${mu:+.4f}  E[r²]=${e2:+.4f}  "
          f"full-K={f_full*100:.1f}%  half-K=${bankroll*f_full/2:.2f}  "
          f"qtr-K=${bankroll*f_full/4:.2f}")


# ─────────────────────────── main ───────────────────────────

def main():
    rows = load_pooled()
    span = (rows[-1]['ts'] - rows[0]['ts']).days
    print(f"\nPooled n={len(rows)}  window={rows[0]['ts'].date()} → {rows[-1]['ts'].date()} ({span}d)\n")

    # ────────── FULL-SAMPLE COMPARISON ──────────
    print("="*88)
    print("FULL-SAMPLE STRATEGY COMPARISON")
    print("="*88)
    res = {}
    res['take_all']  = evaluate(rows, take_all,        'TAKE ALL')
    res['current']   = evaluate(rows, current_takes,   'CURRENT (bot now)')
    res['v3']        = evaluate(rows, v3_takes,        'V3 (proposed)')
    res['v3_strict'] = evaluate(rows, v3_strict_takes, 'V3 STRICT')
    print()

    # ────────── PER-TRADE ECONOMICS ──────────
    print("="*88)
    print("PER-TRADE ECONOMICS (annualized projection)")
    print("="*88)
    for name, r in res.items():
        if not r: continue
        per_trade = r['pnl30']/r['n']
        trades_per_day = r['n']/max(1, span)
        daily_pnl = per_trade * trades_per_day
        print(f"  {name:<12}  ${per_trade:+6.2f}/trade  "
              f"{trades_per_day:5.1f} trades/day  "
              f"${daily_pnl:+7.2f}/day  "
              f"30-day proj: ${daily_pnl*30:+8.2f}")
    print()

    # ────────── KELLY SIZING ──────────
    print("="*88)
    print("KELLY SIZING (per $200 bankroll)")
    print("="*88)
    for name, r in res.items():
        if not r: continue
        print(f"  {name}:")
        kelly_calc(r['taken'], bankroll=200)
    print()

    # ────────── RECENT 7-DAY CHECK ──────────
    print("="*88)
    print("RECENT 7-DAY CHECK — regime sensitivity")
    print("="*88)
    for fn, label in [(take_all,'TAKE ALL'),(current_takes,'CURRENT'),
                       (v3_takes,'V3'),(v3_strict_takes,'V3 STRICT')]:
        recent_check(rows, fn, label, days=7)
    print()

    # ────────── WALK-FORWARD CV ──────────
    print("="*88)
    print("WALK-FORWARD CV (4 EXPANDING FOLDS)")
    print("="*88)
    walk_forward(rows, current_takes,   'CURRENT')
    walk_forward(rows, v3_takes,        'V3')
    walk_forward(rows, v3_strict_takes, 'V3 STRICT')
    print()

    # ────────── EDGE DELTA ──────────
    print("="*88)
    print("EDGE DELTA: V3 vs CURRENT")
    print("="*88)
    cur, v3 = res['current'], res['v3']
    if cur and v3:
        print(f"  CURRENT:  {cur['n']:4d} trades  meanR ${cur['mu']:+.4f}  total ${cur['pnl30']:+.2f}  t={cur['t']:+.2f}")
        print(f"  V3:       {v3['n']:4d} trades  meanR ${v3['mu']:+.4f}  total ${v3['pnl30']:+.2f}  t={v3['t']:+.2f}")
        dn = v3['n'] - cur['n']; dp = v3['pnl30'] - cur['pnl30']
        print(f"  Δ:        {dn:+d} trades  ${dp:+.2f} PnL  "
              f"(${dp/max(1,dn):+.2f}/added-trade)")
    print()


if __name__ == '__main__':
    main()
