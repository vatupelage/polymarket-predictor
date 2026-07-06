"""
LSTM-inv-contra gate — is it still earning its keep?

Compares the agree-cell vs contra-cell edge across two windows:
  - PRE-GATE  (Apr 30 → May 19, 2026): gate off, all trades in trade_history
  - POST-GATE (May 20 → May 22, 2026): gate on; contra in trade_history,
    agree in skip_history (would_have_* fields).

For the gate to still earn its keep we need (in latest data):
  - agree cell meanR significantly < contra cell meanR
  - agree cell meanR < 0 (so skipping is +EV)
  - Effect size large enough to justify forgoing the n agree trades

Stats applied:
  - Wilson 95% CI for win rate
  - Bootstrap 95% CI for mean per-$1 return (10k resamples)
  - Permutation test agree vs contra (two-sided)
  - Bonferroni for 4 cell comparisons → α' = 0.0125
"""

import json
import math
import random
from datetime import datetime

random.seed(42)

TRADES = '/home/vidura/btcpredictor/predictor/trade_history.jsonl'
SKIPS  = '/home/vidura/btcpredictor/predictor/skip_history.jsonl'
GATE_LIVE = datetime(2026, 5, 20)


# ─────────────────────────── helpers ───────────────────────────

def stake_of(r):
    s = r.get('stake_usdc')
    if s not in (None, 0): return float(s)
    if r.get('actual_filled_usdc'): return float(r['actual_filled_usdc'])
    if r.get('won') is False and r.get('pnl', 0) < 0: return abs(r['pnl'])
    return None


def cell_of(direction, lstm_p):
    if lstm_p is None: return None
    if direction == 'UP':
        return 'agree' if lstm_p >= 0.5 else 'contra'
    if direction == 'DOWN':
        return 'agree' if lstm_p < 0.5 else 'contra'
    return None


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n; den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    h = z*math.sqrt((p*(1-p) + z*z/(4*n))/n)/den
    return (c-h, c+h)


def boot(xs, B=10000):
    if not xs: return (0, 0, 0)
    n = len(xs); ms = []
    for _ in range(B):
        s = 0
        for _ in range(n): s += xs[random.randrange(n)]
        ms.append(s/n)
    ms.sort()
    return (sum(xs)/n, ms[int(0.025*B)], ms[int(0.975*B)])


def perm(a, b, B=10000):
    if not a or not b: return 1.0
    obs = abs(sum(a)/len(a) - sum(b)/len(b))
    pool = list(a) + list(b); na = len(a); hits = 0
    for _ in range(B):
        random.shuffle(pool)
        m_a = sum(pool[:na])/na
        m_b = sum(pool[na:])/(len(pool)-na)
        if abs(m_a - m_b) >= obs: hits += 1
    return hits/B


# ─────────────────────────── data load ───────────────────────────

def load_pre_gate():
    """Pre-gate window: all trades in trade_history Apr 30 → May 19."""
    out = []
    for line in open(TRADES):
        try: t = json.loads(line)
        except: continue
        if t.get('direction') not in ('UP', 'DOWN'): continue
        if 'won' not in t: continue
        if t.get('lstm_prob') is None: continue
        if t.get('entry_price') is None: continue
        try: ts = datetime.fromisoformat(t['ts'])
        except: continue
        if not (datetime(2026,4,30) <= ts < GATE_LIVE): continue
        s = stake_of(t)
        if not s: continue
        pnl = t.get('pnl')
        if pnl is None: continue
        out.append({
            'ts': ts,
            'dir': t['direction'],
            'lstm_p': t['lstm_prob'],
            'won': bool(t['won']),
            'r': pnl/s,
            'pnl': pnl,
            'stake': s,
            'ep': t.get('entry_price'),
            'cell': cell_of(t['direction'], t['lstm_prob']),
        })
    return out


def load_post_gate():
    """Post-gate window:
       - contra cell: real trades in trade_history with ts >= GATE_LIVE
       - agree cell:  lstm_inv_contra skips in skip_history
    """
    out = []
    # Contra (real, executed): trade_history with ts >= GATE_LIVE
    for line in open(TRADES):
        try: t = json.loads(line)
        except: continue
        if t.get('direction') not in ('UP', 'DOWN'): continue
        if 'won' not in t: continue
        if t.get('lstm_prob') is None: continue
        try: ts = datetime.fromisoformat(t['ts'])
        except: continue
        if ts < GATE_LIVE: continue
        s = stake_of(t)
        if not s: continue
        pnl = t.get('pnl')
        if pnl is None: continue
        c = cell_of(t['direction'], t['lstm_prob'])
        out.append({
            'ts': ts,
            'dir': t['direction'],
            'lstm_p': t['lstm_prob'],
            'won': bool(t['won']),
            'r': pnl/s,
            'pnl': pnl,
            'stake': s,
            'ep': t.get('entry_price'),
            'cell': c,
            'source': 'real',
        })
    # Agree (shadow, skipped): skip_history with skip_reason=lstm_inv_contra
    for line in open(SKIPS):
        try: t = json.loads(line)
        except: continue
        if t.get('skip_reason') != 'lstm_inv_contra': continue
        if t.get('would_have_won') is None: continue
        if t.get('would_have_pnl') is None: continue
        if t.get('entry_price') is None: continue
        try: ts = datetime.fromisoformat(t['ts'])
        except: continue
        s = t.get('stake_usdc')
        if not s: continue
        pnl = t['would_have_pnl']
        out.append({
            'ts': ts,
            'dir': t['direction'],
            'lstm_p': t['lstm_prob'],
            'won': bool(t['would_have_won']),
            'r': pnl/s,
            'pnl': pnl,
            'stake': s,
            'ep': t.get('entry_price'),
            'cell': 'agree',
            'source': 'shadow',
        })
    return sorted(out, key=lambda x: x['ts'])


# ─────────────────────────── reporting ───────────────────────────

def cell_stats(rows, label):
    if not rows:
        print(f"  {label}: n=0")
        return None
    n = len(rows); W = sum(1 for r in rows if r['won'])
    rs = [r['r'] for r in rows]
    mu, lo, hi = boot(rs)
    wlo, whi = wilson(W, n)
    print(f"  {label}: n={n:3d}  W={W:3d}/{n} ({W/n*100:4.1f}%) "
          f"[{wlo*100:.0f}%,{whi*100:.0f}%]  "
          f"meanR ${mu:+.4f} [${lo:+.4f},${hi:+.4f}]  "
          f"$30EV ${mu*30:+.2f}")
    return (n, W, mu, lo, hi, rs)


def main():
    pre = load_pre_gate()
    post = load_post_gate()

    print("="*78)
    print("LSTM-INV GATE HEALTH CHECK")
    print("="*78)
    print(f"Pre-gate  (Apr 30 → May 19): n={len(pre)}")
    print(f"Post-gate (May 20 → May 22): n={len(post)}  "
          f"({sum(1 for r in post if r.get('source')=='real')} real + "
          f"{sum(1 for r in post if r.get('source')=='shadow')} shadow)")
    print()

    # PRE-GATE BREAKDOWN
    print("="*78)
    print("PRE-GATE — agree vs contra (where the gate's edge claim came from)")
    print("="*78)
    pre_agree  = [r for r in pre if r['cell'] == 'agree']
    pre_contra = [r for r in pre if r['cell'] == 'contra']
    a_pre = cell_stats(pre_agree, 'agree (skip)  ')
    c_pre = cell_stats(pre_contra, 'contra (keep) ')
    if pre_agree and pre_contra:
        p = perm([r['r'] for r in pre_agree], [r['r'] for r in pre_contra])
        print(f"  Perm test agree vs contra: p={p:.4f} "
              f"({'SIG' if p<0.05 else 'NS'})")
    print()

    # POST-GATE BREAKDOWN
    print("="*78)
    print("POST-GATE — agree (shadow) vs contra (real) since May 20")
    print("="*78)
    post_agree  = [r for r in post if r['cell'] == 'agree']
    post_contra = [r for r in post if r['cell'] == 'contra']
    a_post = cell_stats(post_agree, 'agree (skip)  ')
    c_post = cell_stats(post_contra, 'contra (keep) ')
    if post_agree and post_contra:
        p = perm([r['r'] for r in post_agree], [r['r'] for r in post_contra])
        print(f"  Perm test agree vs contra: p={p:.4f} "
              f"({'SIG' if p<0.05 else 'NS'})")
    print()

    # GATE PnL DELTA POST-GATE
    print("="*78)
    print("POST-GATE COUNTERFACTUAL — does gate STILL earn its keep?")
    print("="*78)
    contra_pnl = sum(r['pnl'] for r in post_contra)
    agree_pnl_skipped = sum(r['pnl'] for r in post_agree)  # would-have-pnl, the gate saved/cost us this
    print(f"  Real PnL (gate on, contra only):     ${contra_pnl:+.2f}  (n={len(post_contra)})")
    print(f"  Counterfactual: had we taken agree:  ${agree_pnl_skipped:+.2f}  (n={len(post_agree)})")
    print(f"  Total if gate off:                   ${contra_pnl + agree_pnl_skipped:+.2f}")
    print()
    delta = -agree_pnl_skipped   # gate saves this much (positive means gate helped)
    print(f"  GATE PnL DELTA (vs no-gate): ${delta:+.2f}  "
          f"({'gate HELPED' if delta>0 else 'gate HURT'})")
    print()

    # Regime change test: pre vs post agree-cell edge
    print("="*78)
    print("REGIME CHANGE — has the agree-cell edge changed?")
    print("="*78)
    if pre_agree and post_agree:
        rs_pre  = [r['r'] for r in pre_agree]
        rs_post = [r['r'] for r in post_agree]
        mu_pre  = sum(rs_pre)/len(rs_pre)
        mu_post = sum(rs_post)/len(rs_post)
        p = perm(rs_pre, rs_post)
        print(f"  agree-cell meanR  pre={mu_pre:+.4f}  post={mu_post:+.4f}  "
              f"Δ={mu_post-mu_pre:+.4f}  perm p={p:.4f}")
    if pre_contra and post_contra:
        rs_pre  = [r['r'] for r in pre_contra]
        rs_post = [r['r'] for r in post_contra]
        mu_pre  = sum(rs_pre)/len(rs_pre)
        mu_post = sum(rs_post)/len(rs_post)
        p = perm(rs_pre, rs_post)
        print(f"  contra-cell meanR pre={mu_pre:+.4f}  post={mu_post:+.4f}  "
              f"Δ={mu_post-mu_pre:+.4f}  perm p={p:.4f}")
    print()

    # Bonferroni adjusted thresholds (4 cells tested)
    print("="*78)
    print("VERDICT — Bonferroni-adjusted (4 cells, α'=0.0125)")
    print("="*78)

    # Decide
    if not post_agree or not post_contra:
        print("  Insufficient post-gate data for verdict.")
        return

    p_post = perm([r['r'] for r in post_agree], [r['r'] for r in post_contra])
    post_agree_mu = sum(r['r'] for r in post_agree)/len(post_agree)
    post_contra_mu = sum(r['r'] for r in post_contra)/len(post_contra)

    print(f"  Latest agree meanR: ${post_agree_mu:+.4f}  ({'<0 ✓' if post_agree_mu<0 else '≥0 ✗'} — need <0 for skip to be +EV)")
    print(f"  Latest contra mean: ${post_contra_mu:+.4f}  ({'>0 ✓' if post_contra_mu>0 else '≤0 ✗'} — need >0)")
    print(f"  Latest agree vs contra perm p={p_post:.4f}  "
          f"({'PASSES Bonferroni' if p_post<0.0125 else 'fails Bonferroni'})")
    print()

    if post_agree_mu < 0 and p_post < 0.0125:
        print("  ✓ KEEP GATE — agree still bleeding, contra still earning, sig.")
    elif post_agree_mu < 0:
        print("  ~ KEEP GATE (provisional) — agree mean is negative but not yet sig.")
        print("    Effect is in the right direction; insufficient post-gate sample size.")
    elif post_agree_mu > 0 and len(post_agree) > 50:
        print("  ✗ TURN OFF GATE — agree cell is now PROFITABLE in latest regime.")
        print("    Gate is filtering trades that would make money.")
    else:
        print("  ? AMBIGUOUS — collect more data before deciding.")
    print()


if __name__ == '__main__':
    main()
