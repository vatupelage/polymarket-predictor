"""
Shadow-flip analysis: how would the "flip on lstm_inv_contra skip" strategy
have performed if we'd been live-trading it?

Reads skip_history.jsonl, filters to skip_reason == "lstm_inv_contra".
For records written before the shadow-flip bot patch, flip_* fields are
absent — we derive them from top_ask_up/top_ask_down + would_have_won.

Reports:
  - Running totals + Wilson + bootstrap CI
  - Per-entry-bucket breakdown (the [0.40, 0.50] loss zone test)
  - Direction split
  - "Graduate to live" verdict — has the strategy hit significance yet?

Usage:
  python3 shadow_flip_analysis.py
  python3 shadow_flip_analysis.py --bucket 0.20,0.40   # restrict to ep_other range
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime

random.seed(42)

SKIP_LOG = '/home/vidura/btcpredictor/predictor/skip_history.jsonl'


def derive_flip(rec):
    """Compute flip_* fields for old records lacking them."""
    if rec.get('flip_direction'):
        return (rec['flip_direction'], rec.get('flip_entry_price'),
                rec.get('flip_would_have_won'), rec.get('flip_would_have_pnl'))
    d = rec.get('direction')
    if d not in ('UP', 'DOWN'): return (None, None, None, None)
    ep_other = rec.get('top_ask_down') if d == 'UP' else rec.get('top_ask_up')
    if ep_other is None or not (0 < ep_other < 1): return (None, None, None, None)
    wh_won = rec.get('would_have_won')
    if wh_won is None: return (None, None, None, None)
    flip_dir = 'DOWN' if d == 'UP' else 'UP'
    flip_won = not wh_won
    stake = rec.get('stake_usdc') or 30.0
    flip_shares = stake / ep_other
    flip_pnl = (flip_shares - stake) if flip_won else -stake
    return (flip_dir, ep_other, flip_won, flip_pnl)


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n; den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    h = z*math.sqrt((p*(1-p) + z*z/(4*n))/n)/den
    return (c-h, c+h)


def boot_ci(xs, B=10000):
    if not xs: return (0, 0, 0)
    n = len(xs); ms = []
    for _ in range(B):
        s = 0
        for _ in range(n): s += xs[random.randrange(n)]
        ms.append(s/n)
    ms.sort()
    return (sum(xs)/n, ms[int(0.025*B)], ms[int(0.975*B)])


def sign_perm_p(xs, B=10000):
    """Sign-flip permutation p-value vs H0: mean=0."""
    if not xs: return 1.0
    obs = abs(sum(xs)/len(xs))
    hits = 0
    for _ in range(B):
        s = sum(random.choice([-1, 1])*x for x in xs)
        if abs(s/len(xs)) >= obs: hits += 1
    return hits/B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bucket', help='Filter to ep_other in range e.g. "0.20,0.40"')
    args = ap.parse_args()

    bucket = None
    if args.bucket:
        lo, hi = [float(x) for x in args.bucket.split(',')]
        bucket = (lo, hi)

    rows = [json.loads(l) for l in open(SKIP_LOG)]
    lstm = [r for r in rows if r.get('skip_reason') == 'lstm_inv_contra']

    flips = []
    for r in lstm:
        fd, fep, fw, fpnl = derive_flip(r)
        if fd is None or fpnl is None: continue
        if bucket and not (bucket[0] <= fep < bucket[1]): continue
        stake = r.get('stake_usdc') or 30.0
        flips.append({
            'ts': r['ts'],
            'orig_dir': r['direction'],
            'flip_dir': fd,
            'ep_other': fep,
            'flip_won': fw,
            'flip_pnl': fpnl,
            'stake': stake,
            'r': fpnl / stake,        # per-$1 return
            'orig_pnl': r.get('would_have_pnl'),  # bleed counterfactual
        })

    if not flips:
        print("No usable lstm_inv_contra skips with resolvable flip outcomes.")
        sys.exit(1)

    n = len(flips)
    print(f"\n{'='*72}")
    if bucket:
        print(f"SHADOW FLIP — ep_other ∈ [{bucket[0]:.2f}, {bucket[1]:.2f})")
    else:
        print(f"SHADOW FLIP — ALL lstm_inv_contra skips")
    print('='*72)
    print(f"Window:  {flips[0]['ts'][:19]} → {flips[-1]['ts'][:19]}")
    print(f"N flips: {n}")
    print()

    # Headline
    rs = [f['r'] for f in flips]
    pnls = [f['flip_pnl'] for f in flips]
    W = sum(1 for f in flips if f['flip_won'])
    mu, lo, hi = boot_ci(rs)
    wlo, whi = wilson(W, n)
    p = sign_perm_p(rs)

    print(f"Win rate:      {W}/{n} = {W/n*100:.1f}%   Wilson95% [{wlo*100:.0f}%, {whi*100:.0f}%]")
    print(f"meanR per-$1:  ${mu:+.4f}   boot95% [${lo:+.4f}, ${hi:+.4f}]")
    print(f"Sign-perm p:   {p:.4f}   (vs H0: mean=0)")
    print(f"Total $ PnL:   ${sum(pnls):+.2f}   (avg ${sum(pnls)/n:+.2f}/trade)")
    print()

    # Counterfactual: vs SKIP (gate today) and vs ORIG (take w/o gate)
    orig_pnl = sum(f['orig_pnl'] for f in flips if f['orig_pnl'] is not None)
    print(f"Comparison on this same n={n} window:")
    print(f"  Current gate (SKIP):  $0.00")
    print(f"  Take orig dir (no-gate, the bleed): ${orig_pnl:+.2f}")
    print(f"  Flip (shadow):        ${sum(pnls):+.2f}")
    print()

    # By direction
    print("BY ORIGINAL DIRECTION (= side bot wanted):")
    for d in ('UP', 'DOWN'):
        sub = [f for f in flips if f['orig_dir'] == d]
        if not sub: continue
        rs_ = [f['r'] for f in sub]
        W_ = sum(1 for f in sub if f['flip_won'])
        mu_, lo_, hi_ = boot_ci(rs_, B=5000)
        print(f"  orig={d:>4} (flip to {'DOWN' if d=='UP' else 'UP'}): "
              f"n={len(sub):3d}  W={W_}/{len(sub)} ({W_/len(sub)*100:4.1f}%)  "
              f"meanR ${mu_:+.4f} [{lo_:+.4f},{hi_:+.4f}]")
    print()

    # By ep_other bucket
    if not bucket:
        print("BY ep_other BUCKET (flip entry price):")
        buckets = [(0.0,0.20),(0.20,0.30),(0.30,0.40),(0.40,0.50),(0.50,1.0)]
        for lo_b, hi_b in buckets:
            sub = [f for f in flips if lo_b <= f['ep_other'] < hi_b]
            if not sub: continue
            rs_ = [f['r'] for f in sub]
            pnls_ = [f['flip_pnl'] for f in sub]
            W_ = sum(1 for f in sub if f['flip_won'])
            mu_ = sum(rs_)/len(rs_)
            print(f"  [{lo_b:.2f},{hi_b:.2f}): n={len(sub):3d}  "
                  f"W={W_}/{len(sub)} ({W_/len(sub)*100:4.1f}%)  "
                  f"meanR ${mu_:+.4f}  $PnL=${sum(pnls_):+.2f}")
        print()

    # Graduation criteria
    print("="*72)
    print("GRADUATE TO LIVE? — checking statistical thresholds")
    print("="*72)
    print(f"Current n:           {n}")
    print(f"Lower 95% CI bound:  ${lo:+.4f}   (need > 0 to claim positive edge)")
    print(f"p-value:             {p:.4f}")
    print(f"5% sig threshold:    p < 0.05    "
          f"({'PASS' if p < 0.05 else 'FAIL'})")
    print(f"Bonferroni (3 strats): p < 0.0167  "
          f"({'PASS' if p < 0.0167 else 'FAIL'})")
    print()

    # Estimate trades-to-significance
    if mu > 0 and lo <= 0 and len(rs) > 5:
        # std of mean = sqrt(var/n). Want CI to exclude 0 → 1.96*se < mu → n > (1.96*sd/mu)^2
        var = sum((x-mu)**2 for x in rs) / (len(rs) - 1) if len(rs) > 1 else 0
        sd = math.sqrt(var)
        if mu > 0:
            n_needed = (1.96 * sd / mu) ** 2
            print(f"Approx n needed (parametric estimate): {n_needed:.0f} trades total")
            print(f"  → ~{max(0, n_needed - n):.0f} MORE flips needed to clear 5% significance")
            print(f"  At ~{n/max(1, (datetime.fromisoformat(flips[-1]['ts']) - datetime.fromisoformat(flips[0]['ts'])).days or 1):.0f} flips/day, "
                  f"that's ~{max(0, (n_needed-n))/max(1, n/max(1,(datetime.fromisoformat(flips[-1]['ts']) - datetime.fromisoformat(flips[0]['ts'])).days or 1)):.0f} more days")
    print()

    # Verdict
    if lo > 0 and p < 0.0167:
        print("✓ GRADUATE: lower bound > 0 AND Bonferroni-passed.")
        print("  Recommend enabling BOT_FLIP_AGREE=true in .env.")
    elif lo > 0 and p < 0.05:
        print("~ MARGINAL: significant at 5% but not Bonferroni.")
        print("  Wait for more data or accept the multiple-testing risk.")
    elif mu > 0:
        print("• STILL SHADOW: positive mean but CI crosses zero. Keep collecting.")
    else:
        print("✗ ABANDON CANDIDATE: mean is non-positive. Flip is not paying off here.")
    print()


if __name__ == '__main__':
    main()
