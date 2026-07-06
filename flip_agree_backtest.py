"""
Test: when LSTM-inv-contra gate fires (agree → bot skips),
      can we BET THE OPPOSITE direction instead?

For each "agree" trade in pre-gate data we know:
  - bot's chosen direction D, entry_price ep_D, won_D (actual outcome)
  - top_ask of the OPPOSITE side ep_other (from same orderbook snapshot)

Binary market: won_other = NOT won_D (exactly one side resolves to $1).
Flip per-$1 return:
   if won_other: (1/ep_other) - 1
   else:         -1

Caveats:
  - top_ask is the *displayed* best ask. Slippage may apply (typical 1-2¢).
  - Only valid on PRE-GATE data (May 19 and earlier): those agree trades
    actually executed so we have a real orderbook snapshot. Post-gate
    agree trades were skipped → no orderbook snapshot to use.

Stats applied (per transcripts):
  - Wilson 95% CI for win rate
  - Bootstrap 95% CI for mean (10k resamples)
  - Permutation test vs the "keep contra only" alternative
  - Walk-forward CV (4 expanding folds)
  - Bonferroni: 3 strategies tested → α' = 0.05/3 ≈ 0.0167
  - Slippage stress test (1¢ and 2¢ haircut)
"""

import json
import math
import random
from collections import defaultdict
from datetime import datetime

random.seed(42)

DATA = '/home/vidura/btcpredictor/predictor/trade_history.jsonl'
GATE_LIVE_DATE = datetime(2026, 5, 20)


# ─────────────────────────── load & clean ───────────────────────────

def stake_of(t):
    if t.get('stake_usdc') not in (None, 0): return float(t['stake_usdc'])
    if t.get('actual_filled_usdc'): return float(t['actual_filled_usdc'])
    if t.get('won') is False and t.get('pnl', 0) < 0: return abs(t['pnl'])
    return None


def load_clean():
    rows = []
    for line in open(DATA):
        try: rows.append(json.loads(line))
        except: pass
    out = []
    for t in rows:
        if t.get('direction') not in ('UP','DOWN'): continue
        if 'won' not in t: continue
        s = stake_of(t)
        if not s or s <= 0: continue
        if t.get('pnl') is None: continue
        ep = t.get('entry_price')
        if ep is None or ep <= 0 or ep >= 1: continue
        if t.get('top_ask_up') is None or t.get('top_ask_down') is None: continue
        if t.get('lstm_prob') is None: continue
        try: ts = datetime.fromisoformat(t['ts'])
        except: continue
        out.append({
            'ts': ts,
            'dir': t['direction'],
            'won': bool(t['won']),
            'ep': ep,
            'ep_up': float(t['top_ask_up']),
            'ep_dn': float(t['top_ask_down']),
            'lstm_p': float(t['lstm_prob']),
            'pnl': t['pnl'],
            'stake': s,
            'r_orig': t['pnl'] / s,            # per-$1 return as bot took it
        })
    return sorted(out, key=lambda x: x['ts'])


def cell(r):
    """BOT's gate logic, threshold = 0.5."""
    if r['dir'] == 'UP':
        return 'agree' if r['lstm_p'] >= 0.5 else 'contra'
    else:
        return 'agree' if r['lstm_p'] < 0.5 else 'contra'


def flip_return(r, slippage_cents=0.0):
    """Per-$1 return if we'd bet the opposite direction at top_ask_other.
    Slippage adds to the entry price (worse fill)."""
    ep_other = r['ep_dn'] if r['dir'] == 'UP' else r['ep_up']
    ep_other = min(0.99, ep_other + slippage_cents)
    if ep_other <= 0:
        return 0.0
    won_other = not r['won']
    return (1.0 / ep_other - 1.0) if won_other else -1.0


# ─────────────────────────── stats ───────────────────────────

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n; den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    h = z*math.sqrt((p*(1-p) + z*z/(4*n))/n) / den
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


def t_test_vs0(xs):
    n = len(xs)
    if n < 2: return 0
    mu = sum(xs)/n
    var = sum((x-mu)**2 for x in xs) / (n-1)
    se = math.sqrt(var/n)
    return mu/se if se else 0


def perm_test(a, b, B=10000):
    if not a or not b: return 1.0
    obs = abs(sum(a)/len(a) - sum(b)/len(b))
    pool = list(a) + list(b)
    na = len(a); hits = 0
    for _ in range(B):
        random.shuffle(pool)
        m_a = sum(pool[:na])/na
        m_b = sum(pool[na:])/(len(pool)-na)
        if abs(m_a - m_b) >= obs: hits += 1
    return hits/B


# ─────────────────────────── analysis ───────────────────────────

def main():
    rows = load_clean()
    print(f"Loaded {len(rows)} clean trades (need top_ask + lstm_prob)\n")

    # Restrict to PRE-GATE so agree trades actually executed
    pre = [r for r in rows if r['ts'] < GATE_LIVE_DATE]
    agree   = [r for r in pre if cell(r) == 'agree']
    contra  = [r for r in pre if cell(r) == 'contra']

    print(f"Pre-gate window: {pre[0]['ts'].date()} → {pre[-1]['ts'].date()}")
    print(f"  agree (gate would skip):  n={len(agree)}")
    print(f"  contra (gate keeps):      n={len(contra)}")
    print()

    # ── Step 1: orig agree returns ──
    print("="*78)
    print("STEP 1 — bot's ORIGINAL direction on agree cell (the bleed)")
    print("="*78)
    rs = [r['r_orig'] for r in agree]
    W = sum(1 for r in agree if r['won'])
    mu, lo, hi = boot(rs)
    wlo, whi = wilson(W, len(agree))
    print(f"n={len(agree)}  W={W}/{len(agree)} ({W/len(agree)*100:.1f}%) "
          f"Wilson [{wlo*100:.0f}%,{whi*100:.0f}%]")
    print(f"meanR ${mu:+.4f} boot95% [${lo:+.4f},${hi:+.4f}]  t={t_test_vs0(rs):+.2f}")
    print(f"$30 EV/trade: ${mu*30:+.2f}   →   total over n=81: ${mu*30*len(agree):+.2f}")
    print()

    # ── Step 2: flip returns ──
    print("="*78)
    print("STEP 2 — FLIP direction on agree cell (bet opposite at top_ask_other)")
    print("="*78)

    for slip_label, slip in (('zero',0.0), ('1¢ haircut',0.01), ('2¢ haircut',0.02)):
        rs_f = [flip_return(r, slip) for r in agree]
        W_f = sum(1 for x in rs_f if x > 0)
        mu, lo, hi = boot(rs_f)
        wlo, whi = wilson(W_f, len(rs_f))
        t = t_test_vs0(rs_f)
        print(f"slippage={slip_label}:")
        print(f"  n={len(rs_f)} W={W_f}/{len(rs_f)} ({W_f/len(rs_f)*100:.1f}%) "
              f"Wilson [{wlo*100:.0f}%,{whi*100:.0f}%]")
        print(f"  meanR ${mu:+.4f}  boot95% [${lo:+.4f},${hi:+.4f}]  t={t:+.2f}")
        print(f"  $30 EV/trade: ${mu*30:+.2f}   total n={len(agree)}: ${mu*30*len(agree):+.2f}")
        print()

    # Use zero-slippage flip for downstream tests
    flip_rs = [flip_return(r, 0.0) for r in agree]

    # ── Step 3: vs contra (the kept cell) ──
    print("="*78)
    print("STEP 3 — vs CONTRA cell (what gate already keeps)")
    print("="*78)
    contra_rs = [r['r_orig'] for r in contra]
    cmu, clo, chi = boot(contra_rs)
    fmu, flo, fhi = boot(flip_rs)
    print(f"contra (kept, original dir): n={len(contra)} meanR ${cmu:+.4f} [{clo:+.4f},{chi:+.4f}]")
    print(f"flipped agree (proposal):    n={len(flip_rs)} meanR ${fmu:+.4f} [{flo:+.4f},{fhi:+.4f}]")
    p = perm_test(contra_rs, flip_rs)
    print(f"Perm test contra vs flip: p={p:.4f} "
          f"({'DIFFERENT' if p<0.05 else 'no detectable diff'})")
    print()

    # ── Step 4: walk-forward CV on flip strategy ──
    print("="*78)
    print("STEP 4 — walk-forward CV (4 expanding folds, IS-train, OOS-test)")
    print("="*78)
    n = len(agree)
    if n >= 20:
        k = 4
        size = n // (k+1)
        total_oos_n = 0
        total_oos_pnl = 0
        for fold in range(1, k+1):
            is_end = fold*size
            oos_end = (fold+1)*size if fold < k else n
            IS, OOS = agree[:is_end], agree[is_end:oos_end]
            if not OOS: break
            # IS: estimate flip mean
            is_flip = [flip_return(r, 0.0) for r in IS]
            is_mean = sum(is_flip)/len(is_flip)
            # Only "trade the flip" in OOS if IS edge > 0
            oos_flip = [flip_return(r, 0.0) for r in OOS]
            if is_mean > 0:
                take = oos_flip
                act = "TAKE"
            else:
                take = []
                act = "SKIP"
            W_oos = sum(1 for x in oos_flip if x > 0)
            mu_oos = sum(oos_flip)/len(oos_flip)
            taken_pnl = sum(take)*30
            total_oos_n += len(take)
            total_oos_pnl += taken_pnl
            print(f"fold {fold}: IS n={len(IS)} meanR ${is_mean:+.4f} → {act}  "
                  f"OOS n={len(OOS)} W={W_oos}/{len(OOS)} meanR ${mu_oos:+.4f} "
                  f"$30PnL ${taken_pnl:+.2f}")
        avg = total_oos_pnl/total_oos_n if total_oos_n else 0
        print(f"\nAggregate OOS taken: n={total_oos_n}  $30PnL=${total_oos_pnl:+.2f}  "
              f"avg/trade=${avg:+.2f}")
    print()

    # ── Step 5: 3-way comparison (skip vs flip vs original) ──
    print("="*78)
    print("STEP 5 — agree-cell strategy comparison ($30 stake)")
    print("="*78)
    base = sum(r['r_orig'] for r in contra) * 30
    skip_pnl = base                                    # what gate does today
    take_pnl = base + sum(r['r_orig'] for r in agree) * 30   # no gate
    flip_pnl = base + sum(flip_rs) * 30                # flip agree
    print(f"  SKIP agree (current):     contra-only PnL = ${skip_pnl:+.2f}")
    print(f"  TAKE agree as-is (nogate): TOTAL PnL =      ${take_pnl:+.2f}")
    print(f"  FLIP agree (proposed):    contra + flip =   ${flip_pnl:+.2f}")
    print()
    print(f"  FLIP vs SKIP delta: ${flip_pnl - skip_pnl:+.2f} on the 81-trade window")
    print()

    # ── Step 6: significance (Bonferroni 3-way) ──
    print("="*78)
    print("STEP 6 — significance of FLIP edge (Bonferroni for 3 strategies)")
    print("="*78)
    t = t_test_vs0(flip_rs)
    # Bonferroni: α=0.05/3=0.0167 → two-sided |t| > ~2.40 (df large)
    p_vs_zero_perm = sum(1 for _ in range(10000)
                         if abs(sum(random.choice([-1,1])*x for x in flip_rs)/len(flip_rs))
                         >= abs(sum(flip_rs)/len(flip_rs))) / 10000
    mu = sum(flip_rs)/len(flip_rs)
    print(f"Flip meanR=${mu:+.4f}  t-stat={t:+.2f}")
    print(f"Sign-flip permutation p (two-sided): {p_vs_zero_perm:.4f}")
    print(f"Threshold for sig at 5%:               p < 0.0500   "
          f"({'PASS' if p_vs_zero_perm<0.05 else 'FAIL'})")
    print(f"Threshold w/ Bonferroni (3 strats):    p < 0.0167   "
          f"({'PASS' if p_vs_zero_perm<0.0167 else 'FAIL'})")
    print()

    # ── Step 7: by sub-direction (does the flip work both ways?) ──
    print("="*78)
    print("STEP 7 — flip edge BY ORIGINAL DIRECTION (does asymmetry exist?)")
    print("="*78)
    for d in ('UP', 'DOWN'):
        sub = [r for r in agree if r['dir'] == d]
        if not sub: continue
        rs_f = [flip_return(r, 0.0) for r in sub]
        rs_o = [r['r_orig'] for r in sub]
        W_f = sum(1 for x in rs_f if x > 0)
        mu_f, lo_f, hi_f = boot(rs_f, B=5000)
        mu_o = sum(rs_o)/len(rs_o)
        print(f"agree-cell where bot wanted {d}: n={len(sub)}")
        print(f"  ORIG dir {d}: meanR=${mu_o:+.4f}  (this was the bleed)")
        print(f"  FLIP to {'DOWN' if d=='UP' else 'UP'}: n={len(rs_f)} "
              f"W={W_f}/{len(rs_f)} ({W_f/len(rs_f)*100:.0f}%) "
              f"meanR=${mu_f:+.4f} [{lo_f:+.4f},{hi_f:+.4f}]")
    print()

    # ── Step 8: entry price sensitivity ──
    print("="*78)
    print("STEP 8 — flip edge BY OPPOSITE-SIDE entry price")
    print("="*78)
    print("(If ep_other is cheap = strong underdog, flip pays more on rare wins.)")
    print()
    buckets = [(0,0.20),(0.20,0.30),(0.30,0.40),(0.40,0.50),(0.50,1.0)]
    for lo, hi in buckets:
        sub = [r for r in agree if lo <= (r['ep_dn'] if r['dir']=='UP' else r['ep_up']) < hi]
        if not sub: continue
        rs_f = [flip_return(r, 0.0) for r in sub]
        W_f = sum(1 for x in rs_f if x > 0)
        mu_f = sum(rs_f)/len(rs_f)
        print(f"ep_other [{lo:.2f},{hi:.2f}): n={len(sub):3d}  "
              f"W={W_f}/{len(sub)} ({W_f/len(sub)*100:4.1f}%)  "
              f"flip meanR ${mu_f:+.4f}  $30EV ${mu_f*30:+.2f}")
    print()


if __name__ == '__main__':
    main()
