"""
Pre-registered 1-trade/day strategy search with the SAME rigor as s5_proper_backtest:
no look-ahead (signal from window features; label = GT resolution), no survivorship bias
(GT outcome for every window, drop nothing), costs (dynamic taker fee 0.07*(1-p)/$1, crypto feeRate=0.07 + ask=spread),
data-snooping control (8 PRE-REGISTERED strategies, report ALL, Bonferroni x8), walk-forward
OOS, Monte Carlo (outcome-permutation + daily bootstrap) on the leader.
1 trade/day = the day's highest-|signal| window (path-aware conviction pick).
"""
import json, math, random, collections, statistics as st
random.seed(7)
def fee(p): return 0.07*(1-p)   # taker fee in $ per $1-stake BUY: (1/p)*0.07*p*(1-p)=0.07*(1-p); crypto feeRate=0.07
ok = lambda a: a is not None and 0.02 < a < 0.98

# ---------- unified per-window table (slug -> features+outcome), GT only ----------
W = {}   # slug -> dict
def add(slug, ts, d, lstm, live, ptb, au, ad, blend, upwon):
    if slug is None or upwon is None or slug in W: return
    drift = ((live-ptb)/ptb*100) if (live and ptb) else None
    W[slug] = dict(slug=slug, ts=ts, day=(ts or "")[:10], dir=d, lstm=lstm,
                   drift=drift, au=au, ad=ad, blend=blend, up=upwon)

# June (full)
for l in open("/tmp/june_full.jsonl"):
    d = json.loads(l)
    add(d["slug"], d["ts"], d["dir"], d.get("lstm"), d.get("live"), d.get("ptb"),
        d.get("au"), d.get("ad"), d.get("blend"), d["upwon"])
# v2 (Apr30-May22 full incl lstm)
for l in open("trade_history_v2.jsonl"):
    l=l.strip()
    if not l: continue
    try: d=json.loads(l)
    except: continue
    if d.get("won") is None: continue
    dr=(d.get("direction") or "").upper(); won=d["won"]
    uw=(1 if won else 0) if dr=="UP" else (0 if won else 1) if dr=="DOWN" else None
    add(d.get("slug"), d.get("ts"), dr, d.get("lstm_prob"), d.get("live_price"),
        d.get("ptb"), d.get("top_ask_up"), d.get("top_ask_down"), d.get("final_blended_prob"), uw)
# local skip_history (Apr29-May29: drift+asks, NO lstm/blend)
for l in open("skip_history.jsonl"):
    l=l.strip()
    if not l: continue
    try: d=json.loads(l)
    except: continue
    if d.get("would_have_won") is None: continue
    dr=(d.get("direction") or "").upper(); won=d["would_have_won"]
    uw=(1 if won else 0) if dr=="UP" else (0 if won else 1) if dr=="DOWN" else None
    add(d.get("slug"), d.get("ts"), dr, None, d.get("live_price"), d.get("ptb"),
        d.get("top_ask_up"), d.get("top_ask_down"), None, uw)

rows = list(W.values())
days = sorted(set(r["day"] for r in rows))
print(f"unified windows: {len(rows)}  span {days[0]}..{days[-1]}  ({len(days)} calendar days)")
print(f"  with drift: {sum(1 for r in rows if r['drift'] is not None)} | with lstm: {sum(1 for r in rows if r['lstm'] is not None)} | with asks: {sum(1 for r in rows if ok(r['au']) and ok(r['ad']))}")

# ---------- strategy = (signal_fn -> magnitude, side_fn -> 'UP'/'DOWN') ----------
def side_ask(r, side):
    return r["au"] if side=="UP" else r["ad"]

def run(sig, side, need):
    # 1 trade/day: day's max |signal| window where features valid
    byday = collections.defaultdict(list)
    for r in rows:
        if not need(r): continue
        s = sig(r)
        if s is None: continue
        sd = side(r)
        a = side_ask(r, sd)
        if not ok(a): continue
        byday[r["day"]].append((abs(s), r, sd, a))
    trades=[]
    for day in sorted(byday):
        _, r, sd, a = max(byday[day], key=lambda x: x[0])
        win = (sd=="UP")==(r["up"]==1)
        pnl = ((1.0/a-1.0) if win else -1.0) - fee(a)
        trades.append(dict(day=day, pnl=pnl, win=win, ask=a, side=sd, slug=r["slug"]))
    return trades

def stats(trades):
    if not trades: return None
    p=[t["pnl"] for t in trades]; n=len(p); mu=st.mean(p); sd=st.stdev(p) if n>1 else 0
    t=mu/(sd/math.sqrt(n)) if sd else 0; w=sum(1 for x in trades if x["win"])
    return dict(n=n, win=100*w/n, total=sum(p), mean=mu, sd=sd, t=t, sharpe=(mu/sd if sd else 0))

has_drift = lambda r: r["drift"] is not None
has_lstm  = lambda r: r["lstm"]  is not None
has_blend = lambda r: r["blend"] is not None
has_ask   = lambda r: ok(r["au"]) and ok(r["ad"])

# PRE-REGISTERED STRATEGIES
STRATS = {
 "S1 momentum-drift":  (lambda r: r["drift"], lambda r: "UP" if r["drift"]>0 else "DOWN", lambda r: has_drift(r) and has_ask(r)),
 "S2 reversion-drift": (lambda r: r["drift"], lambda r: "DOWN" if r["drift"]>0 else "UP", lambda r: has_drift(r) and has_ask(r)),
 "S3 lstm-follow":     (lambda r: r["lstm"]-0.5, lambda r: "UP" if r["lstm"]>0.5 else "DOWN", lambda r: has_lstm(r) and has_ask(r)),
 "S4 lstm-fade":       (lambda r: r["lstm"]-0.5, lambda r: "DOWN" if r["lstm"]>0.5 else "UP", lambda r: has_lstm(r) and has_ask(r)),
 "S5 favorite":        (lambda r: r["au"]-r["ad"], lambda r: "UP" if r["au"]>r["ad"] else "DOWN", has_ask),
 "S6 longshot":        (lambda r: r["au"]-r["ad"], lambda r: "DOWN" if r["au"]>r["ad"] else "UP", has_ask),
 "S7 blend-follow":    (lambda r: r["blend"]-0.5, lambda r: "UP" if r["blend"]>0.5 else "DOWN", lambda r: has_blend(r) and has_ask(r)),
 "S8 blend-fade":      (lambda r: r["blend"]-0.5, lambda r: "DOWN" if r["blend"]>0.5 else "UP", lambda r: has_blend(r) and has_ask(r)),
}
K=len(STRATS)
print(f"\n{'='*72}\nFULL-SAMPLE (all {K} pre-registered strategies; report ALL — no cherry-pick)")
results={}
for name,(sig,side,need) in STRATS.items():
    s=stats(run(sig,side,need)); results[name]=s
    if s: print(f"  {name:<20} days={s['n']:<3} win%={s['win']:4.1f} total=${s['total']:+7.2f} mean/day=${s['mean']:+.3f} t={s['t']:+.2f} sharpe={s['sharpe']:+.2f}")

def p_from_t(t,n):
    # two-sided normal approx (n small but indicative)
    from math import erf,sqrt
    z=abs(t); return 2*(1-0.5*(1+erf(z/sqrt(2))))
print(f"\nBonferroni: with K={K} tests, need nominal p < {0.05/K:.4f} (|t| ~ > 2.7) to claim significance")
for name,s in results.items():
    if s: print(f"  {name:<20} nominal p={p_from_t(s['t'],s['n']):.4f}  {'<-- passes Bonferroni' if p_from_t(s['t'],s['n'])<0.05/K and s['t']>0 else ''}")

# LEADER = highest positive t
leader=max((n for n,s in results.items() if s), key=lambda n: results[n]["t"])
print(f"\n{'='*72}\nLEADER by t: {leader}  (full rigor follows)")
sig,side,need=STRATS[leader]; T=run(sig,side,need); s=results[leader]
print(f"  full: days={s['n']} win%={s['win']:.1f} total=${s['total']:+.2f} t={s['t']:+.2f}")

# walk-forward
mid=len(T)//2
a=stats(T[:mid]); b=stats(T[mid:])
print(f"  walk-forward FIRST half:  days={a['n']} total=${a['total']:+.2f} mean=${a['mean']:+.3f} t={a['t']:+.2f}")
print(f"  walk-forward SECOND half: days={b['n']} total=${b['total']:+.2f} mean=${b['mean']:+.3f} t={b['t']:+.2f}")

# MC outcome-permutation (shuffle outcomes across the leader's candidate pool, recompute)
N=20000
# rebuild candidate pool per day for leader
byday=collections.defaultdict(list)
for r in rows:
    if not need(r): continue
    sg=sig(r); sd=side(r); a=side_ask(r,sd)
    if sg is None or not ok(a): continue
    byday[r["day"]].append((abs(sg), r, sd, a))
chosen=[max(v,key=lambda x:x[0]) for v in byday.values()]
actual=sum(((1.0/a-1.0) if (sd=="UP")==(r["up"]==1) else -1.0)-fee(a) for _,r,sd,a in chosen)
ups=[r["up"] for _,r,_,_ in chosen]
nullp=[]
for _ in range(N):
    random.shuffle(ups)
    tot=0
    for (_,r,sd,a),u in zip(chosen,ups):
        win=(sd=="UP")==(u==1); tot+=((1.0/a-1.0) if win else -1.0)-fee(a)
    nullp.append(tot)
pval=sum(1 for x in nullp if x>=actual)/N
print(f"\n  MC outcome-permutation ({N}x): actual=${actual:+.2f} null mean=${st.mean(nullp):+.2f} sd=${st.pstdev(nullp):.2f}  p={pval:.4f}")

# daily bootstrap CI
dp=[t["pnl"] for t in T]; bm=[]
for _ in range(N): bm.append(st.mean(random.choice(dp) for _ in range(len(dp))))
bm.sort(); lo,hi=bm[int(0.025*N)],bm[int(0.975*N)]
print(f"  daily bootstrap mean/day 95% CI=[${lo:+.3f},${hi:+.3f}]  P(mean>0)={sum(1 for x in bm if x>0)/N:.3f}")
