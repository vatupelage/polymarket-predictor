"""
Does PM ORDER-BOOK DEPTH predict the outcome? Rigorous per-window test on mm_book.jsonl.
Features: microprice tilt, depth imbalance, spread, book-skew. Label = terminal resolution
(every window, no dropping). Strategy = bet the side the feature favors, at its ask, net
dynamic fee. Monte-Carlo outcome-permutation null + Bonferroni for #features.
CAVEAT: book depth only spans ~6.8h (June 1) = ONE regime; can't do 1-trade/day, can't
generalize across regimes (Markov caveat). This only asks: ANY signal in the book at all?
"""
import json, math, random, collections, statistics as st
random.seed(11)
def fee(p): return 0.07*(1-p)   # taker fee in $ per $1-stake BUY = 0.07*(1-p); crypto feeRate=0.07
ok=lambda a: a is not None and 0.02<a<0.98

bk=collections.defaultdict(dict)   # (slug,ts) -> token -> snapshot
lastmid={}
for l in open("mm_book.jsonl"):
    try: d=json.loads(l)
    except: continue
    s,t,ts,s2c=d.get("slug"),d.get("token"),d.get("ts"),d.get("secs_to_close")
    if None in (s,t,ts): continue
    bk[(s,ts)][t]=dict(bb=d.get("best_bid"),ba=d.get("best_ask"),mid=d.get("mid"),
                       mp=d.get("microprice"),bd=d.get("bid_depth_2c"),ad=d.get("ask_depth_2c"),s2c=s2c)
    k=(s,t)
    if s2c is not None and (k not in lastmid or s2c<lastmid[k][0]):
        lastmid[k]=(s2c,d.get("mid"))

# terminal winner (UP up_mid>0.5 at last snapshot, must be near close & converged)
winner={}
for (s,t),(s2c,mid) in lastmid.items():
    if t=="UP" and mid is not None and s2c is not None and s2c<=45 and (mid>0.8 or mid<0.2):
        winner[s]=1 if mid>0.5 else 0

TARGET=90  # decision point: secs_to_close ~90
def snap_near(slug):
    best=None
    for (s,ts),toks in bk.items():
        if s!=slug or "UP" not in toks or "DOWN" not in toks: continue
        s2c=toks["UP"]["s2c"]
        if s2c is None: continue
        if best is None or abs(s2c-TARGET)<abs(best[0]-TARGET): best=(s2c,toks)
    return best

rows=[]
for slug in winner:
    nb=snap_near(slug)
    if nb is None or abs(nb[0]-TARGET)>30: continue
    toks=nb[1]; U,D=toks["UP"],toks["DOWN"]
    if None in (U["mid"],U["mp"],U["bd"],U["ad"],U["ba"],D["ba"]): continue
    imb=(U["bd"]-U["ad"])/(U["bd"]+U["ad"]) if (U["bd"]+U["ad"])>0 else 0
    rows.append(dict(slug=slug, y=winner[slug], mid=U["mid"], mptilt=U["mp"]-U["mid"],
                     imb=imb, spread=(U["ba"]-U["bb"]) if U["bb"] else None,
                     au=U["ba"], ad=D["ba"]))
print(f"windows with book@~{TARGET}s + terminal label: {len(rows)}  (ALL from one ~6.8h session)")
base=st.mean(r["y"] for r in rows)
print(f"base rate UP={base:.3f}; mid Brier={st.mean((r['mid']-r['y'])**2 for r in rows):.3f} (0.25=coinflip)")

def strat(name, side_fn, need=lambda r:True):
    tr=[]
    for r in rows:
        if not need(r): continue
        sd=side_fn(r)
        if sd is None: continue
        a=r["au"] if sd=="UP" else r["ad"]
        if not ok(a): continue
        win=(sd=="UP")==(r["y"]==1)
        tr.append((((1.0/a-1.0) if win else -1.0)-fee(a), win, sd, a, r))
    if not tr: print(f"  {name}: no trades"); return None
    p=[x[0] for x in tr]; n=len(p); mu=st.mean(p); sd_=st.stdev(p) if n>1 else 0
    t=mu/(sd_/math.sqrt(n)) if sd_ else 0; w=sum(1 for x in tr if x[1])
    # permutation null: shuffle outcomes
    N=20000; ys=[x[4]["y"] for x in tr]; act=sum(p); nz=[]
    for _ in range(N):
        random.shuffle(ys); tot=0
        for x,yy in zip(tr,ys):
            sd2=x[2]; a=x[3]; win=(sd2=="UP")==(yy==1); tot+=((1.0/a-1.0) if win else -1.0)-fee(a)
        nz.append(tot)
    pv=sum(1 for z in nz if z>=act)/N
    print(f"  {name:<22} n={n:<3} win%={100*w/n:4.1f} total=${act:+6.2f} mean=${mu:+.3f} t={t:+.2f} | perm-p={pv:.4f}")
    return t

print("\nPRE-REGISTERED book strategies (bet the side the feature favors):")
ts=[]
ts.append(strat("microprice-tilt",  lambda r: "UP" if r["mptilt"]>0 else "DOWN"))
ts.append(strat("depth-imbalance",  lambda r: "UP" if r["imb"]>0 else "DOWN"))
ts.append(strat("microprice-strong",lambda r: ("UP" if r["mptilt"]>0 else "DOWN") if abs(r["mptilt"])>0.01 else None))
ts.append(strat("imbalance-strong", lambda r: ("UP" if r["imb"]>0 else "DOWN") if abs(r["imb"])>0.3 else None))
ts.append(strat("follow-mid(fav)",  lambda r: "UP" if r["mid"]>0.5 else "DOWN"))
ts.append(strat("fade-mid(longshot)",lambda r: "DOWN" if r["mid"]>0.5 else "UP"))
K=len([t for t in ts if t is not None])
print(f"\nBonferroni: K={K} tests -> need perm-p < {0.05/K:.4f} for significance")

# Also: correlation of each feature with residual (outcome - mid), Bonferroni
def corr(a,b):
    n=len(a); ma=st.mean(a); mb=st.mean(b)
    cov=sum((x-ma)*(y-mb) for x,y in zip(a,b))/n
    sa=st.pstdev(a); sb=st.pstdev(b)
    return cov/(sa*sb) if sa*sb>0 else 0
resid=[r["y"]-r["mid"] for r in rows]
print("\nfeature corr with residual (outcome - mid); ~0 = no info beyond mid:")
for f in ("mptilt","imb"):
    print(f"  {f:<10} corr={corr([r[f] for r in rows],resid):+.3f}")
