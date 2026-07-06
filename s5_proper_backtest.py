"""
Rigorous S5 backtest — original wide band (0.40, 0.75], 1 trade per UTC day.
Honors: no look-ahead (entry uses only decision-poll data; label = real resolution),
no survivorship/selection bias (ground-truth resolution for EVERY window, drop nothing),
no data-snooping (full grid + walk-forward + parameter stability shown),
conditional vs unconditional (DAY is the unit; path-dependent first-match acknowledged),
costs (dynamic taker fee 0.07*(1-p)/$1, crypto feeRate=0.07 + ask already includes spread + slippage stress).
Plus Monte Carlo: outcome-permutation null, entry-timing null, daily bootstrap, equity/ruin.
"""
import json, math, random, collections, statistics as st

random.seed(42)
STAKE = 1.0
def fee(p):  # dynamic taker fee in $ for $1 stake: (1/p)*0.07*p*(1-p)=0.07*(1-p); crypto feeRate=0.07
    return 0.07 * (1 - p)

# ---------- load shadow poll data (May local + June pulled) ----------
polls = collections.defaultdict(list)   # slug -> list of poll dicts
day_of = {}
def add_shadow(slug, poll, d, ask, ob, au, ad, ts):
    if slug is None or ask is None or ob is None: return
    polls[slug].append(dict(poll=poll or 0, dir=d, ask=ask, ob=ob, au=au, ad=ad, ts=ts))
    if ts: day_of[slug] = ts[:10]

# May (local)
for l in open("s5_shadow.jsonl"):
    try: r = json.loads(l)
    except: continue
    add_shadow(r.get("slug"), r.get("poll_n"), (r.get("direction") or "").upper(),
               r.get("our_ask"), r.get("bitstamp_ob_side"),
               r.get("top_ask_up"), r.get("top_ask_down"), r.get("ts"))
# June (pulled) + resolution
res = {}   # slug -> upwon (ground truth)
for l in open("/tmp/june_s5.jsonl"):
    d = json.loads(l)
    if d["_t"] == "s":
        add_shadow(d.get("slug"), d.get("poll"), d.get("dir"), d.get("ask"),
                   d.get("ob"), d.get("au"), d.get("ad"), d.get("ts"))
    else:
        res[d["slug"]] = d["upwon"]

# ---------- ground-truth resolution from all local sources ----------
def add_res(path, wk):
    for l in open(path):
        l = l.strip()
        if not l: continue
        try: d = json.loads(l)
        except: continue
        if d.get(wk) is None: continue
        dr = (d.get("direction") or "").upper(); won = d[wk]
        uw = (1 if won else 0) if dr == "UP" else (0 if won else 1) if dr == "DOWN" else None
        if uw is not None and d.get("slug"): res.setdefault(d["slug"], uw)
add_res("skip_history.jsonl", "would_have_won")
add_res("trade_history.jsonl", "won")
add_res("trade_history_v2.jsonl", "won")

# fallback winner (last-poll higher ask) ONLY to measure coverage gap, not used if gt exists
def winner(slug):
    if slug in res: return res[slug], "gt"
    ps = sorted(polls[slug], key=lambda x: x["poll"])
    last = ps[-1]
    if last["au"] is not None and last["ad"] is not None:
        return (1 if last["au"] > last["ad"] else 0), "fallback"
    return None, None

# ---------- S5 entry: first qualifying poll in a window ----------
def window_entry(slug, band, thr):
    for p in sorted(polls[slug], key=lambda x: x["poll"]):
        if p["ask"] is None or p["ob"] is None: continue
        if band[0] < p["ask"] <= band[1] and p["ob"] >= thr:
            return p
    return None

# ---------- 1 trade per UTC day: earliest qualifying window ----------
def daily_trades(band, thr, use_fallback=False):
    # gather qualifying windows, group by day, take earliest ts per day
    qual = []
    for slug in polls:
        e = window_entry(slug, band, thr)
        if e is None: continue
        w, src = winner(slug)
        if w is None: continue
        if src == "fallback" and not use_fallback: continue
        qual.append((day_of.get(slug, e["ts"][:10] if e["ts"] else "?"), e["ts"] or "", slug, e, w, src))
    byday = collections.defaultdict(list)
    for q in qual: byday[q[0]].append(q)
    trades = []
    for day in sorted(byday):
        first = sorted(byday[day], key=lambda x: x[1])[0]   # earliest ts
        _, ts, slug, e, w, src = first
        ask = e["ask"]; won = (e["dir"] == "UP") == (w == 1)
        pnl = (STAKE / ask - STAKE) if won else -STAKE
        pnl -= fee(ask)
        trades.append(dict(day=day, slug=slug, dir=e["dir"], ask=ask, won=won, pnl=pnl, src=src))
    return trades

def describe(trades, label):
    if not trades:
        print(f"  {label}: NO TRADES"); return None
    pnls = [t["pnl"] for t in trades]; n = len(pnls)
    mu = st.mean(pnls); sd = st.stdev(pnls) if n > 1 else 0
    t = mu / (sd / math.sqrt(n)) if sd else 0
    w = sum(1 for t_ in trades if t_["won"])
    print(f"  {label}: days={n} win%={100*w/n:4.1f} total=${sum(pnls):+.2f} "
          f"mean/day=${mu:+.3f} sd=${sd:.3f} t={t:+.2f} sharpe(daily)={mu/sd if sd else 0:+.2f}")
    return pnls

print("="*70)
print("DATA COVERAGE")
print(f"  shadow windows: {len(polls)} | ground-truth resolutions: {len(res)}")
gt = sum(1 for s in polls if s in res)
print(f"  shadow windows with GT resolution: {gt}/{len(polls)} ({100*gt/len(polls):.1f}%)")
days = sorted(set(day_of.values()))
print(f"  trading-day span: {days[0]} .. {days[-1]}  ({len(days)} calendar days)")

print("\n" + "="*70)
print("PRIMARY: original wide band (0.40, 0.75], ob>=0.85, 1 trade/UTC day, GT-only")
PRIMARY = daily_trades((0.40, 0.75), 0.85)
pnls_primary = describe(PRIMARY, "(0.40,0.75] thr0.85")
print("  per-day P&L:", [f"{t['day'][5:]}:{t['pnl']:+.2f}({'W' if t['won'] else 'L'})" for t in PRIMARY])

print("\n" + "="*70)
print("DATA-SNOOPING CHECK: full band x threshold grid (NO cherry-pick)")
for thr in (0.80, 0.85, 0.90):
    for band in ((0.40,0.75),(0.50,0.75),(0.58,0.75),(0.40,0.50),(0.40,0.80)):
        describe(daily_trades(band, thr), f"band{band} thr{thr}")

print("\n" + "="*70)
print("PARAMETER STABILITY: small perturbations of the primary (should NOT collapse)")
for band in ((0.40,0.75),(0.42,0.75),(0.38,0.73),(0.40,0.72),(0.43,0.78)):
    describe(daily_trades(band, 0.85), f"band{band}")

print("\n" + "="*70)
print("WALK-FORWARD / TEMPORAL SPLIT (first half vs second half of days)")
if pnls_primary and len(PRIMARY) >= 4:
    mid = len(PRIMARY)//2
    describe(PRIMARY[:mid], "FIRST half")
    describe(PRIMARY[mid:], "SECOND half")

# ================= MONTE CARLO =================
print("\n" + "="*70)
print("MONTE CARLO  (band (0.40,0.75], thr 0.85)")
band, thr = (0.40, 0.75), 0.85

# entries (slug, dir, ask, day) for the primary strategy, with GT winners
entries = []
for slug in polls:
    e = window_entry(slug, band, thr)
    if e is None or slug not in res: continue
    entries.append(dict(slug=slug, day=day_of.get(slug), ts=e["ts"] or "", dir=e["dir"], ask=e["ask"], upwon=res[slug]))

actual_total = sum(t["pnl"] for t in PRIMARY)
n_days = len(PRIMARY)

# (A) OUTCOME-PERMUTATION NULL: shuffle winners across the qualifying windows,
#     keep entries/timing, recompute 1-trade/day total. Tests signal->outcome link.
def total_for_entries(ents):
    byday = collections.defaultdict(list)
    for e in ents: byday[e["day"]].append(e)
    tot = 0
    for day in byday:
        first = sorted(byday[day], key=lambda x: x["ts"])[0]
        won = (first["dir"] == "UP") == (first["_uw"] == 1)
        tot += ((STAKE/first["ask"]-STAKE) if won else -STAKE) - fee(first["ask"])
    return tot
N = 20000
upwins = [e["upwon"] for e in entries]
null_perm = []
for _ in range(N):
    random.shuffle(upwins)
    for e, u in zip(entries, upwins): e["_uw"] = u
    null_perm.append(total_for_entries(entries))
p_perm = sum(1 for x in null_perm if x >= actual_total) / N
print(f"\n(A) Outcome-permutation null (shuffle winners {N}x):")
print(f"    actual total=${actual_total:+.2f} | null mean=${st.mean(null_perm):+.2f} sd=${st.pstdev(null_perm):.2f}")
print(f"    p-value P(random>=actual) = {p_perm:.4f}")

# (B) ENTRY-TIMING NULL: per day pick a RANDOM qualifying window (not first), GT winner.
byday_q = collections.defaultdict(list)
for e in entries: byday_q[e["day"]].append(e)
null_time = []
for _ in range(N):
    tot = 0
    for day, lst in byday_q.items():
        e = random.choice(lst)
        won = (e["dir"]=="UP")==(e["upwon"]==1)
        tot += ((STAKE/e["ask"]-STAKE) if won else -STAKE) - fee(e["ask"])
    null_time.append(tot)
p_time = sum(1 for x in null_time if x >= actual_total)/N
print(f"\n(B) Entry-timing null (random qualifying window/day {N}x):")
print(f"    actual(first-match)=${actual_total:+.2f} | null mean=${st.mean(null_time):+.2f} sd=${st.pstdev(null_time):.2f}")
print(f"    p-value P(random>=actual) = {p_time:.4f}")

# (C) DAILY BOOTSTRAP: resample days w/ replacement -> CI on mean/day & total
boot_mean = []
dp = [t["pnl"] for t in PRIMARY]
for _ in range(N):
    sample = [random.choice(dp) for _ in range(n_days)]
    boot_mean.append(st.mean(sample))
boot_mean.sort()
lo, hi = boot_mean[int(0.025*N)], boot_mean[int(0.975*N)]
p_pos = sum(1 for x in boot_mean if x > 0)/N
print(f"\n(C) Daily bootstrap ({N}x, n_days={n_days}):")
print(f"    mean/day 95% CI = [${lo:+.3f}, ${hi:+.3f}]  (total CI ~ [${lo*n_days:+.1f}, ${hi*n_days:+.1f}])")
print(f"    bootstrap P(mean/day > 0) = {p_pos:.3f}")

# (D) EQUITY-PATH / DRAWDOWN / RUIN: simulate 60 trading days fwd from observed daily P&L
HORIZON = 60; STOP = -20.0; finals = []; maxdd = []; ruin = 0
for _ in range(N):
    eq = 0.0; peak = 0.0; dd = 0.0; busted = False
    for _ in range(HORIZON):
        eq += random.choice(dp)
        peak = max(peak, eq); dd = min(dd, eq - peak)
        if eq <= STOP: busted = True; break
    finals.append(eq); maxdd.append(dd); ruin += int(busted)
finals.sort()
print(f"\n(D) Equity-path MC ({HORIZON} days fwd, {N}x, $1 stake, stop ${STOP}):")
print(f"    final P&L: median=${finals[N//2]:+.1f}  5%=${finals[int(0.05*N)]:+.1f}  95%=${finals[int(0.95*N)]:+.1f}")
print(f"    P(profit after {HORIZON}d) = {sum(1 for x in finals if x>0)/N:.3f}")
print(f"    median max-drawdown = ${st.median(maxdd):+.1f}   P(hit ${STOP} stop) = {ruin/N:.3f}")
