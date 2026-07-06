"""S5 STOP-LOSS backtest on full-window PM book paths (mm_book.jsonl).

Question: after an S5-style entry (buy the favorite when its ask is in the
(0.58, 0.75] band), does cutting the position early when the PM book moves
against us beat holding to resolution — net of the dynamic taker fee on BOTH
the entry and the exit, and net of the bid/ask spread we cross to sell?

Data: mm_book.jsonl — 82 full windows, each with UP & DOWN top-of-book
(best_bid/best_ask/mid) sampled from ~299s to 0s before close. We use the
REAL best_bid as the price we'd sell into (no approximation).

CAVEAT: these windows have the full PM path but NOT the Bitstamp ob>=0.85
signal that the live S5 gate also requires. So this measures the stop-loss
*economics on the entry-band population* — the right question for "do
reversals persist (stop helps) or revert (stop hurts)". The bitstamp filter
changes the base win-rate, not the post-entry reversal dynamics.
"""
import json, collections

def fee(p):                      # dynamic taker fee, ~3.5% of notional at p=0.5
    return 0.07 * p * (1 - p)

ENTRY_BAND = (0.58, 0.75)        # S5 entry-band: buy favorite when its ask is here

# ---- load: W[slug][token] -> {secs_to_close: snapshot} -----------------------
W = collections.defaultdict(lambda: {"UP": {}, "DOWN": {}})
for ln in open("mm_book.jsonl"):
    try: r = json.loads(ln)
    except Exception: continue
    s = r.get("slug"); tok = r.get("token"); secs = r.get("secs_to_close")
    if not s or tok not in ("UP", "DOWN") or secs is None: continue
    W[s][tok][secs] = r          # last write per secs wins (fine: ~1 sample/secs)

def outcome(up_snaps):
    """winner from the final (smallest secs_to_close) UP mid."""
    near = [s for s in up_snaps if up_snaps[s].get("mid") is not None and s <= 20]
    if not near: return None
    final = up_snaps[min(near)]                     # closest to close
    return "UP" if final["mid"] > 0.5 else "DOWN"

# ---- simulate one window: find S5 entry, then hold vs each stop rule ----------
def simulate(slug):
    up = W[slug]["UP"]; dn = W[slug]["DOWN"]
    if not up or not dn: return None
    win = outcome(up)
    if win is None: return None
    secs_desc = sorted(set(up) & set(dn), reverse=True)   # aligned timeline
    # entry: first snapshot where the FAVORITE's ask is in the band
    entry = None
    for sc in secs_desc:
        u, d = up[sc], dn[sc]
        au, ad = u.get("best_ask"), d.get("best_ask")
        mu, md = u.get("mid"), d.get("mid")
        if au is None or ad is None or mu is None or md is None: continue
        fav, fav_ask = ("UP", au) if mu >= md else ("DOWN", ad)
        if ENTRY_BAND[0] < fav_ask <= ENTRY_BAND[1]:
            entry = (sc, fav, fav_ask); break
    if entry is None: return None
    esecs, side, eask = entry
    won = (side == win)
    snaps = up if side == "UP" else dn
    path = [sc for sc in secs_desc if sc <= esecs]        # post-entry, incl entry

    # hold-to-resolution: entry fee only (settlement isn't a taker trade)
    hold_pnl = ((1 - eask) if won else (-eask)) - fee(eask)

    def stop_at(level, max_secs=10**9):
        """exit (sell into best_bid) the first time our bid <= level AND we're
        within max_secs of close (time-gate); else hold."""
        for sc in path:
            if sc > max_secs: continue
            bid = snaps[sc].get("best_bid")
            if bid is None: continue
            if bid <= level:
                pnl = (bid - eask) - fee(eask) - fee(bid)   # 2 fees + spread crossed
                return pnl, True, won                       # stopped; did it go on to lose?
        return hold_pnl, False, won
    return {"slug": slug, "side": side, "eask": eask, "esecs": esecs,
            "won": won, "hold": hold_pnl, "stop_at": stop_at}

sims = [s for s in (simulate(sl) for sl in W) if s]
n = len(sims)
hold_total = sum(s["hold"] for s in sims)
hold_win = sum(1 for s in sims if s["won"])
print(f"windows with a valid S5 band-entry: {n}")
print(f"BASELINE (hold to resolution): win={hold_win}/{n} ({hold_win/n*100:.1f}%)  "
      f"net={hold_total/n*100:+.2f}%/trade  total={hold_total*100:+.1f}% of $1/trade\n")

print("============ STOP-LOSS by ABSOLUTE bid level (sell when our bid <= L) ============")
print(f"{'level':>6} {'net/trade':>10} {'total':>8} {'#stopped':>9} {'saved':>6} {'cost':>6}  read")
for L in (0.50, 0.45, 0.40, 0.35, 0.30):
    tot = 0.0; stopped = 0; saved = 0; cost = 0
    for s in sims:
        pnl, did_stop, won = s["stop_at"](L)
        tot += pnl
        if did_stop:
            stopped += 1
            if not won: saved += 1     # stopped a trade that would've lost -> good
            else:       cost  += 1     # stopped a trade that would've won  -> bad
    delta = (tot - hold_total) * 100
    print(f"{L:>6.2f} {tot/n*100:>+9.2f}% {tot*100:>+7.1f}% {stopped:>9d} {saved:>6d} {cost:>6d}  "
          f"{'BETTER' if delta>0 else 'worse'} than hold by {delta:+.1f}%")

print("\n============ STOP-LOSS by DRAWDOWN from entry (sell when bid <= eask - D) ============")
print(f"{'drawdn':>6} {'net/trade':>10} {'total':>8} {'#stopped':>9} {'saved':>6} {'cost':>6}  read")
for D in (0.10, 0.15, 0.20, 0.25):
    tot = 0.0; stopped = 0; saved = 0; cost = 0
    for s in sims:
        pnl, did_stop, won = s["stop_at"](s["eask"] - D)
        tot += pnl
        if did_stop:
            stopped += 1
            saved += int(not won); cost += int(won)
    delta = (tot - hold_total) * 100
    print(f"{D:>6.2f} {tot/n*100:>+9.2f}% {tot*100:>+7.1f}% {stopped:>9d} {saved:>6d} {cost:>6d}  "
          f"{'BETTER' if delta>0 else 'worse'} than hold by {delta:+.1f}%")

print("\n============ TIME-GATED STOP (only cut when bid<=L AND late in window) ============")
print("steelman: early dips revert, but a late adverse move has no time to recover")
print(f"{'level':>6} {'<=secs':>7} {'net/trade':>10} {'total':>8} {'#stop':>6} {'saved':>6} {'cost':>6}  read")
for L in (0.50, 0.40):
    for T in (90, 60, 30):
        tot = 0.0; stopped = 0; saved = 0; cost = 0
        for s in sims:
            pnl, did_stop, won = s["stop_at"](L, max_secs=T)
            tot += pnl
            if did_stop:
                stopped += 1; saved += int(not won); cost += int(won)
        delta = (tot - hold_total) * 100
        print(f"{L:>6.2f} {T:>7d} {tot/n*100:>+9.2f}% {tot*100:>+7.1f}% {stopped:>6d} {saved:>6d} {cost:>6d}  "
              f"{'BETTER' if delta>0 else 'worse'} by {delta:+.1f}%")

print("\nread: 'saved' = stops that avoided a full loss (good); 'cost' = stops on")
print("trades that would have recovered to win (bad). Stop-loss beats hold only if")
print("the level/drawdown column shows BETTER net than the baseline above.")
