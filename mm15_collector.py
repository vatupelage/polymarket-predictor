"""Standalone MM data collector for a FIXED basket of markets (mm15_markets.json).
Logs order books (all tokens) + executed tape (per condition_id) to ~/arb/mmcollect/,
keyed by token_id/cond + ts so a reward-aware MM backtest can join them. Reuses the
proven client primitives: get_full_book(token_id) and get_tape(condition_id).

Usage:  python mm15_collector.py [duration_seconds]   (0 / omitted = run forever)
"""
import json, time, os, datetime, sys

HOME = os.path.expanduser("~")
ARB = os.path.join(HOME, "arb")
sys.path.insert(0, ARB)
os.chdir(ARB)
from live_trader.config import load_config
from live_trader.polymarket import PolymarketBotClient

OUT = os.path.join(ARB, "mmcollect")
os.makedirs(OUT, exist_ok=True)
BOOK = os.path.join(OUT, "mm15_book.jsonl")
TAPE = os.path.join(OUT, "mm15_tape.jsonl")
ERR = os.path.join(OUT, "mm15_collector_errors.jsonl")

DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 0
BOOK_CADENCE = 5.0          # book snapshot every 5s
TAPE_EVERY = 3              # tape every 3rd cycle (~15s)

markets = json.load(open(os.path.join(ARB, "mm15_markets.json")))
cfg = load_config(dotenv_path=os.path.join(ARB, ".env"))
c = PolymarketBotClient(cfg)

if not hasattr(c, "get_full_book"):
    print("FATAL: client has no get_full_book"); raise SystemExit(1)

def now():
    return datetime.datetime.now().isoformat(timespec="milliseconds")

def append(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")

# flatten to (slug, cond, token_id, outcome)
tokens = []
for m in markets:
    outs = m.get("outs") or []
    for i, tid in enumerate(m["toks"]):
        tokens.append((m["slug"], m["cond"], tid, outs[i] if i < len(outs) else str(i)))
conds = [(m["slug"], m["cond"]) for m in markets]

start = time.time(); cyc = 0; seen = set()
nbook = ntape = nbookerr = ntapeerr = nempty = 0
print(f"[{now()}] START {len(markets)} markets / {len(tokens)} tokens / {len(conds)} conds "
      f"| dur={DURATION or 'inf'}s book@{BOOK_CADENCE}s tape@{BOOK_CADENCE*TAPE_EVERY}s -> {OUT}", flush=True)

while True:
    if DURATION and time.time() - start >= DURATION:
        break
    t0 = time.time(); cyc += 1
    for slug, cond, tid, outcome in tokens:
        try:
            book = c.get_full_book(tid)
            if not book:
                nempty += 1; continue
            bids = book.get("bids") or []; asks = book.get("asks") or []
            bb = bids[0][0] if bids else None
            ba = asks[0][0] if asks else None
            mid = ((bb + ba) / 2.0) if (bb is not None and ba is not None) else None
            spread = (ba - bb) if (bb is not None and ba is not None) else None
            append(BOOK, {"ts": now(), "slug": slug, "cond": cond, "token_id": tid,
                          "outcome": outcome, "best_bid": bb, "best_ask": ba, "mid": mid,
                          "spread": spread, "bids": bids[:5], "asks": asks[:5],
                          "bid_depth10": sum(s for _, s in bids[:10]),
                          "ask_depth10": sum(s for _, s in asks[:10]),
                          "hash": book.get("hash")})
            nbook += 1
        except Exception as e:
            nbookerr += 1; append(ERR, {"ts": now(), "book_err": f"{type(e).__name__}: {e}", "tid": tid})
    if cyc % TAPE_EVERY == 1:
        for slug, cond in conds:
            try:
                tape = c.get_tape(cond)
                if not tape:
                    continue
                for tr in tape:
                    key = (tr.get("tx"), tr.get("asset"), tr.get("ts"), tr.get("price"), tr.get("size"))
                    if key in seen:
                        continue
                    seen.add(key)
                    append(TAPE, {"ts": now(), "slug": slug, "cond": cond, "trade": tr})
                    ntape += 1
            except Exception as e:
                ntapeerr += 1; append(ERR, {"ts": now(), "tape_err": f"{type(e).__name__}: {e}", "cond": cond})
    if cyc % 6 == 0:
        print(f"[{now()}] cyc={cyc} book={nbook} tape={ntape} | book_err={nbookerr} "
              f"tape_err={ntapeerr} empty={nempty}", flush=True)
    time.sleep(max(0.0, BOOK_CADENCE - (time.time() - t0)))

print(f"[{now()}] DONE cyc={cyc} book={nbook} tape={ntape} | book_err={nbookerr} "
      f"tape_err={ntapeerr} empty={nempty}", flush=True)
