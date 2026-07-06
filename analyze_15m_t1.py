"""Go/no-go backtest for the 15m 1-min-in (s2c=840) paper logs.

For each market: realized win% vs avg entry ask (= break-even win%), $/trade,
t-stat, and a sweep over confidence/ask-cap gates. Run after ~1-2 weeks of
dry-run logging. Usage: python analyze_15m_t1.py [paper_glob]"""
import glob
import json
import math
import statistics
import sys


def _ask(r):
    if r.get("our_ask"):
        return float(r["our_ask"])
    d = r.get("direction")
    if d == "UP" and r.get("top_ask_up"):
        return float(r["top_ask_up"])
    if d == "DOWN" and r.get("top_ask_down"):
        return float(r["top_ask_down"])
    return None


def summarize(records):
    res = [r for r in records if r.get("won") is not None and r.get("pnl") is not None]
    n = len(res)
    if n == 0:
        return {"n": 0, "win_rate": 0.0, "avg_ask": 0.0, "edge_pts": 0.0,
                "total_pnl": 0.0, "per_trade": 0.0, "t": 0.0}
    wins = sum(1 for r in res if r.get("won"))
    asks = [a for a in (_ask(r) for r in res) if a is not None]
    pnls = [float(r["pnl"]) for r in res]
    win_rate = wins / n
    avg_ask = statistics.mean(asks) if asks else 0.0
    mu = statistics.mean(pnls)
    sd = statistics.pstdev(pnls) if n > 1 else 0.0
    t = (mu / (sd / math.sqrt(n))) if sd else 0.0
    return {"n": n, "win_rate": win_rate, "avg_ask": avg_ask,
            "edge_pts": (win_rate - avg_ask) * 100.0,
            "total_pnl": sum(pnls), "per_trade": mu, "t": t}


def _load(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "paper_*_15m.jsonl"
    print("%-9s %4s %5s %7s %8s %9s %6s" %
          ("market", "n", "win%", "avgAsk", "edge_pts", "$/trade", "t"))
    for f in sorted(glob.glob(pat)):
        mkt = f.replace("paper_", "").replace(".jsonl", "")
        s = summarize(_load(f))
        if s["n"] == 0:
            continue
        print("%-9s %4d %4.0f%% %7.3f %+8.1f %+9.4f %+6.2f" %
              (mkt, s["n"], 100 * s["win_rate"], s["avg_ask"],
               s["edge_pts"], s["per_trade"], s["t"]))


if __name__ == "__main__":
    main()
