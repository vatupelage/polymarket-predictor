"""Analyze skip_history.jsonl: per-reason WR, counterfactual PnL, and the
'cost' of each skip rule (sum of profits we'd have made if the rule were off).

Run:
    python3 analyze_skips_v2.py

Reads:
    skip_history.jsonl  — one entry per skipped prediction with real outcome
"""

import json
import os
from collections import defaultdict


def main():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skip_history.jsonl")
    if not os.path.exists(path):
        print(f"no skip log yet at {path}")
        return

    skips = [json.loads(l) for l in open(path) if l.strip()]
    if not skips:
        print("skip log is empty")
        return

    print(f"total skipped predictions with outcomes: {len(skips)}\n")

    print(f"  {'reason':<24}{'n':>5}{'WR%':>8}{'sum_pnl':>12}{'avg_pnl':>10}")
    print("  " + "-" * 60)

    by_reason = defaultdict(list)
    for s in skips:
        by_reason[s["skip_reason"]].append(s)

    grand_n, grand_w, grand_pnl = 0, 0, 0.0
    for reason, group in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        n = len(group)
        w = sum(1 for s in group if s.get("would_have_won"))
        pnls = [s["would_have_pnl"] for s in group if s.get("would_have_pnl") is not None]
        sum_pnl = sum(pnls) if pnls else 0.0
        avg_pnl = sum_pnl / len(pnls) if pnls else 0.0
        wr = 100 * w / n
        print(f"  {reason:<24}{n:>5}{wr:>7.1f}%{'$'+f'{sum_pnl:+.2f}':>12}{'$'+f'{avg_pnl:+.3f}':>10}")
        grand_n += n
        grand_w += w
        grand_pnl += sum_pnl

    print("  " + "-" * 60)
    grand_wr = 100 * grand_w / grand_n if grand_n else 0
    print(f"  {'ALL SKIPS':<24}{grand_n:>5}{grand_wr:>7.1f}%{'$'+f'{grand_pnl:+.2f}':>12}")

    # Verdict: each rule is "good" if avg_pnl < 0 (skipping these losers helped).
    print(f"\nverdict per skip rule (good = blocking these would have lost money):")
    for reason, group in sorted(by_reason.items()):
        pnls = [s["would_have_pnl"] for s in group if s.get("would_have_pnl") is not None]
        if not pnls:
            continue
        avg = sum(pnls) / len(pnls)
        verdict = "GOOD skip" if avg < -0.05 else ("BAD skip" if avg > 0.05 else "neutral")
        print(f"  {reason:<24} avg=${avg:+.3f}  -> {verdict}")


if __name__ == "__main__":
    main()
