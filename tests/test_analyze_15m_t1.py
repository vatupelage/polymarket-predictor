from analyze_15m_t1 import summarize


def test_summarize_breakeven_and_edge():
    # 3 resolved trades, all entered at ask 0.50; 2 wins, 1 loss.
    recs = [
        {"won": True,  "pnl": 1.0,  "our_ask": 0.50, "direction": "UP"},
        {"won": True,  "pnl": 1.0,  "our_ask": 0.50, "direction": "UP"},
        {"won": False, "pnl": -1.0, "our_ask": 0.50, "direction": "DOWN"},
    ]
    s = summarize(recs)
    assert s["n"] == 3
    assert abs(s["win_rate"] - 2/3) < 1e-9
    assert abs(s["avg_ask"] - 0.50) < 1e-9        # break-even win rate
    assert s["edge_pts"] > 0                       # win_rate - avg_ask, in points
    assert abs(s["total_pnl"] - 1.0) < 1e-9


def test_summarize_skips_unresolved():
    recs = [{"won": None, "pnl": None, "our_ask": 0.5}]
    s = summarize(recs)
    assert s["n"] == 0
