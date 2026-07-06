from edgelab.seqgap import SeqGapTracker

def test_no_gap_on_contiguous():
    t = SeqGapTracker()
    assert t.check("binance_trade", 10) is None     # first seen
    assert t.check("binance_trade", 11) is None
    assert t.check("binance_trade", 12) is None

def test_detects_forward_hole():
    t = SeqGapTracker()
    t.check("coinbase_match", 100)
    g = t.check("coinbase_match", 104)
    assert g == {"source": "coinbase_match", "gap_start": 101,
                 "gap_end": 103, "count": 3}

def test_none_and_out_of_order_and_dupes_ignored():
    t = SeqGapTracker()
    t.check("s", 5)
    assert t.check("s", None) is None
    assert t.check("s", 5) is None          # dup
    assert t.check("s", 3) is None          # out of order
    assert t.check("s", 6) is None          # resumes contiguous from last max(5)

def test_sources_independent():
    t = SeqGapTracker()
    t.check("a", 1); t.check("b", 100)
    assert t.check("a", 2) is None
    assert t.check("b", 102)["count"] == 1
