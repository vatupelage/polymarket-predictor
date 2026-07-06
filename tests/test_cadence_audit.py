import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from cadence_audit import sec_per_bar, audit

def test_sec_per_bar_basic():
    # $86,400 over 24h = $1/sec; $20 threshold -> 20s/bar
    assert abs(sec_per_bar(86400, 20) - 20.0) < 1e-6

def test_audit_flags_too_slow():
    spb, ok = audit(threshold=125000, vol24h=0.10e9)   # bnb today ~105s
    assert spb > 30 and ok is False

def test_audit_passes_in_band():
    spb, ok = audit(threshold=24000, vol24h=0.10e9)     # ~20s
    assert 18 <= spb <= 30 and ok is True
