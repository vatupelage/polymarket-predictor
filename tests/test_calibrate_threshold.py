import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
from calibrate_threshold import median_duration

def test_median_duration_lower_threshold_shorter_bars():
    # synthetic trades: $1000 of volume per second for 200s
    trades = [(i*1000, 100.0, 10.0) for i in range(200)]  # ts ms, price, qty -> $1000/trade @1/s
    med_lo, n_lo = median_duration(trades, 2000)   # 2 trades/bar -> ~1-2s
    med_hi, n_hi = median_duration(trades, 20000)  # 20 trades/bar -> ~19-20s
    assert n_lo > n_hi
    assert med_hi > med_lo
