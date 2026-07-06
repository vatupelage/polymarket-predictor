import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
from train_db_model import reconstruct_windows
from live_trader.db_features import FEATURE_NAMES


def test_reconstruct_windows_shapes_and_labels():
    # 3 full 900s windows of synthetic 1/sec trades, gently trending up
    rows = []
    for s in range(2700):
        rows.append((s * 1000, 100.0 + s * 0.001, 50.0))   # ts ms, price, qty ($5000/s)
    df = pd.DataFrame(rows, columns=["ts", "price", "qty"])
    X, y, ws = reconstruct_windows(df, threshold=20000, window_s=900, monitor_start_s=840)
    assert X.shape[1] == len(FEATURE_NAMES)
    assert X.shape[0] >= 1                       # 3 synthetic windows must reconstruct, not silently empty
    assert X.shape[0] == len(y) == len(ws)
    assert set(np.unique(y)).issubset({0, 1})
