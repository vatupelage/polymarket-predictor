# tests/test_validate_metrics.py
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
from validate_threshold import bootstrap_diff_ci

def test_bootstrap_diff_ci_detects_real_improvement():
    rng = np.random.default_rng(0)
    # model A errors smaller than B -> mean_diff (A-B) negative, CI excludes 0
    a = rng.normal(0.30, 0.02, 400)   # per-window log-loss of A
    b = rng.normal(0.45, 0.02, 400)   # per-window log-loss of B
    md, lo, hi = bootstrap_diff_ci(a, b, n=2000, seed=1)
    assert md < 0 and hi < 0          # A strictly better, CI clears zero

def test_bootstrap_diff_ci_ties_straddle_zero():
    rng = np.random.default_rng(2)
    a = rng.normal(0.40, 0.05, 400); b = rng.normal(0.40, 0.05, 400)
    md, lo, hi = bootstrap_diff_ci(a, b, n=2000, seed=3)
    assert lo < 0 < hi                # indistinguishable -> CI straddles 0


# ---------------------------------------------------------------------------
# Tests for align_by_ws helper
# ---------------------------------------------------------------------------
from validate_threshold import align_by_ws

def test_align_by_ws_intersection_only():
    """Only windows present in BOTH sides are returned."""
    X2 = np.array([[1.0], [2.0], [3.0]])
    y2 = np.array([0, 1, 0])
    ws2 = [100, 200, 300]
    X1 = np.array([[10.0], [20.0], [30.0]])
    y1 = np.array([1, 0, 1])
    ws1 = [200, 300, 400]
    X2a, y2a, X1a, y1a, common = align_by_ws(X2, y2, ws2, X1, y1, ws1)
    assert list(common) == [200, 300]
    assert len(X2a) == 2 and len(X1a) == 2

def test_align_by_ws_matching_order():
    """Rows from both sides correspond to the same ws_id in the same order."""
    X2 = np.array([[1.0], [2.0], [3.0]])
    y2 = np.array([0, 1, 0])
    ws2 = [300, 100, 200]   # not sorted
    X1 = np.array([[10.0], [20.0], [30.0]])
    y1 = np.array([1, 0, 1])
    ws1 = [200, 100, 400]
    X2a, y2a, X1a, y1a, common = align_by_ws(X2, y2, ws2, X1, y1, ws1)
    # common sorted = [100, 200]
    assert list(common) == [100, 200]
    # ws2: 100->idx1, 200->idx2
    np.testing.assert_array_equal(X2a[:, 0], [2.0, 3.0])
    # ws1: 100->idx1, 200->idx0
    np.testing.assert_array_equal(X1a[:, 0], [20.0, 10.0])

def test_align_by_ws_drops_non_common():
    """Windows present on only one side are absent from both aligned outputs."""
    X2 = np.array([[1.0], [2.0], [3.0]])
    y2 = np.array([0, 1, 0])
    ws2 = [100, 200, 300]
    X1 = np.array([[10.0], [20.0]])
    y1 = np.array([1, 0])
    ws1 = [100, 400]   # only 100 is common
    X2a, y2a, X1a, y1a, common = align_by_ws(X2, y2, ws2, X1, y1, ws1)
    assert list(common) == [100]
    np.testing.assert_array_equal(X2a[:, 0], [1.0])
    np.testing.assert_array_equal(X1a[:, 0], [10.0])
    np.testing.assert_array_equal(y2a, [0])
    np.testing.assert_array_equal(y1a, [1])
