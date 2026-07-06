"""Unit tests for the pure fav90 gate logic. No network."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from live_trader.fav90 import pick_favorite, depth_within_tick, fav90_decision


def test_pick_favorite_up():
    assert pick_favorite(0.91, 0.11) == ("UP", 0.91)

def test_pick_favorite_down():
    assert pick_favorite(0.12, 0.90) == ("DOWN", 0.90)

def test_pick_favorite_missing():
    assert pick_favorite(None, 0.9) == (None, None)
    assert pick_favorite(0.9, None) == (None, None)

def test_depth_sums_within_one_tick():
    # best ask 0.90 with 8 sh, 0.91 with 5 sh (within tick), 0.93 with 100 (beyond)
    asks = [[0.90, 8], [0.91, 5], [0.93, 100]]
    assert depth_within_tick(asks, tick=0.01) == 13.0

def test_depth_empty():
    assert depth_within_tick([], tick=0.01) == 0.0
    assert depth_within_tick(None, tick=0.01) == 0.0

def test_depth_only_best_when_gap():
    asks = [[0.90, 4], [0.95, 999]]
    assert depth_within_tick(asks, tick=0.01) == 4.0

# fav90_decision gate matrix --------------------------------------------------
PARAMS = dict(entry_max_s=100, min_ask=0.88, max_ask=0.92, min_depth=10)

def test_fires_when_all_gates_pass():
    d = fav90_decision(secs_to_close=90, top_ask_up=0.90, top_ask_down=0.11,
                       fav_asks=[[0.90, 12]], **PARAMS)
    assert d["fire"] is True and d["side"] == "UP" and d["fav_ask"] == 0.90

def test_no_fire_too_early():
    d = fav90_decision(secs_to_close=180, top_ask_up=0.90, top_ask_down=0.11,
                       fav_asks=[[0.90, 12]], **PARAMS)
    assert d["fire"] is False and d["timing_ok"] is False

def test_no_fire_ask_below_band():
    d = fav90_decision(secs_to_close=90, top_ask_up=0.85, top_ask_down=0.16,
                       fav_asks=[[0.85, 50]], **PARAMS)
    assert d["fire"] is False and d["price_ok"] is False

def test_no_fire_ask_above_band():
    d = fav90_decision(secs_to_close=90, top_ask_up=0.94, top_ask_down=0.07,
                       fav_asks=[[0.94, 50]], **PARAMS)
    assert d["fire"] is False and d["price_ok"] is False

def test_no_fire_thin_depth():
    d = fav90_decision(secs_to_close=90, top_ask_up=0.90, top_ask_down=0.11,
                       fav_asks=[[0.90, 3]], **PARAMS)
    assert d["fire"] is False and d["depth_ok"] is False and d["depth"] == 3.0

def test_picks_down_favorite():
    d = fav90_decision(secs_to_close=80, top_ask_up=0.10, top_ask_down=0.89,
                       fav_asks=[[0.89, 25]], **PARAMS)
    assert d["fire"] is True and d["side"] == "DOWN"
