"""Tests for the positions-pagination helper used by sweep_orphan_winners.

The redeemer was missing real winners because the /positions fetch was a single
uncapped request that the data-api truncates (~100), and the wallet's old-loser
backlog crowded out fresh winners. This helper paginates until exhausted.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trader._pagination import paginate_all


def test_accumulates_across_full_then_short_page():
    pages = {0: list(range(500)), 500: list(range(500, 730))}  # 500 + 230
    got = paginate_all(lambda off: pages.get(off, []), page_size=500)
    assert len(got) == 730
    assert got[0] == 0 and got[-1] == 729


def test_single_short_page_stops_without_extra_fetch():
    calls = []
    def fetch(off):
        calls.append(off)
        return [1, 2, 3] if off == 0 else []
    got = paginate_all(fetch, page_size=500)
    assert got == [1, 2, 3]
    assert calls == [0]            # exactly one fetch (short page ends it)


def test_exact_multiple_stops_on_empty_page():
    # two full pages then an empty page (total a multiple of page_size)
    pages = {0: list(range(10)), 10: list(range(10, 20)), 20: []}
    got = paginate_all(lambda off: pages.get(off, []), page_size=10)
    assert len(got) == 20


def test_empty_first_page_returns_empty():
    assert paginate_all(lambda off: [], page_size=500) == []


def test_max_pages_caps_runaway():
    # fetch always returns a full page; max_pages must bound it
    got = paginate_all(lambda off: list(range(off, off + 100)),
                       page_size=100, max_pages=3)
    assert len(got) == 300


def test_non_list_response_stops_gracefully():
    got = paginate_all(lambda off: [1, 2] if off == 0 else {"error": "x"},
                       page_size=500)
    assert got == [1, 2]
