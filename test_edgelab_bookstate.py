"""TDD for edgelab.bookstate.BookState — both-side L2 book reconstruction from
the Polymarket CLOB `book` snapshot + `price_change` deltas (verified protocol,
see edgelab/ACCESS_NOTES.md). Correctness here is load-bearing: a silent bug
would fabricate or destroy the OFI signal we are trying to measure."""

from edgelab.bookstate import BookState


def test_best_from_snapshot():
    b = BookState()
    b.apply_snapshot(
        bids=[{"price": "0.40", "size": "100"}, {"price": "0.45", "size": "50"}],
        asks=[{"price": "0.55", "size": "70"}, {"price": "0.52", "size": "30"}],
    )
    # best bid = highest price; best ask = lowest price
    assert b.best_bid() == (0.45, 50.0)
    assert b.best_ask() == (0.52, 30.0)


def test_empty_book_returns_none():
    b = BookState()
    assert b.best_bid() == (None, 0.0)
    assert b.best_ask() == (None, 0.0)


def test_apply_change_adds_and_updates_level():
    b = BookState()
    b.apply_snapshot(bids=[{"price": "0.45", "size": "50"}],
                     asks=[{"price": "0.55", "size": "70"}])
    # a more aggressive bid arrives at 0.46
    b.apply_change(side="BUY", price=0.46, size=20.0)
    assert b.best_bid() == (0.46, 20.0)
    # size at 0.46 updated in place
    b.apply_change(side="BUY", price=0.46, size=35.0)
    assert b.best_bid() == (0.46, 35.0)


def test_apply_change_size_zero_removes_level():
    b = BookState()
    b.apply_snapshot(
        bids=[{"price": "0.45", "size": "50"}, {"price": "0.44", "size": "10"}],
        asks=[{"price": "0.55", "size": "70"}],
    )
    # top bid pulled
    b.apply_change(side="BUY", price=0.45, size=0.0)
    assert b.best_bid() == (0.44, 10.0)


def test_ask_side_change_and_removal():
    b = BookState()
    b.apply_snapshot(bids=[{"price": "0.45", "size": "50"}],
                     asks=[{"price": "0.55", "size": "70"}, {"price": "0.56", "size": "5"}])
    b.apply_change(side="SELL", price=0.54, size=12.0)   # tighter ask
    assert b.best_ask() == (0.54, 12.0)
    b.apply_change(side="SELL", price=0.54, size=0.0)    # pulled
    assert b.best_ask() == (0.55, 70.0)


def test_snapshot_replaces_not_merges():
    b = BookState()
    b.apply_snapshot(bids=[{"price": "0.45", "size": "50"}], asks=[{"price": "0.55", "size": "70"}])
    b.apply_snapshot(bids=[{"price": "0.30", "size": "9"}], asks=[{"price": "0.70", "size": "9"}])
    # stale 0.45/0.55 levels must be gone
    assert b.best_bid() == (0.30, 9.0)
    assert b.best_ask() == (0.70, 9.0)


def test_mid_and_spread():
    b = BookState()
    b.apply_snapshot(bids=[{"price": "0.48", "size": "10"}], asks=[{"price": "0.52", "size": "10"}])
    assert abs(b.mid() - 0.50) < 1e-12
    assert abs(b.spread() - 0.04) < 1e-12
