"""TDD for edgelab.recorder.WindowRecorder — turns verified WS messages into
top-of-book rows with per-token CKS OFI. This is the glue that must route each
price_change to the RIGHT token's book and accumulate OFI per token; a routing
bug here silently mixes Up/Down flow and fabricates signal."""

from edgelab.recorder import WindowRecorder

UP = "UPTOKEN"
DOWN = "DOWNTOKEN"


def _rec():
    return WindowRecorder(slug="btc-updown-5m-1000", horizon="5m",
                          up_token=UP, down_token=DOWN,
                          open_ts=1000.0, close_ts=1300.0)


def _book(tok, bids, asks):
    return {"event_type": "book", "asset_id": tok,
            "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
            "asks": [{"price": str(p), "size": str(s)} for p, s in asks]}


def _pc(*changes):
    return {"event_type": "price_change",
            "price_changes": [{"asset_id": t, "price": str(p), "side": sd, "size": str(s)}
                              for (t, p, sd, s) in changes]}


def test_book_snapshot_emits_topofbook_row():
    r = _rec()
    r.on_ws_message(_book(UP, [(0.45, 50)], [(0.55, 70)]), recv_ts=1001.0)
    assert len(r.rows) == 1
    row = r.rows[0]
    assert row["side"] == "up"
    assert row["best_bid"] == 0.45 and row["best_bid_sz"] == 50.0
    assert row["best_ask"] == 0.55 and row["best_ask_sz"] == 70.0
    assert row["ts"] == 1001.0
    assert row["ofi_inc"] == 0.0   # first state, no predecessor


def test_price_change_routed_to_correct_token():
    r = _rec()
    r.on_ws_message(_book(UP, [(0.45, 50)], [(0.55, 70)]), recv_ts=1001.0)
    r.on_ws_message(_book(DOWN, [(0.40, 10)], [(0.60, 10)]), recv_ts=1001.0)
    # a bid-up change on UP only
    r.on_ws_message(_pc((UP, 0.46, "BUY", 20)), recv_ts=1002.0)
    up_rows = [x for x in r.rows if x["side"] == "up"]
    down_rows = [x for x in r.rows if x["side"] == "down"]
    assert up_rows[-1]["best_bid"] == 0.46
    assert up_rows[-1]["ofi_inc"] == 20.0      # bid up by full new size
    # DOWN untouched -> its last row still the snapshot, no new down row from this pc
    assert down_rows[-1]["best_bid"] == 0.40


def test_ofi_accumulates_per_token():
    r = _rec()
    r.on_ws_message(_book(UP, [(0.45, 50)], [(0.55, 70)]), recv_ts=1001.0)
    r.on_ws_message(_pc((UP, 0.46, "BUY", 20)), recv_ts=1002.0)   # +20
    r.on_ws_message(_pc((UP, 0.54, "SELL", 12)), recv_ts=1003.0)  # ask undercut -> -12
    up_rows = [x for x in r.rows if x["side"] == "up"]
    assert up_rows[-1]["ofi_cum"] == 8.0


def test_no_row_when_topofbook_unchanged():
    r = _rec()
    r.on_ws_message(_book(UP, [(0.45, 50), (0.44, 10)], [(0.55, 70)]), recv_ts=1001.0)
    n0 = len(r.rows)
    # change a DEEPER level on UP (0.44) -> best quote unchanged -> no new row
    r.on_ws_message(_pc((UP, 0.44, "BUY", 99)), recv_ts=1002.0)
    assert len(r.rows) == n0


def test_metadata_finalize():
    r = _rec()
    r.on_ws_message(_book(UP, [(0.45, 50)], [(0.55, 70)]), recv_ts=1001.0)
    meta = r.finalize(strike=64000.0, terminal=64010.0, up_won=True, feed="binance_proxy")
    assert meta["slug"] == "btc-updown-5m-1000"
    assert meta["horizon"] == "5m"
    assert meta["up_won"] is True
    assert meta["strike"] == 64000.0
    assert meta["feed"] == "binance_proxy"
    assert meta["n_rows"] == len(r.rows)
