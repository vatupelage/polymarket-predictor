# predictor/test_edgelab_collect.py
import json
from edgelab.collect import build_rows, BuildCtx
from edgelab.clockstamp import ClockStamper
from edgelab.seqgap import SeqGapTracker

def _ctx(token_index=None):
    return BuildCtx(stamper=ClockStamper("eu-west-1", reader=lambda: (0, 1)),
                    gaps=SeqGapTracker(), symbol="BTC", symbols={"BTC"},
                    token_index=token_index or {})

def test_binance_event_row_is_fully_stamped():
    ctx = _ctx()
    raw = {"stream": "btcusdt@bookTicker",
           "data": {"u": 10, "s": "BTCUSDT", "b": "1", "B": "2", "a": "3", "A": "4"}}
    rows = build_rows("binance", raw, ctx)
    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == "binance_bookticker" and r["symbol"] == "BTC"
    assert r["region_id"] == "eu-west-1" and r["clock_err_ns"] == 1
    assert r["best_bid"] == 1.0 and r["best_ask"] == 3.0
    assert json.loads(r["payload_json"])["data"]["u"] == 10
    assert "symbol_raw" not in r and "asset_id" not in r

def test_gap_row_emitted_before_event_on_hole():
    ctx = _ctx()
    base = {"stream": "btcusdt@bookTicker",
            "data": {"s": "BTCUSDT", "b": "1", "B": "2", "a": "3", "A": "4"}}
    build_rows("binance", {**base, "data": {**base["data"], "u": 10}}, ctx)
    rows = build_rows("binance", {**base, "data": {**base["data"], "u": 14}}, ctx)
    assert [r["source"] for r in rows] == ["gap", "binance_bookticker"]
    assert json.loads(rows[0]["payload_json"])["count"] == 3

def test_oracle_filters_to_configured_symbols():
    ctx = _ctx()
    btc = {"topic": "crypto_prices", "type": "update",
           "payload": {"value": 64000.0, "symbol": "btcusdt"}}
    eth = {"topic": "crypto_prices", "type": "update",
           "payload": {"value": 1700.0, "symbol": "ethusdt"}}
    assert len(build_rows("pm_oracle", btc, ctx)) == 1
    assert build_rows("pm_oracle", eth, ctx) == []     # ETH not configured

def test_pm_clob_tagged_by_token_index_else_dropped():
    ctx = _ctx(token_index={"TKN_UP": ("BTC", "btc-updown-5m-100")})
    known = {"event_type": "book", "asset_id": "TKN_UP",
             "bids": [{"price": "0.48", "size": "100"}],
             "asks": [{"price": "0.52", "size": "200"}]}
    unknown = {"event_type": "book", "asset_id": "NOPE",
               "bids": [], "asks": []}
    rows = build_rows("pm_clob", known, ctx)
    assert rows[0]["window_slug"] == "btc-updown-5m-100" and rows[0]["best_bid"] == 0.48
    assert build_rows("pm_clob", unknown, ctx) == []


import glob
import pyarrow.parquet as pq
from edgelab import harness

def test_dry_run_writes_stamped_parquet(tmp_path):
    sample = tmp_path / "sample.jsonl"
    lines = [
        {"token_index": {"TKN_UP": ["BTC", "btc-updown-5m-100"]}},
        {"family": "binance", "raw": {"stream": "btcusdt@trade",
            "data": {"e": "trade", "s": "BTCUSDT", "t": 1, "p": "64000",
                     "q": "0.5", "m": False}}},
        {"family": "coinbase", "raw": {"type": "match", "price": "63999",
            "size": "0.1", "side": "sell", "sequence": 5, "product_id": "BTC-USD"}},
        {"family": "pm_oracle", "raw": {"topic": "crypto_prices", "type": "update",
            "payload": {"value": 64000.1, "symbol": "btcusdt"}}},
        {"family": "pm_clob", "raw": {"event_type": "book", "asset_id": "TKN_UP",
            "bids": [{"price": "0.48", "size": "100"}],
            "asks": [{"price": "0.52", "size": "200"}]}},
    ]
    sample.write_text("\n".join(json.dumps(x) for x in lines))
    out = tmp_path / "data"
    summary = harness.dry_run(str(sample), str(out))
    assert summary["rows"] == 4
    for src in ("binance_trade", "coinbase_match", "pm_oracle", "pm_clob_book"):
        files = glob.glob(str(out / f"events/day=*/source={src}/*.parquet"))
        assert files, f"no parquet for {src}"
        tbl = pq.read_table(files[0], partitioning=None)
        assert tbl.num_rows >= 1
        col = tbl.column("clock_err_ns").to_pylist()
        assert all(v is not None for v in col)        # clock_err on every row


def test_pump_skips_empty_keepalive_frame():
    # PM oracle WS sends an empty-string frame on connect; json.loads('') would
    # raise and tear down the whole feed. _pump must skip blank frames.
    import asyncio
    from edgelab.collect import _pump

    class _FakeWS:
        def __init__(self, frames):
            self._frames = list(frames)
        async def recv(self):
            return self._frames.pop(0)

    class _Collector:
        def __init__(self):
            self.rows = []
        def write(self, row):
            self.rows.append(row)

    ctx = _ctx()
    w = _Collector()
    ws = _FakeWS([""])                       # one empty keepalive frame
    asyncio.run(_pump(ws, "pm_oracle", ctx, w))   # must NOT raise
    assert w.rows == []                      # nothing written, no crash

    # a real frame after the keepalive still parses
    ws2 = _FakeWS(['{"topic":"crypto_prices","type":"update",'
                   '"payload":{"value":64000.0,"symbol":"btcusdt"}}'])
    asyncio.run(_pump(ws2, "pm_oracle", ctx, w))
    assert any(r["source"] == "pm_oracle" for r in w.rows)
