# predictor/test_edgelab_parse.py
from edgelab import parse

def test_binance_bookticker():
    msg = {"stream": "btcusdt@bookTicker", "data": {
        "u": 96099642589, "s": "BTCUSDT",
        "b": "64059.99000000", "B": "0.98934000",
        "a": "64060.00000000", "A": "3.26974000"}}
    rows = parse.parse_binance(msg)
    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == "binance_bookticker"
    assert r["best_bid"] == 64059.99 and r["best_ask"] == 64060.0
    assert r["best_bid_sz"] == 0.98934 and r["best_ask_sz"] == 3.26974
    assert r["exch_seq"] == 96099642589

def test_binance_trade():
    msg = {"stream": "btcusdt@trade", "data": {
        "e": "trade", "E": 1782030435506, "s": "BTCUSDT", "t": 6427631900,
        "p": "64060.00000000", "q": "0.00033000", "T": 1782030435506,
        "m": False, "M": True}}
    rows = parse.parse_binance(msg)
    assert rows[0]["source"] == "binance_trade"
    assert rows[0]["price"] == 64060.0 and rows[0]["size"] == 0.00033
    assert rows[0]["side"] == "buy"          # m=False => buyer is taker
    assert rows[0]["exch_seq"] == 6427631900

def test_coinbase_match():
    msg = {"type": "last_match", "trade_id": 1040839665, "side": "buy",
           "size": "0.00000015", "price": "63983.99", "product_id": "BTC-USD",
           "sequence": 130993420811, "time": "2026-06-21T08:27:16.364726Z"}
    rows = parse.parse_coinbase(msg)
    assert rows[0]["source"] == "coinbase_match"
    assert rows[0]["price"] == 63983.99 and rows[0]["side"] == "buy"
    assert rows[0]["exch_seq"] == 130993420811

def test_coinbase_ticker():
    msg = {"type": "ticker", "sequence": 130993420900, "product_id": "BTC-USD",
           "price": "63984.00", "best_bid": "63983.99", "best_ask": "63984.01",
           "best_bid_size": "1.5", "best_ask_size": "2.0",
           "time": "2026-06-21T08:27:17.0Z"}
    rows = parse.parse_coinbase(msg)
    assert rows[0]["source"] == "coinbase_ticker"
    assert rows[0]["best_bid"] == 63983.99 and rows[0]["best_ask"] == 63984.01
    assert rows[0]["best_bid_sz"] == 1.5 and rows[0]["best_ask_sz"] == 2.0

def test_pm_oracle_update():
    msg = {"connection_id": "x", "payload": {
        "full_accuracy_value": "64000.12000000", "symbol": "btcusdt",
        "timestamp": 1782030467000, "value": 64000.12},
        "timestamp": 1782030467151, "topic": "crypto_prices", "type": "update"}
    rows = parse.parse_pm_oracle(msg)
    assert rows[0]["source"] == "pm_oracle"
    assert rows[0]["price"] == 64000.12
    assert rows[0]["symbol_raw"] == "btcusdt"

def test_pm_clob_book_top_of_book():
    msg = {"event_type": "book", "asset_id": "TKN_UP",
           "bids": [{"price": "0.48", "size": "100"}, {"price": "0.47", "size": "50"}],
           "asks": [{"price": "0.52", "size": "200"}, {"price": "0.53", "size": "80"}]}
    rows = parse.parse_pm_clob(msg)
    assert len(rows) == 1 and rows[0]["source"] == "pm_clob_book"
    assert rows[0]["asset_id"] == "TKN_UP"
    assert rows[0]["best_bid"] == 0.48 and rows[0]["best_ask"] == 0.52
    assert rows[0]["best_bid_sz"] == 100.0 and rows[0]["best_ask_sz"] == 200.0

def test_pm_clob_price_change_rows():
    msg = {"event_type": "price_change", "asset_id": "TKN_UP",
           "price_changes": [
               {"price": "0.49", "size": "10", "side": "buy"},
               {"price": "0.51", "size": "20", "side": "sell"}]}
    rows = parse.parse_pm_clob(msg)
    assert len(rows) == 2
    assert all(r["source"] == "pm_clob_price_change" for r in rows)
    assert rows[0]["price"] == 0.49 and rows[0]["side"] == "buy"
    assert rows[1]["price"] == 0.51 and rows[1]["side"] == "sell"

def test_pm_clob_book_unsorted_levels():
    msg = {"event_type": "book", "asset_id": "TKN_UP",
           "bids": [{"price": "0.47", "size": "50"},
                    {"price": "0.49", "size": "100"},
                    {"price": "0.48", "size": "10"}],
           "asks": [{"price": "0.54", "size": "80"},
                    {"price": "0.52", "size": "200"},
                    {"price": "0.53", "size": "5"}]}
    rows = parse.parse_pm_clob(msg)
    assert len(rows) == 1 and rows[0]["source"] == "pm_clob_book"
    assert rows[0]["best_bid"] == 0.49 and rows[0]["best_ask"] == 0.52
    assert rows[0]["best_bid_sz"] == 100.0 and rows[0]["best_ask_sz"] == 200.0

def test_unknown_messages_ignored():
    assert parse.parse_binance({"stream": "x", "data": {"e": "depthUpdate"}}) == []
    assert parse.parse_coinbase({"type": "subscriptions"}) == []
    assert parse.parse_pm_oracle({"topic": "other"}) == []
    assert parse.parse_pm_clob({"event_type": "tick_size_change"}) == []
