from edgelab import schema

def test_sources_cover_all_families():
    expected = {
        "binance_trade", "binance_bookticker",
        "coinbase_match", "coinbase_ticker",
        "pm_oracle", "pm_clob_book", "pm_clob_price_change",
        "probe_rpc", "probe_tls", "gap", "session",
    }
    assert expected <= schema.SOURCES

def test_empty_row_has_every_column_and_envelope():
    row = schema.empty_row()
    assert set(row) == set(schema.COLUMNS)
    for k in ("region_id", "source", "symbol", "window_slug",
              "recv_wall_ns", "recv_monotonic_ns", "clock_offset_ns",
              "clock_err_ns", "local_ingest_seq", "exch_seq",
              "payload_json", "price", "size", "side",
              "best_bid", "best_ask", "best_bid_sz", "best_ask_sz", "rtt_ns"):
        assert k in row and row[k] is None

def test_arrow_schema_matches_columns():
    assert [f.name for f in schema.ARROW_SCHEMA] == schema.COLUMNS
