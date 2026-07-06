from edgelab import clockstamp, schema

SAMPLE = """Reference ID    : C0248F88 (time.example)
Stratum         : 3
System time     : 0.000012345 seconds slow of NTP time
Last offset     : -0.000004567 seconds
RMS offset      : 0.000010000 seconds
Root delay      : 0.000500000 seconds
Root dispersion : 0.000250000 seconds
Leap status     : Normal
"""

def test_parse_chrony_offset_and_error_bound():
    offset_ns, err_ns = clockstamp.parse_chrony_tracking(SAMPLE)
    assert offset_ns == -4567               # -0.000004567 s -> ns
    # err = dispersion + delay/2 = 0.000250000 + 0.000250000 = 0.000500000 s
    assert err_ns == 500000

def test_stamp_builds_full_row_with_envelope_and_seq():
    cs = clockstamp.ClockStamper("eu-west-1", reader=lambda: (-4567, 500000))
    r1 = cs.stamp("binance_trade", symbol="BTC", price=64000.0)
    assert set(r1) == set(schema.COLUMNS)
    assert r1["region_id"] == "eu-west-1"
    assert r1["source"] == "binance_trade"
    assert r1["symbol"] == "BTC"
    assert r1["price"] == 64000.0
    assert r1["clock_offset_ns"] == -4567
    assert r1["clock_err_ns"] == 500000
    assert r1["recv_wall_ns"] > 0 and r1["recv_monotonic_ns"] > 0
    assert r1["local_ingest_seq"] == 0
    r2 = cs.stamp("binance_trade", symbol="BTC")
    assert r2["local_ingest_seq"] == 1          # per-source counter advances
    r3 = cs.stamp("coinbase_match", symbol="BTC")
    assert r3["local_ingest_seq"] == 0          # independent per source

def test_stamp_rejects_unknown_column():
    cs = clockstamp.ClockStamper("eu-west-1", reader=lambda: (0, 1))
    try:
        cs.stamp("binance_trade", not_a_column=1)
        assert False, "expected KeyError"
    except KeyError:
        pass
