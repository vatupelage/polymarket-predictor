from edgelab.writer import RotatingParquetWriter
from edgelab import schema, sanity_check

def _row(source, rtt=None):
    r = schema.empty_row()
    r["source"] = source
    r["region_id"] = "eu-west-1"
    r["recv_wall_ns"] = 60 * 1_000_000_000
    r["clock_err_ns"] = 1000
    r["rtt_ns"] = rtt
    return r

def _populate(out_dir):
    w = RotatingParquetWriter(out_dir, clock=lambda: 60.0)
    for s in ("binance_trade", "coinbase_match", "pm_oracle", "pm_clob_book"):
        w.write(_row(s))
    for v in (1_000_000, 2_000_000, 3_000_000):
        w.write(_row("probe_clob", rtt=v))
        w.write(_row("probe_rpc", rtt=v))
        w.write(_row("probe_tls", rtt=v))
    w.flush_all()

def test_summarize_and_gate_pass(tmp_path):
    _populate(str(tmp_path))
    s = sanity_check.summarize(str(tmp_path))
    assert s["clock_err_nulls"] == 0
    assert s["by_source"]["binance_trade"] == 1
    assert s["probe"]["probe_rpc"]["n"] == 3
    assert s["probe"]["probe_rpc"]["p50"] == 2_000_000
    assert s["probe"]["probe_clob"]["n"] == 3
    ok, reasons = sanity_check.check_gate(s)
    assert ok, reasons

def test_gate_fails_when_clob_probe_missing(tmp_path):
    w = RotatingParquetWriter(str(tmp_path), clock=lambda: 60.0)
    for s in ("binance_trade", "coinbase_match", "pm_oracle", "pm_clob_book"):
        w.write(_row(s))
    # binding-constraint probe absent; only secondary probes present
    w.write(_row("probe_rpc", rtt=1)); w.write(_row("probe_tls", rtt=1))
    w.flush_all()
    ok, reasons = sanity_check.check_gate(sanity_check.summarize(str(tmp_path)))
    assert not ok and any("probe_clob" in r for r in reasons)

def test_gate_fails_when_a_family_missing(tmp_path):
    w = RotatingParquetWriter(str(tmp_path), clock=lambda: 60.0)
    w.write(_row("binance_trade"))           # only one family
    w.write(_row("probe_rpc", rtt=1)); w.write(_row("probe_tls", rtt=1))
    w.flush_all()
    ok, reasons = sanity_check.check_gate(sanity_check.summarize(str(tmp_path)))
    assert not ok
    assert any("coinbase" in r for r in reasons)

def test_gate_fails_on_clock_err_null(tmp_path):
    w = RotatingParquetWriter(str(tmp_path), clock=lambda: 60.0)
    for s in ("binance_trade", "coinbase_match", "pm_oracle", "pm_clob_book"):
        w.write(_row(s))
    bad = _row("binance_trade"); bad["clock_err_ns"] = None
    w.write(bad)
    w.write(_row("probe_rpc", rtt=1)); w.write(_row("probe_tls", rtt=1))
    w.flush_all()
    ok, reasons = sanity_check.check_gate(sanity_check.summarize(str(tmp_path)))
    assert not ok and any("clock_err" in r for r in reasons)
