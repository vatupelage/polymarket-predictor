import json, numpy as np
from edgelab import schema, capsim_io
from edgelab.writer import RotatingParquetWriter

def _row(src, wall_ns, **kw):
    r = schema.empty_row(); r["source"] = src; r["region_id"] = "eu-west-1"
    r["recv_wall_ns"] = wall_ns; r["clock_err_ns"] = 1
    r.update(kw); return r

def test_load_capture_builds_series(tmp_path):
    w = RotatingParquetWriter(str(tmp_path), clock=lambda: 60.0)
    w.write(_row("binance_bookticker", 1_000_000_000, best_bid=100.0, best_ask=100.02))
    w.write(_row("binance_bookticker", 2_000_000_000, best_bid=101.0, best_ask=101.02))
    pj = lambda a: json.dumps({"asset_id": a, "market": "0xCOND"})
    w.write(_row("pm_clob_book", 1_500_000_000, best_bid=0.50, best_ask=0.52,
                 best_ask_sz=7.0, payload_json=pj("AAA")))
    w.write(_row("pm_clob_book", 1_800_000_000, best_bid=None, best_ask=None,
                 best_ask_sz=None, payload_json=pj("AAA")))   # one-sided -> mid NaN
    w.write(_row("pm_oracle", 1_200_000_000, price=100.0))
    w.flush_all()

    d = capsim_io.load_capture(str(tmp_path))
    assert list(d["bt"]) == [1.0, 2.0]
    assert abs(d["bmid"][1] - 101.01) < 1e-6
    a = d["assets"]["AAA"]
    assert abs(a["ask"][0] - 0.52) < 1e-9 and abs(a["ask_sz"][0] - 7.0) < 1e-9
    assert abs(a["mid"][0] - 0.51) < 1e-9 and np.isnan(a["mid"][1])
    assert a["condition_id"] == "0xCOND"
    assert list(d["ot"]) == [1.2] and d["oval"][0] == 100.0


def test_resolve_outcomes_uses_fetcher_and_omits_unknown():
    conds = {"AAA": "0x1", "BBB": "0x2", "CCC": "0x3"}
    calls = {"AAA": 1, "BBB": 0, "CCC": None}
    out = capsim_io.resolve_outcomes(conds, fetcher=lambda a, c: calls[a])
    assert out == {"AAA": 1, "BBB": 0}        # CCC unresolved -> omitted


def test_resolve_outcomes_default_fetch_error_omits():
    # a fetcher that raises must be swallowed into omission, never a guess
    def boom(a, c): raise RuntimeError("clob down")
    assert capsim_io.resolve_outcomes({"AAA": "0x1"}, fetcher=boom) == {}
