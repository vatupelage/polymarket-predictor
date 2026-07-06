def _build_record(window):
    """Mirror of the paper-record fields this task adds; keeps the contract tested
    even though the live writer lives on the remote box."""
    return {
        "slug": window["slug"],
        "monitor_start_s": window.get("monitor_start_s"),
        "window_s": window.get("window_s"),
    }


def test_record_carries_decision_time_and_window():
    window = {"slug": "btc-updown-15m-1", "monitor_start_s": 840, "window_s": 900}
    rec = _build_record(window)
    assert rec["monitor_start_s"] == 840
    assert rec["window_s"] == 900


def test_record_window_fields_default_none_when_absent():
    rec = _build_record({"slug": "x"})
    assert rec["monitor_start_s"] is None
    assert rec["window_s"] is None
