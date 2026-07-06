# test_dbmodel_limit_entry.py
# Verifies the dry-run hypothetical-fill entry price honors limit_price, so the
# cheap-entry path books fills at the limit, not the market snapshot ask.
import types
from live_trader import bot as botmod


def test_dryrun_hypo_entry_uses_limit_price(monkeypatch):
    # Build a HighSameBot-like stub exposing only what _execute_dbmodel_trade needs.
    b = botmod.HighSameBot.__new__(botmod.HighSameBot)
    b.cfg = types.SimpleNamespace(dry_run=True, stake_usdc=1.0, dbmodel_delegate_redeem=False)
    b._lock = __import__("threading").Lock()
    b._active = 1

    class FakeClient:
        def resolve_market(self, slug):
            return {"up_token": "UP", "down_token": "DN", "condition_id": "c"}

        def get_top_ask(self, t):
            return 0.62 if t == "UP" else 0.40   # market ask on chosen side = 0.62

        def wait_for_resolution(self, slug, deadline):
            return {"up_won": True}

        def binance_window_prices(self, ws, symbol=None, interval=None):
            return None

    b.client = FakeClient()
    captured = {}

    def fake_paper_log(window, p_up, direction, confidence, drift_pct,
                       our_ask, top_ask_up, top_ask_down, hypo_shares, stake,
                       won, pnl, bw, resolution=None, path=None):
        captured.update(entry_ask=our_ask, shares=hypo_shares, won=won)

    monkeypatch.setattr(b, "_dbmodel_paper_log", fake_paper_log, raising=False)
    monkeypatch.setattr(b, "_dbmodel_log", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(b, "_sample_book_path", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(b, "_record", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(b, "_print_summary", lambda: None, raising=False)

    window = {"slug": "btc-updown-5m-1", "end_ts": 0, "ws": 0,
              "features": {}, "raw_proba": 0.7, "strike": 100.0}
    b._execute_dbmodel_trade(window, "UP", 0.7, limit_price=0.50)

    # hypothetical fill must use the 0.50 limit, NOT the 0.62 market ask
    assert captured["entry_ask"] == 0.50, (
        f"Expected entry_ask=0.50 (limit_price), got {captured.get('entry_ask')}"
    )
    assert abs(captured["shares"] - (1.0 / 0.50)) < 1e-9, (
        f"Expected shares=2.0 (stake/limit_price), got {captured.get('shares')}"
    )
