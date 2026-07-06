import os
from live_trader.config import load_config

def _base_env(monkeypatch):
    # minimal env so load_config builds without external calls
    for k in ("BOT_DBMODEL_MAX_ASK", "BOT_DBMODEL_ENTRY_DEADLINE_S"):
        monkeypatch.delenv(k, raising=False)

def test_defaults_preserve_standard_bot(monkeypatch):
    _base_env(monkeypatch)
    cfg = load_config(dotenv_path=".env")
    assert cfg.dbmodel_max_ask == 1.0
    assert cfg.dbmodel_entry_deadline_s == 0.0

def test_cheap_values_parsed(monkeypatch):
    monkeypatch.setenv("BOT_DBMODEL_MAX_ASK", "0.50")
    monkeypatch.setenv("BOT_DBMODEL_ENTRY_DEADLINE_S", "30")
    cfg = load_config(dotenv_path=".env")
    assert cfg.dbmodel_max_ask == 0.50
    assert cfg.dbmodel_entry_deadline_s == 30.0
