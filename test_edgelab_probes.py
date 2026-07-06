import asyncio
from edgelab import probes

def test_rpc_payload():
    p = probes.rpc_payload()
    assert p["method"] == "eth_blockNumber" and p["jsonrpc"] == "2.0"

def test_parse_rpc_block_hex():
    assert probes.parse_rpc_block({"jsonrpc": "2.0", "id": 1, "result": "0x10"}) == 16

def test_parse_rpc_block_error_raises():
    for bad in ({"error": {"code": -1, "message": "x"}}, {"id": 1}):
        try:
            probes.parse_rpc_block(bad)
            assert False, "expected ValueError"
        except ValueError:
            pass

def test_timed_call_measures_elapsed():
    async def fake():
        await asyncio.sleep(0.01)
        return "ok"
    rtt_ns, res = asyncio.run(probes.timed_call(fake))
    assert res == "ok"
    assert rtt_ns >= 8_000_000          # ~10ms, allow scheduling slack

def test_probe_rpc_uses_injected_caller():
    async def run():
        return await probes.probe_rpc("http://unused", caller=lambda: 12345)
    rtt = asyncio.run(run())
    assert isinstance(rtt, int) and rtt >= 0

def test_probe_rpc_returns_none_on_failure():
    def boom():
        raise RuntimeError("down")
    rtt = asyncio.run(probes.probe_rpc("http://unused", caller=boom))
    assert rtt is None

def test_clob_time_url_appends_time():
    assert probes.clob_time_url("https://clob.polymarket.com") == \
        "https://clob.polymarket.com/time"
    assert probes.clob_time_url("https://clob.polymarket.com/") == \
        "https://clob.polymarket.com/time"

def test_parse_clob_time_numeric():
    assert probes.parse_clob_time("1718900000") == 1718900000
    assert probes.parse_clob_time(" 1718900000\n") == 1718900000

def test_parse_clob_time_garbage_raises():
    for bad in ("", "not-a-number", "<html>error</html>"):
        try:
            probes.parse_clob_time(bad)
            assert False, "expected ValueError"
        except ValueError:
            pass

def test_probe_clob_uses_injected_caller():
    async def run():
        return await probes.probe_clob("http://unused", caller=lambda: 1718900000)
    rtt = asyncio.run(run())
    assert isinstance(rtt, int) and rtt >= 0

def test_probe_clob_returns_none_on_failure():
    def boom():
        raise RuntimeError("down")
    rtt = asyncio.run(probes.probe_clob("http://unused", caller=boom))
    assert rtt is None
