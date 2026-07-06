import glob, os
import pyarrow.parquet as pq
from edgelab.writer import RotatingParquetWriter
from edgelab import schema

def _row(source, wall_min):
    r = schema.empty_row()
    r["source"] = source
    r["region_id"] = "eu-west-1"
    r["recv_wall_ns"] = wall_min * 60 * 1_000_000_000
    r["clock_err_ns"] = 1000
    return r

def test_rotation_on_minute_change_writes_immutable_file(tmp_path):
    clock = {"t": 60.0}  # epoch_minute 1
    w = RotatingParquetWriter(str(tmp_path), clock=lambda: clock["t"])
    w.write(_row("binance_trade", 1))
    w.write(_row("binance_trade", 1))
    clock["t"] = 125.0   # epoch_minute 2 -> triggers flush of minute 1 buffer
    w.write(_row("binance_trade", 2))
    files = glob.glob(str(tmp_path / "events/day=*/source=binance_trade/*.parquet"))
    assert len(files) == 1
    tbl = pq.read_table(files[0], partitioning=None)
    assert tbl.num_rows == 2
    assert tbl.schema.names == schema.COLUMNS

def test_flush_all_drains_open_buffers(tmp_path):
    w = RotatingParquetWriter(str(tmp_path), clock=lambda: 60.0)
    w.write(_row("coinbase_match", 1))
    w.flush_all()
    files = glob.glob(str(tmp_path / "events/day=*/source=coinbase_match/*.parquet"))
    assert len(files) == 1 and pq.read_table(files[0], partitioning=None).num_rows == 1

def test_never_overwrites_existing_file(tmp_path):
    w = RotatingParquetWriter(str(tmp_path), clock=lambda: 60.0)
    w.write(_row("pm_oracle", 1)); w.flush_all()
    w2 = RotatingParquetWriter(str(tmp_path), clock=lambda: 60.0)
    w2.write(_row("pm_oracle", 1)); w2.flush_all()
    files = glob.glob(str(tmp_path / "events/day=*/source=pm_oracle/*.parquet"))
    assert len(files) == 2          # second got a -1 suffix, original intact

def test_region_prefixed_filename(tmp_path):
    w = RotatingParquetWriter(str(tmp_path), region_id="eu-west-1", clock=lambda: 60.0)
    w.write(_row("binance_trade", 1))
    w.flush_all()
    files = glob.glob(str(tmp_path / "events/day=*/source=binance_trade/*.parquet"))
    assert len(files) == 1
    assert os.path.basename(files[0]).startswith("eu-west-1-")

def test_rotate_n_triggers_flush(tmp_path):
    w = RotatingParquetWriter(str(tmp_path), rotate_n=3, clock=lambda: 60.0)
    for _ in range(7):
        w.write(_row("probe_rpc", 1))
    files = glob.glob(str(tmp_path / "events/day=*/source=probe_rpc/*.parquet"))
    assert len(files) == 2          # 3 + 3 flushed, 1 still buffered
