import pandas as pd
import pytest

from ayojna.contracts import EXTENT_BYTES, EXTENT_HOURLY_COLUMNS
from ayojna.ingest.build import build
from ayojna.ingest.msr import TICKS_PER_HOUR, IngestError, read_msr_csv, to_extent_hourly
from ayojna.ingest.synth import generate

T0 = 128_166_372_000_000_000


def _write(tmp_path, rows, name="web_0.csv"):
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, header=False, index=False)
    return p


def test_reader_keeps_good_rows_and_names_volume(tmp_path):
    p = _write(tmp_path, [[T0, "web", 0, "Read", 0, 4096, 10], [T0 + 1, "web", 0, "Write", 4096, 4096, 10]])
    df = read_msr_csv(p)
    assert len(df) == 2 and set(df["volume"]) == {"web_0"}


def test_reader_rejects_broken_file(tmp_path):
    p = _write(tmp_path, [[T0, "web", 0, "Read", -5, 4096, 10], [T0, "web", 0, "Oops", 0, 0, 10]])
    with pytest.raises(IngestError):
        read_msr_csv(p)


def test_extent_hourly_counts_and_sequential():
    io = pd.DataFrame({
        "volume": ["v_0"] * 4,
        "ts": [T0, T0 + 10, T0 + 20, T0 + TICKS_PER_HOUR],   # 3 in hour 0, 1 in hour 1
        "is_read": [True, True, False, True],
        "offset": [0, 4096, EXTENT_BYTES, 0],                  # 2nd I/O is sequential to the 1st
        "size": [4096, 4096, 4096, 4096],
    })
    eh = to_extent_hourly(io)
    assert list(eh.columns) == list(EXTENT_HOURLY_COLUMNS)
    e0h0 = eh[(eh.extent_id == 0) & (eh.hour == 0)].iloc[0]
    assert (e0h0.reads, e0h0.writes, e0h0.rand_ratio) == (2, 0, 0.5)
    assert eh[(eh.extent_id == 1) & (eh.hour == 0)].iloc[0].writes == 1
    assert eh[(eh.extent_id == 0) & (eh.hour == 1)].iloc[0].reads == 1


def test_build_on_synthetic_traces(tmp_path):
    generate(tmp_path / "raw", days=1, seed=3)
    table = build(tmp_path / "raw", tmp_path / "lake" / "extent_hourly.csv")
    assert table["hour"].max() <= 23
    assert table["volume"].nunique() == 6
