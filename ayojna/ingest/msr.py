"""Read MSR Cambridge block traces and turn them into the extent_hourly table.

One raw row = one I/O. One output row = one 256 MB extent in one hour:
how many reads/writes it got, how many bytes, average I/O size, and how
random the access was. Every model downstream starts from this table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ayojna.contracts import EXTENT_BYTES, EXTENT_HOURLY_COLUMNS, validate_frame

MSR_COLUMNS = ["Timestamp", "Hostname", "DiskNumber", "Type", "Offset", "Size", "ResponseTime"]
TICKS_PER_HOUR = 3600 * 10_000_000  # Windows filetime: 100 ns ticks
MAX_REJECT_SHARE = 0.05             # more than 5% bad rows = treat the file as broken


class IngestError(RuntimeError):
    pass


def read_msr_csv(path: str | Path, volume: str | None = None) -> pd.DataFrame:
    """Load one MSR CSV. Returns volume, ts, is_read, offset, size (clean rows only)."""
    path = Path(path)
    raw = pd.read_csv(path, header=None, names=MSR_COLUMNS)
    n_raw = len(raw)
    if n_raw == 0:
        raise IngestError(f"{path.name}: empty file")

    ts = pd.to_numeric(raw["Timestamp"], errors="coerce")
    offset = pd.to_numeric(raw["Offset"], errors="coerce")
    size = pd.to_numeric(raw["Size"], errors="coerce")
    kind = raw["Type"].astype(str).str.strip().str.lower()

    ok = ts.notna() & offset.notna() & size.notna() & (offset >= 0) & (size > 0) & kind.isin(["read", "write"])
    rejected = n_raw - int(ok.sum())
    if rejected / n_raw > MAX_REJECT_SHARE:
        raise IngestError(f"{path.name}: {rejected}/{n_raw} rows rejected, file looks broken")

    df = pd.DataFrame({
        "volume": volume or path.stem,
        "ts": ts[ok].astype("int64"),
        "is_read": (kind[ok] == "read"),
        "offset": offset[ok].astype("int64"),
        "size": size[ok].astype("int64"),
    })
    before = len(df)
    df = df.drop_duplicates(subset=["ts", "offset", "size", "is_read"])
    df.attrs["rejected_rows"] = rejected
    df.attrs["duplicate_rows"] = before - len(df)
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def to_extent_hourly(io: pd.DataFrame, start_ts: int | None = None) -> pd.DataFrame:
    """Aggregate I/Os into one row per (volume, extent, hour).

    start_ts: tick that counts as hour 0. Pass the earliest timestamp across ALL
    volumes so every volume shares the same clock.
    """
    if io.empty:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in EXTENT_HOURLY_COLUMNS.items()})
    io = io.sort_values(["volume", "ts"], kind="stable").copy()
    start = int(io["ts"].min()) if start_ts is None else int(start_ts)
    if (io["ts"] < start).any():
        raise IngestError("events before start_ts: pass the global minimum timestamp")

    io["hour"] = (io["ts"] - start) // TICKS_PER_HOUR
    io["extent_id"] = io["offset"] // EXTENT_BYTES
    # sequential = starts exactly where the previous I/O on the same volume ended
    prev_end = (io["offset"] + io["size"]).groupby(io["volume"]).shift(1)
    io["sequential"] = (io["offset"] == prev_end)
    io["read_bytes"] = np.where(io["is_read"], io["size"], 0)
    io["write_bytes"] = np.where(io["is_read"], 0, io["size"])

    g = io.groupby(["volume", "extent_id", "hour"], sort=True)
    out = g.agg(
        reads=("is_read", "sum"),
        ios=("is_read", "size"),
        read_bytes=("read_bytes", "sum"),
        write_bytes=("write_bytes", "sum"),
        avg_io_size=("size", "mean"),
        seq_share=("sequential", "mean"),
    ).reset_index()
    out["writes"] = out["ios"] - out["reads"]
    out["rand_ratio"] = 1.0 - out["seq_share"]
    return validate_frame(out, EXTENT_HOURLY_COLUMNS, "extent_hourly")
