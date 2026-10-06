"""Build data/lake/extent_hourly from a folder of MSR CSV files.

Run:  python -m ayojna.ingest.build --raw data/raw/synth --out data/lake/extent_hourly.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ayojna.ingest.msr import read_msr_csv, to_extent_hourly
from ayojna.io import write_table


def build(raw_dir: str | Path, out_path: str | Path) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no .csv files in {raw_dir}")
    ios = []
    for f in files:
        df = read_msr_csv(f)
        print(f"{f.name}: {len(df):,} I/Os  (rejected {df.attrs['rejected_rows']}, duplicates {df.attrs['duplicate_rows']})")
        ios.append(df)
    io = pd.concat(ios, ignore_index=True)
    start = int(io["ts"].min())          # one clock for every volume
    table = to_extent_hourly(io, start_ts=start)
    write_table(table, out_path)
    print(f"\nextent_hourly: {len(table):,} rows, {table['volume'].nunique()} volumes, "
          f"{table.groupby('volume')['extent_id'].nunique().sum():,} active extents, "
          f"hours 0-{int(table['hour'].max())} -> {out_path}")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MSR CSVs -> extent_hourly table")
    ap.add_argument("--raw", default="data/raw/synth")
    ap.add_argument("--out", default="data/lake/extent_hourly.parquet")
    a = ap.parse_args()
    build(a.raw, a.out)
