"""Synthetic block I/O traces in MSR Cambridge CSV format.

Use this while the real traces download, and in tests. Each volume gets a
realistic shape: a few hot extents, a warm middle, a long cold tail, a daily
rhythm, and some special patterns (nightly backup, a late spike).

MSR format, one I/O per line, no header:
    Timestamp,Hostname,DiskNumber,Type,Offset,Size,ResponseTime
Timestamp and ResponseTime are Windows filetime units (100 ns ticks).

Run:  python -m ayojna.ingest.synth --days 3 --out data/raw/synth
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ayojna.contracts import EXTENT_BYTES

TICKS_PER_SECOND = 10_000_000
TRACE_START_TICKS = 128_166_372_000_000_000  # Feb 2007, like the real MSR traces


@dataclass(frozen=True)
class VolumeProfile:
    name: str            # MSR style: host_disk, e.g. "web_0"
    n_extents: int
    ios_per_hour: float  # average at the daily peak
    write_share: float
    seq_share: float     # share of I/Os that are sequential runs
    io_kb: int
    pattern: str         # "office", "nightly", "steady", "late_spike"


PROFILES = [
    VolumeProfile("web_0", 120, 900, 0.35, 0.2, 8, "office"),
    VolumeProfile("src1_2", 80, 500, 0.5, 0.3, 16, "office"),
    VolumeProfile("prn_0", 60, 300, 0.9, 0.8, 64, "nightly"),
    VolumeProfile("hm_0", 50, 250, 0.7, 0.5, 8, "steady"),
    VolumeProfile("usr_0", 150, 400, 0.4, 0.2, 16, "office"),
    VolumeProfile("proj_0", 100, 350, 0.3, 0.6, 32, "late_spike"),
]


def _hour_factor(pattern: str, hour: int, n_hours: int) -> float:
    h = hour % 24
    day = hour // 24
    if pattern == "office":
        return 1.0 if 9 <= h <= 18 else 0.15
    if pattern == "nightly":
        return 1.0 if 1 <= h <= 3 else 0.05
    if pattern == "late_spike":
        return 3.0 if day == (n_hours // 24) - 1 and 10 <= h <= 16 else 0.3
    return 0.6  # steady


def _extent_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    """~5% hot, ~15% warm, the rest a long cold tail (many never touched)."""
    w = np.full(n, 0.002)
    order = rng.permutation(n)
    n_hot, n_warm = max(1, n // 20), max(1, n * 3 // 20)
    w[order[:n_hot]] = 1.0
    w[order[n_hot:n_hot + n_warm]] = 0.08
    w[order[n_hot + n_warm:][: n // 2]] = 0.0  # half the tail is never touched
    return w / w.sum()


def generate_volume(p: VolumeProfile, days: int, rng: np.random.Generator) -> pd.DataFrame:
    host, disk = p.name.rsplit("_", 1)
    n_hours = days * 24
    weights = _extent_weights(p.n_extents, rng)
    io_bytes = p.io_kb * 1024
    rows = []
    for hour in range(n_hours):
        n = rng.poisson(p.ios_per_hour * _hour_factor(p.pattern, hour, n_hours))
        if n == 0:
            continue
        extents = rng.choice(p.n_extents, size=n, p=weights)
        within = rng.integers(0, EXTENT_BYTES // io_bytes - 64, size=n) * io_bytes
        offsets = extents.astype(np.int64) * EXTENT_BYTES + within
        # sequential runs: some I/Os continue right after the previous one
        seq = rng.random(n) < p.seq_share
        for i in np.flatnonzero(seq):
            if i > 0:
                offsets[i] = offsets[i - 1] + io_bytes
        secs = np.sort(rng.uniform(0, 3600, size=n)) + hour * 3600
        rows.append(pd.DataFrame({
            "Timestamp": TRACE_START_TICKS + (secs * TICKS_PER_SECOND).astype(np.int64),
            "Hostname": host,
            "DiskNumber": int(disk),
            "Type": np.where(rng.random(n) < p.write_share, "Write", "Read"),
            "Offset": offsets,
            "Size": io_bytes,
            "ResponseTime": rng.lognormal(mean=9.0, sigma=0.6, size=n).astype(np.int64),
        }))
    return pd.concat(rows, ignore_index=True)


def generate(out_dir: str | Path, days: int = 3, seed: int = 7) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for p in PROFILES:
        df = generate_volume(p, days, rng)
        path = out_dir / f"{p.name}.csv"
        df.to_csv(path, header=False, index=False)
        paths.append(path)
        print(f"{path}: {len(df):,} I/Os")
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Write synthetic MSR-format traces.")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/raw/synth")
    args = ap.parse_args()
    generate(args.out, args.days, args.seed)
