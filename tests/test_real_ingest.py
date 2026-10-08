import gzip
import shutil
import tarfile
import zipfile

import numpy as np
import pandas as pd

from ayojna.ingest.msr import read_msr_csv, to_extent_hourly
from ayojna.ingest.real import build_real, discover
from ayojna.ingest.synth import generate


def _make_download(tmp_path):
    """Fake SNIA download: a .tar with gzipped CSVs in a sub-folder, plus one loose .csv.gz."""
    paths = generate(tmp_path / "plain", days=2, seed=3)
    pack = tmp_path / "pack"
    pack.mkdir()
    for p in paths:
        with open(p, "rb") as f, gzip.open(pack / f"{p.stem}.csv.gz", "wb") as g:
            shutil.copyfileobj(f, g)
    raw = tmp_path / "raw"
    raw.mkdir()
    with tarfile.open(raw / "msr-cambridge1.tar", "w") as tar:
        for p in sorted(pack.glob("*.csv.gz"))[:-1]:
            tar.add(p, arcname=f"MSR-Cambridge/{p.name}")
    last = sorted(pack.glob("*.csv.gz"))[-1]
    shutil.copy(last, raw / last.name)
    return paths, raw


def test_discover_finds_tar_members_and_loose_files(tmp_path):
    paths, raw = _make_download(tmp_path)
    found = discover(raw)
    assert set(found) == {p.stem for p in paths}
    assert sum(s.member is not None for s in found.values()) == len(paths) - 1


def test_streaming_equals_in_memory_ingest(tmp_path):
    paths, raw = _make_download(tmp_path)
    vols = ["web_0", "prn_0", "proj_0"]
    got = build_real(raw, tmp_path / "eh.csv", vols, chunksize=5_000)  # many small chunks
    ios = [read_msr_csv(p) for p in paths if p.stem in vols]
    io = pd.concat(ios, ignore_index=True)
    want = to_extent_hourly(io, start_ts=int(io["ts"].min()))
    keys = ["volume", "extent_id", "hour"]
    got = got.sort_values(keys).reset_index(drop=True)
    want = want.sort_values(keys).reset_index(drop=True)
    assert len(got) == len(want)
    for c in ["extent_id", "hour", "reads", "writes", "read_bytes", "write_bytes"]:
        assert (got[c].to_numpy() == want[c].to_numpy()).all(), c
    assert np.allclose(got["avg_io_size"], want["avg_io_size"])
    assert np.allclose(got["rand_ratio"], want["rand_ratio"], atol=1e-3)


def test_missing_volumes_are_skipped(tmp_path):
    _, raw = _make_download(tmp_path)
    t = build_real(raw, tmp_path / "eh.csv", ["web_0", "nope_9"])
    assert set(t["volume"]) == {"web_0"}


def test_zip_and_nested_folders(tmp_path):
    paths = generate(tmp_path / "plain", days=2, seed=3)
    raw = tmp_path / "raw" / "downloads" / "msr"
    raw.mkdir(parents=True)
    with zipfile.ZipFile(raw / "msr.zip", "w") as z:
        z.write(paths[0], arcname=f"MSR/{paths[0].name}")
    shutil.copy(paths[1], raw / paths[1].name)
    found = discover(tmp_path / "raw")
    assert set(found) == {paths[0].stem, paths[1].stem}
    t = build_real(tmp_path / "raw", tmp_path / "eh.csv", [paths[0].stem, paths[1].stem])
    assert t["volume"].nunique() == 2