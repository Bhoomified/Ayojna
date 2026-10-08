"""Read and write tables in the data lake (data/lake), plus safe whole-file writes."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)  # needs pyarrow (in requirements.txt)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"unsupported table format: {path.suffix}")
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"table not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table format: {path.suffix}")


def atomic_write_text(path: str | Path, text: str, tries: int = 20) -> None:
    """Write a whole file atomically (temp file + rename). Readers never see half a file.

    On Windows the rename fails while another process (e.g. the dashboard) has the
    target open for a moment, so retry briefly instead of crashing.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for i in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.05)