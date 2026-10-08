"""Where extents physically live: one bucket (or folder) per tier.

Two backends with the same 4 methods, so the executor does not care which one runs:
  FsTierStore    - folders on disk (default, no services needed)
  MinioTierStore - real object storage, one bucket per tier (docker compose up -d)
"""

from __future__ import annotations

import io
import os
from pathlib import Path

from ayojna.contracts import TIER_ORDER

TIERS = [t.value for t in TIER_ORDER]


def object_key(volume: str, extent_id: int) -> str:
    return f"{volume}/{int(extent_id):06d}"


class FsTierStore:
    def __init__(self, root: str | Path = "data/tiers"):
        self.root = Path(root)

    def _path(self, tier: str, key: str) -> Path:
        return self.root / tier / key

    def put(self, tier: str, key: str, data: bytes) -> None:
        p = self._path(tier, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)  # atomic: readers never see half an object

    def get(self, tier: str, key: str) -> bytes:
        return self._path(tier, key).read_bytes()

    def exists(self, tier: str, key: str) -> bool:
        return self._path(tier, key).exists()

    def delete(self, tier: str, key: str) -> None:
        self._path(tier, key).unlink(missing_ok=True)


class MinioTierStore:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, prefix: str = "ayojna"):
        from minio import Minio  # optional dependency: pip install minio

        self.client = Minio(
            endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=False
        )
        self.buckets = {t: f"{prefix}-{t}" for t in TIERS}
        for b in self.buckets.values():
            if not self.client.bucket_exists(bucket_name=b):
                self.client.make_bucket(bucket_name=b)

    def put(self, tier: str, key: str, data: bytes) -> None:
        self.client.put_object(
            bucket_name=self.buckets[tier], object_name=key, data=io.BytesIO(data), length=len(data)
        )

    def get(self, tier: str, key: str) -> bytes:
        resp = self.client.get_object(bucket_name=self.buckets[tier], object_name=key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def exists(self, tier: str, key: str) -> bool:
        from minio.error import S3Error

        try:
            self.client.stat_object(bucket_name=self.buckets[tier], object_name=key)
            return True
        except S3Error as exc:
            if exc.code in ("NoSuchKey", "NoSuchObject", "ResourceNotFound"):
                return False
            raise

    def delete(self, tier: str, key: str) -> None:
        self.client.remove_object(bucket_name=self.buckets[tier], object_name=key)


def make_store(kind: str = "fs", root: str | Path = "data/tiers"):
    if kind == "fs":
        return FsTierStore(root)
    if kind == "minio":
        return MinioTierStore(
            os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            os.getenv("MINIO_ROOT_USER", "ayojna"),
            os.getenv("MINIO_ROOT_PASSWORD", "change-me-please"),
        )
    raise ValueError(f"unknown tier store: {kind}")