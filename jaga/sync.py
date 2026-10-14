"""Nebius Object Storage: the only path by which home-derived data reaches Nebius.

What goes up: window margins, z, labels (`labelled_margins.csv`). What comes down: `thresholds.json`.
Never frames, never vectors, never thumbnails. The calibration Job never sees credentials: it gets one
presigned GET URL for its input and one presigned PUT URL for its result.
"""
from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

from . import config

S3_REGION = os.environ.get("JAGA_S3_REGION", "eu-west1")
S3_ENDPOINT = os.environ.get("JAGA_S3_ENDPOINT", f"https://storage.{S3_REGION}.nebius.cloud")

# Columns the Job may receive. Anything else is a bug: the egress table lists exactly these.
MARGIN_COLUMNS = ("item_id", "tripwire", "m", "label", "source")


@dataclass(frozen=True)
class Location:
    bucket: str
    prefix: str = ""

    def key(self, name: str) -> str:
        return f"{self.prefix}{name}"


def _read_env_file(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_s3_creds() -> tuple[str, str] | None:
    """Access key from the environment, else from an env file outside the repo (never printed)."""
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"]
    for c in (os.environ.get("JAGA_S3_ENV_FILE"), str(config.config_dir() / "s3.env")):
        if c and Path(c).expanduser().exists():
            env = _read_env_file(Path(c).expanduser())
            key = env.get("NEBIUS_S3_ACCESS_KEY_ID") or env.get("AWS_ACCESS_KEY_ID")
            secret = env.get("NEBIUS_S3_SECRET_ACCESS_KEY") or env.get("AWS_SECRET_ACCESS_KEY")
            if key and secret:
                return key, secret
    return None


def s3_client():
    import boto3
    from botocore.config import Config

    creds = load_s3_creds()
    if not creds:
        raise RuntimeError("no S3 access key (AWS_* env, JAGA_S3_ENV_FILE or ~/.config/jaga/s3.env)")
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        region_name=S3_REGION,
        aws_access_key_id=creds[0],
        aws_secret_access_key=creds[1],
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def storage_file() -> Path:
    return config.jaga_home() / "storage.json"


def resolve_location() -> Location | None:
    """Where calibration data lives: env overrides, else what day-0 stage S3 decided."""
    if os.environ.get("JAGA_BUCKET"):
        return Location(os.environ["JAGA_BUCKET"], os.environ.get("JAGA_PREFIX", ""))
    f = storage_file()
    if f.exists():
        d = json.loads(f.read_text())
        return Location(d["bucket"], d.get("prefix", ""))
    return None


def save_location(loc: Location) -> None:
    storage_file().write_text(json.dumps({"bucket": loc.bucket, "prefix": loc.prefix}, indent=2) + "\n")


def presign(client, method: str, loc: Location, name: str, expires_s: int = 3600 * 6) -> str:
    op = {"PUT": "put_object", "GET": "get_object"}[method]
    return client.generate_presigned_url(op, Params={"Bucket": loc.bucket, "Key": loc.key(name)}, ExpiresIn=expires_s)


def margins_csv(rows: list[dict]) -> bytes:
    """Serialise labelled margins for the Job. Refuses any column outside MARGIN_COLUMNS."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=MARGIN_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        extra = set(r) - set(MARGIN_COLUMNS)
        if extra:
            raise ValueError(f"refusing to upload columns outside the egress table: {sorted(extra)}")
        w.writerow({k: r.get(k, "") for k in MARGIN_COLUMNS})
    return buf.getvalue().encode()


def put_bytes(url: str, data: bytes, content_type: str = "application/octet-stream") -> int:
    import httpx

    r = httpx.put(url, content=data, headers={"Content-Type": content_type}, timeout=60)
    return r.status_code


def get_bytes(url: str) -> tuple[int, bytes]:
    import httpx

    r = httpx.get(url, timeout=60)
    return r.status_code, r.content
