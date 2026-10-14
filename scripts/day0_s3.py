#!/usr/bin/env python3
"""Day-0 stage S3 ($0): bucket (or prefix), one presigned PUT and one presigned GET of 1 KB.

Tries to create bucket `jaga-calib` in eu-west1 with the S3 key it is given. If the key cannot create
buckets, it falls back to the prefix `jaga/` in an existing bucket (`--fallback-bucket`). The choice is
saved to ~/.jaga/storage.json for jaga/sync.py. Presigned URLs carry signatures, so the evidence file
records only status codes, sizes and the object key — never a URL or a credential.

Usage:  JAGA_S3_ENV_FILE=<env file> python scripts/day0_s3.py --fallback-bucket <bucket>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaga import sync  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="jaga-calib")
    ap.add_argument("--fallback-bucket", default=os.environ.get("JAGA_FALLBACK_BUCKET", ""))
    ap.add_argument("--prefix", default="jaga/")
    ap.add_argument("--out", default="evidence/day0")
    args = ap.parse_args()

    client = sync.s3_client()
    ev = {"stage": "S3", "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "endpoint": sync.S3_ENDPOINT,
          "region": sync.S3_REGION}
    try:
        client.create_bucket(Bucket=args.bucket)
        ev["create_bucket"] = "created"
        loc = sync.Location(args.bucket, "")
    except Exception as e:  # botocore ClientError
        code = getattr(e, "response", {}).get("Error", {}).get("Code", type(e).__name__)
        ev["create_bucket"] = f"failed: {code}"
        if code in ("BucketAlreadyOwnedByYou",):
            loc = sync.Location(args.bucket, "")
        elif args.fallback_bucket:
            loc = sync.Location(args.fallback_bucket, args.prefix)
        else:
            print(json.dumps(ev))
            return 1
    ev["mode"] = "bucket" if loc.prefix == "" else "prefix"
    ev["bucket"], ev["prefix"] = loc.bucket, loc.prefix

    payload = os.urandom(1024)
    name = f"day0/s3-check-{int(time.time())}.bin"
    t0 = time.perf_counter()
    put_status = sync.put_bytes(sync.presign(client, "PUT", loc, name, 600), payload)
    t1 = time.perf_counter()
    get_status, body = sync.get_bytes(sync.presign(client, "GET", loc, name, 600))
    t2 = time.perf_counter()
    ev.update({
        "object_key": loc.key(name),
        "presigned_put_status": put_status,
        "presigned_get_status": get_status,
        "bytes": len(payload),
        "roundtrip_equal": body == payload,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "put_s": round(t1 - t0, 3),
        "get_s": round(t2 - t1, 3),
    })
    try:
        client.delete_object(Bucket=loc.bucket, Key=loc.key(name))
        ev["cleanup"] = "deleted"
    except Exception as e:
        ev["cleanup"] = f"failed: {type(e).__name__}"
    ev["pass"] = put_status in (200, 201) and get_status == 200 and ev["roundtrip_equal"]
    if ev["pass"]:
        sync.save_location(loc)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "s3-stage.json").write_text(json.dumps(ev, indent=2) + "\n")
    print(json.dumps(ev))
    return 0 if ev["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
