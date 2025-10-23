# src/utils/register_s3_metadata.py
"""
Scan S3 for images, extract metadata (size, last_modified, content-type, sha256, width/height),
derive label from folder name, then MERGE into Snowflake IMAGE_RESULTS.

Usage (from project root, venv active):
  python src/utils/register_s3_metadata.py --prefix raw/
  python src/utils/register_s3_metadata.py --prefix processed/    # optional second run
"""

import os
import io
import sys
import argparse
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Iterable, Tuple

from dotenv import load_dotenv
load_dotenv()

import boto3
from botocore.exceptions import ClientError
from PIL import Image

import snowflake.connector

AWS_REGION   = os.getenv("AWS_REGION", "us-east-2")
S3_BUCKET    = os.getenv("S3_BUCKET")
AWS_KEY      = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET   = os.getenv("AWS_SECRET_ACCESS_KEY")

SNOW_USER    = os.getenv("SNOWFLAKE_USER")
SNOW_PASS    = os.getenv("SNOWFLAKE_PASSWORD")
SNOW_ACCT    = os.getenv("SNOWFLAKE_ACCOUNT")
SNOW_WH      = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOW_DB      = os.getenv("SNOWFLAKE_DATABASE")
SNOW_SCHEMA  = os.getenv("SNOWFLAKE_SCHEMA")

ALLOWED_EXT = (".jpg", ".jpeg", ".png")

def s3_client():
    if AWS_KEY and AWS_SECRET:
        return boto3.client("s3", region_name=AWS_REGION,
                            aws_access_key_id=AWS_KEY,
                            aws_secret_access_key=AWS_SECRET)
    return boto3.client("s3", region_name=AWS_REGION)

def list_objects(bucket: str, prefix: str) -> Iterable[Dict[str, Any]]:
    s3 = s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(ALLOWED_EXT):
                yield {
                    "Key": key,
                    "Size": obj["Size"],
                    "LastModified": obj["LastModified"]
                }

def head_object(bucket: str, key: str) -> Dict[str, Any]:
    s3 = s3_client()
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        print("head_object error:", e)
        return {}

def read_bytes_for_probe(bucket: str, key: str, max_bytes: int = 2_000_000) -> bytes:
    """Read up to max_bytes of object for SHA and image size probe."""
    s3 = s3_client()
    # If file is small, just get full; else use Range to keep it light
    try:
        return s3.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{max_bytes-1}")["Body"].read()
    except ClientError:
        # Fallback to full get if Range not supported
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

def probe_image_dims(img_bytes: bytes) -> Tuple[int, int]:
    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        return None, None

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def derive_label_from_key(key: str, source_prefix: str) -> str:
    """
    Assumes keys like 'raw/<label>/file.jpg' or 'processed/<label>/file.jpg'
    Returns <label> (folder name immediately under prefix), else ''.
    """
    cleaned = key[len(source_prefix):] if key.startswith(source_prefix) else key
    parts = cleaned.split("/")
    return parts[0] if len(parts) > 1 else ""

def connect_snowflake():
    return snowflake.connector.connect(
        user=SNOW_USER,
        password=SNOW_PASS,
        account=SNOW_ACCT,
        warehouse=SNOW_WH,
        database=SNOW_DB,
        schema=SNOW_SCHEMA,
    )

def ensure_table(cur):
    cur.execute("""
    CREATE TABLE IF NOT EXISTS IMAGE_RESULTS (
      IMAGE_ID        STRING,
      FILE_NAME       STRING,
      S3_PATH         STRING,
      SIZE_BYTES      NUMBER,
      LAST_MODIFIED   TIMESTAMP_NTZ,
      CONTENT_TYPE    STRING,
      SHA256          STRING,
      WIDTH           NUMBER,
      HEIGHT          NUMBER,
      LABEL           STRING,
      SOURCE          STRING,
      PREDICTED_LABEL STRING,
      CONFIDENCE      FLOAT,
      GPT_SUMMARY     STRING,
      INFERRED_AT     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP
    );
    """)

def merge_batch(rows: Iterable[Tuple]):
    """
    rows: tuples in this order:
      (IMAGE_ID, FILE_NAME, S3_PATH, SIZE_BYTES, LAST_MODIFIED, CONTENT_TYPE,
       SHA256, WIDTH, HEIGHT, LABEL, SOURCE)
    """
    if not rows:
        return

    conn = connect_snowflake()
    cur  = conn.cursor()

    ensure_table(cur)

    # Create a temp table for staging the batch
    cur.execute("""
      CREATE TEMP TABLE TMP_IMAGE_META (
        IMAGE_ID STRING,
        FILE_NAME STRING,
        S3_PATH STRING,
        SIZE_BYTES NUMBER,
        LAST_MODIFIED TIMESTAMP_NTZ,
        CONTENT_TYPE STRING,
        SHA256 STRING,
        WIDTH NUMBER,
        HEIGHT NUMBER,
        LABEL STRING,
        SOURCE STRING
      )
    """)

    cur.executemany("""
      INSERT INTO TMP_IMAGE_META
      (IMAGE_ID, FILE_NAME, S3_PATH, SIZE_BYTES, LAST_MODIFIED, CONTENT_TYPE, SHA256, WIDTH, HEIGHT, LABEL, SOURCE)
      VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, list(rows))

    # Idempotent upsert on IMAGE_ID
    cur.execute("""
      MERGE INTO IMAGE_RESULTS AS tgt
      USING TMP_IMAGE_META AS src
      ON tgt.IMAGE_ID = src.IMAGE_ID
      WHEN MATCHED THEN UPDATE SET
        tgt.FILE_NAME     = src.FILE_NAME,
        tgt.S3_PATH       = src.S3_PATH,
        tgt.SIZE_BYTES    = src.SIZE_BYTES,
        tgt.LAST_MODIFIED = src.LAST_MODIFIED,
        tgt.CONTENT_TYPE  = src.CONTENT_TYPE,
        tgt.SHA256        = src.SHA256,
        tgt.WIDTH         = src.WIDTH,
        tgt.HEIGHT        = src.HEIGHT,
        tgt.LABEL         = src.LABEL,
        tgt.SOURCE        = src.SOURCE
      WHEN NOT MATCHED THEN INSERT
        (IMAGE_ID, FILE_NAME, S3_PATH, SIZE_BYTES, LAST_MODIFIED, CONTENT_TYPE, SHA256, WIDTH, HEIGHT, LABEL, SOURCE)
      VALUES
        (src.IMAGE_ID, src.FILE_NAME, src.S3_PATH, src.SIZE_BYTES, src.LAST_MODIFIED, src.CONTENT_TYPE, src.SHA256, src.WIDTH, src.HEIGHT, src.LABEL, src.SOURCE)
    """)

    conn.commit()
    cur.close()
    conn.close()

def run(prefix: str, compute_sha_and_dims: bool = True, batch_size: int = 200):
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set in .env")

    print(f"Scanning s3://{S3_BUCKET}/{prefix} ...")
    batch = []
    count = 0

    for obj in list_objects(S3_BUCKET, prefix):
        key   = obj["Key"]
        sz    = int(obj["Size"])
        lm    = obj["LastModified"].astimezone(timezone.utc).replace(tzinfo=None)  # TIMESTAMP_NTZ
        head  = head_object(S3_BUCKET, key)
        ctype = head.get("ContentType")

        sha   = None
        width = None
        height= None

        if compute_sha_and_dims:
            # Read a small chunk (or full if small) to compute sha and probe dims
            b = read_bytes_for_probe(S3_BUCKET, key)
            sha = sha256_bytes(b)
            w,h = probe_image_dims(b)
            width, height = (int(w) if w else None), (int(h) if h else None)

        label  = derive_label_from_key(key, prefix)
        source = prefix.split("/")[0] if "/" in prefix else prefix.rstrip("/")

        row = (
            key,                        # IMAGE_ID
            os.path.basename(key),      # FILE_NAME
            f"s3://{S3_BUCKET}/{key}",  # S3_PATH
            sz,                         # SIZE_BYTES
            lm,                         # LAST_MODIFIED
            ctype,                      # CONTENT_TYPE
            sha,                        # SHA256
            width,                      # WIDTH
            height,                     # HEIGHT
            label,                      # LABEL
            source                      # SOURCE
        )
        batch.append(row)

        if len(batch) >= batch_size:
            merge_batch(batch)
            count += len(batch)
            print(f"Upserted {count} rows...")
            batch = []

    if batch:
        merge_batch(batch)
        count += len(batch)
        print(f"Upserted {count} rows total.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="raw/", help="S3 prefix to scan (e.g. raw/ or processed/)")
    parser.add_argument("--fast", action="store_true", help="Skip SHA256 and dimension probe for speed")
    args = parser.parse_args()

    run(prefix=args.prefix, compute_sha_and_dims=not args.fast)
    print("✅ Metadata registration complete.")
