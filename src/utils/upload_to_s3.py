# src/utils/upload_to_s3.py
import os
import sys
import math
import logging
from pathlib import Path
from dotenv import load_dotenv
import argparse
import boto3
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm

# Optional: import for Snowflake registration (uncomment if used)
# import snowflake.connector

# Load .env from project root (one level up from src/)
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)

# config from env
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
BUCKET = os.getenv("S3_BUCKET")

# local folder to upload (change as needed)
LOCAL_FOLDER = "data/raw_images"
S3_PREFIX = "raw/"  # files will be uploaded under this prefix in the bucket

# transfer settings: tune for your network / file sizes
MB = 1024 ** 2
transfer_config = TransferConfig(
    multipart_threshold=10 * MB,      # multipart uploads for files > 10MB
    multipart_chunksize=8 * MB,       # chunk size
    max_concurrency=4,                # parallel threads
    use_threads=True
)

# Setup boto3 client - will use env creds if present, else default credential chain (profile, instance role, etc.)
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
else:
    # fallback to default AWS credential chain (aws configure, role, etc.)
    s3_client = boto3.client("s3", region_name=AWS_REGION)

logger = logging.getLogger("uploader")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(handler)


class TqdmUpload:
    """Progress callback for boto3 uploads which updates a tqdm bar."""
    def __init__(self, filename, filesize):
        self._filename = filename
        self._filesize = filesize
        self._tqdm = tqdm(total=filesize, unit='B', unit_scale=True, desc=Path(filename).name)

    def __call__(self, bytes_amount):
        self._tqdm.update(bytes_amount)
        if self._tqdm.n >= self._filesize:
            self._tqdm.close()


def upload_file(local_path: str, bucket: str, s3_key: str) -> bool:
    """Uploads a single file with a progress bar. Returns True on success."""
    try:
        filesize = os.path.getsize(local_path)
        progress = TqdmUpload(local_path, filesize)
        # set content-type metadata (optional) so browser / clients see correct MIME
        content_type = None
        if local_path.lower().endswith(".jpg") or local_path.lower().endswith(".jpeg"):
            content_type = "image/jpeg"
        elif local_path.lower().endswith(".png"):
            content_type = "image/png"

        extra_args = {"ACL": "private"}
        if content_type:
            extra_args["ContentType"] = content_type

        s3_client.upload_file(
            Filename=local_path,
            Bucket=bucket,
            Key=s3_key,
            ExtraArgs=extra_args,
            Callback=progress,
            Config=transfer_config
        )
        logger.info(f"Uploaded {local_path} -> s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        logger.exception(f"Failed to upload {local_path}: {e}")
        return False


def upload_folder(local_folder: str = LOCAL_FOLDER, bucket: str = BUCKET, prefix: str = S3_PREFIX, allowed_ext=(".jpg", ".jpeg", ".png")):
    """Recursively upload files from local_folder to s3://bucket/prefix/ preserving subfolders."""
    if not bucket:
        raise ValueError("S3_BUCKET not set in environment (.env).")

    uploaded_keys = []
    local_folder = Path(local_folder)
    if not local_folder.exists():
        raise FileNotFoundError(f"Local folder {local_folder} not found.")

    for root, _, files in os.walk(local_folder):
        for fname in files:
            if not fname.lower().endswith(allowed_ext):
                continue
            local_path = Path(root) / fname
            # compute key relative to local_folder and prefix
            rel_path = os.path.relpath(local_path, local_folder).replace("\\", "/")
            s3_key = f"{prefix.rstrip('/')}/{rel_path}"
            success = upload_file(str(local_path), bucket, s3_key)
            if success:
                uploaded_keys.append(s3_key)
                # Optional: register metadata in Snowflake by calling a function here
                # register_in_snowflake(s3_key, bucket, local_path)
    return uploaded_keys


# Optional example: register an S3 path in Snowflake (uncomment and edit if you want)
# def register_in_snowflake(s3_key, bucket, local_path):
#     conn = snowflake.connector.connect(
#         user=os.getenv("SNOWFLAKE_USER"),
#         password=os.getenv("SNOWFLAKE_PASSWORD"),
#         account=os.getenv("SNOWFLAKE_ACCOUNT"),
#         warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
#         database=os.getenv("SNOWFLAKE_DATABASE"),
#         schema=os.getenv("SNOWFLAKE_SCHEMA"),
#     )
#     cur = conn.cursor()
#     cur.execute("""
#     INSERT INTO IMAGE_RESULTS (IMAGE_ID, FILE_NAME, S3_PATH, SIZE_BYTES, LAST_MODIFIED)
#     VALUES (%s, %s, %s, %s, current_timestamp())
#     """, (s3_key, os.path.basename(local_path), f"s3://{bucket}/{s3_key}", os.path.getsize(local_path)))
#     conn.commit()
#     cur.close()
#     conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a local folder to S3 (recursive). Values default to environment/.env settings.")
    parser.add_argument("--bucket", type=str, help="S3 bucket name (overrides S3_BUCKET env var)")
    parser.add_argument("--folder", type=str, help="Local folder to upload (overrides LOCAL_FOLDER)")
    parser.add_argument("--prefix", type=str, help="S3 prefix/key prefix to use (overrides S3_PREFIX)")
    args = parser.parse_args()

    bucket = args.bucket or BUCKET
    folder = args.folder or LOCAL_FOLDER
    prefix = args.prefix or S3_PREFIX

    if not bucket:
        logger.error("S3 bucket not set. Please set S3_BUCKET in your environment or pass --bucket. Example: S3_BUCKET=my-bucket in .env")
        sys.exit(1)

    if not Path(folder).exists():
        logger.error(f"Local folder '{folder}' not found. Create the folder or pass --folder to specify an existing folder.")
        sys.exit(1)

    try:
        logger.info("Starting upload...")
        keys = upload_folder(local_folder=folder, bucket=bucket, prefix=prefix)
        logger.info(f"Done. Uploaded {len(keys)} objects.")
        for k in keys[:50]:
            print(k)
    except Exception as exc:
        logger.exception("Upload failed: %s", exc)
        sys.exit(1)
