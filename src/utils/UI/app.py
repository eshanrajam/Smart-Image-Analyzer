# src/utils/UI/app.py
import os
import io
import json
import base64
import datetime
import contextlib
import numpy as np
import streamlit as st
from PIL import Image

# ---- dotenv (optional) ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- TensorFlow (soft hint; not shown to user) ----
import tensorflow as tf
from tensorflow import keras
load_model = keras.models.load_model

# ---- OpenAI (vision) ----
from openai import OpenAI

# ---- Optional: Snowflake + S3 ----
with contextlib.suppress(Exception):
    import snowflake.connector
with contextlib.suppress(Exception):
    import boto3
    from botocore.exceptions import ClientError

# =========================
# Config from environment
# =========================
MODEL_PATH   = os.getenv("MODEL_PATH", "models/tf_saved_model")
LABELS_PATH  = os.getenv("LABELS_PATH", "models/label_classes.npy")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_SEND_IMAGE = os.getenv("OPENAI_SEND_IMAGE", "true").lower() in ("1","true","yes")

# S3 config (for optional upload)
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
S3_BUCKET  = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Snowflake config (optional logging)
SNOW_USER   = os.getenv("SNOWFLAKE_USER")
SNOW_PASS   = os.getenv("SNOWFLAKE_PASSWORD")
SNOW_ACCT   = os.getenv("SNOWFLAKE_ACCOUNT")
SNOW_WH     = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOW_DB     = os.getenv("SNOWFLAKE_DATABASE")
SNOW_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
TABLE_FQN   = f"{SNOW_DB}.{SNOW_SCHEMA}.IMAGE_RESULTS" if SNOW_DB and SNOW_SCHEMA else None

# Tone/length policy
MAX_WORDS = 25
ONE_SENTENCE = True

# Optional label aliases (e.g., "class_4" -> "Golden retriever")
LABEL_ALIASES_PATH = os.getenv("LABEL_ALIASES_PATH", "models/label_aliases.json")

# =========================
# Cached resources
# =========================
@st.cache_resource(show_spinner="Loading model…")
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH)

@st.cache_resource(show_spinner=False)
def get_labels():
    if not os.path.exists(LABELS_PATH):
        return None
    return np.load(LABELS_PATH, allow_pickle=True)

@st.cache_resource(show_spinner=False)
def get_label_aliases():
    if os.path.exists(LABEL_ALIASES_PATH):
        try:
            with open(LABEL_ALIASES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

@st.cache_resource(show_spinner=False)
def get_openai():
    if not OPENAI_API_KEY:
        return None
    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_s3():
    # Return None if bucket not configured
    if not S3_BUCKET:
        return None
    try:
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            return boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION
            )
        return boto3.client("s3", region_name=AWS_REGION)
    except Exception:
        return None

def connect_snowflake():
    if not (SNOW_USER and SNOW_PASS and SNOW_ACCT and SNOW_WH and SNOW_DB and SNOW_SCHEMA):
        return None
    try:
        return snowflake.connector.connect(
            user=SNOW_USER, password=SNOW_PASS, account=SNOW_ACCT,
            warehouse=SNOW_WH, database=SNOW_DB, schema=SNOW_SCHEMA
        )
    except Exception:
        return None

# =========================
# Helper functions
# =========================
def preprocess_bytes(img_bytes: bytes) -> np.ndarray:
    """Resize to 128x128 RGB and scale to [0,1] (matches your training)."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((128, 128))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 128, 128, 3)

def predict_hint(model, labels, img_bytes: bytes, aliases: dict) -> str | None:
    """Optional soft hint from classifier; never shown directly to user."""
    if model is None or labels is None:
        return None
    arr4d = preprocess_bytes(img_bytes)
    probs = model.predict(arr4d, verbose=0)[0]
    idx = int(np.argmax(probs))
    raw = str(labels[idx])
    return aliases.get(raw, raw.replace("_", " ").title())

def image_to_data_url(img_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("utf-8")

def count_words(text: str) -> int:
    return len([w for w in text.strip().split() if w])

def enforce_policy(text: str, target_language: str) -> str:
    """
    Ensure one sentence and <= MAX_WORDS words.
    If the model overshoots, we rewrite with a strict instruction once.
    As a last resort, we truncate hard at MAX_WORDS.
    """
    words = count_words(text)
    if ONE_SENTENCE and (("." in text and text.strip().count(".") > 1) or any(p in text for p in ["?", "!"])):
        # We'll try a rewrite below
        pass
    if words <= MAX_WORDS and (not ONE_SENTENCE or text.strip().count(".") <= 1):
        return text

    client = get_openai()
    if client:
        try:
            rewrite = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Rewrite the following description in {target_language} as exactly ONE sentence, "
                        f"at most {MAX_WORDS} words, preserving meaning, no extra commentary:\n\n{text}"
                    ),
                }],
                temperature=0.0,
                max_tokens=80,
            ).choices[0].message.content.strip()
            if count_words(rewrite) <= MAX_WORDS and rewrite.count(".") <= 1:
                return rewrite
            # fall through to final hard truncate
        except Exception:
            pass

    # Hard truncate as a final guardrail
    toks = text.split()
    return " ".join(toks[:MAX_WORDS]).rstrip(",;:") + "."

def summarize_image_alt_text(img_bytes: bytes, hint: str | None, target_language: str) -> str:
    """
    Ask OpenAI Vision for an alt-text-style description for a blind user.
    Exactly one sentence, <= MAX_WORDS words, in target_language.
    """
    client = get_openai()
    if not client:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env and restart the app.")
    if not OPENAI_SEND_IMAGE:
        raise RuntimeError("OPENAI_SEND_IMAGE is false. Set it true to allow vision captions.")

    data_url = image_to_data_url(img_bytes)
    prompt = (
        "Act as an expert writer of alt text for screen readers. "
        f"Respond in {target_language}. "
        f"Describe the image in one sentence of at most {MAX_WORDS} words. "
        "Focus on visible details for a blind user: notable objects, colors, textures, and spatial layout "
        "(e.g., foreground/background, left/right). State actions if any. "
        "Avoid technical jargon, brands, model IDs, or guesses about identities, ages, or private details. "
        "Do not mention confidence, probabilities, or classification terms."
    )
    if hint:
        prompt += f" Optional context (do not state explicitly): {hint}."

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }],
        temperature=0.3,
        max_tokens=90,
    )
    text = resp.choices[0].message.content.strip()
    return enforce_policy(text, target_language)

def s3_upload_and_get_path(file_bytes: bytes, filename: str) -> str | None:
    """
    Upload the image to S3 under uploads/YYYY/MM/DD/<filename>, return s3://... path.
    """
    s3 = get_s3()
    if not s3 or not S3_BUCKET:
        return None
    today = datetime.datetime.utcnow()
    key = f"uploads/{today:%Y/%m/%d}/{filename}"
    try:
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=file_bytes, ContentType="image/jpeg")
        return f"s3://{S3_BUCKET}/{key}"
    except ClientError:
        return None

def maybe_log_to_snowflake(image_id: str, file_name: str, summary: str, s3_path: str | None):
    if not TABLE_FQN:
        return
    conn = connect_snowflake()
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_FQN} (
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
        cur.execute(f"""
          MERGE INTO {TABLE_FQN} AS tgt
          USING (SELECT %s AS IMAGE_ID, %s AS FILE_NAME, %s AS SUMM, %s AS S3P) AS src
          ON tgt.IMAGE_ID = src.IMAGE_ID
          WHEN MATCHED THEN UPDATE SET
            tgt.FILE_NAME   = src.FILE_NAME,
            tgt.GPT_SUMMARY = src.SUMM,
            tgt.S3_PATH     = COALESCE(src.S3P, tgt.S3_PATH),
            tgt.INFERRED_AT = CURRENT_TIMESTAMP
          WHEN NOT MATCHED THEN INSERT (IMAGE_ID, FILE_NAME, GPT_SUMMARY, S3_PATH, INFERRED_AT)
          VALUES (src.IMAGE_ID, src.FILE_NAME, src.SUMM, src.S3P, CURRENT_TIMESTAMP)
        """, (image_id, file_name, summary, s3_path))
        conn.commit()
    except Exception:
        pass
    finally:
        with contextlib.suppress(Exception):
            cur.close()
            conn.close()

# =========================
# UI
# =========================
st.set_page_config(page_title="Smart Image Analyzer — Alt Text", layout="centered")
st.title("🖼️ Smart Image Analyzer — Descriptive Alt Text")

uploaded = st.file_uploader("Drop a JPG/PNG image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

col1, col2 = st.columns([2,1])
with col1:
    language = st.selectbox(
        "Target language",
        ["English", "Spanish", "French", "German", "Hindi", "Arabic", "Chinese", "Japanese", "Korean", "Portuguese", "Italian"],
        index=0
    )
with col2:
    save_s3 = st.checkbox("Save upload to S3 and link in Snowflake", value=False, help="Stores the image in your S3 bucket and records the s3:// path.")

analyze_btn = st.button("Generate Description", type="primary", use_container_width=True, disabled=(uploaded is None))

# Preview
if uploaded is not None:
    st.image(uploaded, caption=uploaded.name, use_column_width=True)

if uploaded is None:
    st.info("Upload an image, pick a language, then click **Generate Description**.")
elif analyze_btn:
    try:
        if not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY is not set in your environment.")
        elif not OPENAI_SEND_IMAGE:
            st.error("OPENAI_SEND_IMAGE=false. Set it true to enable vision captions.")
        else:
            # Optional classifier hint (never shown directly)
            model   = get_model()
            labels  = get_labels()
            aliases = get_label_aliases()

            img_bytes = uploaded.getvalue()
            hint = predict_hint(model, labels, img_bytes, aliases)

            # Generate alt text with strict policy
            summary = summarize_image_alt_text(img_bytes, hint=hint, target_language=language)

            # Optionally upload to S3 and capture path
            s3_path = None
            if save_s3:
                s3_path = s3_upload_and_get_path(img_bytes, uploaded.name)
                if s3_path:
                    st.caption(f"S3 stored at: {s3_path}")
                else:
                    st.warning("Could not upload to S3 (check AWS creds, bucket, and region).")

            st.success(summary)

            # Best-effort log to Snowflake (stores GPT_SUMMARY and S3_PATH if available)
            image_id = f"upload:{uploaded.name}"
            maybe_log_to_snowflake(image_id=image_id, file_name=uploaded.name, summary=summary, s3_path=s3_path)

    except Exception as e:
        st.error(f"Description failed: {e}")

with st.expander("Notes"):
    st.caption(f"• Generates alt-text style captions: exactly one sentence, up to {MAX_WORDS} words, in your chosen language.")
    st.caption("• We avoid sensitive attributes, identities, probabilities, and brand/model IDs.")
    st.caption("• The classifier (if present) is used only as a soft hint to improve faithfulness; it is never shown.")
    st.caption("• Toggle S3 saving to archive the image and store its s3:// path in Snowflake (if configured).")
