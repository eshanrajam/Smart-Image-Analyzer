# src/api/main.py
import os
import io
import time
import numpy as np
from typing import Optional

from dotenv import load_dotenv
load_dotenv()


from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware



# Keras 3.x only: do not use tensorflow.keras
import keras
from keras.models import load_model
print("Using standalone keras (Keras 3.x)")

# Image handling
from PIL import Image

# S3 (optional, to read by s3_key)
import boto3
from botocore.exceptions import ClientError


# Snowflake logging (for results)
# import snowflake.connector


# ---- Lazy load model + labels ----
MODEL_PATH = "models/model.keras"
LABELS_PATH = "models/label_classes.npy"
MODEL = None
LABELS = None
MODEL_LOAD_ERROR = None
LABELS_LOAD_ERROR = None
def get_model():
    global MODEL, MODEL_LOAD_ERROR
    if MODEL is None:
        if not load_model:
            MODEL_LOAD_ERROR = "Keras not available."
            return None
        if not os.path.exists(MODEL_PATH):
            MODEL_LOAD_ERROR = f"Model not found at {MODEL_PATH}. Train first."
            return None
        try:
            MODEL = load_model(MODEL_PATH)
        except Exception as e:
            MODEL_LOAD_ERROR = str(e)
            MODEL = None
    return MODEL
def get_labels():
    global LABELS, LABELS_LOAD_ERROR
    if LABELS is None:
        if not os.path.exists(LABELS_PATH):
            LABELS_LOAD_ERROR = f"Labels not found at {LABELS_PATH}. Train first."
            return None
        try:
            LABELS = np.load(LABELS_PATH, allow_pickle=True)
        except Exception as e:
            LABELS_LOAD_ERROR = str(e)
            LABELS = None
    return LABELS


app = FastAPI(title="Smart Image Analyzer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Optional S3 client (if using s3_key mode) ----
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

S3 = None
if S3_BUCKET:
    try:
        S3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
    except Exception:
        S3 = None  # allow local-only mode



# ---- Snowflake connector helper ----
# (Removed for robustness. Add back if needed and ensure import is present.)
if __name__ == "__main__":
    # Startup check for model and labels
    m = get_model()
    l = get_labels()
    if not m:
        print(f"[WARNING] Model not loaded: {MODEL_LOAD_ERROR}")
    if not l:
        print(f"[WARNING] Labels not loaded: {LABELS_LOAD_ERROR}")

# ---- Preprocess helper ----


def preprocess_bytes(img_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((128, 128))
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)  # (1, 128, 128, 3)
        if arr.shape != (1, 128, 128, 3):
            raise ValueError(f"Image array shape {arr.shape} is not (1, 128, 128, 3)")
        return arr
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Image preprocessing error: {ve}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")



def predict_np(array4d: np.ndarray):
    model = get_model()
    labels = get_labels()
    if not model:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {MODEL_LOAD_ERROR}")
    if not labels:
        raise HTTPException(status_code=500, detail=f"Labels not loaded: {LABELS_LOAD_ERROR}")
    try:
        probs = model.predict(array4d, verbose=0)[0]  # (num_classes,)
        if len(probs) != len(labels):
            raise ValueError(f"Model output shape {probs.shape} does not match number of labels {len(labels)}")
        idx = int(np.argmax(probs))
        label = str(labels[idx])
        conf = float(probs[idx])
        return label, conf, probs.tolist()
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=f"Prediction value error: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/")
def root():
    return {"message": "Smart Image Analyzer API is running."}

@app.get("/health")
def health():
    model_ok = get_model() is not None
    labels_ok = get_labels() is not None
    return {
        "model_loaded": model_ok,
        "labels_loaded": labels_ok,
        "model_error": MODEL_LOAD_ERROR,
        "labels_error": LABELS_LOAD_ERROR
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(None, description="Upload an image file"),
    s3_key: Optional[str] = Query(default=None, description="Alternative: S3 key to image, e.g., raw/class/file.jpg")
):
    """
    Either upload an image file OR pass ?s3_key=... (if S3 is configured).
    Returns predicted label + confidence, and logs to Snowflake.
    """
    try:
        start = time.time()
        s3_path = None
        size_bytes = None
        file_name = None
        image_id = None

        if file is not None:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"Uploaded file is not an image: {file.content_type}")
            file_bytes = await file.read()
            size_bytes = len(file_bytes)
            file_name = file.filename or "upload.jpg"
            image_id = f"upload:{file_name}"
        elif s3_key:
            if not (S3 and S3_BUCKET):
                raise HTTPException(status_code=400, detail="S3 not configured. Provide a file upload instead.")
            try:
                obj = S3.get_object(Bucket=S3_BUCKET, Key=s3_key)
                file_bytes = obj["Body"].read()
                size_bytes = obj.get("ContentLength")
                file_name = s3_key.split("/")[-1]
                image_id = f"s3:{s3_key}"
                s3_path = f"s3://{S3_BUCKET}/{s3_key}"
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"S3 error: {e}")
        else:
            raise HTTPException(status_code=400, detail="Provide either a file or an s3_key.")

        arr = preprocess_bytes(file_bytes)
        label, conf, probs = predict_np(arr)

        # Log to Snowflake (best-effort)
        # Snowflake logging disabled

        elapsed = round((time.time() - start) * 1000)
        return JSONResponse({
            "predicted_label": label,
            "confidence": conf,
            "probs": probs,
            "latency_ms": elapsed,
            "file_name": file_name,
            "s3_path": s3_path
        })

    except HTTPException:
        raise
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
