# Smart Image Analyzer with AI-Generated Insights

Smart Image Analyzer (Alt-Text + Vision Pipeline)

Upload an image → get a concise, accessibility-grade description.
The app uses OpenAI Vision to produce one-sentence alt text (≤25 words, multilingual), with optional TensorFlow/Keras classification as a soft hint. Results can be logged to Snowflake and uploads archived to AWS S3. Includes batch/ETL utilities and (optional) a FastAPI inference service with MLflow tracking.

✨ Features

Streamlit UI (primary): drag-and-drop a JPG/PNG and receive an alt-text style summary (colors, layout, objects), suitable for screen readers and SEO.

Policy guardrails: exactly one sentence and ≤ 25 words, in a selected language (English, Spanish, French, etc.).

OpenAI Vision integration: generates natural, human-friendly descriptions.

Optional TF/Keras hint: a local MobileNetV2 classifier can provide a private label “hint” (never shown) to make captions more specific if your classes are domain-specific.

Optional S3 archiving: save uploads to S3 under uploads/YYYY/MM/DD/... and store the s3:// path.

Snowflake logging: upsert summaries (and predictions/metadata, if enabled) into IMAGE_RESULTS.

Pipelines included:

S3 → Snowflake metadata ingestor (size, MIME, sha256, width/height).

Batch inference (read from Snowflake, predict, and write back).

FastAPI microservice for real-time inference (if you need an API).

MLflow for experiment tracking (optional).

🧱 Tech Stack

Languages: Python, SQL
ML/CV: TensorFlow/Keras (MobileNetV2), scikit-learn, OpenCV/Pillow
Generative AI: OpenAI API (Vision + Chat)
Serving/UI: Streamlit (UI), FastAPI (optional API)
MLOps: MLflow (optional), Docker
Cloud/Data: AWS S3 (boto3), Snowflake (Snowflake Connector for Python, MERGE/UPSERT)
Tooling: Jupyter, Pandas/NumPy, python-dotenv

📁 Repository Structure (high-level)
.
├─ src/
│  ├─ utils/
│  │  ├─ UI/
│  │  │  └─ app.py                    # Streamlit: upload → OpenAI Vision (multilingual, ≤25 words)
│  │  ├─ register_s3_metadata.py      # S3 → Snowflake metadata ingestor (size, MIME, sha256, WxH)
│  │  └─ upload_to_s3.py              # Helper to bulk upload local images to S3 (optional)
│  ├─ api/
│  │  └─ main.py                      # FastAPI inference (optional)
│  ├─ pipeline/
│  │  └─ batch_infer.py               # Batch predictions + optional OpenAI summaries (optional)
│  ├─ preprocessing/
│  │  └─ preprocess.py                # Example preprocessing to NumPy (optional)
│  └─ train.py                        # TF/Keras MobileNetV2 training + MLflow autolog (optional)
├─ models/
│  ├─ tf_saved_model/                 # Saved Keras model (if training used)
│  ├─ label_classes.npy               # Numpy array of class names (for TF hint)
│  └─ label_aliases.json              # Optional map: {"class_4": "Golden retriever", ...}
├─ docker/
│  └─ Dockerfile                      # Optional container for API/UI
├─ requirements.txt
├─ .env.example
└─ README.md

✅ Prerequisites

Python 3.10–3.11

OpenAI API key with access to vision models

(Optional) AWS S3 bucket & credentials if you want to archive uploads

(Optional) Snowflake account & warehouse if you want to log results

(Optional) Docker if you want containers

⚙️ Setup

Clone & create venv
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
Install dependencies
pip install -r requirements.txt
Create .env from the template:
# ==== OpenAI ====
OPENAI_API_KEY=sk-...
OPENAI_SEND_IMAGE=true          # set false to avoid sending images (caption quality will be lower)

# ==== Streamlit / optional TF hint ====
# USE_TF_HINT=false             # set true if you trained/saved a TF model at models/tf_saved_model

# ==== Optional: Snowflake logging ====
SNOWFLAKE_USER=...
SNOWFLAKE_PASSWORD=...
SNOWFLAKE_ACCOUNT=youracct-xy123
SNOWFLAKE_WAREHOUSE=IMAGE_ANALYZER_WH
SNOWFLAKE_DATABASE=IMAGE_ANALYZER_DB
SNOWFLAKE_SCHEMA=IMAGE_SCHEMA

# ==== Optional: S3 archiving of uploads ====
AWS_REGION=us-east-2
S3_BUCKET=your-bucket
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# ==== Optional: Training & MLflow ====
MLFLOW_TRACKING_URI=http://localhost:5000

🚀 Run the Streamlit App
# If your UI path is src/utils/UI/app.py
streamlit run src/utils/UI/app.py

# If you placed it at src/ui/app.py, use:
# streamlit run src/ui/app.py
Use the app:

Upload a JPG/PNG.

Pick a language (e.g., English).

(Optional) Check “Save upload to S3 and link in Snowflake” to archive the file and store its s3:// path.

Click Generate Description → you’ll get a one-sentence, <=25-word alt-text style caption.

If Snowflake is configured, a row will be upserted into IMAGE_RESULTS with GPT_SUMMARY (and S3 path if enabled).

🔒 Security & Privacy

OpenAI Vision: When OPENAI_SEND_IMAGE=true, images are sent to OpenAI to generate captions. If you need to avoid this, set OPENAI_SEND_IMAGE=false (captions become label-based and less descriptive), or disable the OpenAI step entirely.

PII & sensitive attributes: The prompt instructs the model not to guess identities, ages, or private attributes.

Secrets: Keep secrets in .env (never commit it). Use role-based access for S3/Snowflake.

Data retention: With S3 archiving enabled, you control where uploads are stored and for how long.

🙌 Acknowledgments

Imagenette (optional sample dataset)

OpenAI Vision for generative captions

TensorFlow/Keras MobileNetV2 for classification hinting
