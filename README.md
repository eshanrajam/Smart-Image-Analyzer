# Smart Image Analyzer — AI-generated alt text

Upload an image → get a concise, accessibility-grade alt text (one sentence, ≤25 words). Uses OpenAI Vision for human-friendly captions with an optional local TensorFlow/Keras classifier as a private hint. Results can be archived to S3 and logged to Snowflake.

## Key features
- Streamlit UI: drag-and-drop JPG/PNG → one-sentence alt text (multilingual).  
- Policy guardrails: exactly one sentence and ≤25 words.  
- OpenAI Vision for natural captions.  
- Optional TF/Keras MobileNetV2 hint (private, not shown).  
- Optional S3 archiving: uploads/YYYY/MM/DD/... and s3:// path saved.  
- Snowflake logging: upsert into IMAGE_RESULTS (GPT_SUMMARY + metadata).  
- Optional: FastAPI microservice, batch inference pipeline, MLflow tracking.

## Tech stack
- Python 3.10–3.11
- OpenAI API (Vision + Chat)
- TensorFlow/Keras, scikit-learn, OpenCV/Pillow
- Streamlit (UI), FastAPI (optional)
- AWS S3 (boto3), Snowflake connector, MLflow, Docker

## Prerequisites
- Python 3.10–3.11
- OpenAI API key with vision access
- (Optional) AWS S3 credentials and bucket
- (Optional) Snowflake account
- (Optional) Docker

## Quick start
1. Clone and create virtualenv:
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # macOS / Linux:
   # source .venv/bin/activate
2. Install:
   pip install -r requirements.txt
3. Copy and edit .env from the template below.
4. Run Streamlit:
   streamlit run src/utils/UI/app.py
   (or streamlit run src/ui/app.py depending on location)

## .env (examples)
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_SEND_IMAGE=true  # false = local-label-only captions

# TF hint
# USE_TF_HINT=false     # set true if models/tf_saved_model exists

# Snowflake (optional)
SNOWFLAKE_USER=...
SNOWFLAKE_PASSWORD=...
SNOWFLAKE_ACCOUNT=youracct-xy123
SNOWFLAKE_WAREHOUSE=IMAGE_ANALYZER_WH
SNOWFLAKE_DATABASE=IMAGE_ANALYZER_DB
SNOWFLAKE_SCHEMA=IMAGE_SCHEMA

# S3 (optional)
AWS_REGION=us-east-2
S3_BUCKET=your-bucket
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# MLflow (optional)
MLFLOW_TRACKING_URI=http://localhost:5000

## Usage
- Upload a JPG/PNG in the Streamlit app.
- Select language and options (S3 archive, Snowflake logging).
- Click "Generate Description" → receives a one-sentence alt-text (≤25 words).
- If Snowflake is enabled, the result and metadata are upserted to IMAGE_RESULTS.

## Security & privacy
- If OPENAI_SEND_IMAGE=true, images are sent to OpenAI; set false to avoid sending images.  
- Prompts avoid guessing identities, ages, or sensitive attributes.  
- Keep secrets in .env and out of version control. Use role-based access controls for S3/Snowflake.

## Acknowledgments
- OpenAI Vision, TensorFlow/Keras MobileNetV2, Imagenette (sample dataset)
