import snowflake.connector
from dotenv import load_dotenv
import os

# Load credentials
load_dotenv()

# Connect to Snowflake
conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA"),
)

cur = conn.cursor()
cur.execute("SELECT CURRENT_VERSION()")
version = cur.fetchone()
print(f"✅ Connected to Snowflake! Version: {version[0]}")

cur.close()
conn.close()
