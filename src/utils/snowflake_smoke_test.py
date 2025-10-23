# src/utils/snowflake_smoke_test.py
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
import snowflake.connector

def main():
    load_dotenv()  # reads .env in project root

    # Gather env vars
    user = os.getenv("SNOWFLAKE_USER")
    pwd  = os.getenv("SNOWFLAKE_PASSWORD")
    acct = os.getenv("SNOWFLAKE_ACCOUNT")
    wh   = os.getenv("SNOWFLAKE_WAREHOUSE")
    db   = os.getenv("SNOWFLAKE_DATABASE")
    sch  = os.getenv("SNOWFLAKE_SCHEMA")
    bucket = os.getenv("S3_BUCKET", "your-bucket-name")  # optional, used for S3_PATH

    missing = [k for k,v in {
        "SNOWFLAKE_USER":user, "SNOWFLAKE_PASSWORD":pwd, "SNOWFLAKE_ACCOUNT":acct,
        "SNOWFLAKE_WAREHOUSE":wh, "SNOWFLAKE_DATABASE":db, "SNOWFLAKE_SCHEMA":sch
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}. Check your .env file.")

    # Connect
    conn = snowflake.connector.connect(
        user=user,
        password=pwd,
        account=acct,
        warehouse=wh,
        database=db,
        schema=sch,
    )
    cur = conn.cursor()

    try:
        # Print context
        cur.execute("SELECT CURRENT_ROLE(), CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA()")
        role, warehouse, database, schema = cur.fetchone()
        print(f"✅ Context:\n  ROLE={role}\n  WAREHOUSE={warehouse}\n  DATABASE={database}\n  SCHEMA={schema}")

        # Prepare a test row
        image_id = f"smoke-{uuid.uuid4()}"
        file_name = "smoke_test.jpg"
        s3_path = f"s3://{bucket}/raw/{file_name}"
        size_bytes = 12345
        now = datetime.utcnow()

        # Fully-qualified table to avoid context issues
        fq_table = f"{db}.{sch}.IMAGE_RESULTS"

        # Insert test row
        insert_sql = f"""
        INSERT INTO {fq_table}
          (IMAGE_ID, FILE_NAME, S3_PATH, SIZE_BYTES, LAST_MODIFIED, PREDICTED_LABEL, CONFIDENCE, GPT_SUMMARY)
        VALUES
          (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(insert_sql, (
            image_id, file_name, s3_path, size_bytes, now,
            "demo_label", 0.99, "Smoke test insert from Python."
        ))
        conn.commit()
        print(f"✅ Inserted test row with IMAGE_ID={image_id}")

        # Read it back
        select_sql = f"SELECT IMAGE_ID, FILE_NAME, S3_PATH, PREDICTED_LABEL, CONFIDENCE, INFERRED_AT FROM {fq_table} WHERE IMAGE_ID = %s"
        cur.execute(select_sql, (image_id,))
        row = cur.fetchone()
        if row:
            print("✅ Read back row:")
            print("  IMAGE_ID      :", row[0])
            print("  FILE_NAME     :", row[1])
            print("  S3_PATH       :", row[2])
            print("  PREDICTED_LABEL:", row[3])
            print("  CONFIDENCE    :", row[4])
            print("  INFERRED_AT   :", row[5])
        else:
            print("❌ Could not read back the inserted row (unexpected).")

        # OPTIONAL: clean up the test row (uncomment to delete after verifying)
        # cur.execute(f"DELETE FROM {fq_table} WHERE IMAGE_ID = %s", (image_id,))
        # conn.commit()
        # print(f"🧹 Deleted test row IMAGE_ID={image_id}")

    except Exception as e:
        print("❌ Smoke test failed:", e)
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()
