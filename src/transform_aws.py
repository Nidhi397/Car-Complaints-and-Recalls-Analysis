import sys
import io
import os
import json
import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytz
from datetime import datetime, timezone
from awsglue.utils import getResolvedOptions


# ── Timezone ──────────────────────────────────────────────────────────────────

CST = pytz.timezone("America/Chicago")


# ── Args ──────────────────────────────────────────────────────────────────────

args           = getResolvedOptions(sys.argv, ["RAW_BUCKET", "CURATED_BUCKET", "CURATED_PREFIX"])
RAW_BUCKET     = args["RAW_BUCKET"].rstrip("/")
CURATED_BUCKET = args["CURATED_BUCKET"].rstrip("/")
CURATED_PREFIX = args["CURATED_PREFIX"].strip("/")


# ── Constants ─────────────────────────────────────────────────────────────────

NULL_DATE         = pd.Timestamp("9999-12-31")
NULL_INT          = -1
NULL_TEXT         = "N/A"
NULL_REPLACEMENTS = ["", "None", "nan", "NaN", "NULL", "null"]

MAKE_KEYWORD_MAP = [
    ("toyota", "TOYOTA", 0),
    ("ford",   "FORD",   1),
    ("tesla",  "TESLA",  2),
]


# ── Find latest raw file ──────────────────────────────────────────────────────

def find_latest_raw_key(bucket: str) -> str:
    """Find the most recently uploaded JSON file in the bucket root."""
    s3       = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket)

    json_files = [
        obj for obj in response.get("Contents", [])
        if obj["Key"].endswith(".json")
        and "/" not in obj["Key"]
    ]

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in s3://{bucket}/")

    latest = sorted(json_files, key=lambda x: x["LastModified"], reverse=True)[0]
    print(f"  Latest raw file : {latest['Key']} ({latest['LastModified']})")
    return latest["Key"]


# ── Load ──────────────────────────────────────────────────────────────────────

def load_from_s3(bucket: str, key: str) -> pd.DataFrame:
    s3       = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    records  = json.loads(response["Body"].read().decode("utf-8"))
    print(f"  Loaded {len(records)} records from s3://{bucket}/{key}")
    return pd.DataFrame(records)


# ── Cleaning ──────────────────────────────────────────────────────────────────

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df     = df.drop_duplicates()
    print(f"  Duplicates removed : {before - len(df)}")
    return df


def drop_null_fields(df: pd.DataFrame) -> pd.DataFrame:
    null_fields = [
        col for col in df.columns
        if df[col].isnull().sum() + (df[col].astype(str).str.strip() == "").sum() == len(df)
    ]
    if null_fields:
        df = df.drop(columns=null_fields)
        print(f"  Dropped 100% null fields : {null_fields}")
    else:
        print(f"  ✓ No 100% null fields found")
    return df


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_fields = ["injuries", "deaths", "units_affected"]
    for field in numeric_fields:
        if field not in df.columns:
            continue
        df[field] = pd.to_numeric(df[field], errors="coerce")
        filled    = df[field].isnull().sum()
        df[field] = df[field].fillna(NULL_INT).astype("int64")
        print(f"  {field} : {filled} nulls filled with {NULL_INT}")
    return df


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    unknown_fields = ["component", "model"]
    na_fields      = ["body", "consequence", "remedy"]

    for field in unknown_fields:
        if field not in df.columns:
            continue
        df[field] = df[field].astype(str).str.strip()
        df[field] = df[field].replace(to_replace=NULL_REPLACEMENTS, value="Unknown")
        count     = (df[field] == "Unknown").sum()
        print(f"  {field} : {count} nulls filled with 'Unknown'")

    for field in na_fields:
        if field not in df.columns:
            continue
        df[field] = df[field].astype(str).str.strip()
        df[field] = df[field].replace(to_replace=NULL_REPLACEMENTS, value=NULL_TEXT)
        count     = (df[field] == NULL_TEXT).sum()
        print(f"  {field} : {count} nulls filled with '{NULL_TEXT}'")

    return df


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    date_fields = ["incident_date", "recall_date"]

    for field in date_fields:
        if field not in df.columns:
            continue
        missing_flag     = f"{field}_missing"
        parsed           = pd.to_datetime(df[field], errors="coerce", utc=False)
        df[missing_flag] = parsed.isnull()
        missing_count    = df[missing_flag].sum()
        df[field]        = parsed.fillna(NULL_DATE)
        df[field]        = pd.to_datetime(df[field]).dt.normalize()
        print(f"  {field} : {missing_count} nulls → {NULL_DATE.date()}, "
              f"flag stored in '{missing_flag}'")

    if "scraped_at" in df.columns:
        df["scraped_at"] = (
            pd.to_datetime(df["scraped_at"], utc=True, errors="coerce")
            .dt.tz_convert(CST)
        )
        null_count = df["scraped_at"].isnull().sum()
        if null_count > 0:
            print(f"  ⚠ scraped_at : {null_count} nulls — check ingestion script")
        else:
            print(f"  scraped_at : converted to CST")

    return df


def add_event_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unified event date — incident_date for complaints, recall_date for recalls.
    Keeps both originals for individual tracking.
    """
    df["event_date"] = pd.to_datetime(
        np.where(
            df["source"] == "nhtsa_complaints",
            df["incident_date"].astype(str),
            df["recall_date"].astype(str)
        )
    )
    df["event_date_missing"] = df["event_date"] >= pd.Timestamp("9999-12-31")
    missing_count            = df["event_date_missing"].sum()
    print(f"  event_date : {missing_count} records with no valid event date")
    return df


def clean_booleans(df: pd.DataFrame) -> pd.DataFrame:
    bool_fields = ["crash_involved", "fire_involved"]
    for field in bool_fields:
        if field not in df.columns:
            continue
        df[field] = df[field].map(
            lambda x: True if str(x).strip().lower() in ("true", "yes", "1") else False
        )
    return df


# ── Make normalisation ────────────────────────────────────────────────────────

def map_make(raw: str) -> tuple[str, int]:
    normalised = raw.strip().lower().rstrip(".")
    for keyword, code, make_id in MAKE_KEYWORD_MAP:
        if keyword in normalised:
            return code, make_id
    return "UNKNOWN", -1


def normalise_make(df: pd.DataFrame) -> pd.DataFrame:
    df["make_raw"] = df["make"].astype(str).str.strip()
    mapped         = df["make_raw"].apply(map_make)
    df["make"]     = mapped.apply(lambda x: x[0])
    df["make_id"]  = mapped.apply(lambda x: x[1]).astype("int64")

    print(f"  Make mapping results:")
    for code, count in df["make"].value_counts().items():
        print(f"    {code} : {count} records")

    unmapped = df[df["make"] == "UNKNOWN"]["make_raw"].unique()
    if len(unmapped) > 0:
        print(f"  ⚠ Unmapped makes : {unmapped.tolist()}")
    else:
        print(f"  ✓ All makes mapped successfully")

    return df


# ── ETL timestamp ─────────────────────────────────────────────────────────────

def add_etl_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df["etl_timestamp"] = datetime.now(tz=CST)
    print(f"  etl_timestamp : {df['etl_timestamp'].iloc[0]} CST")
    return df


# ── Schema enforcement ────────────────────────────────────────────────────────

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    int_fields    = ["year", "make_id", "week_added", "injuries",
                     "deaths", "units_affected"]
    float_fields  = ["source_weight"]
    string_fields = ["record_id", "source", "make_raw", "make",
                     "model", "component", "body"]
    bool_fields   = ["crash_involved", "fire_involved",
                     "incident_date_missing", "recall_date_missing",
                     "event_date_missing"]

    for field in ["consequence", "remedy"]:
        if field in df.columns:
            string_fields.append(field)

    for field in int_fields:
        if field in df.columns:
            df[field] = (pd.to_numeric(df[field], errors="coerce")
                         .fillna(NULL_INT).astype("int64"))

    for field in float_fields:
        if field in df.columns:
            df[field] = (pd.to_numeric(df[field], errors="coerce")
                         .fillna(0.0).astype("float64"))

    for field in string_fields:
        if field in df.columns:
            df[field] = df[field].astype(str).str.strip()

    for field in bool_fields:
        if field in df.columns:
            df[field] = df[field].astype(bool)

    print(f"  Schema enforced across {len(df.columns)} columns")
    return df


# ── Dimension tables ──────────────────────────────────────────────────────────

def build_dim_make() -> pd.DataFrame:
    return pd.DataFrame([
        {"make_id": 0,  "make": "TOYOTA",  "make_label": "Toyota"},
        {"make_id": 1,  "make": "FORD",    "make_label": "Ford"},
        {"make_id": 2,  "make": "TESLA",   "make_label": "Tesla"},
        {"make_id": -1, "make": "UNKNOWN", "make_label": "Unknown"},
    ])


# ── Save ──────────────────────────────────────────────────────────────────────

def save_to_s3(df: pd.DataFrame, dim_make: pd.DataFrame,
               bucket: str, prefix: str, stem: str):
    s3 = boto3.client("s3")

    # save dim_make
    dim_buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(dim_make), dim_buffer)
    dim_buffer.seek(0)
    dim_key = f"{prefix}/dim_make.parquet" if prefix else "dim_make.parquet"
    s3.put_object(Bucket=bucket, Key=dim_key, Body=dim_buffer.getvalue())
    print(f"  dim_make → s3://{bucket}/{dim_key}")

    # save facts partitioned by make and year
    partitions_written = 0
    for year in make_df["year"].unique():
        partition_df = make_df[make_df["year"] == year]

        # drop partition columns — already encoded in folder path
        partition_df = partition_df.drop(columns=["make", "year"], errors="ignore")

        buffer = io.BytesIO()
        pq.write_table(
            pa.Table.from_pandas(partition_df, preserve_index=False),
            buffer
        )
        buffer.seek(0)
        key = f"facts_{stem}/make={make}/year={year}/data.parquet"
        s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        partitions_written += 1

    print(f"  Facts → {partitions_written} partitions written")
    print(f"  Path  → s3://{bucket}/{prefix}/facts_{stem}/make=*/year=*/")
    print(f"  Shape → {df.shape[0]} rows × {df.shape[1]} columns")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\nAutoPulse Glue Transform")
    print("=" * 60)

    print("\n[ 1 / 9 ] Find latest raw file")
    raw_key = find_latest_raw_key(RAW_BUCKET)
    stem    = os.path.splitext(os.path.basename(raw_key))[0]

    print(f"\n  Source  : s3://{RAW_BUCKET}/{raw_key}")
    print(f"  Target  : s3://{CURATED_BUCKET}/{CURATED_PREFIX}/")

    print("\n[ 2 / 9 ] Load")
    df = load_from_s3(RAW_BUCKET, raw_key)

    print("\n[ 3 / 9 ] Drop duplicates")
    df = drop_duplicates(df)

    print("\n[ 4 / 9 ] Drop null fields")
    df = drop_null_fields(df)

    print("\n[ 5 / 9 ] Clean")
    df = clean_numeric(df)
    df = clean_text(df)
    df = clean_dates(df)
    df = add_event_date(df)
    df = clean_booleans(df)

    print("\n[ 6 / 9 ] Normalise make")
    df = normalise_make(df)

    print("\n[ 7 / 9 ] ETL timestamp")
    df = add_etl_timestamp(df)

    print("\n[ 8 / 9 ] Enforce schema")
    df = enforce_schema(df)

    print("\n[ 9 / 9 ] Save")
    dim_make = build_dim_make()
    save_to_s3(df, dim_make, CURATED_BUCKET, CURATED_PREFIX, stem)

    print("\n" + "=" * 60)
    print("Transform complete.")
    print(f"  Records    : {len(df)}")
    print(f"  Columns    : {len(df.columns)}")
    print(f"  Makes      : {df['make'].value_counts().to_dict()}")
    print(f"  ETL run at : {df['etl_timestamp'].iloc[0]} CST")


main()