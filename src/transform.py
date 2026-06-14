import os
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timezone


# ── Constants ────────────────────────────────────────────────────────────────

NULL_DATE            = pd.Timestamp("9999-12-31")
NULL_INT             = -1
NULL_TEXT            = "N/A"
NULL_SENTIMENT_SCORE = 0.0
NULL_SENTIMENT_ID    = -1
NULL_SENTIMENT_LABEL = "N/A"

SENTIMENT_THRESHOLDS = {
    "critical": -0.5,
    "negative": -0.2,
    "neutral":   0.05,
}

SENTIMENT_IDS = {
    "positive": 3,
    "neutral":  2,
    "negative": 1,
    "critical": 0,
    "N/A":     -1,
}

# order matters — first keyword match wins
MAKE_KEYWORD_MAP = [
    ("toyota", "TOYOTA", 0),
    ("ford",   "FORD",   1),
    ("tesla",  "TESLA",  2),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_paths(json_filename: str) -> dict:
    base_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    validate_dir = os.path.join(base_dir,"validate")
    processed_dir  = os.path.join(base_dir, "data", "processed")
    raw_dir      = os.path.join(base_dir, "data", "raw")
    stem         = os.path.splitext(json_filename)[0]

    return {
        "raw_file":    os.path.join(raw_dir, json_filename),
        "val_file":    os.path.join(validate_dir, f"val_{stem}.txt"),
        "processed_dir": processed_dir,
    }


def find_latest_json(raw_dir: str) -> str:
    files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
    if not files:
        raise FileNotFoundError(f"No JSON files found in {raw_dir}")
    return sorted(files)[-1]


def load_raw(raw_filepath: str) -> pd.DataFrame:
    with open(raw_filepath, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} records from {os.path.basename(raw_filepath)}")
    return pd.DataFrame(records)


def parse_null_fields_from_report(val_filepath: str) -> list[str]:
    """Read validation report and return fields flagged as 100% missing."""
    null_fields = []
    if not os.path.exists(val_filepath):
        print(f"  ⚠ Validation report not found at {val_filepath} — skipping null field removal")
        return null_fields

    with open(val_filepath, "r", encoding="utf-8") as f:
        for line in f:
            if "100.0%" in line and "✗ HIGH" in line:
                field = line.strip().split()[0]
                null_fields.append(field)

    if null_fields:
        print(f"  Fields flagged 100% null in report : {null_fields}")
    else:
        print(f"  ✓ No 100% null fields found in report")

    return null_fields


# ── Cleaning ──────────────────────────────────────────────────────────────────

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df     = df.drop_duplicates()
    after  = len(df)
    print(f"  Duplicates removed : {before - after}")
    return df


def drop_null_fields(df: pd.DataFrame, null_fields: list[str]) -> pd.DataFrame:
    to_drop = [f for f in null_fields if f in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"  Dropped 100% null fields : {to_drop}")
    else:
        print(f"  ✓ No fields dropped")
    return df


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_fields = ["injuries", "deaths", "units_affected"]
    for field in numeric_fields:
        if field not in df.columns:
            continue
        # coerce non-numeric strings to NaN first, then fill NaN with -1
        df[field] = pd.to_numeric(df[field], errors="coerce")
        df[field] = df[field].fillna(NULL_INT).astype("int64")
        null_count = (df[field] == NULL_INT).sum()
        print(f"  {field} : {null_count} nulls filled with {NULL_INT}")
    return df


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    text_fields = ["component", "body", "consequence", "remedy", "model"]
    for field in text_fields:
        if field not in df.columns:
            continue
        df[field] = df[field].astype(str).str.strip()
        # only replace truly empty or Python null string representations
        # preserves existing meaningful defaults like "Unknown", "OTHER" etc
        df[field] = df[field].replace(
            to_replace=["", "None", "nan", "NaN", "NULL", "null"],
            value=NULL_TEXT
        )
        null_count = (df[field] == NULL_TEXT).sum()
        print(f"  {field} : {null_count} nulls filled with {NULL_TEXT}")
    return df


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    date_fields = ["incident_date", "recall_date"]

    for field in date_fields:
        if field not in df.columns:
            continue

        missing_flag = f"{field}_missing"

        # coerce to datetime — invalid or empty strings become NaT
        parsed = pd.to_datetime(df[field], errors="coerce", utc=False)

        # flag missing before filling so we know which were genuinely null
        df[missing_flag] = parsed.isnull()
        missing_count    = df[missing_flag].sum()

        # fill nulls with sentinel date and normalise to date only (no time)
        df[field] = parsed.fillna(NULL_DATE)
        df[field] = pd.to_datetime(df[field]).dt.normalize()

        print(f"  {field} : {missing_count} nulls filled with {NULL_DATE.date()}, "
              f"flag stored in '{missing_flag}'")

    # scraped_at — full UTC timestamp, should never be null
    if "scraped_at" in df.columns:
        df["scraped_at"] = pd.to_datetime(df["scraped_at"], utc=True, errors="coerce")
        null_count = df["scraped_at"].isnull().sum()
        if null_count > 0:
            print(f"  ⚠ scraped_at : {null_count} nulls found — check ingestion script")

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
    """Match raw make string to code and id using keyword search."""
    normalised = raw.strip().lower().rstrip(".")
    for keyword, code, make_id in MAKE_KEYWORD_MAP:
        if keyword in normalised:
            return code, make_id
    return "UNKNOWN", -1


def normalise_make(df: pd.DataFrame) -> pd.DataFrame:
    # preserve raw value before any changes
    df["make_raw"] = df["make"].astype(str).str.strip()

    mapped       = df["make_raw"].apply(map_make)
    df["make"]   = mapped.apply(lambda x: x[0])
    df["make_id"]= mapped.apply(lambda x: x[1]).astype("int64")

    # report mapping results
    print(f"  Make mapping results:")
    for code, count in df["make"].value_counts().items():
        print(f"    {code} : {count} records")

    unmapped = df[df["make"] == "UNKNOWN"]["make_raw"].unique()
    if len(unmapped) > 0:
        print(f"  ⚠ Unmapped make variants — add keyword to MAKE_KEYWORD_MAP: {unmapped.tolist()}")
    else:
        print(f"  ✓ All makes mapped successfully")

    return df


# ── Sentiment ─────────────────────────────────────────────────────────────────

def score_sentiment(text, analyzer: SentimentIntensityAnalyzer) -> tuple[float, str, int]:
    """Score a single text string and return score, label, id."""
    # guard against float NaN slipping through from PyArrow backed columns
    if text is None or not isinstance(text, str) or text.strip() == "" or text == NULL_TEXT:
        return NULL_SENTIMENT_SCORE, NULL_SENTIMENT_LABEL, NULL_SENTIMENT_ID

    score = analyzer.polarity_scores(text)["compound"]

    if score < SENTIMENT_THRESHOLDS["critical"]:
        label = "critical"
    elif score < SENTIMENT_THRESHOLDS["negative"]:
        label = "negative"
    elif score < SENTIMENT_THRESHOLDS["neutral"]:
        label = "neutral"
    else:
        label = "positive"

    return round(score, 4), label, SENTIMENT_IDS[label]

def apply_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    analyzer         = SentimentIntensityAnalyzer()
    sentiment_fields = ["body", "consequence", "remedy"]

    for field in sentiment_fields:
        if field not in df.columns:
            continue

        # force to string first — PyArrow backed columns can carry native float NaN
        df[field] = df[field].astype(str).fillna(NULL_TEXT).replace("nan", NULL_TEXT)

        print(f"  Scoring '{field}'...")
        results = df[field].apply(lambda t: score_sentiment(t, analyzer))

        df[f"{field}_sentiment_score"] = results.apply(lambda x: x[0]).astype("float64")
        df[f"{field}_sentiment_label"] = results.apply(lambda x: x[1])
        df[f"{field}_sentiment_id"]    = results.apply(lambda x: x[2]).astype("int64")

        dist = df[f"{field}_sentiment_label"].value_counts().to_dict()
        print(f"    Distribution : {dist}")

    return df

# ── Schema enforcement ────────────────────────────────────────────────────────

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    int_fields    = ["year", "make_id", "week_added", "injuries",
                     "deaths", "units_affected", "body_sentiment_id"]
    float_fields  = ["source_weight", "body_sentiment_score"]
    string_fields = ["record_id", "source", "make_raw", "make", "model",
                     "component", "body", "body_sentiment_label"]

    # add recall-only fields if present
    for field in ["consequence_sentiment_id", "remedy_sentiment_id"]:
        if field in df.columns:
            int_fields.append(field)

    for field in ["consequence_sentiment_score", "remedy_sentiment_score"]:
        if field in df.columns:
            float_fields.append(field)

    for field in ["consequence", "remedy",
                  "consequence_sentiment_label", "remedy_sentiment_label"]:
        if field in df.columns:
            string_fields.append(field)

    for field in int_fields:
        if field in df.columns:
            df[field] = (pd.to_numeric(df[field], errors="coerce")
                         .fillna(NULL_INT)
                         .astype("int64"))

    for field in float_fields:
        if field in df.columns:
            df[field] = (pd.to_numeric(df[field], errors="coerce")
                         .fillna(0.0)
                         .astype("float64"))

    for field in string_fields:
        if field in df.columns:
            df[field] = df[field].astype(str).str.strip()

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


def build_dim_sentiment() -> pd.DataFrame:
    return pd.DataFrame([
        {"sentiment_id": 3,  "sentiment_label": "positive", "threshold": ">= 0.05"},
        {"sentiment_id": 2,  "sentiment_label": "neutral",  "threshold": ">= -0.2 and < 0.05"},
        {"sentiment_id": 1,  "sentiment_label": "negative", "threshold": ">= -0.5 and < -0.2"},
        {"sentiment_id": 0,  "sentiment_label": "critical", "threshold": "< -0.5"},
        {"sentiment_id": -1, "sentiment_label": "N/A",      "threshold": "unscored"},
    ])


def save_dim_tables(processed_dir: str):
    os.makedirs(processed_dir, exist_ok=True)

    pq.write_table(
        pa.Table.from_pandas(build_dim_make()),
        os.path.join(processed_dir, "dim_make.parquet")
    )
    pq.write_table(
        pa.Table.from_pandas(build_dim_sentiment()),
        os.path.join(processed_dir, "dim_sentiment.parquet")
    )
    print(f"  Saved dim_make.parquet and dim_sentiment.parquet")


# ── Save facts ────────────────────────────────────────────────────────────────

def save_facts(df: pd.DataFrame, curated_dir: str, json_filename: str):
    os.makedirs(curated_dir, exist_ok=True)
    stem     = os.path.splitext(json_filename)[0]
    out_path = os.path.join(curated_dir, f"facts_{stem}")

    table = pa.Table.from_pandas(df, preserve_index=False)

    pq.write_to_dataset(
        table,
        root_path=out_path,
        partition_cols=["make", "year"]
    )

    print(f"  Facts saved to     : {out_path}")
    print(f"  Shape              : {df.shape[0]} rows × {df.shape[1]} columns")
    return out_path

# ── Orchestrator ──────────────────────────────────────────────────────────────

def run():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir  = os.path.join(base_dir, "data", "raw")

    json_filename = find_latest_json(raw_dir)
    paths         = get_paths(json_filename)

    print(f"\nTransforming : {json_filename}")
    print("=" * 60)

    print("\n[ 1 / 8 ] Load")
    df = load_raw(paths["raw_file"])

    print("\n[ 2 / 8 ] Validation report")
    null_fields = parse_null_fields_from_report(paths["val_file"])

    print("\n[ 3 / 8 ] Drop duplicates")
    df = drop_duplicates(df)

    print("\n[ 4 / 8 ] Drop null fields")
    df = drop_null_fields(df, null_fields)

    print("\n[ 5 / 8 ] Clean")
    df = clean_numeric(df)
    df = clean_text(df)
    df = clean_dates(df)
    df = clean_booleans(df)

    print("\n[ 6 / 8 ] Normalise make")
    df = normalise_make(df)

    print("\n[ 7 / 8 ] Sentiment")
    df = apply_sentiment(df)

    print("\n[ 8 / 8 ] Schema + save")
    df = enforce_schema(df)
    save_dim_tables(paths["processed_dir"])
    save_facts(df, paths["processed_dir"], json_filename)

    print("\n" + "=" * 60)
    print("Transform complete.")
    print(f"  Records processed  : {len(df)}")
    print(f"  Columns            : {len(df.columns)}")
    print(f"  Makes              : {df['make'].value_counts().to_dict()}")
    print(f"  Body sentiment     : {df['body_sentiment_label'].value_counts().to_dict()}")


if __name__ == "__main__":
    run()