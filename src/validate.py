import os
import json
import pandas as pd
from datetime import datetime, timezone


def load_raw(data_dir: str = None) -> tuple[pd.DataFrame, list[str]]:
    if data_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data", "raw")

    all_records  = []
    files_loaded = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                records = json.load(f)
                all_records.extend(records)
                files_loaded.append(filename)

    print(f"Loaded {len(files_loaded)} file(s), {len(all_records)} total records\n")
    return pd.DataFrame(all_records), files_loaded


def check_all_fields(df: pd.DataFrame, lines: list):
    header = "FULL FIELD REPORT"
    lines.append("=" * 75)
    lines.append(header)
    lines.append("=" * 75)

    total     = len(df)
    col_width = max(len(col) for col in df.columns) + 2
    divider   = "-" * (col_width + 50)

    lines.append(f"Total records : {total}")
    lines.append(f"Total columns : {len(df.columns)}")
    lines.append(f"Sources       : {df['source'].value_counts().to_dict()}")
    lines.append(f"Makes         : {df['make'].value_counts().to_dict()}")
    lines.append("")
    lines.append(divider)
    lines.append(
        f"{'Field':<{col_width}} {'Total':>7} {'Nulls':>7} "
        f"{'Empty':>7} {'Missing%':>9} {'Dupes':>7}  {'Dtype'}"
    )
    lines.append(divider)

    for col in df.columns:
        null_count  = df[col].isnull().sum()
        empty_count = (df[col].astype(str).str.strip() == "").sum() - null_count
        empty_count = max(empty_count, 0)
        missing     = null_count + empty_count
        missing_pct = round(missing / total * 100, 1)
        dupes       = df[col].duplicated().sum()
        dtype       = str(df[col].dtype)

        flag = ""
        if missing_pct > 50:
            flag = "  ✗ HIGH"
        elif missing_pct > 10:
            flag = "  ⚠ WARN"

        lines.append(
            f"{col:<{col_width}} {total:>7} {null_count:>7} {empty_count:>7} "
            f"{missing_pct:>8}% {dupes:>7}  {dtype}{flag}"
        )

    lines.append(divider)


def check_identity_keys(df: pd.DataFrame, lines: list):
    lines.append("")
    lines.append("=" * 75)
    lines.append("IDENTITY KEY CHECKS")
    lines.append("=" * 75)

    lines.append("\n[ record_id ]")
    null_ids  = df["record_id"].isnull().sum()
    empty_ids = (
        (df["record_id"] == "complaint_") |
        (df["record_id"] == "recall_")
    ).sum()

    lines.append(f"  Null         : {null_ids}")
    lines.append(f"  Empty prefix : {empty_ids}  (prefix exists but no value after it)")

    lines.append("\n[ Full row duplicates ]")
    full_dupes = df.duplicated().sum()
    lines.append(f"  Exact duplicate rows : {full_dupes}")

    if full_dupes > 0:
        lines.append("\n  Duplicate samples:")
        lines.append(
            df[df.duplicated(keep=False)]
            .head(10)
            .to_string(index=False)
        )
        lines.append(f"\n  ⚠ WARNING: {full_dupes} exact duplicate rows — likely caused by running ingestion more than once")
    else:
        lines.append("  ✓ No exact duplicate rows found")


def check_dates(df: pd.DataFrame, lines: list):
    lines.append("")
    lines.append("=" * 75)
    lines.append("DATE FIELD CHECKS")
    lines.append("=" * 75)

    date_fields = {
        "scraped_at":    "all",
        "incident_date": "nhtsa_complaints",
        "recall_date":   "nhtsa_recalls"
    }

    for field, source_filter in date_fields.items():
        if field not in df.columns:
            lines.append(f"\n[ {field} ] — not found, skipping")
            continue

        subset      = df if source_filter == "all" else df[df["source"] == source_filter]
        total       = len(subset)
        null_count  = subset[field].isnull().sum()
        empty_count = (subset[field].astype(str).str.strip() == "").sum() - null_count
        empty_count = max(empty_count, 0)
        missing     = null_count + empty_count
        pct         = round(missing / total * 100, 1) if total > 0 else 0

        lines.append(f"\n[ {field} ] — source: {source_filter}")
        lines.append(f"  Total   : {total}")
        lines.append(f"  Null    : {null_count}")
        lines.append(f"  Empty   : {empty_count}")
        lines.append(f"  Missing : {missing} ({pct}%)")

        valid = subset[field].dropna()
        valid = valid[valid.astype(str).str.strip() != ""]
        if len(valid) > 0:
            lines.append(f"  Sample  : {valid.iloc[0]}")

        if pct > 10:
            lines.append(f"  ⚠ WARNING: {pct}% missing — time series analysis will be affected")


def check_descriptions(df: pd.DataFrame, lines: list):
    lines.append("")
    lines.append("=" * 75)
    lines.append("DESCRIPTION FIELD CHECKS")
    lines.append("=" * 75)

    desc_fields = {
        "body":        "all",
        "consequence": "nhtsa_recalls",
        "remedy":      "nhtsa_recalls"
    }

    for field, source_filter in desc_fields.items():
        if field not in df.columns:
            lines.append(f"\n[ {field} ] — not found, skipping")
            continue

        subset      = df if source_filter == "all" else df[df["source"] == source_filter]
        total       = len(subset)
        null_count  = subset[field].isnull().sum()
        empty_count = (subset[field].astype(str).str.strip() == "").sum() - null_count
        empty_count = max(empty_count, 0)
        missing     = null_count + empty_count
        pct         = round(missing / total * 100, 1) if total > 0 else 0

        lines.append(f"\n[ {field} ] — source: {source_filter}")
        lines.append(f"  Total      : {total}")
        lines.append(f"  Null       : {null_count}")
        lines.append(f"  Empty      : {empty_count}")
        lines.append(f"  Missing    : {missing} ({pct}%)")

        valid = subset[field].dropna()
        valid = valid[valid.astype(str).str.strip() != ""]
        if len(valid) > 0:
            avg_len = round(valid.str.len().mean())
            min_len = valid.str.len().min()
            max_len = valid.str.len().max()
            lines.append(f"  Avg length : {avg_len} chars")
            lines.append(f"  Min length : {min_len} chars")
            lines.append(f"  Max length : {max_len} chars")
            lines.append(f"  Sample     : {valid.iloc[0][:120]}...")

        if pct > 10:
            lines.append(f"  ⚠ WARNING: {pct}% missing — sentiment scoring will have gaps")


def check_numeric_anomalies(df: pd.DataFrame, lines: list):
    lines.append("")
    lines.append("=" * 75)
    lines.append("NUMERIC ANOMALY CHECKS")
    lines.append("=" * 75)

    checks = {
        "injuries":       {"min_valid": 0, "max_valid": 1000},
        "deaths":         {"min_valid": 0, "max_valid": 100},
        "units_affected": {"min_valid": 0, "max_valid": 10_000_000},
    }

    for field, bounds in checks.items():
        if field not in df.columns:
            continue

        col         = pd.to_numeric(df[field], errors="coerce")
        null_count  = col.isnull().sum()
        type_errors = null_count - df[field].isnull().sum()
        negatives   = (col < bounds["min_valid"]).sum()
        outliers    = (col > bounds["max_valid"]).sum()

        lines.append(f"\n[ {field} ]")
        lines.append(f"  Null          : {df[field].isnull().sum()}")
        lines.append(f"  Type errors   : {type_errors}  (non-numeric strings coerced to null)")
        lines.append(f"  Negatives     : {negatives}  (expected >= {bounds['min_valid']})")
        lines.append(f"  Outliers      : {outliers}  (expected <= {bounds['max_valid']})")

        if type_errors > 0:
            bad = df[field][
                pd.to_numeric(df[field], errors="coerce").isnull() &
                df[field].notnull()
            ]
            lines.append(f"  Sample bad values : {bad.unique()[:5].tolist()}")

        if negatives > 0:
            lines.append(f"  ⚠ WARNING: negative values found — check ingestion logic")

        if outliers > 0:
            lines.append(f"  ⚠ WARNING: outliers found — verify against source data")


def check_make_normalisation(df: pd.DataFrame, lines: list):
    lines.append("")
    lines.append("=" * 75)
    lines.append("MAKE NORMALISATION CHECKS")
    lines.append("=" * 75)
    lines.append("")
    lines.append("[ Raw make values found in data ]")

    # gets every unique raw string exactly as it appears in the data
    raw_makes = df["make"].unique()

    for make in sorted(raw_makes):
        count = (df["make"] == make).sum()
        lines.append(f"  '{make}' — {count} records")

    lines.append("")
    lines.append(f"  Total unique make variants : {len(raw_makes)}")
    lines.append(f"  → Map each variant to a make_code in transform.py")


def save_report(lines: list, json_filenames: list):
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir  = os.path.join(base_dir, "validate")
    os.makedirs(output_dir, exist_ok=True)

    # derive report name from json filename(s)
    if len(json_filenames) == 1:
        stem      = os.path.splitext(json_filenames[0])[0]
        out_name  = f"val_{stem}.txt"
    else:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_name  = f"val_combined_{timestamp}.txt"

    filepath = os.path.join(output_dir, out_name)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath


def run():
    df, filenames = load_raw()

    lines = []
    lines.append(f"NHTSA Complaints and Recalls Validation Report")
    lines.append(f"Generated : {datetime.now(tz=timezone.utc).isoformat()}")
    lines.append(f"Source files : {', '.join(filenames)}")
    lines.append("")

    check_all_fields(df, lines)
    check_identity_keys(df, lines)
    check_dates(df, lines)
    check_descriptions(df, lines)
    check_numeric_anomalies(df, lines)
    check_make_normalisation(df, lines)

    lines.append("")
    lines.append("=" * 75)
    lines.append("RECORD COUNT SUMMARY")
    lines.append("=" * 75)
    lines.append(
        df.groupby(["source", "make"])
        .size()
        .rename("record_count")
        .reset_index()
        .to_string(index=False)
    )
    lines.append("\nValidation complete.")

    # print to terminal
    print("\n".join(lines))

    # save to file
    filepath = save_report(lines, filenames)
    print(f"\nReport saved to {filepath}")


if __name__ == "__main__":
    run()