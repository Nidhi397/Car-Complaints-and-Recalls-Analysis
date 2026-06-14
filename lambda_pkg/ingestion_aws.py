import json
import os
import boto3
import requests
from datetime import datetime, timezone

MAKES = ["Toyota", "Ford", "Tesla"]
YEARS = list(range(2015, 2026))

BASE_URLS = {
    "complaints": "https://api.nhtsa.gov/complaints/complaintsByVehicle",
    "recalls":    "https://api.nhtsa.gov/recalls/recallsByVehicle"
}

MODELS = {
    "Toyota": ["Camry", "Corolla", "RAV4", "Prius", "C-HR"],
    "Ford":   ["F-150", "Mustang", "Explorer", "Escape", "Fiesta"],
    "Tesla":  ["Model 3", "Model S", "Model X", "Model Y"]
}

# read from Lambda environment variables
RAW_BUCKET = os.environ.get("RAW_BUCKET", "car-complaints-recalls-analysis-raw-253545")


def fetch_complaints(make: str, model: str, year: int) -> list[dict]:
    try:
        response = requests.get(
            BASE_URLS["complaints"],
            params={"make": make, "model": model, "modelYear": year},
            timeout=10
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        return [
            {
                "record_id":        f"complaint_{r.get('odiNumber', '')}",
                "source":           "nhtsa_complaints",
                "source_weight":    1.0,
                "make":             make,
                "model":            model,
                "year":             year,
                "component":        r.get("components", ""),
                "body":             r.get("cdescription", ""),
                "incident_date":    r.get("dateOfIncident", ""),
                "crash_involved":   r.get("crash", False),
                "fire_involved":    r.get("fire", False),
                "injuries":         r.get("numberOfInjuries", 0),
                "deaths":           r.get("numberOfDeaths", 0),
                "scraped_at":       datetime.now(tz=timezone.utc).isoformat(),
                "week_added":       1
            }
            for r in results
        ]

    except requests.exceptions.RequestException as e:
        print(f"  ✗ Complaints error {make} {model} {year}: {e}")
        return []


def fetch_recalls(make: str, model: str, year: int) -> list[dict]:
    try:
        response = requests.get(
            BASE_URLS["recalls"],
            params={"make": make, "model": model, "modelYear": year},
            timeout=10
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        return [
            {
                "record_id":        f"recall_{r.get('NHTSACampaignNumber', '')}",
                "source":           "nhtsa_recalls",
                "source_weight":    2.0,
                "make":             make,
                "model":            model,
                "year":             year,
                "component":        r.get("Component", ""),
                "body":             r.get("Summary", ""),
                "consequence":      r.get("Consequence", ""),
                "remedy":           r.get("Remedy", ""),
                "recall_date":      r.get("ReportReceivedDate", ""),
                "units_affected":   r.get("PotentialNumberOfUnitsAffected", 0),
                "scraped_at":       datetime.now(tz=timezone.utc).isoformat(),
                "week_added":       1
            }
            for r in results
        ]

    except requests.exceptions.RequestException as e:
        print(f"  ✗ Recalls error {make} {model} {year}: {e}")
        return []


def save_to_s3(records: list[dict]) -> str:
    s3        = boto3.client("s3")
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    key       = f"nhtsa_{timestamp}.json"

    s3.put_object(
        Bucket      = RAW_BUCKET,
        Key         = key,
        Body        = json.dumps(records, indent=2, ensure_ascii=False),
        ContentType = "application/json"
    )

    print(f"Saved {len(records)} records to s3://{RAW_BUCKET}/{key}")
    return key


def run():
    all_records = []
    total       = len(MAKES) * sum(len(m) for m in MODELS.values()) * len(YEARS)
    processed   = 0

    for make in MAKES:
        for model in MODELS[make]:
            for year in YEARS:
                processed += 1
                print(f"[{processed}/{total}] {make} {model} {year}...")

                complaints = fetch_complaints(make, model, year)
                recalls    = fetch_recalls(make, model, year)
                all_records.extend(complaints)
                all_records.extend(recalls)

                print(f"  → {len(complaints)} complaints, {len(recalls)} recalls")

    return save_to_s3(all_records)


def notify(subject: str, message: str):
    """Send SNS notification — fails silently if topic not configured."""
    if not SNS_TOPIC_ARN:
        return
    try:
        boto3.client("sns").publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=message
        )
    except Exception as e:
        print(f"  !!! SNS failed : {e}!!!")

def lambda_handler(event, context):
    """AWS Lambda entry point."""
    print("NHTSA ingestion Lambda started")
    notify("NHTSA — Ingestion started", f"Ingestion started for {', '.join(MAKES)}")
    try:
        key, record_count = run()
        notify(
            "NHTSA Ingestion complete",
            f"Records saved : {record_count}\nS3 key : s3://{RAW_BUCKET}/{key}"
        )
        return {"statusCode": 200, "body": json.dumps({"s3_key": key})}

    except Exception as e:
        notify("ALERT: NHTSA Ingestion failed", f"Error : {str(e)}")
        raise


if __name__ == "__main__":
    run()