#Importing libraries for ingestion
import requests
import json
import os
from datetime import datetime, timezone

MAKES = ["Toyota", "Ford", "Tesla"]
YEARS = list(range(2015, 2026))
base_dir = r"C:\Users\nkopp\car-sentiment-intelligence-reddit"
BASE_URLS = {
    "complaints": "https://api.nhtsa.gov/complaints/complaintsByVehicle",
    "recalls":    "https://api.nhtsa.gov/recalls/recallsByVehicle"
}

MODELS = {
    "Toyota": ["Camry", "Corolla", "RAV4", "Prius", "C-HR"],
    "Ford":   ["F-150", "Mustang", "Explorer", "Escape", "Fiesta"],
    "Tesla":  ["Model 3", "Model S", "Model X", "Model Y"]
}

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
                "source_weight":    1.5,
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

def save_raw(records: list[dict]) -> str:
    output_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"nhtsa_{timestamp}.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    return filepath

def run():
    all_records = []
    total_combinations = len(MAKES) * sum(len(m) for m in MODELS.values()) * len(YEARS)
    processed = 0

    for make in MAKES:
        for model in MODELS[make]:
            for year in YEARS:
                processed += 1
                print(f"[{processed}/{total_combinations}] {make} {model} {year}...")

                complaints = fetch_complaints(make, model, year)
                recalls    = fetch_recalls(make, model, year)

                all_records.extend(complaints)
                all_records.extend(recalls)

                print(f"  → {len(complaints)} complaints, {len(recalls)} recalls")

    filepath = save_raw(all_records)
    print(f"\nDone. {len(all_records)} total records saved to {filepath}")

if __name__ == "__main__":
    run()