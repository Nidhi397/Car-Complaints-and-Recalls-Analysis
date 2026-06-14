# Car Complaints and Recalls Analysis

An end-to-end ETL data pipeline processing publicly available vehicle complaint and recall data from the [NHTSA Open Data API](https://api.nhtsa.gov) across 4 vehicle models for brands Toyota, Ford, and Tesla (2015–2025).

Built as a personal learning project to demonstrate AWS data engineering and architecture decision-making. Non-commercial.

> **Work in Progress** — The pipeline is functional end to end on AWS. Production-grade monitoring practices, robust sentiment analysis strategy, streaming, and BI layer coming in following weeks.

---

## Pipeline on AWS

```
Eventbridge has a recurring run schedule every 6 hours.
```

```
NHTSA API → Lambda (ingest) → S3 raw → Lambda (trigger) → Glue (transform) → S3 processed → Athena
```

Runs every 6 hours via EventBridge. Full-refresh loads currently, incremental strategy in progress.

---

## Stack

| Layer | Service |
|-------|---------|
| Ingestion | AWS Lambda |
| Storage | S3 (1 bucket for raw + 1 bucket for processed) |
| Transform | AWS Glue Python Shell |
| Catalog | AWS Glue Data Catalog |
| Query | Amazon Athena |
| Schedule | Amazon EventBridge |
| Monitoring | Amazon CloudWatch |

---

## Local Setup

```bash
git clone https://github.com/Nidhi397/Car-Complaints-and-Recalls-Analysis.git
cd Car-Complaints-and-Recalls-Analysis
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run locally:

```bash
python src/ingestion.py   # pull data from NHTSA API
python src/validate.py          # data quality checks
python src/transform.py         # clean and transform
python src/analysis.py          # run analysis queries
```

---

## Analysis

Questions answered via Athena currently:

- Which brands and models generate the most complaints and recalls?
- What proportion of each brand's complaints does each model account for?

SQL in `queries/` folder.

---

## Architecture Tradeoffs

| Decision | Choice | Why |
|----------|--------|-----|
| Transform layer | Glue Python Shell over Lambda | Lambda 250MB package limit exceeded by pandas + PyArrow. Glue has no size limit, built-in dependencies, and a clean PySpark upgrade path for week 3 |
| Storage format | Parquet over CSV | Schema enforcement, columnar compression, and Athena cost control via partition pruning. Overkill at 35k records — chosen to establish the correct production pattern |
| Lambda architecture | ARM for ingestion, x86 for transform | ARM is 20% cheaper for pure Python workloads. x86 chosen for transform due to PyArrow binary compatibility when packaging on Windows |
| Orchestration | EventBridge over Step Functions | Linear single-job pipeline — full orchestration is overkill at week 1. Step Functions upgrade planned post week 1 |
| IAM design | Least privilege custom policies over managed | Scoped to specific bucket prefixes and job ARNs — no AmazonS3FullAccess anywhere |

## Data Ethics

All data is publicly available from the NHTSA — a US federal government open data initiative. No proprietary data, no commercial use. Built in personal time.

---

## License

MIT
