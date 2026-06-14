import os
import duckdb
import pandas as pd
import pyarrow.parquet as pq


# ── Paths ─────────────────────────────────────────────────────────────────────

def get_paths() -> dict:
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    curated_dir = os.path.join(base_dir, "data", "processed")
    analysis_dir = os.path.join(base_dir, "data", "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    pattern = os.path.join(curated_dir, "facts_*", "**", "*.parquet").replace("\\", "/")

    return {
        "pattern":      pattern,
        "analysis_dir": analysis_dir,
        "curated_dir":  curated_dir,
    }


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data(pattern: str) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    conn.execute(f"""
        CREATE VIEW facts AS
        SELECT *
        FROM read_parquet('{pattern}', hive_partitioning=true)
    """)
    count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    print(f"  Loaded {count:,} records into DuckDB view")
    return conn


# ── Question 1: Recall to complaint ratio per brand ───────────────────────────

def analyse_recall_complaint_ratio(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    print("\n[ Q1 ] Recall to complaint ratio per brand")

    df = conn.execute("""
        SELECT
            make,
            COUNT(*) FILTER (WHERE source = 'nhtsa_complaints') AS complaint_count,
            COUNT(*) FILTER (WHERE source = 'nhtsa_recalls')    AS recall_count,
            ROUND(
                COUNT(*) FILTER (WHERE source = 'nhtsa_recalls') * 100.0 /
                NULLIF(COUNT(*) FILTER (WHERE source = 'nhtsa_complaints'), 0),
                2
            ) AS recall_to_complaint_pct,
            SUM(CASE WHEN source = 'nhtsa_recalls'
                THEN units_affected ELSE 0 END) AS total_units_affected
        FROM facts
        WHERE make != 'UNKNOWN'
        GROUP BY make
        ORDER BY recall_to_complaint_pct DESC
    """).df()

    print(df.to_string(index=False))
    return df


# ── Question 2: Component crash rate per brand ────────────────────────────────

def analyse_component_crash_rate(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    print("\n[ Q2 ] Component crash rate per brand")

    # step 1 — crash rate per component across all brands
    component_crash = conn.execute("""
        SELECT
            component,
            COUNT(*)                                                AS complaint_count,
            SUM(CASE WHEN crash_involved = true THEN 1 ELSE 0 END) AS crash_count,
            ROUND(
                SUM(CASE WHEN crash_involved = true THEN 1 ELSE 0 END) * 100.0 /
                NULLIF(COUNT(*), 0),
                2
            ) AS crash_rate_pct
        FROM facts
        WHERE source    = 'nhtsa_complaints'
          AND component != 'N/A'
          AND make      != 'UNKNOWN'
        GROUP BY component
        HAVING COUNT(*) >= 10
        ORDER BY crash_rate_pct DESC
        LIMIT 20
    """).df()

    print("\n  Top 20 components by crash rate:")
    print(component_crash.to_string(index=False))

    # step 2 — crash rate per component broken down by brand
    component_brand = conn.execute("""
        SELECT
            make,
            component,
            COUNT(*)                                                AS complaint_count,
            SUM(CASE WHEN crash_involved = true THEN 1 ELSE 0 END) AS crash_count,
            ROUND(
                SUM(CASE WHEN crash_involved = true THEN 1 ELSE 0 END) * 100.0 /
                NULLIF(COUNT(*), 0),
                2
            ) AS crash_rate_pct
        FROM facts
        WHERE source    = 'nhtsa_complaints'
          AND component != 'N/A'
          AND make      != 'UNKNOWN'
        GROUP BY make, component
        HAVING COUNT(*) >= 5
        ORDER BY make, crash_rate_pct DESC
    """).df()

    print("\n  Component crash rate by brand:")
    print(component_brand.to_string(index=False))

    # step 3 — weighted brand crash risk score
    brand_risk = conn.execute("""
        WITH component_stats AS (
            SELECT
                component,
                COUNT(*) AS total_complaints,
                SUM(CASE WHEN crash_involved = true THEN 1 ELSE 0 END) * 1.0 /
                NULLIF(COUNT(*), 0) AS crash_rate
            FROM facts
            WHERE source    = 'nhtsa_complaints'
              AND component != 'N/A'
            GROUP BY component
            HAVING COUNT(*) >= 10
        ),
        brand_component AS (
            SELECT
                f.make,
                f.component,
                COUNT(*) AS brand_component_complaints
            FROM facts f
            JOIN component_stats cs ON f.component = cs.component
            WHERE f.source    = 'nhtsa_complaints'
              AND f.component != 'N/A'
              AND f.make      != 'UNKNOWN'
            GROUP BY f.make, f.component
        )
        SELECT
            bc.make,
            ROUND(
                SUM(bc.brand_component_complaints * cs.crash_rate) /
                NULLIF(SUM(bc.brand_component_complaints), 0) * 100,
                2
            ) AS weighted_crash_risk_pct
        FROM brand_component bc
        JOIN component_stats cs ON bc.component = cs.component
        GROUP BY bc.make
        ORDER BY weighted_crash_risk_pct DESC
    """).df()

    print("\n  Weighted brand crash risk score:")
    print(brand_risk.to_string(index=False))

    return component_crash, component_brand, brand_risk


# ── Question 3: Model level complaint and recall ranking per brand ─────────────

def analyse_model_rankings(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    print("\n[ Q3 ] Model level complaint and recall ranking per brand")

    # complaints per model per brand
    model_complaints = conn.execute("""
        SELECT
            make,
            model,
            COUNT(*) AS complaint_count,
            SUM(CASE WHEN crash_involved = true THEN 1 ELSE 0 END) AS crash_count,
            ROUND(
                SUM(CASE WHEN crash_involved = true THEN 1 ELSE 0 END) * 100.0 /
                NULLIF(COUNT(*), 0),
                2
            ) AS crash_rate_pct
        FROM facts
        WHERE source  = 'nhtsa_complaints'
          AND model  != 'N/A'
          AND make   != 'UNKNOWN'
        GROUP BY make, model
        ORDER BY make, complaint_count DESC
    """).df()

    print("\n  Complaints per model:")
    print(model_complaints.to_string(index=False))

    # recalls per model per brand
    model_recalls = conn.execute("""
        SELECT
            make,
            model,
            COUNT(*)                    AS recall_count,
            SUM(units_affected)         AS total_units_affected,
            AVG(units_affected)         AS avg_units_per_recall
        FROM facts
        WHERE source  = 'nhtsa_recalls'
          AND model  != 'N/A'
          AND make   != 'UNKNOWN'
        GROUP BY make, model
        ORDER BY make, recall_count DESC
    """).df()

    print("\n  Recalls per model:")
    print(model_recalls.to_string(index=False))

    # combined model scorecard — complaints + recalls joined
    model_scorecard = conn.execute("""
        WITH complaints AS (
            SELECT make, model, COUNT(*) AS complaint_count
            FROM facts
            WHERE source = 'nhtsa_complaints'
              AND model != 'N/A'
              AND make  != 'UNKNOWN'
            GROUP BY make, model
        ),
        recalls AS (
            SELECT make, model,
                   COUNT(*)         AS recall_count,
                   SUM(units_affected) AS units_affected
            FROM facts
            WHERE source = 'nhtsa_recalls'
              AND model != 'N/A'
              AND make  != 'UNKNOWN'
            GROUP BY make, model
        )
        SELECT
            c.make,
            c.model,
            c.complaint_count,
            COALESCE(r.recall_count, 0)    AS recall_count,
            COALESCE(r.units_affected, 0)  AS units_affected,
            ROUND(
                COALESCE(r.recall_count, 0) * 100.0 /
                NULLIF(c.complaint_count, 0),
                2
            ) AS recall_to_complaint_pct
        FROM complaints c
        LEFT JOIN recalls r
          ON c.make = r.make AND c.model = r.model
        ORDER BY c.make, c.complaint_count DESC
    """).df()

    print("\n  Model scorecard (complaints + recalls combined):")
    print(model_scorecard.to_string(index=False))

    return model_complaints, model_recalls, model_scorecard


# ── Supporting: Year over year trend ─────────────────────────────────────────

def analyse_year_trend(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    print("\n[ Supporting ] Year over year complaint trend per brand")

    df = conn.execute("""
        SELECT
            make,
            year,
            COUNT(*) FILTER (WHERE source = 'nhtsa_complaints') AS complaints,
            COUNT(*) FILTER (WHERE source = 'nhtsa_recalls')    AS recalls
        FROM facts
        WHERE make != 'UNKNOWN'
          AND year  != 9999
          AND year BETWEEN 2015 AND 2025
        GROUP BY make, year
        ORDER BY make, year
    """).df()

    print(df.to_string(index=False))
    return df


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(results: dict, analysis_dir: str):
    for name, df in results.items():
        if df is None or not isinstance(df, pd.DataFrame):
            continue
        out_path = os.path.join(analysis_dir, f"{name}.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved {name}.csv — {len(df)} rows")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run():
    paths = get_paths()

    print(f"\nAnalysis")
    print("=" * 60)

    print("\n[ Load ]")
    conn = load_data(paths["pattern"])

    q1 = analyse_recall_complaint_ratio(conn)

    q2_component, q2_brand, q2_risk = analyse_component_crash_rate(conn)

    q3_complaints, q3_recalls, q3_scorecard = analyse_model_rankings(conn)

    q4 = analyse_year_trend(conn)

    print("\n[ Save ]")
    save_results({
        "q1_recall_complaint_ratio":  q1,
        "q2_component_crash_rate":    q2_component,
        "q2_component_brand":         q2_brand,
        "q2_weighted_brand_risk":     q2_risk,
        "q3_model_complaints":        q3_complaints,
        "q3_model_recalls":           q3_recalls,
        "q3_model_scorecard":         q3_scorecard,
        "q4_year_trend":              q4,
    }, paths["analysis_dir"])

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print(f"Results saved to : {paths['analysis_dir']}")


if __name__ == "__main__":
    run()