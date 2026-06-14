
SELECT
    make,
    COUNT(*) FILTER (WHERE source = 'nhtsa_complaints') AS complaints,
    COUNT(*) FILTER (WHERE source = 'nhtsa_recalls')    AS recalls,
    COUNT(*)                                             AS total
FROM facts_nhtsa_20260606_200341
WHERE make != 'UNKNOWN'
GROUP BY make
ORDER BY total DESC;

SELECT
    make,
    model,
    COUNT(*) FILTER (WHERE source = 'nhtsa_complaints') AS complaints,
    COUNT(*) FILTER (WHERE source = 'nhtsa_recalls')    AS recalls,
    COUNT(*)                                             AS total
FROM facts_nhtsa_20260606_200341
WHERE make  != 'UNKNOWN'
  AND model != 'Unknown'
GROUP BY make, model
ORDER BY make, total DESC;

WITH brand_totals AS (
    SELECT
        make,
        COUNT(*) AS brand_complaint_total
    FROM facts_nhtsa_20260606_200341
    WHERE source = 'nhtsa_complaints'
      AND make   != 'UNKNOWN'
    GROUP BY make
),
model_complaints AS (
    SELECT
        make,
        model,
        COUNT(*) AS model_complaint_count
    FROM facts_nhtsa_20260606_200341
    WHERE source = 'nhtsa_complaints'
      AND make   != 'UNKNOWN'
      AND model  != 'Unknown'
    GROUP BY make, model
)
SELECT
    mc.make,
    mc.model,
    mc.model_complaint_count                               AS complaints,
    bt.brand_complaint_total                               AS brand_total,
    ROUND(
        mc.model_complaint_count * 100.0 /
        NULLIF(bt.brand_complaint_total, 0),
        2
    )                                                      AS pct_of_brand_complaints
FROM model_complaints mc
JOIN brand_totals bt ON mc.make = bt.make
ORDER BY mc.make, pct_of_brand_complaints DESC;

WITH component_complaints AS (
    SELECT
        make,
        component,
        COUNT(*)                                                AS complaint_count,
        SUM(CASE WHEN crash_involved = true THEN 1 ELSE 0 END) AS crash_count,
        SUM(CASE WHEN injuries > 0 THEN injuries ELSE 0 END)   AS total_injuries
    FROM facts_nhtsa_20260606_200341
    WHERE source    = 'nhtsa_complaints'
      AND make      != 'UNKNOWN'
      AND component != 'Unknown'
    GROUP BY make, component
),
component_recalls AS (
    SELECT
        make,
        component,
        COUNT(*)            AS recall_count,
        SUM(units_affected) AS units_affected
    FROM facts_nhtsa_20260606_200341
    WHERE source    = 'nhtsa_recalls'
      AND make      != 'UNKNOWN'
      AND component != 'Unknown'
    GROUP BY make, component
)
SELECT
    cc.make,
    cc.component,
    cc.complaint_count,
    cc.crash_count,
    cc.total_injuries,
    COALESCE(cr.recall_count, 0)   AS recall_count,
    COALESCE(cr.units_affected, 0) AS units_affected,
    CASE
        WHEN COALESCE(cr.recall_count, 0) = 0
         AND cc.complaint_count >= 50
         AND cc.crash_count > 0
        THEN 'high concern — no recall'
        WHEN COALESCE(cr.recall_count, 0) = 0
        THEN 'no recall issued'
        ELSE 'recall exists'
    END                            AS recall_status,
    ROUND(
        cc.complaint_count * 100.0 /
        NULLIF(SUM(cc.complaint_count) OVER (PARTITION BY cc.make), 0),
        2
    )                              AS pct_of_brand_complaints
FROM component_complaints cc
LEFT JOIN component_recalls cr
    ON cc.make = cr.make
   AND cc.component = cr.component
ORDER BY cc.make, cc.crash_count DESC, cc.complaint_count DESC;

WITH complaints AS (
    SELECT make, model, COUNT(*) AS complaint_count
    FROM facts_nhtsa_20260606_200341
    WHERE source = 'nhtsa_complaints'
      AND model != 'Unknown'
      AND make  != 'UNKNOWN'
    GROUP BY make, model
),
recalls AS (
    SELECT make, model,
           COUNT(*)            AS recall_count,
           SUM(units_affected) AS units_affected
    FROM facts_nhtsa_20260606_200341
    WHERE source = 'nhtsa_recalls'
      AND model != 'Unknown'
      AND make  != 'UNKNOWN'
    GROUP BY make, model
)
SELECT
    c.make,
    c.model,
    c.complaint_count,
    COALESCE(r.recall_count, 0)   AS recall_count,
    COALESCE(r.units_affected, 0) AS units_affected,
    ROUND(
        COALESCE(r.recall_count, 0) * 100.0 /
        NULLIF(c.complaint_count, 0),
        2
    ) AS recall_to_complaint_pct
FROM complaints c
LEFT JOIN recalls r
  ON c.make = r.make AND c.model = r.model
ORDER BY c.make, c.complaint_count DESC;