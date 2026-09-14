-- SQLite presentation query. Bind :action_rows_json to the complete joined
-- action rows produced by scripts/build_sisa_overall_report.py from the
-- strict metrics and expansion paired_summary CSVs. Preserve input ordering.
SELECT
    json_extract(value, '$.model') AS model,
    json_extract(value, '$.dataset') AS dataset,
    json_extract(value, '$.label') AS label,
    json_extract(value, '$.baseline_auc') AS baseline_auc,
    json_extract(value, '$.sisa_auc') AS sisa_auc,
    json_extract(value, '$.baseline_logloss') AS baseline_logloss,
    json_extract(value, '$.sisa_logloss') AS sisa_logloss,
    json_extract(value, '$.source') AS source,
    json_extract(value, '$.delta_auc') AS delta_auc,
    json_extract(value, '$.delta_logloss') AS delta_logloss,
    json_extract(value, '$.evidence_scope') AS evidence_scope
FROM json_each(:action_rows_json)
ORDER BY CAST(key AS INTEGER);
