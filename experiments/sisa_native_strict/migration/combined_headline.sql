WITH
strict_runs AS (
  SELECT * FROM read_csv_auto('experiments/sisa_native_strict/results/runs.csv')
),
expansion_runs AS (
  SELECT * FROM read_csv_auto('experiments/sisa_expansion_acd/results/runs.csv')
),
strict_pairs AS (
  SELECT COUNT(*) / 2 AS pair_count
  FROM read_csv_auto('experiments/sisa_native_strict/results/metrics.csv')
),
expansion_pairs AS (
  SELECT * FROM read_csv_auto('experiments/sisa_expansion_acd/results/paired_summary.csv')
),
baseline_audit AS (
  SELECT * FROM read_csv_auto('experiments/sisa_expansion_acd/results/baseline_audit.csv')
)
SELECT
  (SELECT COUNT(*) FROM strict_runs) + (SELECT COUNT(*) FROM expansion_runs) AS completed_tasks,
  (SELECT pair_count FROM strict_pairs) + (SELECT COUNT(*) FROM expansion_pairs) AS paired_labels,
  1000 * (SELECT AVG(delta_auc) FROM expansion_pairs) AS expansion_average_auc_delta_milli,
  1000 * (SELECT MAX(abs_delta_auc) FROM baseline_audit) AS baseline_max_abs_delta_milli;
