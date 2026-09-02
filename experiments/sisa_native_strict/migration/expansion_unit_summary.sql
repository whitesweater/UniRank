SELECT
  model || ' · ' || CASE dataset
    WHEN 'TencentGR_10M_Action' THEN 'TencentGR'
    WHEN 'QK_Video_Action' THEN 'QK-Video'
    WHEN 'KuaiRand_Video_Action' THEN 'KuaiRand'
    WHEN 'Taobao_Action' THEN 'Taobao'
    WHEN 'MerRec_Action' THEN 'MerRec'
  END AS unit,
  model,
  CASE dataset
    WHEN 'TencentGR_10M_Action' THEN 'TencentGR'
    WHEN 'QK_Video_Action' THEN 'QK-Video'
    WHEN 'KuaiRand_Video_Action' THEN 'KuaiRand'
    WHEN 'Taobao_Action' THEN 'Taobao'
    WHEN 'MerRec_Action' THEN 'MerRec'
  END AS dataset,
  labels,
  mean_delta_auc AS delta_auc,
  auc_improved AS improved_labels,
  mean_delta_logloss AS delta_logloss,
  max_abs_baseline_delta AS max_abs_baseline_deviation
FROM read_csv_auto('experiments/sisa_expansion_acd/results/unit_summary.csv')
ORDER BY mean_delta_auc DESC;
