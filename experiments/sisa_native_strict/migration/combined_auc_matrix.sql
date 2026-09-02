WITH strict_labels AS (
  SELECT
    model,
    dataset,
    label,
    MAX(CASE WHEN setting = 'baseline' THEN AUC END) AS baseline_auc,
    MAX(CASE WHEN setting = 'sisa' THEN AUC END) AS sisa_auc
  FROM read_csv_auto('experiments/sisa_native_strict/results/metrics.csv')
  GROUP BY model, dataset, label
),
strict_units AS (
  SELECT
    model,
    CASE dataset
      WHEN 'QK_Video_Action' THEN 'QK-Video'
      WHEN 'KuaiRand_Video_Action' THEN 'KuaiRand'
      WHEN 'Taobao_Action' THEN 'Taobao'
      WHEN 'MerRec_Action' THEN 'MerRec'
    END AS dataset,
    AVG(sisa_auc - baseline_auc) * 1000 AS delta_auc_milli,
    'digai' AS provenance
  FROM strict_labels
  GROUP BY model, dataset
),
expansion_units AS (
  SELECT
    model,
    CASE dataset
      WHEN 'TencentGR_10M_Action' THEN 'TencentGR'
      WHEN 'QK_Video_Action' THEN 'QK-Video'
      WHEN 'KuaiRand_Video_Action' THEN 'KuaiRand'
      WHEN 'Taobao_Action' THEN 'Taobao'
      WHEN 'MerRec_Action' THEN 'MerRec'
    END AS dataset,
    mean_delta_auc * 1000 AS delta_auc_milli,
    'hpc3' AS provenance
  FROM read_csv_auto('experiments/sisa_expansion_acd/results/unit_summary.csv')
),
all_units AS (
  SELECT * FROM strict_units
  UNION ALL
  SELECT * FROM expansion_units
)
SELECT
  dataset,
  MAX(CASE WHEN model = 'UltraHSTU' THEN delta_auc_milli END) AS UltraHSTU,
  MAX(CASE WHEN model = 'HiFormer' THEN delta_auc_milli END) AS HiFormer,
  MAX(CASE WHEN model = 'HyFormer' THEN delta_auc_milli END) AS HyFormer,
  MAX(CASE WHEN model = 'OneTrans' THEN delta_auc_milli END) AS OneTrans,
  MAX(CASE WHEN model = 'UniMixer' THEN delta_auc_milli END) AS UniMixer,
  MAX(CASE WHEN model = 'RankMixer' THEN delta_auc_milli END) AS RankMixer,
  MAX(CASE WHEN model = 'Zenith' THEN delta_auc_milli END) AS Zenith,
  MAX(CASE WHEN model = 'UltraHSTU' THEN provenance END) AS _source_UltraHSTU,
  MAX(CASE WHEN model = 'HiFormer' THEN provenance END) AS _source_HiFormer,
  MAX(CASE WHEN model = 'HyFormer' THEN provenance END) AS _source_HyFormer,
  MAX(CASE WHEN model = 'OneTrans' THEN provenance END) AS _source_OneTrans,
  MAX(CASE WHEN model = 'UniMixer' THEN provenance END) AS _source_UniMixer,
  MAX(CASE WHEN model = 'RankMixer' THEN provenance END) AS _source_RankMixer,
  MAX(CASE WHEN model = 'Zenith' THEN provenance END) AS _source_Zenith
FROM all_units
GROUP BY dataset
ORDER BY CASE dataset
  WHEN 'MerRec' THEN 1
  WHEN 'Taobao' THEN 2
  WHEN 'QK-Video' THEN 3
  WHEN 'KuaiRand' THEN 4
  WHEN 'TencentGR' THEN 5
END;
