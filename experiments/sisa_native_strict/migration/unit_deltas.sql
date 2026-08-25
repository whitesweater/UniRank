WITH unit_deltas(unit, model, dataset, delta_auc, improved_labels, delta_logloss) AS (
  VALUES
    ('OneTrans · QK-Video', 'OneTrans', 'QK-Video', 0.000437, '3/4', -0.000193),
    ('OneTrans · KuaiRand', 'OneTrans', 'KuaiRand', 0.007690, '5/6', -0.008736),
    ('OneTrans · Taobao', 'OneTrans', 'Taobao', -0.003796, '1/4', 0.000127),
    ('OneTrans · MerRec', 'OneTrans', 'MerRec', 0.009145, '4/5', -0.001963),
    ('HiFormer · QK-Video', 'HiFormer', 'QK-Video', -0.002100, '1/4', -0.000029),
    ('HiFormer · KuaiRand', 'HiFormer', 'KuaiRand', 0.003684, '3/6', -0.003161),
    ('HiFormer · Taobao', 'HiFormer', 'Taobao', -0.025167, '0/4', 0.000411),
    ('HiFormer · MerRec', 'HiFormer', 'MerRec', 0.001194, '3/5', -0.001831),
    ('RankMixer · QK-Video', 'RankMixer', 'QK-Video', 0.001231, '4/4', -0.000764),
    ('RankMixer · KuaiRand', 'RankMixer', 'KuaiRand', 0.008597, '6/6', -0.004843),
    ('RankMixer · Taobao', 'RankMixer', 'Taobao', -0.000596, '0/4', 0.000016),
    ('RankMixer · MerRec', 'RankMixer', 'MerRec', 0.007506, '4/5', -0.002186),
    ('Zenith · QK-Video', 'Zenith', 'QK-Video', 0.000736, '4/4', -0.000664),
    ('Zenith · KuaiRand', 'Zenith', 'KuaiRand', 0.006733, '6/6', -0.004647),
    ('Zenith · Taobao', 'Zenith', 'Taobao', 0.003102, '3/4', -0.000028),
    ('Zenith · MerRec', 'Zenith', 'MerRec', 0.003184, '3/5', -0.001337)
)
SELECT * FROM unit_deltas;
