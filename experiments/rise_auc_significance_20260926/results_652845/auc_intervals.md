This is a post hoc exploratory comparison of fixed predictions; positive AUC(Full) - AUC(RMS) favors Full.

| Model | Seed | Endpoint | Full AUC | RMS AUC | Difference | 95% interval | 99.375% Bonferroni-8 interval | Adjusted direction |
|---|---:|---|---:|---:|---:|---|---|---|
| RankMixer | 20262028 | macro_auc | 0.845647400 | 0.845790490 | -0.000143090 | [-0.000941349, +0.000454945] | [-0.001345721, +0.000678204] | unresolved |
| RankMixer | 20262028 | click_auc | 0.801748681 | 0.801596910 | +0.000151771 | [+0.000040452, +0.000272563] | [-0.000006625, +0.000322627] | unresolved |
| RankMixer | 20262029 | macro_auc | 0.850337091 | 0.849722446 | +0.000614645 | [-0.000033655, +0.001310126] | [-0.000315770, +0.001687952] | unresolved |
| RankMixer | 20262029 | click_auc | 0.802496846 | 0.802101811 | +0.000395035 | [+0.000199882, +0.000615041] | [+0.000121416, +0.000708194] | Full_higher_AUC |
| HiFormer | 20262028 | macro_auc | 0.866749291 | 0.867427237 | -0.000677947 | [-0.001265396, -0.000076938] | [-0.001518500, +0.000159089] | unresolved |
| HiFormer | 20262028 | click_auc | 0.846241374 | 0.847454201 | -0.001212827 | [-0.001905766, -0.000530238] | [-0.002228007, -0.000267454] | RMS_higher_AUC |
| HiFormer | 20262029 | macro_auc | 0.862540830 | 0.862153801 | +0.000387029 | [-0.000235317, +0.001041062] | [-0.000487353, +0.001257177] | unresolved |
| HiFormer | 20262029 | click_auc | 0.841338212 | 0.841019728 | +0.000318484 | [-0.000406416, +0.001051551] | [-0.000692842, +0.001331646] | unresolved |

HiFormer seed 20262029 was selected after inspecting its AUC/BCE point estimates; no confirmatory or prespecified-study claim

Conditional on these trained checkpoints, chosen windows and estimated shared templates; no training-seed replication uncertainty, population generalization or template re-estimation inside bootstrap

10000 draws give about31 observations per adjusted tail; reported bootstrap endpoints have Monte Carlo uncertainty and are not exact coverage guarantees

No p-values are reported. An interval crossing zero means unresolved, not equivalent. No labels or missing-class draws are silently omitted.
