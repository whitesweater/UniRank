# SISA single-seed 20262028 result snapshot

- Complete tasks: **16/16**
- Final label metrics: **68**
- Seed bundle: base `20262028`, dataloader `2027`, SISA parameters `20260822`
- Protocol assignments: `ws2_bs16384_acc1`=16
- Error classes: none

## Runs

| Task | Model | Dataset | Protocol | Attempt | Job | State | GPU | Complete | Error / incomplete evidence |
|---:|---|---|---|---:|---|---|---|---|---|
| 0 | Zenith | MerRec_Action | ws2_bs16384_acc1 | 1 | 548166_0 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 1 | HyFormer | MerRec_Action | ws2_bs16384_acc1 | 1 | 548166_1 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 2 | HiFormer | MerRec_Action | ws2_bs16384_acc1 | 1 | 548166_2 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 3 | RankMixer | MerRec_Action | ws2_bs16384_acc1 | 1 | 548166_3 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 4 | Zenith | KuaiRand_Video_Action | ws2_bs16384_acc1 | 1 | 548166_4 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 5 | HyFormer | KuaiRand_Video_Action | ws2_bs16384_acc1 | 1 | 548166_5 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 6 | HiFormer | KuaiRand_Video_Action | ws2_bs16384_acc1 | 1 | 548166_6 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 7 | RankMixer | KuaiRand_Video_Action | ws2_bs16384_acc1 | 1 | 548166_7 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 8 | Zenith | QK_Video_Action | ws2_bs16384_acc1 | 1 | 548166_8 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 9 | HyFormer | QK_Video_Action | ws2_bs16384_acc1 | 1 | 548166_9 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 10 | HiFormer | QK_Video_Action | ws2_bs16384_acc1 | 1 | 548166_10 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 11 | RankMixer | QK_Video_Action | ws2_bs16384_acc1 | 1 | 548166_11 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 12 | Zenith | TencentGR_10M_Action | ws2_bs16384_acc1 | 1 | 548166_12 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 13 | HyFormer | TencentGR_10M_Action | ws2_bs16384_acc1 | 1 | 548166_13 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 14 | HiFormer | TencentGR_10M_Action | ws2_bs16384_acc1 | 1 | 548166_14 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 15 | RankMixer | TencentGR_10M_Action | ws2_bs16384_acc1 | 1 | 548166_15 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |

## Final test metrics

| Task | Model | Dataset | Label | AUC | Logloss | Protocol |
|---:|---|---|---|---:|---:|---|
| 0 | Zenith | MerRec_Action | Cart | 0.816580 | 0.052110 | ws2_bs16384_acc1 |
| 0 | Zenith | MerRec_Action | Checkout | 0.846522 | 0.011335 | ws2_bs16384_acc1 |
| 0 | Zenith | MerRec_Action | Like | 0.758067 | 0.324592 | ws2_bs16384_acc1 |
| 0 | Zenith | MerRec_Action | Offer | 0.775302 | 0.021595 | ws2_bs16384_acc1 |
| 0 | Zenith | MerRec_Action | Purchase | 0.851195 | 0.008151 | ws2_bs16384_acc1 |
| 1 | HyFormer | MerRec_Action | Cart | 0.810189 | 0.052477 | ws2_bs16384_acc1 |
| 1 | HyFormer | MerRec_Action | Checkout | 0.855579 | 0.004954 | ws2_bs16384_acc1 |
| 1 | HyFormer | MerRec_Action | Like | 0.750689 | 0.331361 | ws2_bs16384_acc1 |
| 1 | HyFormer | MerRec_Action | Offer | 0.731796 | 0.021512 | ws2_bs16384_acc1 |
| 1 | HyFormer | MerRec_Action | Purchase | 0.856109 | 0.003084 | ws2_bs16384_acc1 |
| 2 | HiFormer | MerRec_Action | Cart | 0.818659 | 0.051844 | ws2_bs16384_acc1 |
| 2 | HiFormer | MerRec_Action | Checkout | 0.845414 | 0.009625 | ws2_bs16384_acc1 |
| 2 | HiFormer | MerRec_Action | Like | 0.765716 | 0.323721 | ws2_bs16384_acc1 |
| 2 | HiFormer | MerRec_Action | Offer | 0.769679 | 0.021428 | ws2_bs16384_acc1 |
| 2 | HiFormer | MerRec_Action | Purchase | 0.846914 | 0.006044 | ws2_bs16384_acc1 |
| 3 | RankMixer | MerRec_Action | Cart | 0.814349 | 0.052303 | ws2_bs16384_acc1 |
| 3 | RankMixer | MerRec_Action | Checkout | 0.847677 | 0.004640 | ws2_bs16384_acc1 |
| 3 | RankMixer | MerRec_Action | Like | 0.760581 | 0.323712 | ws2_bs16384_acc1 |
| 3 | RankMixer | MerRec_Action | Offer | 0.734759 | 0.021394 | ws2_bs16384_acc1 |
| 3 | RankMixer | MerRec_Action | Purchase | 0.853851 | 0.002998 | ws2_bs16384_acc1 |
| 4 | Zenith | KuaiRand_Video_Action | is_click | 0.787549 | 0.536917 | ws2_bs16384_acc1 |
| 4 | Zenith | KuaiRand_Video_Action | is_comment | 0.896437 | 0.014745 | ws2_bs16384_acc1 |
| 4 | Zenith | KuaiRand_Video_Action | is_follow | 0.885366 | 0.006155 | ws2_bs16384_acc1 |
| 4 | Zenith | KuaiRand_Video_Action | is_forward | 0.869064 | 0.005672 | ws2_bs16384_acc1 |
| 4 | Zenith | KuaiRand_Video_Action | is_like | 0.925888 | 0.055153 | ws2_bs16384_acc1 |
| 4 | Zenith | KuaiRand_Video_Action | long_view | 0.802694 | 0.457331 | ws2_bs16384_acc1 |
| 5 | HyFormer | KuaiRand_Video_Action | is_click | 0.793189 | 0.532893 | ws2_bs16384_acc1 |
| 5 | HyFormer | KuaiRand_Video_Action | is_comment | 0.893000 | 0.015244 | ws2_bs16384_acc1 |
| 5 | HyFormer | KuaiRand_Video_Action | is_follow | 0.885166 | 0.006285 | ws2_bs16384_acc1 |
| 5 | HyFormer | KuaiRand_Video_Action | is_forward | 0.868786 | 0.005668 | ws2_bs16384_acc1 |
| 5 | HyFormer | KuaiRand_Video_Action | is_like | 0.925453 | 0.056911 | ws2_bs16384_acc1 |
| 5 | HyFormer | KuaiRand_Video_Action | long_view | 0.806051 | 0.446141 | ws2_bs16384_acc1 |
| 6 | HiFormer | KuaiRand_Video_Action | is_click | 0.795981 | 0.524404 | ws2_bs16384_acc1 |
| 6 | HiFormer | KuaiRand_Video_Action | is_comment | 0.898652 | 0.014701 | ws2_bs16384_acc1 |
| 6 | HiFormer | KuaiRand_Video_Action | is_follow | 0.885380 | 0.006254 | ws2_bs16384_acc1 |
| 6 | HiFormer | KuaiRand_Video_Action | is_forward | 0.868933 | 0.005698 | ws2_bs16384_acc1 |
| 6 | HiFormer | KuaiRand_Video_Action | is_like | 0.929336 | 0.053535 | ws2_bs16384_acc1 |
| 6 | HiFormer | KuaiRand_Video_Action | long_view | 0.809744 | 0.448796 | ws2_bs16384_acc1 |
| 7 | RankMixer | KuaiRand_Video_Action | is_click | 0.794903 | 0.525134 | ws2_bs16384_acc1 |
| 7 | RankMixer | KuaiRand_Video_Action | is_comment | 0.890161 | 0.014956 | ws2_bs16384_acc1 |
| 7 | RankMixer | KuaiRand_Video_Action | is_follow | 0.877210 | 0.006344 | ws2_bs16384_acc1 |
| 7 | RankMixer | KuaiRand_Video_Action | is_forward | 0.866626 | 0.005663 | ws2_bs16384_acc1 |
| 7 | RankMixer | KuaiRand_Video_Action | is_like | 0.927295 | 0.053911 | ws2_bs16384_acc1 |
| 7 | RankMixer | KuaiRand_Video_Action | long_view | 0.812814 | 0.441410 | ws2_bs16384_acc1 |
| 8 | Zenith | QK_Video_Action | click | 0.934288 | 0.250450 | ws2_bs16384_acc1 |
| 8 | Zenith | QK_Video_Action | follow | 0.922992 | 0.004942 | ws2_bs16384_acc1 |
| 8 | Zenith | QK_Video_Action | like | 0.948310 | 0.036833 | ws2_bs16384_acc1 |
| 8 | Zenith | QK_Video_Action | share | 0.935292 | 0.005326 | ws2_bs16384_acc1 |
| 9 | HyFormer | QK_Video_Action | click | 0.934075 | 0.250842 | ws2_bs16384_acc1 |
| 9 | HyFormer | QK_Video_Action | follow | 0.913611 | 0.005121 | ws2_bs16384_acc1 |
| 9 | HyFormer | QK_Video_Action | like | 0.947015 | 0.037439 | ws2_bs16384_acc1 |
| 9 | HyFormer | QK_Video_Action | share | 0.930531 | 0.005428 | ws2_bs16384_acc1 |
| 10 | HiFormer | QK_Video_Action | click | 0.931291 | 0.255325 | ws2_bs16384_acc1 |
| 10 | HiFormer | QK_Video_Action | follow | 0.913586 | 0.005112 | ws2_bs16384_acc1 |
| 10 | HiFormer | QK_Video_Action | like | 0.945454 | 0.037497 | ws2_bs16384_acc1 |
| 10 | HiFormer | QK_Video_Action | share | 0.933234 | 0.005447 | ws2_bs16384_acc1 |
| 11 | RankMixer | QK_Video_Action | click | 0.932455 | 0.253516 | ws2_bs16384_acc1 |
| 11 | RankMixer | QK_Video_Action | follow | 0.912944 | 0.005261 | ws2_bs16384_acc1 |
| 11 | RankMixer | QK_Video_Action | like | 0.943098 | 0.038666 | ws2_bs16384_acc1 |
| 11 | RankMixer | QK_Video_Action | share | 0.927964 | 0.005594 | ws2_bs16384_acc1 |
| 12 | Zenith | TencentGR_10M_Action | is_click | 0.839413 | 0.054888 | ws2_bs16384_acc1 |
| 12 | Zenith | TencentGR_10M_Action | is_conversion | 0.888088 | 0.021089 | ws2_bs16384_acc1 |
| 13 | HyFormer | TencentGR_10M_Action | is_click | 0.851790 | 0.047312 | ws2_bs16384_acc1 |
| 13 | HyFormer | TencentGR_10M_Action | is_conversion | 0.897097 | 0.017701 | ws2_bs16384_acc1 |
| 14 | HiFormer | TencentGR_10M_Action | is_click | 0.851302 | 0.043410 | ws2_bs16384_acc1 |
| 14 | HiFormer | TencentGR_10M_Action | is_conversion | 0.890414 | 0.017180 | ws2_bs16384_acc1 |
| 15 | RankMixer | TencentGR_10M_Action | is_click | 0.821511 | 0.066827 | ws2_bs16384_acc1 |
| 15 | RankMixer | TencentGR_10M_Action | is_conversion | 0.874172 | 0.026967 | ws2_bs16384_acc1 |
