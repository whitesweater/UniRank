# SISA single-seed 20262029 result snapshot

- Complete tasks: **16/16**
- Final label metrics: **68**
- Experiment seed: **`20262029`** (one experiment seed only)
- Internal RNG substreams: dataloader `2028`, SISA parameters `20260823`; these are not additional experiment seeds
- Protocol assignments: `ws2_bs16384_acc1`=16
- Error classes: none

## Runs

| Task | Model | Dataset | Protocol | Attempt | Job | State | GPU | Complete | Error / incomplete evidence |
|---:|---|---|---|---:|---|---|---|---|---|
| 0 | Zenith | MerRec_Action | ws2_bs16384_acc1 | 2 | 549797_0 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 1 | HyFormer | MerRec_Action | ws2_bs16384_acc1 | 1 | 549595_1 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 2 | HiFormer | MerRec_Action | ws2_bs16384_acc1 | 1 | 549595_2 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 3 | RankMixer | MerRec_Action | ws2_bs16384_acc1 | 1 | 549595_3 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 4 | Zenith | KuaiRand_Video_Action | ws2_bs16384_acc1 | 1 | 549595_4 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 5 | HyFormer | KuaiRand_Video_Action | ws2_bs16384_acc1 | 1 | 549595_5 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 6 | HiFormer | KuaiRand_Video_Action | ws2_bs16384_acc1 | 1 | 549595_6 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 7 | RankMixer | KuaiRand_Video_Action | ws2_bs16384_acc1 | 1 | 549595_7 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 8 | Zenith | QK_Video_Action | ws2_bs16384_acc1 | 1 | 549595_8 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 9 | HyFormer | QK_Video_Action | ws2_bs16384_acc1 | 1 | 549595_9 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 10 | HiFormer | QK_Video_Action | ws2_bs16384_acc1 | 1 | 549595_10 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 11 | RankMixer | QK_Video_Action | ws2_bs16384_acc1 | 1 | 549595_11 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 12 | Zenith | TencentGR_10M_Action | ws2_bs16384_acc1 | 1 | 549595_12 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 13 | HyFormer | TencentGR_10M_Action | ws2_bs16384_acc1 | 1 | 549595_13 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 14 | HiFormer | TencentGR_10M_Action | ws2_bs16384_acc1 | 1 | 549595_14 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |
| 15 | RankMixer | TencentGR_10M_Action | ws2_bs16384_acc1 | 1 | 549595_15 | COMPLETED | NVIDIA H100 80GB HBM3 | True | none |

## Final test metrics

| Task | Model | Dataset | Label | AUC | Logloss | Protocol |
|---:|---|---|---|---:|---:|---|
| 0 | Zenith | MerRec_Action | Cart | 0.812453 | 0.053386 | ws2_bs16384_acc1 |
| 0 | Zenith | MerRec_Action | Checkout | 0.843908 | 0.012052 | ws2_bs16384_acc1 |
| 0 | Zenith | MerRec_Action | Like | 0.754835 | 0.325968 | ws2_bs16384_acc1 |
| 0 | Zenith | MerRec_Action | Offer | 0.764756 | 0.022429 | ws2_bs16384_acc1 |
| 0 | Zenith | MerRec_Action | Purchase | 0.840841 | 0.007468 | ws2_bs16384_acc1 |
| 1 | HyFormer | MerRec_Action | Cart | 0.811112 | 0.052343 | ws2_bs16384_acc1 |
| 1 | HyFormer | MerRec_Action | Checkout | 0.853285 | 0.006130 | ws2_bs16384_acc1 |
| 1 | HyFormer | MerRec_Action | Like | 0.755749 | 0.326671 | ws2_bs16384_acc1 |
| 1 | HyFormer | MerRec_Action | Offer | 0.756030 | 0.021099 | ws2_bs16384_acc1 |
| 1 | HyFormer | MerRec_Action | Purchase | 0.856738 | 0.003760 | ws2_bs16384_acc1 |
| 2 | HiFormer | MerRec_Action | Cart | 0.820005 | 0.051863 | ws2_bs16384_acc1 |
| 2 | HiFormer | MerRec_Action | Checkout | 0.829334 | 0.014377 | ws2_bs16384_acc1 |
| 2 | HiFormer | MerRec_Action | Like | 0.761906 | 0.323950 | ws2_bs16384_acc1 |
| 2 | HiFormer | MerRec_Action | Offer | 0.773916 | 0.023083 | ws2_bs16384_acc1 |
| 2 | HiFormer | MerRec_Action | Purchase | 0.834625 | 0.008625 | ws2_bs16384_acc1 |
| 3 | RankMixer | MerRec_Action | Cart | 0.810422 | 0.052437 | ws2_bs16384_acc1 |
| 3 | RankMixer | MerRec_Action | Checkout | 0.834801 | 0.004868 | ws2_bs16384_acc1 |
| 3 | RankMixer | MerRec_Action | Like | 0.763228 | 0.320886 | ws2_bs16384_acc1 |
| 3 | RankMixer | MerRec_Action | Offer | 0.738775 | 0.021356 | ws2_bs16384_acc1 |
| 3 | RankMixer | MerRec_Action | Purchase | 0.837675 | 0.003169 | ws2_bs16384_acc1 |
| 4 | Zenith | KuaiRand_Video_Action | is_click | 0.785892 | 0.538663 | ws2_bs16384_acc1 |
| 4 | Zenith | KuaiRand_Video_Action | is_comment | 0.895144 | 0.014827 | ws2_bs16384_acc1 |
| 4 | Zenith | KuaiRand_Video_Action | is_follow | 0.889686 | 0.006170 | ws2_bs16384_acc1 |
| 4 | Zenith | KuaiRand_Video_Action | is_forward | 0.873658 | 0.005579 | ws2_bs16384_acc1 |
| 4 | Zenith | KuaiRand_Video_Action | is_like | 0.927028 | 0.055352 | ws2_bs16384_acc1 |
| 4 | Zenith | KuaiRand_Video_Action | long_view | 0.800178 | 0.462285 | ws2_bs16384_acc1 |
| 5 | HyFormer | KuaiRand_Video_Action | is_click | 0.795028 | 0.524111 | ws2_bs16384_acc1 |
| 5 | HyFormer | KuaiRand_Video_Action | is_comment | 0.889446 | 0.015025 | ws2_bs16384_acc1 |
| 5 | HyFormer | KuaiRand_Video_Action | is_follow | 0.875987 | 0.006358 | ws2_bs16384_acc1 |
| 5 | HyFormer | KuaiRand_Video_Action | is_forward | 0.871262 | 0.005849 | ws2_bs16384_acc1 |
| 5 | HyFormer | KuaiRand_Video_Action | is_like | 0.926478 | 0.054160 | ws2_bs16384_acc1 |
| 5 | HyFormer | KuaiRand_Video_Action | long_view | 0.811544 | 0.442765 | ws2_bs16384_acc1 |
| 6 | HiFormer | KuaiRand_Video_Action | is_click | 0.796055 | 0.523250 | ws2_bs16384_acc1 |
| 6 | HiFormer | KuaiRand_Video_Action | is_comment | 0.896727 | 0.014736 | ws2_bs16384_acc1 |
| 6 | HiFormer | KuaiRand_Video_Action | is_follow | 0.886823 | 0.006182 | ws2_bs16384_acc1 |
| 6 | HiFormer | KuaiRand_Video_Action | is_forward | 0.867965 | 0.005663 | ws2_bs16384_acc1 |
| 6 | HiFormer | KuaiRand_Video_Action | is_like | 0.926401 | 0.054387 | ws2_bs16384_acc1 |
| 6 | HiFormer | KuaiRand_Video_Action | long_view | 0.805943 | 0.456811 | ws2_bs16384_acc1 |
| 7 | RankMixer | KuaiRand_Video_Action | is_click | 0.796813 | 0.522401 | ws2_bs16384_acc1 |
| 7 | RankMixer | KuaiRand_Video_Action | is_comment | 0.894452 | 0.014853 | ws2_bs16384_acc1 |
| 7 | RankMixer | KuaiRand_Video_Action | is_follow | 0.882847 | 0.006237 | ws2_bs16384_acc1 |
| 7 | RankMixer | KuaiRand_Video_Action | is_forward | 0.867082 | 0.005708 | ws2_bs16384_acc1 |
| 7 | RankMixer | KuaiRand_Video_Action | is_like | 0.928654 | 0.053569 | ws2_bs16384_acc1 |
| 7 | RankMixer | KuaiRand_Video_Action | long_view | 0.814307 | 0.439336 | ws2_bs16384_acc1 |
| 8 | Zenith | QK_Video_Action | click | 0.934530 | 0.250160 | ws2_bs16384_acc1 |
| 8 | Zenith | QK_Video_Action | follow | 0.921674 | 0.004998 | ws2_bs16384_acc1 |
| 8 | Zenith | QK_Video_Action | like | 0.947288 | 0.037029 | ws2_bs16384_acc1 |
| 8 | Zenith | QK_Video_Action | share | 0.936623 | 0.005304 | ws2_bs16384_acc1 |
| 9 | HyFormer | QK_Video_Action | click | 0.934379 | 0.250461 | ws2_bs16384_acc1 |
| 9 | HyFormer | QK_Video_Action | follow | 0.912153 | 0.005266 | ws2_bs16384_acc1 |
| 9 | HyFormer | QK_Video_Action | like | 0.946284 | 0.037557 | ws2_bs16384_acc1 |
| 9 | HyFormer | QK_Video_Action | share | 0.931824 | 0.005460 | ws2_bs16384_acc1 |
| 10 | HiFormer | QK_Video_Action | click | 0.932216 | 0.253956 | ws2_bs16384_acc1 |
| 10 | HiFormer | QK_Video_Action | follow | 0.913352 | 0.005280 | ws2_bs16384_acc1 |
| 10 | HiFormer | QK_Video_Action | like | 0.945247 | 0.037883 | ws2_bs16384_acc1 |
| 10 | HiFormer | QK_Video_Action | share | 0.931627 | 0.005564 | ws2_bs16384_acc1 |
| 11 | RankMixer | QK_Video_Action | click | 0.932877 | 0.253871 | ws2_bs16384_acc1 |
| 11 | RankMixer | QK_Video_Action | follow | 0.910454 | 0.005258 | ws2_bs16384_acc1 |
| 11 | RankMixer | QK_Video_Action | like | 0.943617 | 0.038093 | ws2_bs16384_acc1 |
| 11 | RankMixer | QK_Video_Action | share | 0.929672 | 0.005444 | ws2_bs16384_acc1 |
| 12 | Zenith | TencentGR_10M_Action | is_click | 0.839080 | 0.063010 | ws2_bs16384_acc1 |
| 12 | Zenith | TencentGR_10M_Action | is_conversion | 0.870350 | 0.021840 | ws2_bs16384_acc1 |
| 13 | HyFormer | TencentGR_10M_Action | is_click | 0.845539 | 0.052826 | ws2_bs16384_acc1 |
| 13 | HyFormer | TencentGR_10M_Action | is_conversion | 0.889328 | 0.018939 | ws2_bs16384_acc1 |
| 14 | HiFormer | TencentGR_10M_Action | is_click | 0.850097 | 0.047388 | ws2_bs16384_acc1 |
| 14 | HiFormer | TencentGR_10M_Action | is_conversion | 0.889723 | 0.017501 | ws2_bs16384_acc1 |
| 15 | RankMixer | TencentGR_10M_Action | is_click | 0.824030 | 0.067723 | ws2_bs16384_acc1 |
| 15 | RankMixer | TencentGR_10M_Action | is_conversion | 0.877133 | 0.025274 | ws2_bs16384_acc1 |
