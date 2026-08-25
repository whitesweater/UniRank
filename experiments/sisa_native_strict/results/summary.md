# UniRank SISA strict result snapshot

Structurally complete logical tasks: **32/32**.

Exploratory single-GPU results are intentionally excluded.

| Model | Dataset | Label | Baseline AUC | SISA AUC | Delta AUC | Baseline logloss | SISA logloss | Delta logloss |
|---|---|---|---:|---:|---:|---:|---:|---:|
| HiFormer | KuaiRand_Video_Action | is_click | 0.777662 | 0.797238 | +0.019576 | 0.541932 | 0.521549 | -0.020383 |
| HiFormer | KuaiRand_Video_Action | is_comment | 0.898789 | 0.895209 | -0.003580 | 0.014709 | 0.014806 | +0.000097 |
| HiFormer | KuaiRand_Video_Action | is_follow | 0.879913 | 0.881796 | +0.001883 | 0.006364 | 0.006180 | -0.000184 |
| HiFormer | KuaiRand_Video_Action | is_forward | 0.870920 | 0.869450 | -0.001470 | 0.005684 | 0.005692 | +0.000008 |
| HiFormer | KuaiRand_Video_Action | is_like | 0.928496 | 0.925869 | -0.002627 | 0.053637 | 0.054294 | +0.000657 |
| HiFormer | KuaiRand_Video_Action | long_view | 0.795110 | 0.803433 | +0.008323 | 0.460971 | 0.461809 | +0.000838 |
| HiFormer | MerRec_Action | Cart | 0.810019 | 0.813548 | +0.003529 | 0.053089 | 0.052204 | -0.000885 |
| HiFormer | MerRec_Action | Checkout | 0.834515 | 0.838614 | +0.004099 | 0.012448 | 0.011611 | -0.000837 |
| HiFormer | MerRec_Action | Like | 0.749610 | 0.764396 | +0.014786 | 0.328501 | 0.321600 | -0.006901 |
| HiFormer | MerRec_Action | Offer | 0.775777 | 0.761824 | -0.013953 | 0.023378 | 0.023437 | +0.000059 |
| HiFormer | MerRec_Action | Purchase | 0.835563 | 0.833070 | -0.002493 | 0.008751 | 0.008159 | -0.000592 |
| HiFormer | QK_Video_Action | click | 0.932306 | 0.932526 | +0.000220 | 0.253972 | 0.253650 | -0.000322 |
| HiFormer | QK_Video_Action | follow | 0.919914 | 0.913924 | -0.005990 | 0.004997 | 0.005111 | +0.000114 |
| HiFormer | QK_Video_Action | like | 0.946691 | 0.945462 | -0.001229 | 0.037272 | 0.037298 | +0.000026 |
| HiFormer | QK_Video_Action | share | 0.933742 | 0.932340 | -0.001402 | 0.005301 | 0.005366 | +0.000065 |
| HiFormer | Taobao_Action | buy | 0.847202 | 0.807079 | -0.040123 | 0.006887 | 0.006962 | +0.000075 |
| HiFormer | Taobao_Action | cart | 0.825830 | 0.813815 | -0.012015 | 0.018146 | 0.018067 | -0.000079 |
| HiFormer | Taobao_Action | fav | 0.853523 | 0.825288 | -0.028235 | 0.013816 | 0.014294 | +0.000478 |
| HiFormer | Taobao_Action | is_click | 0.645062 | 0.624768 | -0.020294 | 0.186264 | 0.187433 | +0.001169 |
| OneTrans | KuaiRand_Video_Action | is_click | 0.772234 | 0.791105 | +0.018871 | 0.566734 | 0.538148 | -0.028586 |
| OneTrans | KuaiRand_Video_Action | is_comment | 0.896976 | 0.896294 | -0.000682 | 0.014770 | 0.014797 | +0.000027 |
| OneTrans | KuaiRand_Video_Action | is_follow | 0.883080 | 0.886430 | +0.003350 | 0.006201 | 0.006128 | -0.000073 |
| OneTrans | KuaiRand_Video_Action | is_forward | 0.872306 | 0.877678 | +0.005372 | 0.005742 | 0.005611 | -0.000131 |
| OneTrans | KuaiRand_Video_Action | is_like | 0.923693 | 0.928058 | +0.004365 | 0.055033 | 0.053581 | -0.001452 |
| OneTrans | KuaiRand_Video_Action | long_view | 0.795764 | 0.810629 | +0.014865 | 0.469541 | 0.447340 | -0.022201 |
| OneTrans | MerRec_Action | Cart | 0.789147 | 0.820006 | +0.030859 | 0.054460 | 0.051998 | -0.002462 |
| OneTrans | MerRec_Action | Checkout | 0.866320 | 0.872093 | +0.005773 | 0.006042 | 0.005122 | -0.000920 |
| OneTrans | MerRec_Action | Like | 0.755615 | 0.765848 | +0.010233 | 0.325279 | 0.319739 | -0.005540 |
| OneTrans | MerRec_Action | Offer | 0.764039 | 0.757720 | -0.006319 | 0.021467 | 0.021190 | -0.000277 |
| OneTrans | MerRec_Action | Purchase | 0.868531 | 0.873708 | +0.005177 | 0.003837 | 0.003222 | -0.000615 |
| OneTrans | QK_Video_Action | click | 0.935674 | 0.935902 | +0.000228 | 0.248535 | 0.247955 | -0.000580 |
| OneTrans | QK_Video_Action | follow | 0.917042 | 0.918218 | +0.001176 | 0.005088 | 0.005097 | +0.000009 |
| OneTrans | QK_Video_Action | like | 0.948348 | 0.948861 | +0.000513 | 0.036389 | 0.036164 | -0.000225 |
| OneTrans | QK_Video_Action | share | 0.933407 | 0.933237 | -0.000170 | 0.005297 | 0.005320 | +0.000023 |
| OneTrans | Taobao_Action | buy | 0.767930 | 0.753688 | -0.014242 | 0.007003 | 0.007050 | +0.000047 |
| OneTrans | Taobao_Action | cart | 0.742172 | 0.741239 | -0.000933 | 0.020009 | 0.019920 | -0.000089 |
| OneTrans | Taobao_Action | fav | 0.780022 | 0.776298 | -0.003724 | 0.014892 | 0.015006 | +0.000114 |
| OneTrans | Taobao_Action | is_click | 0.629803 | 0.633518 | +0.003715 | 0.187510 | 0.187945 | +0.000435 |
| RankMixer | KuaiRand_Video_Action | is_click | 0.780693 | 0.794759 | +0.014066 | 0.541045 | 0.525008 | -0.016037 |
| RankMixer | KuaiRand_Video_Action | is_comment | 0.884469 | 0.891848 | +0.007379 | 0.015211 | 0.014913 | -0.000298 |
| RankMixer | KuaiRand_Video_Action | is_follow | 0.873795 | 0.880471 | +0.006676 | 0.006422 | 0.006296 | -0.000126 |
| RankMixer | KuaiRand_Video_Action | is_forward | 0.851098 | 0.860017 | +0.008919 | 0.005811 | 0.005742 | -0.000069 |
| RankMixer | KuaiRand_Video_Action | is_like | 0.923963 | 0.927324 | +0.003361 | 0.055584 | 0.053812 | -0.001772 |
| RankMixer | KuaiRand_Video_Action | long_view | 0.800238 | 0.811420 | +0.011182 | 0.453588 | 0.442830 | -0.010758 |
| RankMixer | MerRec_Action | Cart | 0.792999 | 0.810890 | +0.017891 | 0.053894 | 0.052330 | -0.001564 |
| RankMixer | MerRec_Action | Checkout | 0.836097 | 0.838580 | +0.002483 | 0.004732 | 0.004693 | -0.000039 |
| RankMixer | MerRec_Action | Like | 0.744913 | 0.759983 | +0.015070 | 0.331885 | 0.322562 | -0.009323 |
| RankMixer | MerRec_Action | Offer | 0.728575 | 0.732952 | +0.004377 | 0.021398 | 0.021410 | +0.000012 |
| RankMixer | MerRec_Action | Purchase | 0.840563 | 0.838274 | -0.002289 | 0.003050 | 0.003033 | -0.000017 |
| RankMixer | QK_Video_Action | click | 0.931713 | 0.933300 | +0.001587 | 0.254815 | 0.252233 | -0.002582 |
| RankMixer | QK_Video_Action | follow | 0.910614 | 0.911848 | +0.001234 | 0.005177 | 0.005163 | -0.000014 |
| RankMixer | QK_Video_Action | like | 0.944118 | 0.945435 | +0.001317 | 0.037417 | 0.036966 | -0.000451 |
| RankMixer | QK_Video_Action | share | 0.929184 | 0.929971 | +0.000787 | 0.005411 | 0.005404 | -0.000007 |
| RankMixer | Taobao_Action | buy | 0.683724 | 0.682729 | -0.000995 | 0.007292 | 0.007303 | +0.000011 |
| RankMixer | Taobao_Action | cart | 0.675781 | 0.675196 | -0.000585 | 0.020714 | 0.020731 | +0.000017 |
| RankMixer | Taobao_Action | fav | 0.721372 | 0.720740 | -0.000632 | 0.015512 | 0.015515 | +0.000003 |
| RankMixer | Taobao_Action | is_click | 0.604445 | 0.604271 | -0.000174 | 0.188673 | 0.188708 | +0.000035 |
| Zenith | KuaiRand_Video_Action | is_click | 0.781888 | 0.792759 | +0.010871 | 0.538991 | 0.527799 | -0.011192 |
| Zenith | KuaiRand_Video_Action | is_comment | 0.897885 | 0.901214 | +0.003329 | 0.014788 | 0.014609 | -0.000179 |
| Zenith | KuaiRand_Video_Action | is_follow | 0.881683 | 0.890208 | +0.008525 | 0.006761 | 0.006251 | -0.000510 |
| Zenith | KuaiRand_Video_Action | is_forward | 0.871088 | 0.873619 | +0.002531 | 0.005649 | 0.005596 | -0.000053 |
| Zenith | KuaiRand_Video_Action | is_like | 0.928141 | 0.930466 | +0.002325 | 0.054321 | 0.053227 | -0.001094 |
| Zenith | KuaiRand_Video_Action | long_view | 0.797290 | 0.810109 | +0.012819 | 0.459923 | 0.445069 | -0.014854 |
| Zenith | MerRec_Action | Cart | 0.806908 | 0.814763 | +0.007855 | 0.053756 | 0.053028 | -0.000728 |
| Zenith | MerRec_Action | Checkout | 0.852295 | 0.853138 | +0.000843 | 0.010226 | 0.010215 | -0.000011 |
| Zenith | MerRec_Action | Like | 0.744505 | 0.759388 | +0.014883 | 0.329855 | 0.323222 | -0.006633 |
| Zenith | MerRec_Action | Offer | 0.773288 | 0.766747 | -0.006541 | 0.022102 | 0.022682 | +0.000580 |
| Zenith | MerRec_Action | Purchase | 0.853223 | 0.852104 | -0.001119 | 0.006772 | 0.006880 | +0.000108 |
| Zenith | QK_Video_Action | click | 0.933477 | 0.934710 | +0.001233 | 0.252005 | 0.249782 | -0.002223 |
| Zenith | QK_Video_Action | follow | 0.922279 | 0.923107 | +0.000828 | 0.005022 | 0.004978 | -0.000044 |
| Zenith | QK_Video_Action | like | 0.947958 | 0.948299 | +0.000341 | 0.037439 | 0.037061 | -0.000378 |
| Zenith | QK_Video_Action | share | 0.935961 | 0.936504 | +0.000543 | 0.005276 | 0.005265 | -0.000011 |
| Zenith | Taobao_Action | buy | 0.792465 | 0.798329 | +0.005864 | 0.006853 | 0.006768 | -0.000085 |
| Zenith | Taobao_Action | cart | 0.782480 | 0.785999 | +0.003519 | 0.019201 | 0.019057 | -0.000144 |
| Zenith | Taobao_Action | fav | 0.799847 | 0.805541 | +0.005694 | 0.014666 | 0.014559 | -0.000107 |
| Zenith | Taobao_Action | is_click | 0.622431 | 0.619763 | -0.002668 | 0.187359 | 0.187583 | +0.000224 |
