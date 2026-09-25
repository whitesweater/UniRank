# 扩展科研图与英文片段验收

2026-09-23。仅新增 `scripts/plot_extended_results.py`、`paper/extended_analyses.tex`、本记录及 `figures/extended_*` 新文件。未修改原稿、root dossier builder、训练、score_dim或既有绘图脚本。

## 图契约与来源

| 文件前缀 | 分析问题与图型 | 粒度/数量 | 数值与区间口径 |
|---|---|---|---|
| extended_hiformer_placement | HiFormer–QK插入位置；分类点图 | Native/full/target/field四点 | 单seed20262028宏AUC；无误差线；明确zoomed轴 |
| extended_hiformer_history_forest | QK与TAAC原生对照长度效应；双panel forest | 两数据集八个非空组；空组保留 | E−A宏AUC百分点；固定模型的名义95%用户簇区间 |
| extended_inference_cost | A/B/E实际成本；分组点与区间 | 三batch×三variant×两顺序=18点/指标 | 两轮全保留；延迟whisker为10个block均值的IQR，非CI；显存为peak减resident baseline |
| extended_compiled_site_tv | 不同插入点selection再分配；分组点与样本分位区间 | B/E×六站点=12点 | 点为等样本均值；whisker为样本10–90分位，非CI；不宣称AM/PA内部观测 |

所有图的度量值均直接读取已验收CSV，未手工填写点坐标。源表为targeted/new training comparisons、两个HiFormer groups_v1、extended_benchmark_v1和compiled_mechanism_v1。图与TeX另读取正式 `site_replication_comparisons.csv` 的四个selective-placement宏记录，形成英文跨模型表；没有虚构TAAC位置实验。原生A是该表的第一个数值列。

采用Matplotlib 3.10.6，运行于既有独立 `figure_env` 的uv锁定环境；没有更改训练环境或安装额外包。统一DejaVu Serif、蓝色Full RISE、金色Content/目标位置、灰色Native，并以marker形状/填充分辨配置或顺序。此为用户论文的科研图，未加入平台品牌标识。

## 已执行验证

```bash
uv run --project experiments/rise_sample_analysis/figure_env --frozen python experiments/rise_sample_analysis/scripts/plot_extended_results.py
```

已生成四图各PDF/PNG/SVG，以及 `figures/extended_figures_manifest.json`。manifest保存源CSV/完成回执SHA256、图粒度和不确定性定义、脚本与生成TeX哈希。绘图时验证HiFormer分组结果checksum/样本覆盖、成本18记录与六轮完成条件、compiled统计receipt checksum、点估计处于所给区间内等条件。

已逐张视觉检查全部四张PNG，并用 `pdftoppm` 将四张PDF各自渲染后逐张检查。标题、图例、样本数、轴标签、误差线和脚注均可读，无重叠/裁切；概率距离保持0–1范围，成本轴从零开始；forest共同横轴保留TAAC空历史宽区间和两个空组，未截断负端。极窄区间在全范围内显示为小caps，这是来源精度，不是遗漏。

TeX以临时article wrapper加载graphicx/booktabs/amsmath，并设置figure目录，执行两遍 `pdflatex -halt-on-error` 成功。第二遍无undefined references、Overfull/Underfull或错误；临时验证PDF为4页。临时文件位于 `/tmp/rise_extended_qa/`，未作为训练或证据产物混入仓库。

## 英文内容边界

`paper/extended_analyses.tex` 覆盖HiFormer位置选择、QK/TAAC长度组、RankMixer与HyFormer两种fixed-control结果的差异、compiled selection、原生A到RISE的实测成本及Flash→efficient路径变化，以及MerRec原始动作时点不可得的限制。所有展示的结果数字由脚本从CSV替换生成；保留单seed、固定预算、非capacity-matched placement、nominal区间和offline-data边界。未把Content-only称Baseline，未把时钟/顺序波动当作训练置信区间，未宣称AM/PA内部状态观测已通过。

复用片段时需要把本study的figures目录加入论文 `\graphicspath`。现有原稿未被覆盖；若后续更新输入CSV，应重跑本脚本并重新核对manifest和TeX。

最终追加并核验了跨模型表后的解释段：Zenith history-only增益、token-only下降、full仍高于native且Logloss优于history-only；HyFormer full比query-only仅高0.00008675，follow/share在full及两个选择位置均低于native，且其两个sequence阶段不等同field-token消融。该段数值/符号从正式CSV生成并作断言，生成器与TeX/manifest同步。追加后再次两遍pdflatex成功，无warning或overfull/underfull。
