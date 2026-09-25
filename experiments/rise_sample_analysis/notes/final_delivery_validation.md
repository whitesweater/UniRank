# 最终交付验收

日期：2026-09-23。主报告 final_research_report.md 为当前完整结论入口，早期 research_report.md 和 validation.md 已明确为历史阶段文档。

- 14项新增训练全部eligible并且COMPLETED/0:0：5项第一组、3项定向控制、6项跨模型位置复核。六项追加结果已独立复解析，与正式collector的24标签行、48个指标和20行比较一致。
- 主报告1065行、约7.4万字符。生成时23个源SHA、报告SHA、Markdown表格列数、代码围栏及无网页超链接检查通过；修正一处bootstrap表格分隔行缺列，不改变数值。
- 正式MerRec、QK、TAAC配对行数分别11599872、48529408、153157632。原生A与Content-only B、历史2027主表与新2028匹配研究在报告中分别定义。
- 核心CPU回归152项通过，日志 artifacts/rise_sample_analysis/validation/final_core_tests_20260923.log。这是执行时工作区版本的回执，不覆盖其他agent后续独立改动。
- 本研究23个测试模块、144项测试分别在独立进程通过，见 results/isolated_test_receipt.json。保留合并进程的SIGSEGV失败，不将模块隔离结果冒称合并进程通过。
- 根目录与本研究Slurm脚本bash -n通过。六项追加collector结果另由独立审阅检查，无需新训练。
- 科研图10套PDF/PNG/SVG。已有组件4图、新原生2图、扩展4图分别验收；原生两图由root逐张PNG检查，确认坐标、样本占比、零线与区间无遮挡。扩展图由独立agent分别检查PDF渲染和PNG。
- 英文补充材料由3个研究片段组合为独立13页PDF；最终两遍pdflatex成功，无undefined references、Overfull、Underfull或warning。root检查组合PDF第1、8、13页布局，图表自身另有完整视觉记录。材料可供入稿编辑，未改写用户提供的document.pdf。
- 真实AM/PA内部状态未通过前向/分数一致性，仍是明确限制；已接纳的是输出一致的compiled selection观测。固定seed与单次离线数据的结论不宣称训练稳定性、普遍最优或无时点泄漏。
- 未修改核心训练代码、根环境依赖或另一项score_dim消融。交付时其他agent在tests/test_sisa_score_dim.py及其archive产生的工作未被覆盖或归入本研究。

最终图表与英文材料入口：paper/RISE_supplementary_analyses.pdf；LaTeX主入口paper/supplement.tex，从研究目录执行pdflatex。完整中文主报告包含逐动作/逐组表、原始16格主表、14项训练台账、失败尝试、统计定义与复现命令。
