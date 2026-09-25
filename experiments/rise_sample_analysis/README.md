# RISE论文补充研究交付

首先阅读final_research_report.md：独立完整中文报告，含14项新增训练及所有分析记录。paper/RISE_supplementary_analyses.pdf为英文论文补充材料，paper/supplement.tex为组合入口，另含三个可拆分LaTeX片段。figures保存10套PDF、SVG、PNG科研图。results为compact数据与审计回执，scripts/tests为复现代码，notes为方法与审阅记录。

初始research_report.md及archive文件保留历史状态，不作为当前待办。大体量数据、权重、预测和完整运行日志仍保留在仓库的artifacts、checkpoints、logs，未装入交付包。具体边界与验证见notes/final_delivery_validation.md。

编译英文PDF：在本目录执行pdflatex -interaction=nonstopmode -halt-on-error paper/supplement.tex两次。绘图环境使用figure_env下的uv锁文件，交付包不包含虚拟环境。
