# 诊断归档说明

- 主问题：QK-Video 的 HiFormer 是否仍值得优化，以及优先验证什么。
- 受众：模型实验研究者；采用 build-report 的 technical 结构，按用户既有偏好交付无超链接的独立中文 Markdown。
- 结构映射：标题与结论；指标定义；逐标签结果；接入机制假设；固定预算候选；收敛诊断；证据核对与未决问题。方法和边界紧邻相关结论。
- 可复现命令：在仓库根目录执行 `.venv/bin/python experiments/hiformer_qk_diagnostic/archive/scripts/audit_results.py`。
- 控制来源：`checkpoints/QK_Video_Action/*HiFormer*.log` 的最终测试段；先与 `experiments/sisa_paper_report/archive/results/label_results.csv` 及 `source_manifest.csv` 核对，再计算标签宏平均和差值。
- 主配对：14811_8 / 14811_9；辅助运行：548166_10 / 549595_10。样本覆盖来自已归档的 `coverage_audit.csv`，本次未重新遍历数据集。
- 当前实现来源：`model_zoo/HiFormer.py`、`unirank/pytorch/layers/attentions/sisa.py`、`unirank/pytorch/layers/attentions/target_attention.py`、`run_expid.py`、`unirank/pytorch/models/rank_model.py`。
- 模型文件只在 CPU 上以 mmap 方式读取小型参数；旧的主配对 checkpoint 已删除。20262028/29 的 lambda 仅供解释性补充，不能替代主配对参数证据。
- visualize-data 表格设计：逐标签表用于精确核对 AUC 与差值，方案表用于对照配置和实验问题；均为无颜色 Markdown 表，正文紧邻解释符号和比较口径。
- 未画趋势图：每次运行只有一个验证点，不足以支持收敛或过拟合曲线。未画统计误差条：不存在多个同协议配对重复。未附加图片：独立可复制 Markdown 和精确数值查阅优先。
- 补充阅读：核查了 Hiformer 原论文 arXiv:2311.05884 的摘要。报告中接入机制判断以本仓库代码为依据，不把论文中的其他数据集或线上收益作为本次结果证据。
- 本次输出为诊断报告及归档证据；没有执行候选训练或修改模型、配置、启动器和既有报告。
