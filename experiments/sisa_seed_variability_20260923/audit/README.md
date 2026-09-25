本次交付采用用户既有的独立中文 Markdown 报告偏好。

复算：在仓库根目录运行 `.venv/bin/python experiments/sisa_seed_variability_20260923/audit/analyze.py`。同内容 notebook 可供逐项检查。

指标、分母、筛选阈值、异常实例、未入选对照、限制与后续检查均已写入 report.md；使用精确数值表，因为任务是查找具体种子对及差距。没有添加装饰图，也没有改动既有 HTML 报告、训练代码或通用采集器。

verification.json 保留来源文件散列及验证数量。results/run_log_audit.csv 保留双日志散列、运行身份及匹配结果。只读检查已有状态证据，未重新提交任务。
