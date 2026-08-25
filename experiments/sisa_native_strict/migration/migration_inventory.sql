WITH migration_inventory(category, file_count, total_bytes, notes) AS (
  VALUES
    ('最终 strict Slurm 日志', 32, 252589669, '最终 job/task 映射；排除 pilot 和被替代的 14810_10'),
    ('Strict checkpoint 日志', 32, 347871, '训练、测试指标及 checkpoint 删除记录'),
    ('机器可读结果', 3, 37118, 'runs.csv、metrics.csv、summary.md'),
    ('完整 strict 实验报告', 1, 16987, '协议、结果、重试、限制与审计')
)
SELECT * FROM migration_inventory;
