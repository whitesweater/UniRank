# 启动修复及新种子实验

用户授权：修正启动失败项后重提；对完全一致的两项各换一个新种子。

| 模型 | 数据集 | 种子 | 用途 | 作业 |
|---|---|---|---|---|
| HyFormer | MerRec | 20262030 | 新实验种子；原 seed28 所有标签完全复现 | 645444_1 |
| HiFormer | MerRec | 20262030 | 新实验种子；原 seed29 所有标签完全复现 | 645444_2 |
| OneTrans | TencentGR / TAAC-25 | 20262029 | 修复 git 缺失，attempt3 | 645445_5 |

新种子的 dataloader_seed=2029，sisa_parameter_seed=20260824。三项均为 2×H100、每卡 batch=16384、global batch=32768、1 epoch、max_len=100、完整 SISA；其余配置保持相应原实验设置。

已核验 HyFormer–MerRec 和 HiFormer–MerRec 的所有标签 AUC/LogLoss 在同种子复跑中完全一致。HyFormer–KuaiRand 已完成，宏平均 AUC 从 0.8616241667 到 0.8624451667，提高 0.0008210，不在这次新种子任务内。

计算节点未提供 git，645228_5 在 manifest 构造阶段失败，未开始训练。共享运行目录提供管理节点真实 Git 2.43.0 二进制、动态加载器及依赖；各文件散列见 git_environment_fix.json。已有用户 PATH 上的回退入口优先使用节点原生 Git，缺失时才使用共享副本；新提交明确优先使用共享副本。当前提交号验证为 22515f5fd4fa35094e565cb23f6988d2b5cff87b。

原失败日志及空输出目录保留。原批次的 OneTrans–TAAC-25 逻辑任务现指向 645445_5；旧任务/提交清单保存在 prior_tasks.json、prior_submission.json。两项新种子作为新增观测保留，不覆盖原种子结果。三个原有排队任务未重新提交。

原批次仍由原后台进程追踪；本目录三项另行追踪。report.md、results/status.json 是自动核验输出。训练源码和通用启动器未修改。
