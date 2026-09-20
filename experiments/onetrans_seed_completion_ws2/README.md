# OneTrans SISA 两GPU排队备选

2026-09-20 归档状态：八对任务的选择已经结束。QK/MerRec 保留四卡，
KuaiRand/TAAC-25 保留两卡；MerRec seed20262029 后续四卡重提已在主研究报告中记录。
最终诊断入口为 `../onetrans_seed_completion/report.md`，选定结果为该研究的
`results/selected_` 文件。本目录保留两卡配置及分支选择证据。

`notes/race_completion.json` 是已结束监测器的最终状态原样快照，来自
`artifacts/onetrans_seed_completion_ws2/race_state.json`；包含八对任务的选择时间和取消回执。
它记录分支选择完成，不等于所有训练均成功；训练完成及重提核验以主研究选定结果为准。

按用户2026-09-18新增指令，与已有四GPU版本同时提交两GPU版本，覆盖QK-Video、
KuaiRand、TAAC-25、MerRec，每个数据集seed20262028/20262029，共八个任务。
任务编号沿用0、1、2、3、4、5、8、9，Taobao的6、7不提交，baseline不提交。

每任务2×H100、16CPU、480GiB主机内存，最多四个任务并行。每卡batch16384，
全局batch32768，梯度累积1。数据路径、特征、1epoch预算、最大历史长度100、
完整SISA设置、优化器及实验seed/loader子seed/SISA参数子seed沿用四GPU配置。
TAAC保持已恢复的历史group_id=user_id，不自动添加MerRec优化器特殊覆盖。

实际参数从上一研究的冻结runtime配置读取并逐项核对；两GPU分支不修改四GPU
启动器或配置。训练命令只改变world size、每卡batch和独立run ID。
原始输出的run ID包含ws2_bs16384，四GPU包含ws4_bs8192，各自保存manifest和
outcome，禁止覆盖。新Slurm日志使用unirank-onetrans-ws2前缀。

global batch一致不代表逐rank随机流或测试尾批覆盖一致。两卡覆盖重新记录于
configs/data_coverage.csv，比较表为configs/coverage_comparison.csv；后续只可将
实际成功且协议已核验的结果纳入该协议分析，不把两种硬件布局伪装成同一重复运行。
原四GPU采集器拒收两GPU记录；应根据新的manifest及对应两GPU覆盖另行核验。

用户已明确选择“先启动的保留，取消另一版本”。独立监测程序按同一task ID、
同一数据集及seed配对，仅可操作登记的两个数组和八个保留编号。依据实际开始
时间选择，不使用PENDING预计时间或测试指标；同秒开始时优先保留原四GPU版本。
每30秒查询一次，取消配对的另一版本。monitor不改变原四GPU启动器。
选择及取消回执保存在artifacts/onetrans_seed_completion_ws2/race_state.json，
监测完成全部八个选择后退出，最长监测七天。若先启动的保留任务失败，仍应另行
处理其失败/重试；本程序不按测试表现切换版本，也不自动补新训练。
两种协议的结果仍分别记录；实际采用哪种协议必须随每个seed披露。

本独立研究不自动改变现行四模型×四数据集论文证据范围。原四GPU任务、主线程
文件和用户报告的编辑均保持不变。提交回执保存在notes/submission.json。
