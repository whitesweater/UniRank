# QK-Video接入位置跨模型复核：独立结果审阅

审阅日期：2026-09-23。**本次结论是“局部复现并有明确反例”，不是把HiFormer结论完整推广到所有模型。** Zenith复现了“历史汇聚位置优于原生、token-only低于原生”的方向；但Zenith完整RISE仍优于原生，与HiFormer完整配置负向不同。HyFormer两类接入点均属于序列处理，完整双阶段配置宏AUC最好，不能将其query decoder重命名为字段attention来支持“字段位置普遍有害”。

本审阅只读日志、调度状态、冻结配置、构造审计和已有证据；只新增本审阅文档及独立审计副本，没有修改训练、core、score_dim或模型代码，也没有提交任务或启动推理。所有比较严格限定QK-Video、seed20262028；没有使用seed20262027结果补充或替代数值。

## 1. 实际复核范围和完成证据

- 644632_0至644632_5六项Slurm均为COMPLETED/0:0；`started.json`与`outcome.json`对应且returncode=0，均记录两张H100。
- 六份checkpoint log都只有一次最终Test段、包含click/follow/like/share各一组AUC和Logloss；逐项与对应Slurm日志最终Test一致。只读最终Test，不混入validation。
- 六项日志Params逐键等于各自frozen_manifest中的effective_params。相对同骨干历史完整RISE，原生仅改变model_id和sisa_enabled；单位置日志仅改变model_id，处理条件由进程内位置patch与模型构造审计明确记录，并不是从run名称推测配置。
- 两个rank的模型构造审计一致；分别复核实际启用位置、reference、每位置seed offset、输入宽度、head数、r=16和稠密参数数量。六套新训练的冻结源码map完全一致，所列60个源码/环境文件当前SHA仍匹配。
- 两个新骨干的历史完整RISE参照分别为548166_8(Zenith)、548166_9(HyFormer)；HiFormer同种子参照为548166_10。其checkpoint最终Test、Slurm最终Test和rank覆盖已重新核对，参照log SHA与冻结manifest一致。历史完成状态沿用已归档的COMPLETED/0:0证据；本次不冒称重新证明了历史训练源码逐字节身份。
- 本文引用的12份检查点文件均存在，ZIP目录及data.pkl条目可读；这是结构/存在检查，不是所有tensor内容或重新推理的验证。

共同执行协议是每rank batch16,384、2 ranks、global batch32,768、1 epoch、max_len100、loader seed2027、辅助初始化seed20260822、BF16及torch.compile。覆盖从实际Slurm输出复核如下：

| split | rank0步数 | rank1步数 | 实际保留行数 |
|---|---:|---:|---:|
| Train | 12,038 | 12,036 | 394,428,416 |
| Valid | 1,527 | 1,527 | 50,036,736 |
| Test | 1,481 | 1,481 | 48,529,408 |

这些是block-level drop_last后的数量，不是完整footer行数。新六项与三个同种子完整参照的分块/行数/步数记录相同；数量相同并不自动证明历史数据内容逐字节相同。所有AUC以最终Test日志中的六位标签值为输入，宏值是四标签等权平均，不冒称本轮做了逐样本重新计算。

## 2. 核心比较：原生始终在最左比较列

| 骨干 | 原生Native AUC | 完整RISE AUC（相对原生） | 单位置Ⅰ AUC（相对原生） | 单位置Ⅱ AUC（相对原生） |
|---|---:|---:|---:|---:|
| HiFormer | 0.93343800 | 0.93089125（-0.00254675） | 0.93441475（+0.00097675） | 0.92939075（-0.00404725） |
| Zenith | 0.93463475 | 0.93522050（+0.00058575） | 0.93578575（+0.00115100） | 0.93380000（-0.00083475） |
| HyFormer | 0.93085800 | 0.93130800（+0.00045000） | 0.93093600（+0.00007800） | 0.93122125（+0.00036325） |

“单位置Ⅰ/Ⅱ”的语义按模型固定：HiFormer为历史target pooling/feature-task层；Zenith为历史target pooling/聚合token层；HyFormer为历史self-attention/query-to-sequence decoder。前两模型提供相近的“历史汇聚对聚合token交互”对照，HyFormer检验的是两个序列阶段，不能强行赋予相同语义。

### Zenith：全部配置、宏指标与容量

| 配置 | 宏AUC | ΔAUC vs原生 | ΔAUC vs完整 | 宏Logloss | ΔLogloss vs原生 | 可训练稠密参数 |
|---|---:|---:|---:|---:|---:|---:|
| 原生Native | 0.93463475 | +0.00000000 | -0.00058575 | 0.07517975 | +0.00000000 | 11,164,164 |
| 完整RISE | 0.93522050 | +0.00058575 | +0.00000000 | 0.07438775 | -0.00079200 | 11,230,434 |
| 仅历史target pooling | 0.93578575 | +0.00115100 | +0.00056525 | 0.07440950 | -0.00077025 | 11,167,446 |
| 仅聚合token层 | 0.93380000 | -0.00083475 | -0.00142050 | 0.07539900 | +0.00021925 | 11,227,152 |

每个AUC单元格括号中为相对原生的差值；原生AUC位于最左比较列。

| 动作 | 原生AUC | 完整RISE AUC（Δ原生） | 仅历史target pooling AUC（Δ原生） | 仅聚合token层 AUC（Δ原生） |
|---|---:|---:|---:|---:|
| click | 0.932670 | 0.934288（+0.001618） | 0.934345（+0.001675） | 0.932227（-0.000443） |
| follow | 0.923350 | 0.922992（-0.000358） | 0.924854（+0.001504） | 0.921671（-0.001679） |
| like | 0.947279 | 0.948310（+0.001031） | 0.947871（+0.000592） | 0.946704（-0.000575） |
| share | 0.935240 | 0.935292（+0.000052） | 0.936073（+0.000833） | 0.934598（-0.000642） |

### HyFormer：全部配置、宏指标与容量

| 配置 | 宏AUC | ΔAUC vs原生 | ΔAUC vs完整 | 宏Logloss | ΔLogloss vs原生 | 可训练稠密参数 |
|---|---:|---:|---:|---:|---:|---:|
| 原生Native | 0.93085800 | +0.00000000 | -0.00045000 | 0.07569825 | +0.00000000 | 7,505,412 |
| 完整RISE | 0.93130800 | +0.00045000 | +0.00000000 | 0.07470750 | -0.00099075 | 7,631,388 |
| 仅历史self-attention | 0.93093600 | +0.00007800 | -0.00037200 | 0.07526200 | -0.00043625 | 7,568,400 |
| 仅query-to-sequence decoder | 0.93122125 | +0.00036325 | -0.00008675 | 0.07493175 | -0.00076650 | 7,568,400 |

每个AUC单元格括号中为相对原生的差值；原生AUC位于最左比较列。

| 动作 | 原生AUC | 完整RISE AUC（Δ原生） | 仅历史self-attention AUC（Δ原生） | 仅query-to-sequence decoder AUC（Δ原生） |
|---|---:|---:|---:|---:|
| click | 0.932155 | 0.934075（+0.001920） | 0.933257（+0.001102） | 0.933643（+0.001488） |
| follow | 0.913908 | 0.913611（-0.000297） | 0.913477（-0.000431） | 0.913450（-0.000458） |
| like | 0.946032 | 0.947015（+0.000983） | 0.946277（+0.000245） | 0.947079（+0.001047） |
| share | 0.931337 | 0.930531（-0.000806） | 0.930733（-0.000604） | 0.930713（-0.000624） |

### HiFormer：全部配置、宏指标与容量

| 配置 | 宏AUC | ΔAUC vs原生 | ΔAUC vs完整 | 宏Logloss | ΔLogloss vs原生 | 可训练稠密参数 |
|---|---:|---:|---:|---:|---:|---:|
| 原生Native | 0.93343800 | +0.00000000 | +0.00254675 | 0.07563200 | +0.00000000 | 27,824,132 |
| 完整RISE | 0.93089125 | -0.00254675 | +0.00000000 | 0.07584525 | +0.00021325 | 27,889,090 |
| 仅历史target pooling | 0.93441475 | +0.00097675 | +0.00352350 | 0.07478300 | -0.00084900 | 27,826,102 |
| 仅feature/task层 | 0.92939075 | -0.00404725 | -0.00150050 | 0.07632900 | +0.00069700 | 27,887,120 |

每个AUC单元格括号中为相对原生的差值；原生AUC位于最左比较列。

| 动作 | 原生AUC | 完整RISE AUC（Δ原生） | 仅历史target pooling AUC（Δ原生） | 仅feature/task层 AUC（Δ原生） |
|---|---:|---:|---:|---:|
| click | 0.931452 | 0.931291（-0.000161） | 0.933251（+0.001799） | 0.930387（-0.001065） |
| follow | 0.921287 | 0.913586（-0.007701） | 0.921751（+0.000464） | 0.912760（-0.008527） |
| like | 0.946112 | 0.945454（-0.000658） | 0.946900（+0.000788） | 0.943511（-0.002601） |
| share | 0.934901 | 0.933234（-0.001667） | 0.935757（+0.000856） | 0.930905（-0.003996） |

## 3. Zenith：复现位置方向，但不复现完整配置整体负向

Zenith history_only比原生提高0.00115100，token_only下降0.00083475，方向与HiFormer target_only/field_only一致。更具体地，Zenith history_only四个动作都提高，token_only四个动作都下降；它不是只有某一个动作拉动的宏平均现象。作为第二个骨干上的同种子位置复核，这支持把HiFormer观察收窄为“某些骨干中的接入位置具有系统性差异”。

反例同样关键：Zenith完整RISE比原生提高0.00058575，而HiFormer完整RISE下降0.00254675。所以不能把“在QK上完整RISE整体变差”推广到Zenith。Zenith完整配置的follow仍下降0.000358，其他三个动作提高；完整宏正向并非每个反馈任务都正向。

history_only相对完整配置的AUC增量为0.00056525。逐动作是click +0.000057、follow +0.001862、like −0.000439、share +0.000781，表明它牺牲了一部分like表现换取其他动作的平均改善。并且完整配置的宏Logloss为0.07438775，略好于history_only的0.07440950，差0.00002175。故不能将history_only写成所有指标、所有任务上的严格支配者；“AUC最佳”必须保留指标限定。

Zenith的聚合token包含用户、物品、上下文和历史汇聚信息。因此token_only并不意味着完全不使用历史；它改变的是辅助评分接入聚合token交互的位置，不能简化成“有历史信息”和“无历史信息”的比较。

## 4. HyFormer：两阶段协作的点估计最优，不是字段位置有害的复现

HyFormer完整RISE、sequence_only、query_only相对原生分别提高0.00045000、0.00007800和0.00036325，三个配置均为宏AUC正向。完整配置同时拥有最好的宏AUC0.93130800和宏Logloss0.07470750。这与把所有非单一历史位置都视为有害的强解释不相容。

两种单位置变体各保留三个接入点，可训练稠密参数都为7,568,400。因此query_only相对sequence_only的0.00028525差值不受这两者稠密参数总数不同的解释影响；但它们的计算角色、输入、query/key集合和decay reference不同，仍不能称为只隔离了某一个抽象机制。query_only的优势主要来自click和like；follow和share反而略低于sequence_only。

完整RISE比query_only的宏AUC只高0.00008675，是很小的单种子差值。逐动作上，完整配置click/follow更好，但like比query_only低0.000064、share低0.000182。更重要的是，无论完整、sequence_only还是query_only，follow与share都低于原生；这些负面任务方向没有因拆分位置而消失。不能把“完整宏值最高”写成每个动作都需要两个阶段，也不能将这一小宏差宣称为跨seed显著优势。

HyFormer的query decoder仍然读取历史序列，而不是HiFormer/Zenith式的feature/token mixing层。本次结果提供“history encoding和query decoding具有不同作用”的架构内证据，不是对字段attention假说的同义复制。

## 5. 参数量、参考方式和实验单位的解释边界

| 模型 | 原生稠密参数 | 完整RISE新增量 | 单位置Ⅰ新增量 | 单位置Ⅱ新增量 | 解释 |
|---|---:|---:|---:|---:|---|
| HiFormer | 27,824,132 | 64,958 | 1,970 | 62,988 | target/field数量与宽度不同，非等参数 |
| Zenith | 11,164,164 | 66,270 | 3,282 | 62,988 | history/token数量与宽度不同，非等参数 |
| HyFormer | 7,505,412 | 125,976 | 62,988 | 62,988 | 两单位置同稠密参数，完整配置仍更多 |

同一骨干内embedding参数保持一致：HiFormer和HyFormer均140,001,120，Zenith为419,988,768；跨骨干架构和embedding容量并不相同。本次不把不同模型当作同一个等容量处理实验。

在HiFormer/Zenith中，target pooling使用query reference，feature/token层使用sequence-end reference；HyFormer历史self-attention使用sequence-end，query decoder使用query reference。位置比较同时包含原生token布局、投影角色、mask和reference等差异；不能从本表单独推导“query reference胜过sequence-end”或“某种decay公式导致下降”。

三个骨干都只在当前QK数据与seed20262028下比较。四个动作共享样本与训练，不是四个独立seed；三个架构也不构成同一种处理效应的三次独立同分布复制。本审阅没有逐样本预测对照或用户bootstrap，因此不生成置信区间和显著性星号。大量测试行不能替代对训练随机性的复验。

这些位置研究是看到HiFormer现象之后开展的探索性跨模型复核；不能事后选择测试最优位置替换原主表或将它称为原研究预注册验证。现有冻结配置和覆盖匹配支持协议层面对照，但historical_training_source_revision_verified在新manifest中明确为false：没有证据证明历史完整E与新训练使用逐字节相同软件环境。

## 6. 可用于论文的收窄结论

> 在固定QK-Video、seed20262028的探索性位置复核中，HiFormer和Zenith都表现出历史target pooling单独接入优于原生、feature/token交互层单独接入低于原生的方向。然而，Zenith的完整RISE仍提高宏AUC，且完整配置的Logloss略优于history-only，说明HiFormer完整配置的负面结果不能直接推广。HyFormer的两个接入点均属于序列处理，完整双阶段配置取得最高宏AUC，而query-only优于同稠密参数量的sequence-only；与此同时，follow和share在所有RISE位置配置下仍低于原生。结果支持架构与接入位置相关的选择性效果，而不是“非历史位置普遍有害”或“单位置始终更好”的通则。

该段只描述已观察到的单种子结果，不使用因果、在线效益或跨seed显著性措辞。HyFormer的小增量及Zenith的指标取舍均应保留为反例。

## 7. 独立来源与复核档案

下表来自本次独立读取的最终Test日志；source SHA用于锁定本次审阅输入。训练主体没有任何修改。完整配置/coverage/构造审计核查副本保存为 `artifacts/rise_sample_analysis/cross_model_independent_review/review.json`。

| 模型 | 配置 | 最终Test日志 | SHA-256 |
|---|---|---|---|
| Zenith | baseline | `checkpoints/QK_Video_Action/RISE_cross_sites_Zenith_QK_Video_Action_baseline_seed20262028_attempt1.log` | `8823722454a00d82daa43ed69cc1b531f7f4e0da642d5761897d5f255e120745` |
| Zenith | full | `checkpoints/QK_Video_Action/SISA_single_seed_Zenith_QK_Video_Action_sisa_seed20262028_ws2_bs16384_acc1_attempt1.log` | `52b882a8215a3190d0c7c4b74a9b8f7b13c86c45106300f1a372a635d0826366` |
| Zenith | history_only | `checkpoints/QK_Video_Action/RISE_cross_sites_Zenith_QK_Video_Action_history_only_seed20262028_attempt1.log` | `965b17bfd31f4603358241efd9346846e2cb74c8d2f947776578e119db02fa6c` |
| Zenith | token_only | `checkpoints/QK_Video_Action/RISE_cross_sites_Zenith_QK_Video_Action_token_only_seed20262028_attempt1.log` | `5c5f225c02e54a8dc70b952091960f9985dba0eba9cb7cd75fe33ca2681e30c4` |
| HyFormer | baseline | `checkpoints/QK_Video_Action/RISE_cross_sites_HyFormer_QK_Video_Action_baseline_seed20262028_attempt1.log` | `f886b3f9ff33fb3c0e4d84aefaed520ef224a969445646b36287347a1f43c168` |
| HyFormer | full | `checkpoints/QK_Video_Action/SISA_single_seed_HyFormer_QK_Video_Action_sisa_seed20262028_ws2_bs16384_acc1_attempt1.log` | `042df0b5cc2adb6832de580ccb0895a4e253b06a9ee78662dc9d378db578f1ae` |
| HyFormer | sequence_only | `checkpoints/QK_Video_Action/RISE_cross_sites_HyFormer_QK_Video_Action_sequence_only_seed20262028_attempt1.log` | `cd658d41b869df6dba3ac5f52a69d707e72578eb31b12fde8f072a06e3847524` |
| HyFormer | query_only | `checkpoints/QK_Video_Action/RISE_cross_sites_HyFormer_QK_Video_Action_query_only_seed20262028_attempt1.log` | `423971017ef2976fcb9b7be0e20f7c5bd23bb5ad0a2a69b505cde0e9ce3fc371` |
| HiFormer | baseline | `artifacts/rise_sample_analysis/training_checkpoints/QK_Video_Action/RISE_matched_HiFormer_QK_Video_Action_baseline_seed20262028_attempt1.log` | `93fc042ae5f65bccb98dd4634f72ed1c24293e670aee01799dc39230242a6e3f` |
| HiFormer | full | `checkpoints/QK_Video_Action/SISA_single_seed_HiFormer_QK_Video_Action_sisa_seed20262028_ws2_bs16384_acc1_attempt1.log` | `4b682120db61e1719ae5bd40f563c744606a7a95be95aa63a25473202c0a5195` |
| HiFormer | target_only | `checkpoints/QK_Video_Action/RISE_targeted_HiFormer_QK_Video_Action_target_only_seed20262028_attempt1.log` | `c2a75a133a2215fa6c5c400da04b21237546b9776b7f23917154125347aa4803` |
| HiFormer | field_only | `checkpoints/QK_Video_Action/RISE_targeted_HiFormer_QK_Video_Action_field_only_seed20262028_attempt1.log` | `59d5e852f1e3f85db2464c28397974be12eecbb90e62e83306dfc4041333b51d` |

六个新任务的artifact根目录为 `artifacts/rise_site_replication/`，每项含frozen_manifest、started/outcome和双rank model audit。历史完整参照的job/硬件/完成状态来源为 `experiments/sisa_paper_report/archive/results/run_evidence.csv` 中明确的20262028三行；并重新核对了相应Slurm最终Test和覆盖。HiFormer既有位置实验的完成审计还见 `results/targeted_training_runs.csv` 与 `targeted_training_comparisons.csv`，本文的HiFormer指标则独立从原始checkpoint log重解析，未从主表抄写。

本审阅未量化同一个模型内的注意力机制变化，也未把标签指标当作其证据；配置位置、参数数目和Test表现只能支持上述限定的架构相关结论。

## 8. 与随后完成的正式collector交叉校验

独立原始日志重解析完成后，又对 `results/site_replication_runs.csv`、`site_replication_label_metrics.csv` 和 `site_replication_comparisons.csv` 进行逐项对照：6项均eligible/selected、COMPLETED/0:0且无issues；宏AUC/Logloss、稠密/embedding参数和任务号一致；24个新任务标签行的48个最终标量指标全部一致；单位置对原生/完整参照的逐标签与宏平均比较亦一致。该校验结果及三份正式表SHA追加在独立审计JSON中，没有改写正式collector结果。
