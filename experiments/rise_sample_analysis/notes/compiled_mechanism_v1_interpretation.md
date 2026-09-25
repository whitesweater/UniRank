# 已验证compiled selection的学术解读：站点、长度与padding

日期：2026-09-23。结果目录为 `results/compiled_mechanism_v1/`；来源是B的643911已验证子阶段及E的独立成功任务643949。B整项作业仍保留FAILED 11:0状态，不能写为作业全部完成。本文只解释actual compiled Q/K/Δ的selection重构结果，不使用643858的旧eager机制表、643918未通过的E控制重算或尚未验收的内部状态捕获。

## 1. 这次完成的实证链条

512条交互来自8个固定test parts的预先声明窗口，涉及455个用户；各variant有512×6=3,072条站点记录。collector重新读取保存的逐行compiled预测，验证5标签capture/plain以及冻结compiled参照完全相等，而非仅依赖JSON的passed字段。两份来源的diagnostic脚本、输入cache和checkpoint对应关系均核对一致。

三个sequence站点各有511个非空历史样本进入均值；唯一零历史样本的sequence均值保留NaN。query decoder的512条均进入selection统计，但只对有效key计算的Δ指标有511条。长度组为0/1–20/21–50/51–99/100，对应样本量1/37/57/67/350。用户可能跨长度组出现，组内用户数不能相加后当作总体用户数。

所有观测均通过TV上界和top-order保持证书的1e−6数值容差，记录的最大违例均为0。它们是理论与数值实现相容的证据，不说明每个上界都很紧，也不保证排序性能改善。

## 2. 第一层sequence attention发生显著的选择再分配

| 量 | B content-only | E full RISE |
|---|---:|---:|
| 每样本平均Ω，再在非空样本中等权平均 | 2.490413 | 10.435600 |
| 每样本平均TV，再在非空样本中等权平均 | 0.187710 | 0.638117 |
| 样本TV均值的中位数 | 0.194362 | 0.620036 |
| 样本TV均值的95分位 | 0.223148 | 0.826133 |
| 样本等权top-key改变比例 | 0.374518 | 0.849394 |
| centered Δ SD/native logit SD的样本等权均值 | 0.456907 | 1.378773 |

E的第一sequence站点不是一个统一可以视为很小的logit扰动：在这一固定样本上，其centered修正尺度与native scores相当，且分布再分配在中位样本上也很明显。这一描述来自实际Δ与native score的相对尺度，而不是仅看head scale参数。

需要同时保留表示边界：B/E分别训练，表中每一列的native scores和hidden representations都属于该列自己的模型。因此两列说明两个模型中的修正行为不同，不能把E−B解释为在同一个固定表示上打开AM/PA的净作用。

## 3. 作用并不是所有站点一律增强

E首层query decoder的TV均值为0.049932，B为0.088571；该处E的top-key改变比例0.411621，也低于B的0.483398。E后两层query decoder的TV则为0.132770、0.196231。

所以，合适的描述是“修正的作用强度随插入点和层次变化”，而不是“RISE在每一层都加强注意力重排”。将所有站点合成单个平均值会掩盖这个差异，也容易把不同token layout的query数不加区分地池化。

TV与argmax改变也不是同一量。E首层decoder可以在平均TV约0.05时产生约41%的top-key改变；应并列报告概率质量再分配和排序首位变化，避免把任意一个指标当作另一个的充分替代。

## 4. Padding相关再分配很强，但有效历史内部仍有明显改变

当前HyFormer保留native允许的padding keys，所以padding必须进入selection重构。E第一sequence站点的长度分组结果是：

| 有效长度 | 样本数 | 原生padding mass均值 | 加Δ后padding mass均值 | TV均值 | top-key改变比例 |
|---|---:|---:|---:|---:|---:|
| 1–20 | 37 | 0.741354 | 0.071991 | 0.850611 | 0.764588 |
| 21–50 | 57 | 0.506346 | 0.123706 | 0.712744 | 0.854688 |
| 51–99 | 67 | 0.275509 | 0.094921 | 0.624160 | 0.853631 |
| 100 | 350 | 0 | 0 | 0.606172 | 0.856686 |

短历史中，第一层修正伴随明显的padding probability mass移出。与此同时，350条有效长度100的样本没有任何padding key，TV仍为0.606172。因此，selection变化也大量发生在有效历史内部，不能将所有观测到的再分配仅归为对padding的处理。

上述结果没有证明padding再分配是AUC提升的原因。原生处理方式固定不变，本研究只是忠实测量当前实现的条件选择行为。若未来要识别padding处理的性能因果效应，需要另一个保持其他条件一致的实验，不能由本表直接推出。

## 5. 长度关联随站点改变

E第一sequence层的TV从短历史1–20的0.850611降到长度100的0.606172，短历史更多padding的结构参与其中。相对地，第三sequence层的TV从0.134271升到0.317281；第三query decoder从0.144978升到0.213178，top-key改变比例从0.439189升到0.831429。

所以本次机制样本不支持一个不加站点区分的“历史越长，RISE作用越强”规则。更合理的候选解释是：早期sequence层、后期sequence层和query decoding呈现不同的长度关联，需要结合正式大样本分组性能结果讨论。这里的长度关联仍包含样本组成差异，不能当作改变某一用户历史长度的干预效应。

## 6. 正负修正与理论证书的报告边界

E三个sequence站点在有效keys上的正修正比例分别为0.498107、0.498937、0.493684，表明这些site的实际Δ存在较均衡的正负两种取值。B第二sequence层的正修正比例为0.978802。这里说的是**logit修正的符号**；正Δ不保证该key的最终softmax probability上升，因为其他key同时发生变化。

E第一sequence层的mean TV上界为0.976894，实际TV均值0.638117；证书成立但上界较宽。该site的 `native_top_gap>Ω` 充分保持条件触发比例为0，而实际仍有约15%的top-key未改变。这也体现了充分条件的保守性：证书未触发不能反推出一定发生排序改变。

## 7. 可放入论文的英文段落草稿

We instrumented the compiled checkpoints on a fixed diagnostic sample of 512 test interactions from 455 users, drawn from a predeclared window of eight test parts. The instrumented and uninstrumented forward passes produced exactly identical probabilities for all five action labels. At each attention site, we reconstructed pre-dropout selection from the captured native queries, keys, and auxiliary score correction while retaining the backbone's padding semantics. These measurements describe conditional redistribution within each trained model; they are not comparisons against an independently trained native model.

The full model's strongest redistribution occurred in the first sequence-encoding layer, with a mean total variation distance of 0.638, a median sample-level mean of 0.620, and a sample-weighted top-key change rate of 84.9%. The first query-decoding layer showed a smaller mean total variation distance of 0.050, compared with 0.089 for the content-only model. The correction therefore exhibited substantial site dependence. For histories of length 1–20, the first sequence layer's mean probability mass on padding keys decreased from 0.741 to 0.072. A mean total variation distance of 0.606 remained among the 350 examples with 100 valid history keys, where padding mass was identically zero, showing that redistribution also occurred entirely within valid histories. These observations do not establish a causal explanation of ranking gains or population-level generality.

这两段只适用于当前single-seed HyFormer–MerRec诊断，不应泛化为全部4×4场景共同机制，也不包含尚未验证的AM/PA内部状态论断。

## 8. 本次collector与独立复核

新增 `scripts/summarize_compiled_mechanism.py` 与 `tests/test_compiled_mechanism_summary.py`；6项CPU tests通过。测试覆盖错误row/site、旧eager路线或额外控制统计混入、理论证书/有效query计数异常、JSON宣称通过但实际预测差1 ULP、长度边界，以及保留B失败job来源的政策。

实际collector完成后，另行直接读取两份原始selection parquet，独立复算12个variant/site组合的Ω、TV、top-key改变及两类padding mass，共60组mean和P95，与CSV在 `atol=rtol=1e−14` 下全部一致。另确认两variant的长度100首层sequence各350行、padding质量全零；输出completion receipt列出的全部文件SHA256通过。

结果目录包含站点摘要、逐site分位数、按长度×site的摘要/分位数、长度组样本/用户/正例统计、配对描述差值、完整样本身份和source/scheduler审计。该目录没有在完成后手工更改数值；本解读备忘另存于notes，避免破坏已冻结artifact哈希。
