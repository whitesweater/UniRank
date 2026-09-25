# MerRec 分组特征：数据语义审计与执行约定

审计日期：2026-09-23。本文对应独立脚本 `scripts/build_group_features.py` 与 CPU 测试 `tests/test_group_features.py`。本次只执行合成小数据测试、源码检查、真实 schema/少量头部读取及小词表对象提取，未扫描完整训练集、未生成正式全量分组旁表、未启动训练或 GPU 推理。下列全量命令交由主任务的 CPU 作业执行。

## 1. 输入、输出和身份

范围固定为当前 `datasets/MerRec_Action`。`user_index`、`item_index` 是 split 内分块局部索引；样本 `user_id` 及经同块 `item_info` 映射得到的 `item_id` 是全局特征 ID。不能把不同块的局部 item index 合并计数。输出键为 `part_id:int32` 与 `row_offset:int64`；行号来自原始物理 Parquet 块，筛选、分批及 drop_last 不重新编号。

频次阶段只读 `train/data` 的 `user_id,item_index` 和对应 `train/item_info` 的 `item_index,item_id`。统计的是完整预处理训练目标行数，不是优化器实际保留行数、原始平台所有事件数或 embedding 更新次数；不会读取 train `full_item_seq` 来定义频次。当前训练总量为 144,108,235 行。频次为零仅表示训练目标未见。

`count-train` 输出 `frequencies/user_train_target_count.npy`、`item_train_target_count.npy` 与 `manifest.json`。频次数组为 uint64，以全局 ID 直接索引。清单记录实际计数、完整 footer 总数、数据文件路径/大小/修改时间、元数据与程序散列、数组内容散列。路径/大小/修改时间不是完整数据内容散列；不宣称历史内容逐字节已经验证。若使用 `--limit-rows`，清单标为 incomplete；正式 build 默认拒绝它。

`build` 输出 `features/part-xxxxx.parquet`、逐块 manifest 及整体 manifest。可提供唯一的 row-ID Parquet，也可遍历整块。默认每块保留 `floor(rows/16384)*16384` 行，对应既有两卡消融测试的 block-level drop_last：总计 11,599,872 行。`--retained-batch-size 0` 改为不丢尾批，必须单列为新评估口径。提供的 row IDs 若超出选定覆盖则报错，不静默丢弃。重复 row IDs、缺失块、target-anchor 不对齐、未知局部 ID 或频次来源变化均报错。

## 2. 精确窗口、类别与动作列

设当前样本 `seq_len=s`，窗口上限 M=100。pre padding 对应历史切片 `[max(0,s-M),s)`，不含 target。脚本同时校验 `full_item_seq[s] == 当前 item_index`，再以历史 local item index>0 构造有效 mask。`seq_len` 是原始历史索引，`effective_length` 是实际 mask 和，不能把两者混称模型输入长度。

固定列如下：

| 列 | 类型与含义 |
|---|---|
| `part_id,row_offset` | int32/int64，稳定物理样本键 |
| `user_id,item_id,seq_len` | int64，全局 ID 与原始历史索引 |
| `effective_length,padding_fraction` | int16/float32，有效 mask 个数与 `1-L/100` |
| `user_train_target_count,item_train_target_count` | uint64，完整训练目标频次 |
| `c1_known_count,c1_unknown_count,c1_ambiguous_count` | int16，已知位置、被排除位置及其中语义有歧义位置数 |
| `c1_unique_count,c1_entropy` | int16/float32，已知类别种数及自然对数熵；无已知类别时熵为空 |
| `target_c1_id,target_c1_known` | int32/bool，候选的原编码及是否按本策略已知 |
| `target_c1_history_count` | int16，非零候选编码在有效历史中的原始匹配数，包括候选编码309的匹配；候选为0时设0，避免把缺失与缺失当语义一致 |
| `target_c1_history_fraction` | float32，已知候选匹配数/已知历史数；任一不可解释时为空 |
| `history_{Like,Cart,Offer,Checkout,Purchase}_count` | int16，按 meta_data action_vocab 解码的可见历史多标签支持数 |
| `history_action_unknown_count` | int16，有效物品位置中 action=0 的数量，不当成某个动作 |
| `history_action_temporal_status` | 字符串，非空历史为 `unverified_eventual_outcomes`，空历史为 `no_history` |

所有空熵、空比例写为 Arrow null，不填0。action 编码严格取 `meta_data.json` 的多热组合词表；例如 exposure=32 是合法历史动作编码，不能因为旧 generic 词表把32称为OOV就把它丢弃。输出中的语义状态用于告知研究者：历史 view 锚点在过去，不保证附在锚点上的动作结局在预测时已经发生。

## 3. 实际 c1 词表的歧义及保守处理

本轮核查了真实 `feature_vocab.json` 的 `c1_id` 对象：包含310个条目，`0`到`308`为 identity 映射，`__OOV__`为309；没有 `__MISSING__` 原始类别映射。该对象规范 JSON 的 SHA-256 为 `8077fe16eb7d3c596f34ea271d8a45923dc028fd8f7659e0e17f94e53d9c3763`。全文件约1.46 GB。初次误用整文件 JSON 解析的进程被系统终止，没有产生可用核查结果；随后改为 mmap 定位并解析约6 KB的小对象，独立构建脚本只采用这一有界解析路径，不应整文件 json.load。

这个 generic 词表**不是作者预处理的原始类别到编码词表**。`unirank/preprocess/feature_processor.py:252–258` 在 `rebuild_dataset=False` 时用 `range(vocab_size-1)` 建立 identity 字典，再保留最后一位作 generic OOV。与之相对，作者 MerRec 预处理 `data/MerRec/preprocess_MerRec_seq_action.py:425–439,836–837` 排除 `__MISSING__`，把保留原始值编码为1到K，OOV/缺失编码为0，vocab_size=K+1；按此语义309也可能是一个真实类别。当前预编码路径 `unirank/features.py:130–170` 直接使用整数 ID，不需要经过 generic 字典重映射。

因此，当前能确定0是未知，不能单凭旧 generic 词表断言309必为缺失。首轮采用已约定的保守策略：排除0及语义有歧义的309，并将309出现数单独保留；不声称已经还原原始类别语义。将来找到可信原始映射后可补敏感性结果。

由于这里只有一个歧义编码309，无需重扫历史即可把309视为已知并重算：令 n 为 `c1_known_count`，k 为 `c1_ambiguous_count`，H 为现有熵。若n+k>0，新的熵为 `log(n+k)-[n(log(n)-H)+k log(k)]/(n+k)`，约定零计数项为0；新已知类别数为原种数加 `1[k>0]`，新已知位置数为n+k。候选非零时用已保存的 `target_c1_history_count/(n+k)` 重算一致度。n=0、k>0时新增分布只有一种类别、熵为0；两者均0时仍为空。未知0始终排除。这是保守编码策略的敏感性，不是额外模型预测。

## 4. 归因可能包含未来信息：可以证明什么

源程序 `build_view_anchor_samples`（`data/MerRec/preprocess_MerRec_seq_action.py:607–654`）保留每个用户—物品的首个 view；把该锚点之后匹配到的所有支持动作按 OR 写入它的标签，再将多标签组合编码为该 view 的 history action。`process_partition:682–711` 在构造完这些结局标签之后建立完整 history；`:721–727` 此后才按锚点日期划分 train/valid/test。

可构造最小事件序列：同一用户在t=100浏览A，t=200浏览B，t=300点赞A。当前程序使A在t=100的 history action 带有Like。预测t=200的B时，A属于合法过去view窗口，但其Like直到t=300才发生。窗口不含未来view并不能阻止这种未来结局经过去view的action编码进入模型。CPU测试直接调用当前预处理函数验证该反例，没有修改训练数据或模型。

同理，若训练日期的view匹配了验证或测试日期的后续动作，先全轨迹归因再按view日期切分不会主动截断该跨边界归因。当前函数没有显式归因时长上限，也没有按“当前被预测锚点的时间”过滤历史锚点的结局。

**这证明该构造允许此类信息依赖，不证明当前每条真实样本均泄漏，也没有测得实际受影响比例。** processed `full_timestamp_seq` 只记录view锚点时刻，`full_action_seq` 已折叠后续事件，缺少各动作真实发生时间；仅靠现有三张 processed 表无法重建每个历史动作在当前预测时是否已发生。需要原始事件级数据及预处理运行证据才能量化、修复或证明不存在该问题。现有action支持度只能解释当前离线编码条件，不能作为严格point-in-time兴趣证据。现有模型间对照同受该协议，但这不证明协议适合在线部署或无泄漏。

此外，源码以 `_row_id > __anchor_row_id` 判定“之后”（`:628–630`），而 `_row_id` 在读取/清理时赋值（`:349`），之后按时间排序并不重写它。因此还隐含“输入读取顺序与用户事件时间顺序一致”的条件。第二个合成反例令早于view的Like具有更大的原始row_id，当前函数仍把它归因给view。此测试证明实现对原始顺序有依赖；没有审计真实原始文件是否满足顺序条件，不能直接断言当前全量归因已错误。

## 5. 可运行命令与资源边界

先在CPU作业执行一次训练频次统计；没有GPU请求：

```bash
.venv/bin/python experiments/rise_sample_analysis/scripts/build_group_features.py count-train
```

随后按默认既有消融覆盖生成八块特征：

```bash
.venv/bin/python experiments/rise_sample_analysis/scripts/build_group_features.py build
```

如需与一份正式预测清单逐行绑定（清单或目录仅包含唯一 `part_id,row_offset`，不要传四变体预测目录的重复并集）：

```bash
.venv/bin/python experiments/rise_sample_analysis/scripts/build_group_features.py build --row-ids artifacts/rise_sample_analysis/test_row_ids.parquet
```

可先在计算节点试一块的32行；使用独立输出目录，避免小样本输出被误当正式整块：

```bash
.venv/bin/python experiments/rise_sample_analysis/scripts/build_group_features.py --output-dir artifacts/rise_sample_analysis/group_features_smoke build --frequency-dir artifacts/rise_sample_analysis/group_features/frequencies --part-id 0 --limit-rows 32
```

小行数构建仍须加载所选块的投影 side info，因此此命令供CPU计算节点，不是轻量登录节点数据抽样。`build --part-id 0`到`--part-id 7`可以在频次完成后按块独立运行；各块输出互不覆盖，整体manifest文件名也按part区分。显式 `--overwrite` 才能覆盖已存在结果。

算法：PyArrow按批读取，频次用有限跨度bincount或unique-count累计；历史按4096×100向量化取窗，类别直方图最多4096×310；每次只驻留一个测试块的局部ID查找和item/action序列。两份全局uint64频次数组原始负载364,124,800字节，约347 MiB。最终约1160万行特征纯数值负载约1.1 GB，加上字符串字典、null位图、格式和manifest开销；压缩大小需实际执行后记录。一次构建峰值还包含一个块的两条完整历史数组、item查找表、Arrow临时缓冲及映射页，预计数GB量级，尚未在真实完整块上实测。`acd_u12cores96G`的CPU配置有充分预算余量，但不据此承诺具体耗时或实测峰值。应从首块日志和调度器资源记录获得真实成本，再决定并行度。

定向验证：

```bash
.venv/bin/python -m unittest discover -s experiments/rise_sample_analysis/tests -p 'test_group_features.py' -v
.venv/bin/python -m py_compile experiments/rise_sample_analysis/scripts/build_group_features.py experiments/rise_sample_analysis/tests/test_group_features.py
```

本轮9项CPU测试通过，覆盖计数不混入历史、跨块全局ID、rowID稳定、重复/越界覆盖拒绝、不完整频次拒绝、目标锚点及截断、空/未知类别和两个归因反例。真实全量脚本尚未执行；正式完成需核对频次总和、8块输出总数、每块row-key散列、与四变体预测的ID/标签匹配及最终资源记录。
