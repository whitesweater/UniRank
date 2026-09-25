# MerRec 原始事件可用性、动作输入依赖及归因时点审计

日期：2026-09-23。本审计在正式 CPU 特征作业提交后独立进行，未修改或重跑已冻结的 `build_group_features.py`，未全盘搜索、未下载原始数据、未全量扫描训练/测试记录。

## 1. 本地是否具备可量化的原始事件

**在本轮明确访问的范围内没有找到原始 MerRec 事件文件。** 这不是对未挂载存储、未接入远端备份或所有用户目录的不存在证明。机器可读访问记录为 `results/merrec_raw_availability_audit.json`。

| 定位依据 | 实际访问范围 | 结果 |
|---|---|---|
| 下载脚本默认目录 | 主仓 `data/MerRec/20231001` | 不存在；`data/MerRec` 只有预处理、下载和统计程序及运行缓存 |
| 常见数据目录 | 主仓 `datasets/MerRec`、`datasets/MerRec/20231001` | 不存在；现有 `datasets/MerRec_Action` 为预处理三表 |
| 已知工作树 | UniRank-sisa-temporal、ManCAR 的 `data/MerRec` / `20231001` | 有程序，无原始事件目录 |
| 相关旧工程 | ManCAR-stage0、ManCAR-sisa-style-71a3226 的 `data/MerRec`，ManCAR 的 `datasets` | 指定目录不存在 |
| 预处理明确默认路径 | `/mnt/ceph-nj1-csp/bingoozhang/salmonli/data/ft_local/data/MerRec` | 本机不可见/不存在该路径 |
| 下载日志中的旧机器位置 | `/data_nvme/user/ywhao/proj/ManCAR/UniRank/data/MerRec` | 本机不存在 |
| Hugging Face 本地缓存 | 用户 `.cache/huggingface/hub`、`.cache/huggingface/datasets`、`cache/hf_datasets` | 顶层无 MerRec/Mercari/20231001 命名条目；17份小型 dataset_info.json 无对应源说明 |
| 旧下载临时目录 | `.unirank-datasets.wiyHIm` 顶层名称 | 其他商品数据集，无 MerRec |

`datasets/download.log` 明确记录本地下载的是 **salmon1802/MerRec 预处理包**，不是原始 **mercari-us/merrec** 事件。`block_manifest.json` 保留作者机器上的 processed 输出路径，不记录可用的原始事件源版本、原始ID映射或每个动作发生时刻。现有 `feature_vocab.json` 是预编码整数的 generic identity 字典，也不是 raw user/item ID 到处理后 ID 的可信原始映射。

因此，当前不能实现并宣称已运行“真实跨split归因率/历史未来动作率”的事件级审计。没有编造这些比率，也没有重建或修改当前训练数据。是否存在其他存储上的完整原始分区仍未知。

## 2. 当前 processed 数据的限量实证

为区分源码可构造风险与当前数据实际采用的编码，读取 `test/data/part-00000.parquet` 的前32条物理行，并从配对 user_info 中找到它们的完整 action/timestamp 序列。涉及8名用户；为定位相应局部 user index，分批读取了9,728条 side-info 用户记录、1,754,157个 action token。该读取设有20,000用户上限，只选取一块中的32条目标样本，不是代表性总体抽样。

| 核验量 | 实测 |
|---|---:|
| 当前 anchor 的 action 解码与对应样本的五标签完全一致 | 32/32 |
| 32条目标中至少一个正标签的目标行 | 1 |
| 可见过去view中含至少一个正动作结局编码的目标行 | 32/32 |
| 历史view时间戳晚于当前target时间戳 | 0/32 |

逐行记录和源文件身份保存在 `results/merrec_processed_action_encoding_sample.json`。这些结果支持“当前三表中的历史 action 确实承载 view 的多标签结局”这一编码联系。**它们既不能证明32条样本中的过去结局均已及时可用，也不能证明其中一定存在未来动作污染。** 历史view时间早于target，只确定view顺序；它没有提供附着动作的实际发生时间。32/32也不能推广为总体比例。

## 3. 四个正式模型是否实际使用该 action

**四个正式骨干都把 action 作为历史表示的一部分；该边界不是 RISE 独有，也不是仅存在于新增分组统计。** MerRec 配置将 action 标为 active categorical、source=action（`config/dataset_config.yaml:243`）。collator 将历史 action 放入 `item_dict['action']`，同时把target位置action置零（`unirank/pytorch/dataloaders/unirank_dataloader.py:555–580`）。

| 模型 | 动作进入预测的代码路径 |
|---|---|
| HiFormer | source=item/action 都计入 item_info_dim（`:66–68`）；`embedding_layer(item_dict)`组成历史表示并进入target pooling（`model_zoo/HiFormer.py:124–130`） |
| HyFormer | source=action纳入维度（`:77–78`）；item_dict embedding进入历史投影、序列与global query构造（`model_zoo/HyFormer.py:151–179`） |
| RankMixer | source=action纳入维度（`:69–71`）；item_dict embedding进入历史target pooling（`model_zoo/RankMixer.py:131–137`） |
| Zenith | source=action归入物品属性维度（`:68–72`）；feature embedding只排除user_id/item_id，保留action，随后用于历史pooling（`model_zoo/Zenith.py:79–83,144–156`） |

这些 forward 都调用 `get_inputs(inputs)`，默认 `return_multi_masks=False`；`rank_model.py:394–428` 将额外 task masks 留在loader接口层，不返回给这些 forward。因此，不能以“模型没有使用multi_masks”推断模型没有使用动作；动作 embedding 与 multi_masks 是两条不同路径。上述是本轮源码核查，不宣称每份历史执行源码已逐字节冻结；日志中的 active action 配置与当前路径相符。

## 4. 原始读取顺序能否排除 row_id 陷阱

目前**没有证据排除**。`iter_parquet_batches` 只按文件名字典序遍历、按文件原行序迭代，并检查必要列（预处理 `:291–302`）；它没有读取后按时间全局重排，也没有校验原始事件顺序。`clean_chunk:349` 在此读取顺序上分配 `_row_id`。下载程序也只是按远端路径排序下载文件，并不验证文件内部时间顺序。

之后 `process_partition:682` 按用户、stime和_row_id排序，但没有重写_row_id。`build_view_anchor_samples:628–630` 的后续事件谓词仍是 `_row_id > anchor_row_id`，不是时间排序键比较。要排除此隐患，至少需要证明每个原始 `(user_id,item_id)` 的事件时间顺序与读取顺序一致；要求全局时序有序更强但不是必要条件。仅有处理后的 `full_timestamp_seq` 单调性或文件名排序，都不足以证明这个原始输入条件。

现有CPU测试已构造两个互相独立的反例：

1. **未来结局附着于过去view。** A-view(t=100,row=0)、B-view(t=200,row=1)、A-Like(t=300,row=2)：A-view的action被编码为Like，预测B时该view在过去而Like尚未发生。即使原始stream完全按时间排序，这个反例仍成立，因此原始顺序正确也不能排除未来结局的可用性问题。
2. **原始读取顺序不等于时间顺序。** A-view(t=100,row=10)、A-Like(t=50,row=11)：函数仍把更早的Like归因给view。这个反例只证明存在顺序依赖；它不证明历史原始文件确实乱序。

测试位于 `tests/test_group_features.py`，直接调用当前预处理函数。它们是算法行为的确定性证据，不是总体泄漏率估计。真实数据受影响范围仍需原始事件。

## 5. 原始事件可用后，怎样量化而不误计

未来事件审计应先恢复输入文件版本、原读取次序、原始事件时间、user/item ID映射以及用户筛选规则。在这些输入缺失前不能把一小段新下载数据等同于现有processed版本。

对历史view锚点h、动作标签ℓ，先按实际归因规则找到产生最终正标签的原始动作集合。令 `T(h,ℓ)` 为这些动作中的最早可用时间排序键，排序键应明确包含时间戳及同时间戳的事件次序。对于target q的实际最后100个历史view，若最终编码为正而 `T(h,ℓ)>T(q)`，则该历史动作位在q预测时不可用。不能把“某个后续同类动作发生在q之后”直接算污染：若该标签此前已出现，最终的二值正标签可能早已可知。

至少分别报告：

- 有任意未来结局动作位的target比例，分动作标签、split及历史长度；分母为可完成事件追溯的目标样本。
- 未来结局动作位数/全部编码为正的历史动作位数，避免把样本比例与token比例混为一谈。
- 训练/验证anchor中，其正标签的首次归因事件落在后续split的比例；另报“存在任一跨split事件”的较宽定义，二者不混用。
- 按原读取顺序与时间键分别归因时，标签不一致的anchor比例，专门量化row_id问题。

跨split归因并不在所有任务定义中自动等于错误，仍取决于预先声明的归因窗口、标签成熟期与训练截止时点。历史输入在预测时可得性则是另一条需独立审计的约束。若只拿到部分原始事件，对未观测到匹配动作的样本必须标“无法判断”，不能当作“无泄漏”；要以完整用户轨迹或已证明完整的事件窗口为抽样单位。事件级统计与processed样本对齐验证应成为独立审计产物，而不是在现有分组任务中悄悄改写标签。
