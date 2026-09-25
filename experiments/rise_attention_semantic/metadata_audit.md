# KuaiRand 内容注释与 ID 映射审计

日期：2026-09-24。范围限于既有 KuaiRand-27K 数据的官方内容注释，不增加数据集，不修改训练、验证、测试三表或模型输入。

## 原始语义来源

KuaiRand 官方站点提供同一视频 ID 空间的补充 CSV：

- 说明：`https://kuairand.com/` 的 Additional Video-Level Metadata 部分。
- 语义文件：`https://zenodo.org/records/18159199/files/kuairand_video_captions.csv`。
- 字段：`final_video_id, caption, show_cover_text, duration`。caption 是视频原始文案，show_cover_text 是封面文字；两者不是人工根据匿名 ID 编造的类别。
- 文件大小：3,222,932,768 字节。本轮只下载有界前缀及所请求 ID 对应的 HTTP Range，不声称验证了完整文件 MD5。
- 这些补充注释仅用于事后解读，并非训练时新增文本特征。

## 映射依据及交叉核验

当前预处理代码对全部五份原始日志的 video_id 集合升序编号，编码从 1 开始，0 保留为 padding。官方 ID 空间图明确 KuaiRand-27K 的 video_id 范围是连续的 0 至 32,038,724。

本地 metadata 的 27,285 位用户、32,038,725 个视频、323,464,444 次交互，分别与官方完整 27K 的数量一致。由此推导的映射是：

`当前全局 encoded item_id = 官方 final_video_id + 1`

这里的全局 item_id 不是 block-local item_index；查询注释时不能使用后者。

进一步读取官方文件前 8,519,680 字节，仅解析完整记录，并与现有 `test/item_info/part-00000.parquet` 的 item_id / duration_bucket 核验：

| 核验项 | 结果 |
|---|---:|
| 完整官方记录 | 77,495 |
| 官方 final_video_id 范围 | 0–77,494 |
| 可解析且有限、非负的 duration | 62,717 |
| duration 缺失 | 14,406 |
| duration 非法非数值 | 372 |
| 与指定 processed block 的交集 | 44,382 |
| 交集 duration 分箱一致 | 44,382 / 44,382 |

| 原始 duration 分箱 | 当前 encoded duration_bucket | 一致条数 |
|---|---:|---:|
| >60 秒 | 1 | 19,220 |
| (7,15] 秒 | 2 | 9,911 |
| (15,30] 秒 | 3 | 6,825 |
| (30,60] 秒 | 4 | 6,454 |
| (3,7] 秒 | 6 | 1,968 |
| (0,3] 秒 | 7 | 4 |

**此样本未验证 code 5 的意义。** 遇到 code 5 的展示记录先标为未通过，不推测其名称。使用上表固定分箱对应关系，错误地将连接关系偏移一个 ID 时，一致率分别为 26.98% 和 26.93%，没有通过同样检查。

机器可读 receipt、原始前缀 SHA-256、预处理代码 SHA-256、processed block SHA-256、错位对照结果位于 `artifacts/rise_attention_semantic/metadata/mapping_validation.json`。不把这 44,382 条前缀匹配称为全部 3,203 万视频逐项验证。

## 单条展示记录的接受条件

通过正常 TLS 验证获取。每批在开始和结束以 HEAD 核查 Last-Modified 与完整字节数；每个 Range 发送 If-Unmodified-Since 条件，核对 Content-Range 起止、总字节数、实际响应体长度。原始响应、响应头及其 SHA-256 全部保留。

每条记录必须满足：

1. UTF-8 严格解码、严格 CSV 解析、四个字段完整、请求 ID 精确一致；截断记录不补全，不猜测错位字段。
2. 第二个重叠 Range 能恢复四个完全相同的原始字段。
3. caption 优先，封面文字补缺；展示文本至少有四个汉字、字母或数字。
4. duration 是有限非负数，并且与当前展示 token 的 processed duration_bucket 按上表核验一致。

未通过的记录保留失败状态。若所选案例的候选或展示历史记录不满足条件，按照研究协议中事先冻结的候选排序寻找下一个，并记录跳过原因；不依靠语义内容或图形美观度重新挑选。

## 重现

脚本：`scripts/prepare_kuairand_semantic_metadata.py`。

```bash
.venv/bin/python scripts/prepare_kuairand_semantic_metadata.py --audit-prefix
.venv/bin/python scripts/prepare_kuairand_semantic_metadata.py \
    --ids-json artifacts/rise_attention_semantic/metadata/request_ids.json \
    --expected-buckets-csv artifacts/rise_attention_semantic/metadata/request_buckets.csv
.venv/bin/python -m unittest discover -s tests -p test_kuairand_semantic_metadata.py -v
```

ID JSON 是全局 encoded item_id 的整数列表。分箱 CSV 的两列是 `encoded_item_id,duration_bucket`。结果为 `selected_captions.csv` / `.json`，使用 `record_usable=true` 才能进入展示，不能只检查文本非空。每段 Range 的原始文件和独立 receipt 在 `metadata/ranges/`，不存在完整文件校验的伪声明。

## 最终展示记录

两组初选均因真实注释缺失而按冻结排序顺延：改善组 sample 330 的一条展示历史（item 1939039），退化组 sample 18 的候选（item 1563090），文案和封面文字均为空。失败记录和第一轮请求保留在 `figures/selection_round01/`。

最终为改善组 sample 443 与退化组 sample 426，均是各组排序第 2 位。两例候选加六条展示历史，共 14 条记录全部通过精确 ID、两个重叠 Range 完整字段一致、duration_bucket 及可读文字检查。item 4039002 的文案为空，但封面文字可读，按既定备用规则通过。

最终 receipt 为 `artifacts/rise_attention_semantic/metadata/selected_captions_receipt.json`；请求快照在 `figures/selection_round02/`，同时复制到 metadata 下以 `final_` 开头命名。最终元数据 SHA-256 已与图形 selection receipt 的 caption 来源散列核对一致。
