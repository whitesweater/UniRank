# 机制观测 643858 数值门槛失败：诊断与独立恢复方案

日期：2026-09-23。原作业已经 FAILED，错误为 `Numerical gate failed`。以下保留原失败目录与原阈值，不将失败结果改写为成功，也不把 BF16 误差自动认定为可忽略。

## 1. 已验证的失败模式

原目录：`artifacts/rise_sample_analysis/mechanism/observed_643858/`。B、E 各512行，覆盖8个part；各variant每行六站点，共3,072条观测。两份观测均保留 `_unvalidated.parquet` 后缀。原阈值为 `atol=0.0005, rtol=0.005`，即逐行 `abs(error)<=0.0005+0.005*abs(compiled_prediction)`。

| Variant | 标签 | 失败行/512 | 最大绝对误差 | 平均绝对误差 |
|---|---|---:|---:|---:|
| B content-only | Like | 123 | 0.005859375 | 0.0005832762 |
| B | Cart | 21 | 0.003906250 | 0.0001312420 |
| B | Offer | 1 | 0.001220703 | 0.0000540782 |
| B | Checkout | 5 | 0.001708984 | 0.0000333274 |
| B | Purchase | 1 | 0.004882813 | 0.0000202487 |
| E full RISE | Like | 96 | 0.005859375 | 0.0004719496 |
| E | Cart | 17 | 0.001464844 | 0.0001044646 |
| E | Offer | 0 | 0.000732422 | 0.0000466034 |
| E | Checkout | 2 | 0.000976563 | 0.0000174656 |
| E | Purchase | 2 | 0.000976563 | 0.0000123046 |

Like 的误差95分位在B/E中均为0.001953125，99分位均为0.00390625；最大误差0.005859375。所有预测均精确落在BF16概率网格。B Like 的BF16 ULP误差中位数为1，90分位3，最大8；E对应0、3、8。部分低概率动作绝对误差小但可跨更多ULPs，不能只用Like的量级替代每标签审计。

Like失败分散在所有8个part：B每part失败11–19行，E为9–18行。因此没有发现仅一个part或单个row identity错误所导致的集中失败。B关闭AM/PA时仍失败，说明该问题不可能仅用“E的累积控制实现错了”解释。

## 2. 已排查和仍未识别的因素

1. **源代码漂移未发现。** 当前 `observe_mechanism.py` SHA256 为 `e5b36ed694454a40d6f7468652b2716f9de05f8f3812b4b8f274218351251f4d`，与643858 manifest一致；`export_predictions.py` 为 `faa38a594284ee2331f5c6f74503a0cff9f892d0bbab12eb2f289ac7beb44995`，也一致。未修改这两份冻结文件。
2. **state key映射没有结构错误证据。** B/E均205个source key、一一对应205个target key；仅删除完整 `_orig_mod` 路径段，无别名碰撞、缺失或额外键，加载函数还严格检查shape和dtype，并以 `strict=True` 加载。映射直接引用原checkpoint tensor，不重新初始化有效权重。
3. **观测结构完整。** 各variant六站点均有512行，无重复row/site；原数学TV上界与top-order证书没有观测到违例。完整结构和理论不等式成立不代表观测路径已与正式compiled模型等价。
4. **BF16与compile的路径差异是候选解释。** B/E都出现离散BF16概率差异，且原模型以compiled子模块执行、观测模型以eager执行。Inductor融合、算子重排或中间round-off可能改变最终值。但是现有失败文件只包含“compiled无hook”与“eager有hook”两个终点，不能从这两列单独区分compile差异和GPU hook影响。
5. **GPU hook的影响尚未验证。** 原CPU六站点测试证明FP32小模型加旧hook前后bitwise一致，但不能代替实际H100/BF16/checkpoint下的验证。需要同一个eager模型在GPU上先无hook、再加旧hook比较。

E的未验证观测出现部分站点/样本较高clipping（单样本query clipping最高1.0，key最高约0.433735）；B为零。这只能作为后续观测重点，当前不能作为通过gate后的正式机制结果写入论文。

## 3. 新增独立四路诊断

新增 `scripts/diagnose_mechanism_precision.py` 与 `tests/test_mechanism_precision.py`。不修改原observer、export、benchmark、controls或训练core。新程序读原failed manifest中的阈值，没有放宽阈值的CLI入口；复用原样本cache并校验其SHA256，同时校验checkpoint与原失败记录相同。

在同一个GPU任务中，对同一模型、权重与输入执行四路：

| Route | execution | hook | 目的 |
|---|---|---|---|
| compiled_plain | 原compiled模型 | 无 | 正式计算路径参照 |
| compiled_tensor_capture | compiled模型重新编译 | 只保存tensor引用 | 尝试保留compiled路径进行Q/K/Δ观测 |
| eager_plain | 严格映射的eager模型 | 无 | 识别compile/eager差异 |
| eager_original_observer | 同一个eager模型 | 原643858 hooks | 识别旧hook是否改变GPU输出 |

每路保存5标签逐行预测；四个主要比较和与643858旧预测的比较同时报告原threshold gate与bitwise equality。若 eager_plain == eager_original_observer 而二者均不同于 compiled_plain，才可以直接归因于execution模式差异，而不是hooks。若 compiled_tensor_capture 与 compiled_plain 仍不一致，则不能据此恢复正式机制观测，更不能把容差调整到已知最大误差后称作精确匹配。

Compiled capture hook内部只将 `(site, kind, args, kwargs)` 加入Python容器，不进行matmul、softmax、统计、CPU复制或NumPy转换；全部统计在 `model.forward` 返回后完成。新程序仅尝试恢复由**实际compiled Q/K/Δ**重构的selection统计。AM/PA内部状态没有被直接捕捉，明确标记 `am_pa_internal_states_not_recovered=True`，不将eager重算的内部状态误称为compiled实际状态。

## 4. 新CPU检验发现的真实缓存行为

第一版新增测试在已经执行过的 `torch.compile` 模型上注册hooks，结果是 **0 captures，而不是12**，测试失败。这表明当前PyTorch配置会复用已有compiled graph并忽略事后新挂的内部hook，不能假定“register hook成功”就意味着取到了数据。

修复限定在新诊断脚本：注册tensor-only hooks后显式执行 `torch._dynamo.reset()`，强制重新编译，然后再按原预测门槛检验新compiled路径。CPU `torch.compile(..., backend='eager')` 回归在连续两次forward中都收集六站点/12 captures，输出与无hook模型bitwise一致。测试主体用时2.779秒；`py_compile`与 `git diff --check`通过。

该CPU测试检验Dynamo捕获与容器副作用，并未模拟GPU Inductor kernel的数值行为，仍必须由下一次GPU任务决定是否可用。

## 5. 单GPU短诊断命令与验收

```bash
.venv/bin/python experiments/rise_sample_analysis/scripts/diagnose_mechanism_precision.py \
  --failed-dir artifacts/rise_sample_analysis/mechanism/observed_643858 \
  --output-dir artifacts/rise_sample_analysis/mechanism/precision_probe_JOBID \
  --variants E --limit-batches 1
```

此命令使用原cache第一个64行小批进行短诊断；成功后须去掉 `--limit-batches` 并对B/E重新做全部512行检验，不能用64行过gate代表512行。

`E_diagnosis.json` 保存四路比较，`E_four_route_predictions.parquet` 保存逐行数据。只有 compiled capture预测通过原gate，候选selection文件才带 `_validated.parquet` 后缀；它仍仅覆盖selection，不代表全部AM/PA机制已经恢复。`diagnosis_complete.json` 表示四路诊断执行完整，并明确 `mechanism_analysis_complete=False`。失败证据643858保留不动。

后续benchmark可以复用643858的输入cache，因为数值gate失败并不意味着输入身份或缓存失效；效率测量不能据此把未验证机制统计提升为正式结论。
