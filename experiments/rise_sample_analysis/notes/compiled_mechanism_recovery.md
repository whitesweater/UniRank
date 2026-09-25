# Compiled机制观测恢复与 learned-control 重算审计

日期：2026-09-23。本文件接续 `mechanism_643858_diagnosis.md`，保留先前失败证据和所有冻结源码；不追改643858的门槛或观测状态。

## 1. 643882短GPU诊断的实际结论

结果目录：`artifacts/rise_sample_analysis/mechanism/precision_probe_643882/`。E的原cache第一个64行小批、5个标签，在同一次H100任务中得到：

| 比较 | 原门槛 | bitwise equality | 含义 |
|---|---|---|---|
| compiled tensor capture vs compiled plain | 通过 | 是 | 此小批的捕获未改变正式compiled输出 |
| compiled plain vs643858 compiled | 通过 | 是 | 正式参照可重现 |
| eager原hook vs eager plain | 通过 | 是 | 在此实际GPU/checkpoint条件下，旧hook没有改变eager输出 |
| eager原hook vs643858 eager | 通过 | 是 | 旧观测执行路径可重现 |
| eager plain vs compiled plain | 失败 | 否 | Like12行、Cart1行超原阈值，其余标签通过 |

因此，**对这64行E样本，先前的门槛失败已经被定位为compiled与eager执行路径的数值差异，而非hook引入的输出改变**。不能把64行E的证据直接扩展到512行或B；完整检查仍须分别完成。

这里的bitwise指保存为float32后的BF16预测值逐元素完全一致，不是仅通过较宽的allclose。原阈值仍为 `atol=5e-4,rtol=5e-3`。

## 2. 完整512行B/E的独立执行与验收

使用同一冻结诊断脚本、原cache和新目录：

```bash
.venv/bin/python experiments/rise_sample_analysis/scripts/diagnose_mechanism_precision.py \
  --failed-dir artifacts/rise_sample_analysis/mechanism/observed_643858 \
  --output-dir artifacts/rise_sample_analysis/mechanism/precision_full_JOBID \
  --variants B E --limit-batches 8
```

完整验收条件：

1. 每variant完整512行、每行6个站点，共3,072条site记录；row IDs、标签、用户/物品与原cache完全一致且无重复。
2. B/E各5标签的 `compiled_capture_vs_compiled_plain.passed=True`，并单独报告是否bitwise一致；保持原门槛不变。
3. 同样记录 `compiled_plain_vs_failed_run_compiled` 是否复现，排除参照随新attempt改变。
4. 只接受无unvalidated标记且capture gate通过的 `*_compiled_selection_validated.parquet`；最终诊断completion只是执行回执，不能替代逐variant检查。
5. 使用实际compiled Q/K/Δ重构的Ω、TV、top-key变化及padding质量可以作为该固定样本的机制统计；native hard mask和无效query排除规则与原实现一致。
6. TV上界与top-order证书仅容许浮点舍入范围内误差；若出现明显违例，先查重构、mask与scale，不解释成理论失效。

跨eager/compiled比较继续失败是本次预期定位结果，不会否定通过gate的actual compiled selection统计。该结果只覆盖预先固定的512行诊断样本，不提供全测试集群体均值的代表性保证。

## 3. AM/PA内部状态不能从selection恢复直接推出

捕获actual Q/K/Δ恢复了selection层的证据，但它不直接暴露fused graph内的log-decay、prefix phase、centering或clip中间量。尤其是clipping在边界附近的命中与编译数值路径有关，不能仅因最终概率接近就把eager内部状态称为实际compiled状态。

可以做一个更有限、定义清楚的研究：固定actual compiled输入表示 `x`、valid mask与当前checkpoint权重，在模型forward完成后，按照原 `SISAScoreBias` 的eager BF16算法重算完整Δ，并明确称 **learned-control reconstruction from compiled representations**。它对应“这些学到的控制在指定独立数值执行下的行为”，不是fused实现逐算子的直接观测。

即使重算Δ与actual Δ足够接近，也不能由此推出所有中间状态逐位相同。公共相位/gauge抵消、裁剪饱和和双线性压缩都可能使不同中间状态产生接近的最终Δ。因此文件与论文措辞始终保留“重算”，不因gate通过就改称“实际fused AM/PA状态”。

## 4. 新增独立重算审计脚本

新增 `scripts/audit_control_reconstruction.py`，依赖冻结的capture工具与观测统计函数，不修改它们。每variant先跑无hook的compiled预测，再启用tensor-only capture、清理Dynamo缓存并重编译；两路最终预测仍通过原门槛。

每个captured site在完整forward结束后独立调用原 `SISAScoreBias.forward` 直接重算Δ，绕过module hooks，避免把重算误记成新actual capture。随后逐个 `[sample,head,query,key]` 位置比较actual compiled Δ和重算Δ，并计算二者对同一native scores引起的selection TV。

在任何GPU结果产生前，代码固定两组独立门槛：

- **Δ数值重算门槛，单位为attention logit**：每个位置的误差不超过 `5e-4+5e-3*abs(actual_delta)`。
- **selection敏感性门槛**：每个有效attention row的 `TV(softmax(s+actual_delta),softmax(s+reconstructed_delta))<=1e-3`。

这两组门槛与最终预测门槛分开记录。Δ门槛包含所有tensor位置，也另报native实际使用的位置；不能用大量padding零值稀释失败比率。相对L1误差、最大绝对误差、失败位置数和selection变化均保留。若给整个row的Δ添加常数，softmax可不变，但逐位置Δ门槛仍会拒绝；测试显式覆盖此例。

门槛定义了一项事前数值审计准则，不能解释成模型性能上的统计等价。其主要作用是拒绝与actual Δ不一致的重算，而不是证明中间状态唯一或完全相同。

## 5. 短任务命令和输出语义

```bash
.venv/bin/python experiments/rise_sample_analysis/scripts/audit_control_reconstruction.py \
  --reference-dir artifacts/rise_sample_analysis/mechanism/observed_643858 \
  --output-dir artifacts/rise_sample_analysis/mechanism/control_probe_JOBID \
  --variants B E --limit-batches 1
```

完整512行将 `--limit-batches` 改为8；本子任务未提交GPU任务。

输出 `*_capture_prediction_gate.json`、`*_reconstruction_gate.json` 和逐样本/站点的learned-control重算表。只有最终预测capture gate、所有Δ位置门槛、所有有效row的selection门槛全部通过，文件才带 `_validated.parquet`；否则保留 `_unvalidated.parquet` 并保存明确失败计数。`audit_complete.json` 仅表示审计执行完成，其 `all_reconstructions_eligible` 决定是否接受重算；程序成功退出不等于数值重算通过。

如果重算失败，既不放宽门槛，也不改用643858的未验证AM/PA统计。实际compiled selection的已通过证据仍然独立有效。若确有必要恢复内部AM/PA状态，需要另一个事前定义并重新验证输出不变的fused-state捕获方案，不能用命名替换弥补缺失证据。

## 6. 新增CPU验证记录

`tests/test_control_reconstruction.py` 的5项测试通过，主体用时0.031秒：相同Δ逐位置及selection均通过；常数Δ误差不能仅靠softmax不变混过门槛；被hard mask排除的位置仍进入完整逐位置审计；同一hidden/权重的原SISA CPU重算精确相同；NaN重算拒绝。`py_compile`、`git diff --check`通过。

新代码产生后再次核对，`diagnose_mechanism_precision.py` SHA256与643882 manifest完全一致。旧observer、export、benchmark和已保存失败证据均未改变。
