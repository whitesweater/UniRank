# 探索性比较选择时点：文字审计修正记录

本次仅修改四份论文/中文稿及一份AUC结构readout的表述，不更改实验数字、区间、分组边界、展示公式或任何源结果。三项宏组间比较统一说明为：**明确列出的探索性比较，其选择发生在查看初次E−B分组均值之后；原生分析沿用同样定义。** Offer比较仍保留为随后另行追加的posthoc比较；不暗示前三项具有confirmatory地位。

分组边界在正式效果读取前固定、频次切点来自训练数据的事实单独保留。所有区间继续标为名义区间，未做多重比较校正。

## 版本保留与验证

- 归档目录：`artifacts/rise_sample_analysis/wording_audit_20260922T205619Z`。
- `before/` 保存五个文字文件的修改前完整版本；`previous_qa/` 保存两套旧QA目录。原QA目录本身也未修改。
- `qa/component/` 与 `qa/native/` 保存当前版本的独立包装编译、日志、PDF及视觉检查图。
- `wording_changes.patch` 为精确unified diff；`audit_manifest.json` 保存逐处前后文本、旧/新行号、文件SHA及保护的源文件SHA。
- 五个现有文字文件的完整数字token序列逐一相同；展示公式字节相同。46个CSV/JSON/统计脚本SHA逐一相同。
- 新LaTeX两遍编译通过：组件4页、原生3页；没有未定义引用或overfull/underfull排版警告，并完成相关页面视觉复核。
- 原文中的旧字数和旧交付QA事实保留其数字，只补充“初版交付时”的时间限定。原execution/delivery记录保持历史原样；新的文稿与readout hash以本次audit_manifest为准，不回写原始运行记录。
- `population_profile_readout.md` 和 `compiled_mechanism_v1/readout.md` 也已检查；没有这三项宏比较的预注册或事前选择声称，故未修改。

## 逐处精确改动

### 01. `paper/component_analysis.tex`

旧起始行：45；新起始行：48；替换次数：1。

修改前：

```text
The three
supplemental contrasts were fixed before their execution, after the initial
group analysis, and remain exploratory rather than preregistered confirmation.
```

修改后：

```text
The three
supplemental contrasts were explicitly selected after inspecting the initial
grouped mean effects of Full RISE minus Content-only. They are exploratory,
not preregistered or confirmatory comparisons.
```

### 02. `paper/component_analysis.tex`

旧起始行：29；新起始行：29；替换次数：1。

修改前：

```text
Undefined fractions remain separate. Table~\ref{tab:rise_all_conditions} reports
```

修改后：

```text
Undefined fractions remain separate. Group boundaries were fixed before
formal effect inspection, with frequency cut points derived from training counts.
This boundary choice is distinct from selecting group contrasts after inspecting
the initial results. Table~\ref{tab:rise_all_conditions} reports
```

### 03. `paper/component_analysis.tex`

旧起始行：194；新起始行：198；替换次数：1。

修改前：

```text
Full RISE minus Content-only effect $\delta$; the final row is the separately
identified post-hoc Offer comparison of Full RISE (E) and PA-only (D).
```

修改后：

```text
Full RISE minus Content-only effect $\delta$ and were selected after inspecting
the initial grouped mean effects. The final row is the later-added post-hoc
Offer comparison of Full RISE (E) and PA-only (D). All comparisons are exploratory.
```

### 04. `paper/native_merrec_analysis.tex`

旧起始行：29；新起始行：29；替换次数：1。

修改前：

```text
Table~\ref{tab:native_all_conditions} retains all predefined one-dimensional
groups. Effective length excludes the target and counts valid positions in a
```

修改后：

```text
Table~\ref{tab:native_all_conditions} retains all one-dimensional groups.
Their boundaries were fixed before formal effect inspection, with frequency
cut points derived from training counts. Effective length excludes the target
and counts valid positions in a
```

### 05. `paper/native_merrec_analysis.tex`

旧起始行：62；新起始行：64；替换次数：1。

修改前：

```text
the same claim against the native backbone. All three native contrasts are
exploratory, inherited from the component analysis, with no multiplicity correction.
```

修改后：

```text
the same claim against the native backbone. The three contrasts were selected
after inspecting the initial grouped mean effects of Full RISE minus
Content-only; this native analysis reuses those same definitions. They remain
exploratory, not preregistered or confirmatory, with no multiplicity correction.
```

### 06. `paper/native_merrec_analysis.tex`

旧起始行：181；新起始行：185；替换次数：1。

修改前：

```text
$E-A$ effects. Entropy and candidate-support contrasts restrict length to 100.
Shared-user and across-label covariance are retained; no multiplicity correction
```

修改后：

```text
$E-A$ effects. Definitions were selected after inspecting the initial grouped
mean Full RISE-minus-Content-only effects and reused for this native analysis.
Entropy and candidate-support contrasts restrict length to 100.
Shared-user and across-label covariance are retained; no multiplicity correction
```

### 07. `notes/component_analysis_verified_zh.md`

旧起始行：23；新起始行：23；替换次数：1。

修改前：

```text
所有N均是五个任务共用的样本行数；
```

修改后：

```text
分组边界在正式效果读取前固定，频次切点来自训练数据；这与查看分组均值后再选择组间比较是不同层面的决定。所有N均是五个任务共用的样本行数；
```

### 08. `notes/component_analysis_verified_zh.md`

旧起始行：141；新起始行：141；替换次数：1。

修改前：

```text
本次执行前固定的三项补充对比及之后追加的Offer对比如下：
```

修改后：

```text
下面三项宏比较是在查看初次E−B分组均值之后明确列出的探索性比较；Offer比较是在此后另行追加：
```

### 09. `notes/component_analysis_verified_zh.md`

旧起始行：145, 146, 147；新起始行：145, 146, 147；替换次数：3。

修改前：

```text
本次补充执行前固定；仍属探索性
```

修改后：

```text
查看初次E−B分组均值后选择；探索性
```

### 10. `notes/component_analysis_verified_zh.md`

旧起始行：150；新起始行：150；替换次数：1。

修改前：

```text
这里“执行前固定”仅指在此次补充作业运行前固定估计目标；初步分组结果已存在，因此三项仍是探索性分析，不等于在主实验之前预注册。
```

修改后：

```text
三项宏比较的选择发生在初次E−B分组均值检查之后；原生分析沿用同样定义。它们不是看结果前预先指定的比较，也不是原研究预注册或验证性检验。分组边界提前固定与组间比较事后选择须分开表述。
```

### 11. `notes/component_analysis_verified_zh.md`

旧起始行：207；新起始行：207；替换次数：1。

修改前：

```text
英文正文约872个英文词（不计行内/展示公式及表图文字），
```

修改后：

```text
初版交付时英文正文约872个英文词（不计行内/展示公式及表图文字），
```

### 12. `notes/native_merrec_analysis_verified_zh.md`

旧起始行：45；新起始行：45；替换次数：1。

修改前：

```text
分组定义与组件分析相同，以保证比较清晰。
```

修改后：

```text
分组定义与组件分析相同，以保证比较清晰。分组边界在正式效果读取前固定，频次切点来自训练数据；这不意味着后续组间比较在查看初次分组均值前已经指定。
```

### 13. `notes/native_merrec_analysis_verified_zh.md`

旧起始行：181；新起始行：181；替换次数：1。

修改前：

```text
三项原生宏对比沿用组件阶段的定义，属于继承而来的探索性分析，不是主研究之前的预注册验证；名义95%区间没有多重比较校正：
```

修改后：

```text
三项宏比较先是在查看初次E−B分组均值之后明确列出；本次原生分析沿用同样定义，仍属探索性，不是看结果前预先指定或原研究预注册的验证性比较。名义95%区间没有多重比较校正：
```

### 14. `notes/native_merrec_analysis_verified_zh.md`

旧起始行：230；新起始行：230；替换次数：1。

修改前：

```text
英文正文约812词（不计公式和表格），
```

修改后：

```text
原生初版交付时英文正文约812词（不计公式和表格），
```

### 15. `notes/native_merrec_analysis_verified_zh.md`

旧起始行：230；新起始行：230；替换次数：1。

修改前：

```text
原有组件英文稿与中文稿的SHA保持不变。
```

修改后：

```text
原生初版交付时，原有组件英文稿与中文稿的SHA保持不变；本次选择时点的文字修正另有版本差异记录。
```

### 16. `results/auc_structure_v1/readout.md`

旧起始行：22；新起始行：22；替换次数：1。

修改前：

```text
这些比较在补充作业前冻结，但已有分组结果存在，不能称主研究事前预注册验证。
```

修改后：

```text
这些明确列出的探索性比较是在查看初次E−B分组均值之后选择的；原生分析沿用同样定义。它们不是看结果前预先指定的比较，也不是原研究预注册或验证性检验。分组边界本身在正式效果读取前固定，频次切点来自训练数据；这与后续比较选择是不同决定。
```

## 文件身份

| 文件 | 修改前SHA-256 | 修改后SHA-256 |
|---|---|---|
| `paper/component_analysis.tex` | `9211280dc3d0eef8aba0d616418fd5c01e49871d62ddb034d1e50f29f239e686` | `7e319d28a08ac9cc1cb9a4e958634225a91bed69ee7af467efbd0a74a617a3cd` |
| `paper/native_merrec_analysis.tex` | `1b6d44504599d5381be3c14c0b68dac50f5c63e6c5e7b7b2231013eff28e756f` | `608339a4391cbf380b63b0703b6c79d03ecb67517e13023d665b633c0ae5a157` |
| `notes/component_analysis_verified_zh.md` | `3a4efd155982444d3261ad6ed729704164a7b5d2b9c0708ad1da59294302b062` | `933b432e8b39ba02c5d699f985497e1d865fc680fe4e3923ca296fd99cad93c7` |
| `notes/native_merrec_analysis_verified_zh.md` | `039636ea87c5f43875d4771f9a4f99f937a02be60851e48246eacef2d0e1e3f0` | `0be8c641e59078b9aa9907f09de9ede413b4f5c993592409ee5315c68534ab94` |
| `results/auc_structure_v1/readout.md` | `9fc0c2d0ed8e8a936da3ce37717855a376847be55f84ebcd5a261bbcc2d2ed52` | `231ab8c34ed674b5f364f00d70c2c5b4148d638fe39703d2d3d132f66acd740d` |
