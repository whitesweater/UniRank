# 配对 AUC 的用户簇不确定性：实现定义与文献边界

本研究的主要效应是固定已训练模型在同一组样本上的 AUC 差值。一个用户有多条记录，
不同模型共享相同真值和样本；按行独立估计方差会忽略前者，分别给两个AUC估计区间后
相减会丢失后者。相关AUC的结构分量思想见 DeLong、DeLong与Clarke-Pearson（1988），
聚类ROC数据应处理簇内相关的背景见 Obuchowski（1997）。本实现下面直接给出公式，
不声称逐行复刻某个外部软件或其全部有限样本修正。

令正样本数为n+、负样本数为n-，经验AUC为A，平分同分数样本对。正例i的placement为
V_i=(负例低于其分数的数量+半数同分负例)/n-；负例j的placement为
W_j=(正例高于其分数的数量+半数同分正例)/n+。
定义正例行影响ψ_i=(V_i-A)/n+，负例ψ_j=(W_j-A)/n-。
其总和为零；这种归一化包含重采样时正负例分母的变化。

对同一行两个模型，先计算ψ_treatment−ψ_control，再在用户u内求和得到G_u。
若共有K个用户簇，渐近方差估计为

`Var_hat = K/(K−1) × Σ_u (G_u−mean(G))²`。

报告δ±1.95996398454×sqrt(Var_hat)。这是一阶投影/影响函数的用户簇sandwich近似，
不是精确用户bootstrap、不是iid DeLong区间，也不是训练seed的方差。
用户间仍被当作工作上独立的抽样簇；共同物品、日期冲击及训练不确定性未被完全覆盖。

五任务宏平均先平均每标签的用户簇影响，再计算方差；不平均标准误，也不把标签当独立
重复。组间增益差同样先对齐用户ID，在用户簇层面相减，以保留跨组同用户协方差。
存在不可定义标签AUC时不悄悄改变宏平均标签分母。

数值实现 `scripts/statistical_metrics.py` 对分数排序后按ties累计正负样本，避免显式
枚举正负样本对。单元验证涵盖：穷举样本对、与sklearn点估计一致、样本权重有限差分、
相同模型差值方差为零、共享用户组间协方差、单类组，以及宏平均任务完全相关的边界。

独立验证 `scripts/validate_cluster_uncertainty.py` 使用三个合成相关结构，每个600用户、
500次精确用户多项式重采样，与线性化标准误对照；结果保存于
`results/cluster_uncertainty_validation.json`。这些是算法数值检查，不能被当作RISE性能
实验，不能据此宣称所有真实数据场景的区间覆盖率已经得到证明。

正式实测仍应在固定真实用户子样本上比较精确重采样与线性化结果，并报告差异。
对稀有标签、小支持组或有效用户数很少的组，渐近近似可能不足；应公开正负例和用户数，
必要时扩大组或仅作描述。所有组效应先作为探索性效应量报告，不用未校正的显著星号
筛选结论。后续若需要确定性检验，必须先固定检验族与校正规则。

文献核验来源（原始研究的PubMed记录，本轮在线核对）：

- DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics. 1988;44(3):837–845. DOI 10.2307/2531595. PMID 3203132.
- Obuchowski NA. Nonparametric analysis of clustered ROC curve data. Biometrics. 1997;53(2):567–578. DOI 10.2307/2533958. PMID 9192452.

可定位源：PubMed `https://pubmed.ncbi.nlm.nih.gov/3203132/` 和
`https://pubmed.ncbi.nlm.nih.gov/9192452/`。引用用于说明统计方法背景；本实现的具体公式、
代码和数值验证是本研究可复查的方法记录。
