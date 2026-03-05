# TODO: 迁移《How does GPT-2 compute greater-than?》到 `<tool_call>` 首 token 电路定位
> 模型：`/root/data/Qwen/Qwen3-1.7B`
## 0) 范围边界（强制）
- 只做「定位与寻找」：行为验证、路径补丁、组件筛选、电路验证、组件探针、最终电路图。
- 不做：knowledge editing、模型编辑、应用化/产品化实验。

## Part 1. 需要完成的目标（按论文实验思路迁移）

### 1.1 任务与数据基线（对应论文 sec.2 + app: task/dataset/metric）
- [ ] 定义任务为：给定 clean/corrupt 成对提示词，研究模型在 assistant 首 token 位置是否选择 `<tool_call>`。
- [ ] 固定分析位置：`<|im_start|>assistant` 后的第一个待预测 token（首决策位）。
- [ ] 复核数据完整性：
  - [ ] `pair/meta-q*.json` 均满足 clean/corrupt token 长度对齐。
  - [ ] 使用 `pair/first_token_len_eval_qwen3_1.7b.csv` 复现首 token 行为统计。
- [ ] 产出基线统计表（clean 命中 `<tool_call>`、corrupt 非 `<tool_call>`、pair 同时满足比例）。
- [ ] 记录当前快速基线（164 对样本）：clean 命中 `<tool_call>`=89.6%，corrupt 非 `<tool_call>`=83.5%，pair 同时满足=73.2%。
- [ ] 产出行为图（替代论文 probability heatmap/slice）：展示 clean/corrupt 下 `<tool_call>` 概率与 margin 分布。

### 1.2 IPP 第一步：直接到 logits 的组件搜索（对应论文 Fig. logits-ipp）
- [ ] 对所有注意力头 `L{layer}H{head}` 做路径补丁：patch `(component -> logits@首token)`。
- [ ] 对所有 `MLP{layer}` 做同样补丁。
- [ ] 使用主指标计算每个组件的因果影响，得到两张热力图：
  - [ ] `figs/ap_head_heatmap.png`（attention heads）
  - [ ] `figs/ap_mlp_heatmap.png`（MLPs）
- [ ] 从热力图筛出直接贡献最大的候选组件集（头 + MLP）。

### 1.3 IPP 递进搜索：经关键 MLP 的上游定位（对应论文 step-by-step circuit finding）
- [ ] 从最下游关键 `MLP{k}` 开始，做条件路径补丁：`(C, MLP{k}, logits)`。
- [ ] 对存在多条下游路径的 `MLP{k}`，补全组合路径（如 `(C, MLP{k}, MLP{k+1}, logits)`）。
- [ ] 逐层回溯，直到定位到稳定上游注意力头集合。
- [ ] 输出条件热力图：
  - [ ] `figs/ct_head_heatmap.png`（conditioned tracing heads）
  - [ ] `figs/ct_mlp_heatmap.png`（可选但建议）

### 1.4 候选电路评估（对应论文 circuit sufficiency / necessity）
- [ ] 充分性实验：仅候选电路路径保留 clean，其余路径用 corrupt；计算性能恢复率。
- [ ] 必要性实验：仅候选电路路径替换为 corrupt，其余保留 clean；验证性能显著下降。
- [ ] 给出最终候选电路节点与边列表（仅保留有因果证据的边）。

### 1.5 组件语义探针（对应论文 circuit semantics，限定在定位相关）
- [ ] 对入选注意力头输出探针图（至少包含其对 `<tool_call>` logit 的直接效应或等价因果量）。
- [ ] 对入选 MLP 输出探针图（显示其对 `<tool_call>` vs 非 `<tool_call>` 的偏置）。
- [ ] 单组件图命名示例：`figs/L7H14_probe.png`、`figs/MLP12_probe.png`。

### 1.6 最终交付
- [ ] 最终电路图：`figs/final_circuit.png`（节点、边、方向、权重/符号清晰）。
- [ ] 实验报告：`reports/circuit_localization_report.md`，包含
  - [ ] 数据统计
  - [ ] 各阶段热力图
  - [ ] 充分性/必要性结果
  - [ ] 与论文结论的对齐说明（趋势对齐，不强求绝对数值一致）。

## Part 2. 规则（强制）

### 2.1 命名规范（R1，强制）
- 注意力头：`L{layer}H{head}`（例：`L7H14`）。
- MLP：`MLP{layer}`（例：`MLP12`）。
- 其他结点（如确实需要）：`RESID_L{layer}`（默认不进最终电路图）。
- 图文件：
  - 热力图：`ap_head_heatmap.png` / `ct_head_heatmap.png`
  - 组件探针图：`L7H14_probe.png`
  - 最终电路图：`final_circuit.png`

### 2.2 指标规范（迁移自论文 metric 设计）
- 令目标 token 为 `t* = <tool_call>`，在第 `i` 对样本上：
  - `p_clean_i = P(t* | clean_i)`
  - `p_corr_i = P(t* | corrupt_i)`
  - `m_clean_i = logit_clean_i(t*) - max_{v!=t*} logit_clean_i(v)`
  - `m_corr_i = max_{v!=t*} logit_corr_i(v) - logit_corr_i(t*)`
- 主指标（论文 probability difference 的迁移版）：
  - `PD_tool = mean_i[p_clean_i - p_corr_i]`（越大越好）
- 辅指标：
  - `PairSignAcc = mean_i[1(m_clean_i>0 and m_corr_i>0)]`
  - `MarginSep = mean_i[m_clean_i + m_corr_i]`
- 组件影响分数（IPP）：
  - `Impact(C) = Metric(unpatched) - Metric(patch C along specified path)`
- 电路恢复率（论文 recovery 的迁移版）：
  - `Recovery = (M_suff - M_corr) / (M_clean - M_corr)`

### 2.3 路径补丁实验规范
- clean/corrupt 必须成对使用，且对齐长度；仅在同一 pair 内 patch。
- 所有 patch 只在“assistant 首 token”决策位读取指标。
- 每个实验固定随机种子、batch 切分和数值精度；报告必须记录这些配置。
- 先跑小样本调通，再跑全量 164 对；图与报告必须标注样本规模。
- 单样本深挖（single-case study）时，固定使用 `/root/data/R2/sample`（已采样到显著点）作为默认样本来源。

### 2.4 颜色与对比（R2，强制）
- 所有热力图必须使用红/蓝发散色系（`RdBu` 或同类）。
- 0 必须在色带中心（白或浅色）。
- 正负贡献必须一眼可区分。
- 如做裁剪/归一化，必须在图注标注规则和阈值。

### 2.5 图排版与风格（R3，强制）
- 风格对齐论文图：清爽、留白足够、信息密度高但不拥挤。
- 字体统一，字号层级明确（标题 > 轴标签 > tick）。
- 统一坐标语义：
  - 头热力图：`x=head`, `y=layer`
  - MLP 热力图：`x=MLP` 或单列，`y=layer`
- 图注必须写清：patch 路径、指标、样本数、颜色含义。

### 2.6 结论输出规范
- 结论只回答定位问题：
  - 哪些头/MLP 决定 `<tool_call>` 首 token 选择；
  - 它们通过哪些路径协同；
  - 该电路是否具有充分性与必要性证据。
- 不扩展到知识编辑、应用实验、上线策略。
