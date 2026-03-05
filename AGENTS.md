# Project Agent Guide (for Codex)

## 1) 项目目标
- 研究问题：模型为何在 assistant 首 token 位置选择或不选择 `<tool_call>`。
- 核心任务：在 clean/corrupt 最小对数据上，定位并寻找决定该首 token 的内部组件与电路路径。
- 研究边界：仅做定位/寻找，不做 knowledge editing，不做应用化实验。

## 2) 关键目录说明
- `pair/`
  - 主数据目录。包含 `meta-q*.json`（token 对齐、分段、关键 span）与首 token 评估表。
  - `first_token_len_eval_qwen3_1.7b.csv` 是当前行为基线统计来源。
- `sample/`
  - 单题示例（clean/corrupt/meta），用于快速调试 patch 与可视化流程。
- `How does GPT-2 compute greater-than copy/`
  - 参考论文源码与图像。用于对齐实验设计、图像类型和排版风格。
- `figs/`
  - 所有实验图输出目录（热力图、探针图、最终电路图）。
- `reports/`
  - 实验报告、结论与复盘文档目录。
- `src/`
  - 实验脚本目录（数据加载、patch、打分、绘图）。

## 3) 运行环境与资源策略
- Python 环境：`base`
- GPU 策略：
  - 优先使用 RTX 4090 24G
  - 若显存不足，不降级破坏实验一致性；等待 GPU 资源释放后再运行
- 建议每次实验记录：模型版本、batch size、精度、随机种子、样本规模。

## 4) 工作约束
- clean/corrupt 必须成对、等长、同位置评估首 token。
- 任何结论必须有因果证据（路径补丁结果）支撑。
- 单样本研究默认使用 `/root/data/R2/sample`（该目录为已采样的显著样本点）。
- 最终报告仅回答“定位与寻找”的问题，不延展到编辑或应用层结论。
- 模型：`/root/data/Qwen/Qwen3-1.7B`
- 图像风格最好与参考论文一致
- 只在你的工作目录下读写，不要关注其他代码