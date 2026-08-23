# 最终代码实现审计记录（2026-08-12）

> 事实口径：只以当前代码、Git 暂存区、机器上现有数据/结果和实际测试为准；不使用旧论文叙述作为证据。本记录不改写既有实验身份，也没有重新调用模型或运行大规模实验。

## 1. 本轮已完成的代码修复

### 1.1 Entity Resolution Phase 1

- FTS 别名交叉引用从“同一 OCID 中名称相同即可连接”收紧为“名称和角色均相同”。
- 同一 `(OCID, normalised name, role)` 若对应多个官方实体，不再任意选择第一个实体。
- 同一个 FTS raw ID 若在多个 OCID 中分别指向不同官方实体，会保持 unresolved；只有所有观察共同指向唯一官方实体时才建立别名。
- 实现：`src/procurement_graph/er/phase1.py`。

### 1.2 Entity Resolution Phase 2

- 规范化后为空的实体名称不再被 Phase 2 静默丢弃，而是保留为 singleton。
- `canonical_name` 为 `NaN` 时，政府实体查找和名称/地区合并不再抛出 `TypeError`，且该实体仍被保留。
- 实现：`src/procurement_graph/er/phase2.py`。

### 1.3 OCDS 金额提取

- award/contract 的 `amount=0` 现在被视为合法金额，不再因 Python truthiness 被 `amountNet` 覆盖。
- 只有 `amount is None` 时才回退到 `amountNet`。
- 实现：`src/procurement_graph/extract/tables.py`。

### 1.4 比较评估入口

- `scripts/run_compare.py --planner llm` 现在真正实例化 `LLMReasoningPlanner`；此前该选项会静默落入规则规划器。

### 1.5 局部 pipeline 输出保护

- `01_ingest --years` 必须配合独立的 `--output-path`。
- `02_er_phase1 --limit/--sample` 必须配合独立的 `--output-dir`。
- `03_er_phase2 --limit/--sample` 从正式 Phase 1 产物读取，但必须把局部结果写到独立的 `--output-dir`。
- `04_extract --years` 必须配合独立的 `--output-dir`。
- 即使显式传入的路径经解析后仍等于正式路径，也会拒绝运行；完整运行仍保留原默认输出位置。
- `--years` 改为至少接收一个年份，避免裸 `--years` 被解释成完整运行。
- 四个 pipeline 已直接导入 `procurement_graph.*` 的最终 package 实现，不再依赖旧的平铺兼容模块。
- 共享路径保护：`src/procurement_graph/common/pipeline_paths.py`。

## 2. 已执行的轻量验证

- 修复前相关基线：ER Phase 1、ER Phase 2、extract 共 `116 passed`。
- 第一轮修复后：同一组测试 `121 passed`。
- 补齐跨 OCID 冲突与 `NaN` 边界后：同一组测试 `123 passed`。
- 加入四条 pipeline 输出保护及测试后：定向测试合计 `134 passed`。
- 最后一次命令：

```text
OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -B -m pytest -q -p no:cacheprovider \
tests/test_pipeline_output_safety.py tests/test_er_phase1.py \
tests/test_er_phase2.py tests/test_extract.py
```

- 没有重新训练模型、调用云端 LLM 或批量重跑评估。
- 因工作方向切换到论文准备，完整测试套件尚未在这批改动后运行。

## 3. Git 暂存区只读审计

审计时暂存区为 22 个文件，约 224.01 MiB，主要是生成数据、展示文档、本机链接和 Python 缓存；没有已暂存的 Python 源码、测试或 YAML 实现补丁。本轮源代码修复保持为未暂存改动，没有混入用户已有暂存内容。

### 建议保留但分开管理

- `data/training/wtq_sft_C/sft.jsonl` 与 `data/training/wtq_sft_C6/*`：当前 WTQ 数据链路直接使用。
- `data/training/llamafactory_compose_v3/*`：当前 compose/PACS 代码直接使用；已有实验身份不应被静默原地去重。
- `data/qa/wtq/harvest_teacher_reasoning.jsonl`：昂贵的 teacher 历史产物，但不是当前最终 runtime 的直接输入，宜独立归档。
- compose v1/v2：主要用于历史复现，宜与当前实现分开。

### 不应作为可移植实现提交

- `outputs`：指向 `/home/uceeh01/migrated_outputs` 的本机绝对软链接。
- `scripts/wtq/__pycache__/*.pyc`：运行缓存，仓库的 ignore 规则本已排除同类文件。
- 四份展示/讲义文档宜与实现和数据提交分开。

### 已识别的数据边界

- compose-v3 存在 2 条 train/val 完全相同样本，train 内也有精确重复。为了保留已完成实验身份，本轮没有修改；后续应生成带 manifest/hash 的新版本，而不是覆盖 v3。
- WTQ C6 的 7 类 idiom 各复制 10 次是代码中的显式加权设计；validation 是 canonical base 的末 200 条，而不是按 table/question group 隔离，不能表述为严格独立验证集。

## 4. 已发现但在切换论文任务前未继续修改的问题

以下属于下一轮代码修复候选，不应在论文中当成已经修复：

- compose 评分器的 Python `bool`/`int` 等价问题：部分路径会把 `True` 与非零数、`False` 与 `0` 误判为相等。机器上已有结果受影响程度不同，必须先统一严格重评分，才可引用相关准确率。
- raw 文件年份选择仍使用文件名 substring，且 `Path.stem` 会把 `2024.jsonl.gz` 记录成 `2024.jsonl`；修复工作已暂停。
- teacher harvest 的多 JSONL append/resume 不是事务性的：先写 trace 后崩溃可能漏写训练 sink，且 fresh run 会 append 旧文件。
- CICADA compare 的 partial resume 尚缺运行 manifest；不同模型或输入可能误混。
- 金额问答中仍有非 GBP 的 additive 记录，而部分问题文本声称 GBP。直接加过滤会改变 benchmark 语义，应先版本化数据与评估，不能静默修补。
- 多项 `settings.yaml` 配置当前未真正注入实现；应统一清理或集中接线，不能在论文中声称这些设置可配置。

## 5. 论文使用约束

- Methodology 可以引用第 1 节已确认的最终实现设计，但应在完整测试后再将其表述为最终冻结版本。
- Results 只能引用能从当前结果文件按当前评分代码复算的数字。
- 受 bool 评分、货币混合、split 泄漏或缺失 artifact 影响的结果必须先纠正、降级为限制，或明确标记为历史结果。
- 不以旧论文、展示文稿或硬编码绘图数字替代代码和原始预测文件证据。
