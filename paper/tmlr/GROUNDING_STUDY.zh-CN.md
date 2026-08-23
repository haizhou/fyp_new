# Grounding 机制与模型优化影响：代码事实、初步反事实与实验设计

> 状态：工作研究说明。只依据当前实现和已保存的机器产物；没有调用 LLM，也没有把旧论文叙述当作证据。

## 1. 要回答的研究问题

本项目中的 grounding 不是一个单独的“实体链接”步骤，而是一组发生在不同表示层之间的确定性约束：

1. 自然语言字段如何映射到 KG schema；
2. 文本实体和值如何映射到 KG 中实际存在的值；
3. 多跳图计划中的变量如何建立依赖并把上一步输出绑定到下一步；
4. planner 产生的查询如何修补为执行器允许且安全的运行时查询；
5. 执行前后检查如何阻止无覆盖、非穷举或证据不足的答案被发布。

由此形成两个需要分开的研究问题：

- **RQ-G1（运行时）**：在冻结模型输出和 KG 后，每类 grounding 干预分别改变多少查询的可执行性、答案正确性和拒答行为？
- **RQ-G2（优化）**：由 grounding 失败产生的 repair-SFT / preference examples，是否使后续模型更少产生 schema、entity 和 binding 错误？

RQ-G1 可以用现有 trace 做无模型反事实；RQ-G2 必须做受控训练数据消融，不能从现有 SFT、RSFT、DPO checkpoint 的总体差异中直接推断。

## 2. 当前实现中每一步究竟如何 grounding

| 层次 | 输入 → 输出 | 当前确定性行为 | 失败时发生什么 | 主要实现 |
|---|---|---|---|---|
| G1. 语义 schema grounding | 自然语言 `field_text` → canonical slot | alias 候选与字符 3–5 gram 相似度；默认置信度阈值 0.72、top-2 margin 0.08；再用字段类型和值域做 type gate；`awarded by/to` 等方向词约束 buyer/supplier role | 产生带候选和建议的 `IntentIssue`；不猜测低置信映射 | `reasoning/schema_grounding.py:126–264`；`typed_planning.py:1502–1637` |
| G2. Intent 检查与编译 | typed intent/program → graph variables/relations/return | 校验操作、输入类型、answer step；可确定编译时走 fast path；否则调用 Step-2 typed planner；编译后检查问题中的年份、CPV、类别等原子约束是否被遗漏 | invalid intent 进入结构化反馈；可修缺陷允许 bounded replan | `typed_planning.py:1186–1470, 2386–2415, 2583–2799` |
| G3. Graph structural grounding | graph plan → executable dependency DAG | 变量引用转 dependency；丢弃 padding filter；折叠冗余 entity-set 链；建立 reasoning order；拒绝 cycle、未知 dependency 和无约束 bind source | graph compile/grounding failure，不执行该 plan | `reasoning/graph_planning.py:180–485` |
| G4. 实体和值 grounding | literal/entity/filter → KG value | 组织名先 exact；否则要求 score ≥0.85 且与第二候选 margin ≥0.10；同组织表面变体可组成 `IN`；年份、范围、类别、日期写法转 canonical value | unresolved/ambiguous entity 进入显式失败；exact no-result 后只有受限候选修复可改一个 equality | `entity_resolution.py:59–103`；`grounding.py:227–294`；`retrieval.py:198–295` |
| G5. 多跳变量绑定 | 上游 variable output → 下游 hidden constraint | 按拓扑顺序执行；从 source `emit_field`、role 和 relation 决定 bind field；把上游值注入下游 `IN/eq` constraint；每个 variable 查询前再次 grounding | 空/错误 binding 不会被语言模型直接补答案；作为 graph execution failure 返回 | `graph_planning.py:700–838` |
| G6. Runtime-spec grounding | planner query spec → executor-safe spec | 字段 alias；强制 operation 对应 answer field；`sum` 加 `value_is_additive=True`；reduction 强制 exhaustive retrieval；修 `dedupe_key`、`sort_field`、`group_by_field` 和 answer type | 不能安全修复则返回 grounding failure | `reasoning/grounding.py:109–219, 329–342` |
| G7. 执行与 release checks | grounded spec + rows → answer / no verified answer | preflight 检 schema、约束冲突和穷举条件；执行后检查 population coverage、唯一值 cardinality、answer sanity、evidence verdict 和 postflight | 不通过则不发布答案；允许的机械缺陷可带反馈重规划，并完整重跑 ground→execute→check | `verifier.py:38–155`；`executor.py:12–133`；`pipeline.py:184–446` |

这七层中，G1–G5 决定“模型表达的语义如何落到 KG”；G6–G7 是执行器的安全契约。后两层原则上不应完全交给模型学习，因为它们可以由代码稳定保证。

## 3. 两个真实反事实例子

### 3.1 Grounding 把不可执行查询修成正确答案

保存样例 `sum_additive_year_category_0606` 中，planner 的原查询没有声明金额行必须可加总。直接执行原查询得到：

```text
pre-ground status = incomplete_evidence
pre-ground answer = null
```

G6 加入隐藏约束 `value_is_additive = true` 后，同一执行器、同一冻结 KG 得到：

```text
grounded status = passed
grounded answer = 1,532,262,913.14
oracle          = 1,532,262,913.14
```

这说明 grounding 可以把一个语义基本正确但不满足执行安全契约的计划救成 oracle-consistent 答案。它不证明金额具有现实世界的货币一致性：当前 KG 的 additive 行仍可能混有非 GBP 币种，因此该实验只能声称“与当前 benchmark oracle 一致”。

### 3.2 Grounding 把不可执行查询修成了错误答案

保存样例 `nlu_decomp_03_buyer_goods_total` 使用不存在的 answer field `contract_value`。G6 将它映射并强制为 `value_amount`：

```text
pre-ground: schema_error, answer = null
grounded:   passed,       answer = 67,103.42
oracle:                         667,103.42
```

因此：**executability ≠ semantic correctness**。字段 grounding 修复了接口错误，却无法补回 planner 已经遗漏或误解的过滤语义。这一失败例应该进入论文，而不是只展示 grounding 的成功案例。

## 4. 已完成的无模型反事实

脚本 `scripts/analyze_grounding_impact.py` 在三组保存 trace 上执行 `pre_ground_spec` 与 `grounded_spec`。两者使用同一个当前执行器和同一个 `data/kg`，没有重新调用 understander/planner/LLM。

运行命令：

```bash
.venv/bin/python -B scripts/analyze_grounding_impact.py \
  --counterfactual \
  --output paper/tmlr/grounding_impact.json
```

初步结果（重新生成 JSON 时以其中数值为准）：

- 共重放 140 条，其中 136 条有非空答案 oracle，可用于低层答案 exact-match 反事实；无 oracle 的 unsupported/ambiguous 行不能只靠 query spec 重放其完整拒答策略。
- grounding 前 89/136 正确；grounding 后 103/136 正确，绝对提高 10.29 个百分点。
- 18 条保存记录的 spec 被 grounding 修改：14 条从不正确/不可执行变为正确，3 条修改前后都正确，1 条变得可执行但仍错误；没有观察到 correct→incorrect。
- 17 条修改是给 `sum` 增加 `value_is_additive` guard：14 条被救成正确，3 条原本已正确。
- 1 条同时进行了 `contract_value → value_amount` alias 和 sum answer-field 强制：只修复可执行性，没有修复答案。
- 当前代码重放与历史保存 matcher 有 1 条 answerable 记录不一致；论文需把结果绑定到该分析脚本、当前代码状态和生成的 JSON，而不能把历史 summary 与当前 replay 混称完全复现。

这组数据只能估计 **G6 的局部处理效应**，因为现有 trace 几乎没有保存每个 G1–G5 子步骤的完整 before/after state。下一步需要增加统一 grounding event log，才能逐层归因。

## 5. Grounding 怎样进入后续模型优化

当前数据路径是：

```text
model proposal
  → schema/entity/compile/ground/execute checks
  → structured failure feedback
  → successful repaired graph
  → repair-SFT target and/or chosen–rejected preference pair
```

代码与产物显示三种影响。

### 5.1 Canonicalization 改写监督目标

teacher 导出优先取 compiler-normalized graph，而不是原始模型文本。因此后续 SFT 学到的是 canonical slot、规范依赖和可执行 graph，而不是把模型原始错误照抄为 target。实现见 `scripts/run_teacher.py:72–91`。

### 5.2 Grounding 失败成为 repair supervision

`data/qa/teacher_full_v1` 的现有产物包含：

- 9,267 条 trace；
- 6,860 条通过 runtime verification；
- 5,605 条同时 oracle match；
- 5,598 条最终 plan SFT，且其中没有 `oracle_match != true` 的记录；
- 1,725 条 repair-SFT，其中按 failure feedback 可识别出 237 条 grounding-related successful repair：216 条 semantic/schema、17 条 runtime-spec、4 条 entity grounding；
- 390 个 DPO pair，其中 242 个标记为 oracle-gated repair；
- 1,262 个 hard negative，590 个 abstention SFT。

这证明 grounding **确实改变了后续训练数据的组成**。但“237 条进入训练”不是“这 237 条带来多少准确率提升”的因果证据。

### 5.3 确定性 guard 不必转嫁给模型

例如 `value_is_additive` 是运行时自动注入的安全 guard。保留这种职责边界可以让模型专注于问题语义与组合结构，同时由代码保证稳定的执行约束。论文应区分：

- **希望模型学会的 grounding**：正确 field/role、entity mention、bridge dependency、return signature；
- **始终由 runtime 强制的 grounding**：additivity、exhaustive retrieval、deduplication、sort/group keys、release checks。

### 5.4 必须区分“首次规划”与“收到反馈后的重规划”

当前导出逻辑把两种条件分布放进同一次 SFT：

```text
plan-SFT:   p(plan | question, schema context)
repair-SFT: p(repaired plan | question, failed plan, hard-detector feedback)
```

237 条 grounding-related repair 中，按当前 2% validation hash 和 plan-SFT family/bucket cap 重建：

- 230 条进入 repair-SFT train，7 条进入 repair validation；
- 166 条同时进入普通 plan-SFT train，另外 4 条进入 plan validation；
- 因而训练集中有 230 个相关问题、396 次样本行暴露；166 个问题以“直接规划”和“反馈修复”两种 prompt 重复出现；
- teacher DPO 的 390 个 pair 中，只有 17 个 ID 与这 237 条 grounding repair 重合。

所以，现有训练并不是一个单一的“grounding data treatment”。166 条双重暴露可能改变 first-pass plan；其余 repair-only 样本主要训练收到 failure feedback 后的条件策略。DPO 的变化也不能笼统归因于 grounding pair。

还必须区分两个不同强度的结论：

- **完整 hard-filtered curriculum 的效果**：plan-SFT 和 repair-SFT 都来自执行、类型、grounding、oracle/shape 等筛选。zero-shot 与 SFT 的同底座比较可以支持这一整体训练方案有效。
- **grounding-specific repair 的边际效果**：当前 230 个 grounding repair train questions 对应 396 次训练行暴露，只占 2,787 条 plan-SFT train + 1,679 条 repair-SFT train 中的一部分。没有移除/替换这 396 行的 matched checkpoint，就不能把整体提升折算为它们的贡献。

现有 260 题同题 paired comparison 对第一条结论给出强证据：

| 同底座阶段 | 之前→之后 | 错→对 | 对→错 | 净增 | exact McNemar |
|---|---:|---:|---:|---:|---:|
| Qwen zero-shot→SFT | 183→211 | 31 | 3 | +28 | 7.66e-7 |
| Llama zero-shot→SFT | 156→216 | 63 | 3 | +60 | 1.30e-15 |

因此可以严谨地说：**包含 hard-filtered plan 与 repair supervision 的 SFT 显著改变并改善了两个 8B 底座的决策。** 但由于 SFT 同时包含干净 verified plans、非-grounding repairs、grounding repairs 和 abstention examples，这还不是“grounding repair 单独贡献 +X pp”。

后续阶段也不是单调改善：Qwen SFT→RSFT 只有净 +1（19 错→对、18 对→错，p=1.0），RSFT→DPO 净 +5（p=0.50）；Llama RSFT→DPO 净 -14（4 错→对、18 对→错，p=0.0043）。这表明 repair/preference 数据会重排 plan 决策，而“加入更多硬检测衍生数据必然更好”并不成立。

### 5.5 现有学生产物显示的决策变化（描述性、非因果）

在这 237 个曾触发 teacher grounding repair 的**同题训练样例**上：

| 保存的学生 harvest | 首次即成功 | 最终 verified | 最终 oracle match | grounding failure 再次出现后被修复 |
|---|---:|---:|---:|---:|
| Qwen SFT | 54/237 | 216/237 | 204/237 | 140/237 |
| Llama SFT | 54/237 | 227/237 | 217/237 | 151/237 |
| local Step-1 + Qwen DPO | 125/237 | 227/237 | 207/237 | 102/237 |

这里最值得研究的不是总体数值，而是行为分解：普通 SFT 后仍有大量 plan 先触发同类硬失败，但系统通过 feedback-conditioned replan 获得较高最终通过率；换成本地 Step-1 + DPO 后 first-pass success 增加、repair 次数减少，但 oracle match 没有同步单调增加。由于这三组同时改变了 Step-1、planner checkpoint、采样和优化目标，而且评估题就是训练题，这些结果只能证明“行为与预期机制相容”，不能作为 grounding 数据的因果增益。

在另一个 260 题保存的 checkpoint matrix 上，Qwen zero-shot/SFT/RSFT/DPO 为 70.38%/81.15%/81.54%/83.46%；bridge join 则为 10%/40%/80%/55%。这种非单调变化进一步说明不同训练阶段在重排 plan preference，而不是统一提高所有能力。该 matrix 没有逐题 plan trace，不能据此判断 bridge 下降是否来自 grounding 决策。

### 5.6 Hard acceptance 也可能生成 shortcut supervision

对 237 个 repaired target 与源 benchmark gold constraints 做保守静态对齐：229 条可评估，173 条覆盖全部可对齐约束，56 条需要人工复核；整体 constraint recall 为 85.85%。最明显的是 79 条含 `tender_category` 的题中，有 43 个 accepted target 没有保留 category literal。

例如问题明确要求 `goods + CPV 34144900 + year 2025`，accepted target 只保留 CPV 和 year，却仍在冻结 KG 上得到相同 oracle。这个 target 通过了“可执行 + oracle answer”筛选，但在 intensional semantics 上是不完整的。它可能教模型利用当前 KG 中 CPV 与 category 的相关性，而不是忠实保留问题约束。

这里已经定位到一条具体实现原因，而不只是模型随机漏词：repair prompt 确实包含完整 question、failed graph、failure feedback 和 retrieved schema。`plan_consistency_check` 也会发现缺失的 year/CPV/category，但把这三类标为 deterministic-autocompletable。`compile_typed_plan` 随后把缺失 category 补进 `candidate.query_spec`，却没有同步写回 `candidate.graph_plan`。pipeline 看到 graph plan 后优先走 `execute_graph_plan`，因此真正执行和 `_graph_payload_of` 导出的仍是未补 category 的 graph。

对 `coverage_fixed_0141` 的当前代码重构直接显示：

```text
candidate.query_spec: cpv=34144900, year=2025, category=goods
candidate.graph_plan: cpv=34144900, year=2025
pipeline authority:   graph_plan
```

冻结 KG 上 `cpv + year` 与 `cpv + year + goods` 都恰好返回 9 条，而且前者的 distinct category 只有 `goods`，所以 answer-only oracle gate 无法发现这次语义遗漏。teacher exporter 最后保存的是 compiled graph 的 `raw_graph_plan`，遗漏也就进入了 plan-SFT/repair-SFT。

因此 future curation 至少需要双 gate：

1. **extensional gate**：执行结果与 oracle 一致；
2. **intensional gate**：题目/gold 中不可省略的 constraint、role、dependency 和 return signature 在 normalized plan 中仍然存在。

第二个 gate 失败的样本不应直接删除：实体 canonicalization 和等价 graph rewrite 可能造成表面差异；应进入 review/normalization 队列，而不是继续作为 verified target。

## 6. 真正隔离“对模型优化的影响”的最小实验

不能直接比较现有 SFT、RSFT、DPO 总分，因为数据量、目标、筛选和训练过程同时变化。实验必须把 first-pass planning 与 feedback-conditioned replanning 分开。建议用相同 base checkpoint、相同 token budget、相同训练步数和相同 PACS split，先做一个 2×2 数据消融：

| Arm | 训练数据 | 回答的问题 |
|---|---|---|
| M0 | clean canonical plan-SFT；排除 grounding repair IDs | 基础规划能力 |
| M1 | M0 + grounding targets，以普通 plan prompt 暴露 | 是否改善首次 plan 决策 |
| M2 | M0 + 同量 grounding repair prompt，仅在 feedback 条件下暴露 | 是否主要改善 replan，而非 first pass |
| M3 | M0 + M1 与 M2，两类样本各自降采样，保持总 token 数相同 | 两种监督是否互补 |

然后在相同 M3 checkpoint 上增加一个 grounding-only DPO arm M4，检验 preference optimization。每个 arm 还需一个等量、同 family、同 plan depth 的 non-grounding repair control，排除“只是多看了 repair 数据”的解释。

主要指标不能只报 overall accuracy，应报告：

1. schema-slot exact match；
2. buyer/supplier role accuracy；
3. entity-link accepted / ambiguous / unresolved；
4. graph compile rate；
5. dependency/bind correctness；
6. pre-ground executability；
7. grounded executability；
8. exact execution accuracy；
9. correct abstention；
10. `grounding rescued / unchanged / degraded / executable-but-wrong` 转移矩阵。

每个 checkpoint 要跑两个互不混合的 protocol：

- **first-pass protocol**：只给原问题/schema context，禁止 feedback 和 repair；
- **replan protocol**：给同一组 held-out rejected plans 和完全相同的结构化 detector feedback，只评估一次修复输出。

两者之差才能回答硬检测训练数据究竟改变了初始 decision boundary，还是只训练了一个更好的 error-conditioned repair policy。

若要进一步回答“模型内部更偏好哪个 plan”，对 held-out chosen/rejected plan pair 做 teacher-forced likelihood，而不只看生成后的最终正确率：

```text
decision_margin(x) = mean_logp(corrected_plan | x)
                   - mean_logp(rejected_plan  | x)
```

分别令 `x` 为普通 planning prompt 和带相同 hard feedback 的 repair prompt，并在 base/SFT/RSFT/DPO checkpoint 上比较 margin。再把 token loss 分解到 canonical slot、operator、dependency edge、return operation 和 literal value，就能直接看到训练把概率质量从哪个错误决定移向哪个正确决定。当前迁移后的 `outputs` 保留了评估 JSON，但主 planner adapters 不在该 target 中，因此现阶段只能做 plan/answer transition 审计；完成 likelihood 分析需要恢复对应 adapters，或在未来 matched ablation 训练时同步保存。

分析时同时报告 paired bootstrap 置信区间和同题 McNemar 检验。若资源只允许一次轻量探索，可先跑 M0/M2；但论文中的因果结论最好用至少 3 个随机种子，否则明确标为 exploratory。

## 7. 需要补充的 trace 字段

为了逐步回答 G1–G7 的贡献，建议每次运行追加统一的 `grounding_events`，每条 event 至少包含：

```json
{
  "stage": "schema|entity|graph_bind|runtime_spec|preflight|postflight",
  "target": "A.filters[0].field",
  "before": "contract value",
  "after": "value_amount",
  "decision": "accepted|rewritten|rejected",
  "confidence": 0.93,
  "alternatives": ["value_amount", "value_currency"],
  "reason": "alias_and_type_gate",
  "source_variable": null,
  "affected_variable": "A"
}
```

对于 graph binding 还应保存：上游 variable ID、emit field、输出 cardinality、展示用的少量值、下游 bind field 和注入后的约束。这样既能重建 Figure 1 的真实中间状态，也能做逐层 knockout，而无需再次调用模型。

## 8. 论文中怎样呈现

正文 Method 用一张 grounding 表说明 G1–G7 的输入、输出、规则和失败语义；完整 alias、slot、阈值和 operator 规则放附录。Results 的精确数字优先用表格：行是 grounding intervention，列是 changed、rescued、degraded、executable-but-wrong 和 exact accuracy。另放一张两案例图：左边是 additivity guard 的成功反事实，右边是 field alias 只修可执行性但答案仍错的反例。

训练影响另用一张 matched-ablation 表报告 M0–M3，并按 schema/entity/binding failure family 分解。不要把 oracle match 称为在线 verifier，也不要把“执行成功”称为“事实正确”。

## 9. 当前可以与不可以声称的结论

可以声称：

- grounding 是跨语义 schema、实体值、图依赖和运行时安全契约的分层过程；
- 在当前 136 条有答案 oracle 的保存计划重放中，执行时 grounding 将 exact correctness 从 89/136 提高到 103/136；
- grounding repair 明确进入了后续监督数据，现有 repair-SFT 中可识别 237 条相关成功修复。

暂时不可以声称：

- grounding-derived training examples 导致某个 checkpoint 提高了多少；
- verified/executable 等同于 semantic correctness；
- additive guard 已解决跨币种金额语义；
- 当前 140 条局部 trace 能代表完整 benchmark 或完整 G1–G7 分布。
