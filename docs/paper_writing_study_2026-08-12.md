# 代码事实驱动的论文写作研究笔记（2026-08-12）

> 本笔记只把当前最终代码、现有数据与可复核结果作为本项目事实来源；没有读取或继承 `docs/paper_draft_v1.md` 的论点、结构或数字。相关工作部分只学习论文如何建立问题、组织论证和公开方法，不用其写法替代本项目证据。后续为确认系统名称进行的一次文件名搜索意外在结果中显示了其他旧 thesis 文件的若干匹配行；搜索已停止，这些片段没有被用于本笔记的论点、结构、数字或措辞。

## 1. 先确定论文是什么，而不是先填章节

### 1.1 建议的核心研究问题

当前项目最连贯、也最接近顶会论文写法的主问题是：

> **如何在异构、含噪且可能证据不足的公共采购数据上，把自然语言问题转成可执行的组合式图查询，并用确定性执行与验证给出可追溯答案或可靠拒答？**

这比“做了一个 procurement KG”“微调了一个 LLM”或“比较了若干模型”更适合作为主线，因为它能把项目中原本分散的工作串成一个因果链：

1. 原始 OCDS 数据存在来源、结构、实体名称、角色和币种等不一致。
2. 摄取、实体消歧和表格抽取把它们整理成带 provenance 的查询底座。
3. 类型化的组合代数表达过滤、集合、连接、聚合、比较和布尔判断，而不仅是检索几条相似文本。
4. planner 生成计划；grounding、compiler、executor 和 verifier 逐层排除无效或证据不足的计划。
5. 系统只从实际执行结果生成答案，并在前置条件失败时拒答。
6. verifier 接受的轨迹可以作为本地 student 的训练数据，形成部署扩展，但它应是主方法的延伸，而不是掩盖主问题的第二篇论文。

### 1.2 建议的一句话定位

可以先用下面这句作为全篇的“北极星”，以后每一段都检查是否服务于它：

> We study verifiable compositional question answering over noisy public-procurement data, where a system must not only retrieve relevant records, but also exhaustively execute typed set, join, aggregation, and comparison operations and abstain when their evidential preconditions are not met.

这句话目前只是写作定位，不是最终摘要；最终版本必须根据冻结后的代码和重评分结果调整。

## 2. 顶会论文的 Abstract 实际在完成什么

对 KQA Pro、QueryAgent、RoG、GrailQAbility、StructGPT、PICARD 等论文的摘要逐句拆解后，最稳定的结构不是“背景—方法—结果”三个大块，而是六个连续功能：

1. **任务与重要性**：一句话告诉读者研究什么。
2. **精确缺口**：指出现有方法在哪一种可观察条件下失败，不能只写泛化的 “LLMs hallucinate”。
3. **核心主张**：一句话说出本文的关键 insight。
4. **机制**：用一至两句说明系统怎样实现该 insight。
5. **证据**：说明在哪些数据、对照和指标上检验，并只给可复核数字。
6. **意义**：解释结果为什么改变读者对这个任务的认识，或公开了什么资源。

### 2.1 可直接套入本项目的摘要槽位

在评分器与结果审计完成前，摘要应该保留数字占位符，不应把历史文档中的百分比直接抄入：

> Public procurement records could support evidence-based questions about suppliers, buyers, contracts, and spending, but answering compositional questions over these records requires more than retrieving a few relevant passages. Existing language-model and retrieval pipelines can produce plausible answers without establishing that all required records were retrieved, correctly joined, and aggregated. We present **[system name]**, a verifiable procurement-QA framework that integrates noisy OCDS records into a provenance-preserving graph and maps questions to a typed algebra of set, join, aggregation, comparison, and Boolean operations. The framework grounds and compiles each plan, executes it deterministically, verifies structural and evidential preconditions, and abstains when they fail; accepted execution traces additionally supervise a local student model. On **[frozen datasets]**, the framework **[verified main result]**, while **[ablation/baseline]** shows **[verified contrast]**. These results indicate **[narrow implication supported by evidence]**.

这个版本暂时不能定稿，原因有三：

- compose 的部分历史结果受到 Python `bool`/`int` 等价评分问题影响；
- 某些金额问题混入非 GBP 记录，问题文本与执行语义尚需版本化；
- 不同数据版本的 split、重复与 artifact 完整性需要在实验表旁明确披露。

### 2.2 从代表论文中学到的具体写法

| 论文 | 摘要的关键动作 | 本项目可借鉴之处 | 不能照搬之处 |
|---|---|---|---|
| [KQA Pro, ACL 2022](https://aclanthology.org/2022.acl-long.422/) | 把“复杂问题”拆成具体组合技能，再给数据规模、程序语言和诊断结果 | 明确列出 join、set、aggregation、comparison、Boolean 等能力；说明代数为何可解释、可执行 | 不能把合成模板规模本身当作方法贡献；必须证明采购问题与数据条件真实存在 |
| [QueryAgent, ACL 2024](https://aclanthology.org/2024.acl-long.274/) | 从可靠性与效率两个明确失败机制推出环境反馈式 agent | 用执行错误、类型错误、空结果和证据条件驱动修复，而非笼统声称“self-correction” | 不能把任意重试都称为 verifier-guided repair；要定义触发条件、接受条件和停止条件 |
| [RoG, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3e2aeb66481dd63a32421bf032b70384-Abstract-Conference.html) | 用 planning–retrieval–reasoning 三段式解释图上推理 | 先给总体分解，再展开各模块；把 plan 与 evidence retrieval 的关系写成形式化过程 | relation path 不能自然代表本项目的分支、连接和精确聚合，差异必须写清楚 |
| [StructGPT, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.574/) | 先定义结构化数据接口，再说明迭代读取与推理 | 将“访问数据”和“得出答案”分开描述，统一 KG/table/backend 接口 | 迭代读取少量证据不等于穷尽式 COUNT/SUM，不能弱化 retrieval-completeness 条件 |
| [GrailQAbility, ACL 2023](https://aclanthology.org/2023.acl-long.576/) | 从真实部署中的不可回答性推出 benchmark 与拒答评估 | 给 answerable/unanswerable 严格定义，区分空答案、缺字段、缺实体和执行失败 | 不能把所有空结果都等同为不可回答；要由 operator 语义和证据前置条件判断 |
| [PICARD, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.779/) | 用增量解析器拒绝无效输出 token，机制短而清楚 | 解释类型检查、语法验证为何减少不可执行计划 | 本项目不是 token-level constrained decoding；应准确写成 plan validation/compilation |
| [TheyBuyForYou, Semantic Web 2022](https://journals.sagepub.com/doi/10.3233/SW-210442) | 从采购数据碎片化问题推进到 ontology、KG pipeline、服务与用例 | 适合学习采购领域动机、数据源、ontology/KG 和摄取流程的组织方式 | 不能沿用其旧统计或把平台规模替代 QA 方法评估 |

### 2.3 最值得写清楚的理论直觉：相关检索不等于精确聚合

PullNet、UniKGQA、RoG 等方法主要优化答案实体、关系路径或相关子图的召回。对于实体型问题，取回一条包含答案的路径可能已经足够；但 `COUNT`、`SUM`、分组比较和排名要求完整匹配集合。若真实匹配集合为 `R(q)`，检索器只返回 `R_hat(q) ⊂ R(q)`，通常不能推出：

```text
|R_hat(q)| = |R(q)|
sum(value(r) for r in R_hat(q)) = sum(value(r) for r in R(q))
argmax(group_sum(R_hat(q))) = argmax(group_sum(R(q)))
```

因此，本项目不能只比较“是否检索到相关 evidence”，还应分别评估：

- candidate/record coverage；
- 完整集合上的执行正确率；
- incomplete retrieval 是否被 verifier 检出并拒答；
- 在相同 oracle plan 下，top-k retrieval 与 exhaustive backend execution 的差异。

这条论点能把系统和普通 GraphRAG/KGQA 区分开，但论文中只能声称当前实现真实提供的完整性范围；它不是对开放世界中“所有采购事实都已存在”的保证。

## 3. Introduction 不是加长版摘要，而是一条论证链

优秀相关论文的 Introduction 大多按下面的逻辑推进：

```text
现实需求
  → 精确定义任务
  → 现有范式为何在该任务下失败
  → 一个具体问题/初步事实暴露缺口
  → 本文的核心 insight
  → 系统如何实现它
  → 怎样验证
  → 贡献边界
```

### 3.1 建议的逐段布局

**第 1 段：现实任务，而非宏大口号。**

介绍公共采购记录能支持供应商集中度、买方支出、合同变化和程序合规等问题；随即指出，这些答案常需跨记录、跨实体角色和跨表做过滤、连接与聚合。不要在没有当代权威来源时写采购占 GDP 的具体比例。

**第 2 段：把失败条件写具体。**

解释普通 RAG 找到“相关记录”不代表已经找全 COUNT/SUM 所需记录；自由文本 CoT 也不能保证实体角色、时间范围、金额字段和聚合口径正确。使用一个真实可执行的 running example，让读者看到：缺一条记录仍能生成流畅但错误的答案。

**第 3 段：现有方法分成几类并逐类指出边界。**

- text/Table RAG：相关性强，但通常没有 exhaustive retrieval 保证；
- relation-path KGQA：适合多跳路径，但难以表达分支集合、join、arg-extreme 与组合聚合；
- LLM-to-program：程序可执行，但仍会遇到错误 grounding、无效类型、空证据和不可回答问题；
- procurement KG：解决数据集成与服务问题，但不自动解决自然语言到可验证组合查询。

这段是 related work 的“问题导向预告”，不要在 Introduction 堆论文名。

**第 4 段：一句 insight + 一张系统图。**

核心 insight 可表述为：可靠性来自把生成限制在一个类型化计划上，并让最终答案服从数据执行与显式前置条件，而不是服从语言模型的自然语言判断。随后用一段话给出 ingestion/ER/KG → plan → ground/compile → execute → verify/repair → answer/abstain 的完整流。

**第 5 段：说明训练扩展，但保持主次。**

解释 teacher 轨迹只有在计划形状、执行和 oracle/verification 条件满足时才进入训练池；SFT/过滤式训练/偏好训练如何复用这些轨迹。需要等代码审计完成后准确写 acceptance rule，不能先写成“仅 verifier 接受”。

**第 6 段：实验问题与主要发现。**

围绕研究问题组织，而非罗列模型：

- RQ1：端到端回答的正确性与拒答可靠性如何？
- RQ2：typed validation、grounding、verification/exhaustiveness 各自贡献什么？
- RQ3：经验证轨迹能否把能力迁移到本地 student？
- RQ4：系统错误主要来自数据、计划、执行还是语言生成？

只预告重算后成立的发现。

**第 7 段：贡献列表。**

贡献应是可检查的名词短语加结果，而不是“我们首次探索”：

1. 一个面向公共采购组合问题的、带 provenance 的数据/KG 构建流程；
2. 一套覆盖集合、连接、聚合、比较与布尔操作的类型化查询代数，以及确定性 compile/execute/verify/abstain 流程；
3. 一个用已验证执行轨迹训练本地模型的流程；
4. 一套区分答案正确性、可回答性、计划有效性和证据完整性的评估与误差分析。

若实验不能分别证明四项，贡献数应收缩；三项扎实贡献优于五项弱主张。

## 4. Methodology 应该清楚到什么程度

Methodology 的目标不是让读者知道仓库有哪些脚本，而是让读者可以回答四个问题：输入是什么、每一步怎样变换、每一步保证什么、失败时发生什么。

### 4.1 建议章节结构

#### 3.1 Task Definition and Notation

- 定义原始 release/record、entity、contract/award、evidence/provenance。
- 定义问题 `q`、数据图 `G`、计划树 `z`、执行结果 `r`、答案 `a`。
- 区分 answer correctness、plan validity、executability、answerability 和 evidence completeness。
- 给出系统目标：若前置条件满足，则 `a = verbalise(exec(z, G))`；否则输出 abstention，而不是由 LLM 猜答案。

#### 3.2 System Overview

- 放一张端到端图。
- 用一个 running example 贯穿后续全部小节。
- 每个模块只写输入/输出/保证，细节留到对应小节。

#### 3.3 Procurement Data Integration

按真实执行顺序写，而不是按文件顺序写：

1. source discovery 与 raw release ingestion；
2. schema normalization 与来源/时间 provenance 保留；
3. Phase-1 entity resolution：官方标识优先、基于同一 observation 的名称与角色约束映射、冲突保持 unresolved；
4. Phase-2 entity resolution：政府映射、名称/地区合并、无法规范化的实体保持 singleton；
5. award、contract、party、evidence 等规范表抽取；
6. 构建查询图/索引并报告最终冻结版规模与数据质量统计。

每一步都应给一个微型例子。例如同名组织在 buyer/supplier 两种角色下为什么不能仅凭名称合并；金额为零为什么不是缺失值。

#### 3.4 Typed Compositional Query Algebra

- 给 operator family 表：filter、project、set、join、aggregate、compare、arg-extreme、Boolean 等。
- 每个 operator 至少写：输入类型、输出类型、参数、执行语义、失败/空值语义。
- 说明程序为何是树而不只是 relation path：分支结果可以 join、intersect、compare 或 aggregate。
- 给 running example 的完整 plan tree，并逐节点标出中间结果。

#### 3.5 Planning and Grounding

- planner 接收的 schema/operator contract；
- LLM 或规则 planner 输出的结构；
- entity/field/value grounding 如何完成，候选冲突如何处理；
- schema/type validation 在执行前拒绝哪些错误；
- 若有 repair loop，写明错误反馈、最大轮数、接受与停止条件。

#### 3.6 Compilation, Execution, and Verification

- plan 如何编译到 backend 查询或代数 executor；
- 结果完全由 executor 得出，LLM 只负责计划或表述；
- verifier 检查哪些 shape、type、execution、oracle 或 evidence 条件；
- 对 COUNT/SUM 等操作，定义 retrieval-completeness/exhaustive gate，而不是把检索到的局部证据直接聚合；
- 定义异常、空集、缺字段、冲突和不可回答时的行为；
- 最终 evidence trace 怎样关联回 source record。

#### 3.7 Verified-Trace Distillation

- teacher 生成什么；
- acceptance/filter 的真实代码条件；
- positive/negative pair 如何形成；
- SFT、filtered/RSFT、DPO 各自消费哪种数据；
- train/dev/test 如何划分及去重；
- student 在推理时是否仍依赖 compiler/executor/verifier。

这一节必须披露 resume、重复样本和 split 的实际约束；尚未实现的事务性保证不能写进方法。

#### 3.8 Complexity, Safety, and Reproducibility

- 哪些步骤是确定性的，哪些依赖模型；
- 数据版本、配置、随机种子、模型/adapter、prompt 和 evaluator 如何固定；
- provenance、拒答和货币/单位处理边界；
- 代码、数据 manifest 与 artifact 的可用范围。

### 4.2 一个合格的 running example 应展示什么

最终论文宜选一个确实存在于冻结 benchmark、且能从现有代码完整重放的问题。理想例子至少包含两条分支和一个聚合，例如：

> “在 2024 年由某买方发布的服务类合同中，哪个供应商获得的合同总金额最高？”

展示顺序：

1. 原始自然语言问题；
2. grounding：买方实体、年份、类别、金额字段；
3. plan tree：filter → join contract/supplier → group/sum → argmax；
4. 每个节点的输入/输出类型；
5. executor 的中间结果计数；
6. provenance 对应的 source records；
7. verifier 的通过条件；
8. 缺少完整年份文件或发现混合币种时如何拒答/降级。

示例必须从真实冻结数据挑选并重放，不能为了图好看手写一个系统实际执行不了的程序。

## 5. Related Work 应按“差异轴”写，而不是逐篇摘要

建议分四组，每组最后一句落到本项目的明确差异：

1. **Compositional semantic parsing and program induction**：KQA Pro、PICARD，以及 text-to-SQL/program generation；差异是采购领域的脏数据、实体/角色解析与 answerability。
2. **KG retrieval and reasoning with LLMs**：RoG、Think-on-Graph、QueryAgent、StructGPT；差异是 branching typed algebra、穷尽式聚合和确定性执行，而不只是 relation paths 或局部子图。
3. **Unanswerable QA and verification**：GrailQAbility 等；差异是把拒答绑定到 operator-specific evidence preconditions 和 provenance。
4. **Rationale filtering and distillation**：[STaR, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html)、[Distilling Step-by-Step, Findings ACL 2023](https://aclanthology.org/2023.findings-acl.507/)、[DPO, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)；差异是训练信号来自可执行且经筛选的结构化轨迹，而不是只看自然语言 rationale。

其中，[Break It Down / QDMR, TACL 2020](https://aclanthology.org/2020.tacl-1.13/) 最适合学习如何给每个组合 operator 写类型签名、语义和例子；[GraphQ IR, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.394/) 最适合学习如何先定义 graph/query domain，再写 grammar、strong typing 和 compiler；[BINDER, ICLR 2023](https://openreview.net/forum?id=lH1PV42cbF) 则是“语言模型生成程序、符号系统执行确定性运算”的直接比较对象。需要明确：当前代码既不是 QDMR 的自然语言分解，也不是跨 SPARQL/Cypher/SQL 的通用 IR，更不是在执行期间任意回调 LLM API 的 BINDER。

## 6. TMLR/FYP 内容预算

项目现已明确要求使用 TMLR 模板。TMLR 官方没有 8 页限制，而是允许 “any length”；此前用 ACL 八页制约束写作密度的建议不再是本论文的格式规则。初稿可用下面的相对比例控制内容，最终仍应服从学校的字数或页数要求：

| 内容 | 建议占正文比例 | 必须完成的任务 |
|---|---:|---|
| Abstract | 单段 | 问题、缺口、方法、可靠数字、意义 |
| Introduction | 8–12% | 视觉论点、研究缺口、insight、贡献 |
| Related Work | 10–15% | 按差异轴定位，不逐篇复述 |
| Problem/Data | 10–15% | 任务定义、数据/KG、answerability |
| Methodology | 25–30% | 代数、训练、执行、验证及真实例子 |
| Experimental Setup + Results | 25–35% | RQ、主表、分解、ablation、错误分析 |
| Limitations/Ethics/Conclusion | 10–15% | 数据与部署边界、回答研究问题 |

核心方法、主结果和支撑中心 claim 的证据不能只放附录，因为 TMLR 明确提醒审稿人可以不阅读 appendix。完整 operator 表、prompt、额外数据统计和补充案例仍适合放在 references 之后的 appendix。

## 7. 官方 venue 要求对写法的直接影响

- [TMLR Author Guidelines](https://jmlr.org/tmlr/author-guide.html) 与[官方 LaTeX 模板](https://github.com/JmlrOrg/tmlr-style-file)：单栏 US Letter，正文宽 6.5 inch，没有固定页数上限；署名 FYP 但并非 TMLR 录用稿时应使用 `\usepackage[preprint]{tmlr}`，不能使用会声称 “Published in TMLR” 的 `accepted` 选项，也不应修改 `tmlr.sty`、页边距或字体。
- [ACL 2026 Main Conference](https://2026.aclweb.org/calls/main_conference_papers/)：long paper 正文 8 页、short 4 页；双盲；limitations、ethics 和 reproducibility 需遵循 ARR 要求。
- [ACL 2026 Student Research Workshop](https://2026.aclweb.org/calls/student_research_workshop/)：long/short/thesis proposal 分别有明确页限；要求在 references 前有标题明确的 Limitations，缺失可能 desk reject。
- [ICLR 2026 Author Guide](https://iclr.cc/Conferences/2026/AuthorGuide)：投稿正文最多 9 页、双盲；摘要必须真实且信息充分，因为还用于 reviewer bidding；强烈鼓励 reproducibility statement。
- [KDD 2026 Research Track](https://kdd2026.kdd.org/research-track-call-for-papers/)：8 页正文、双盲；前 8 页必须自洽，reviewer 无义务读附录；评价同时覆盖技术性、原创性、影响、执行、表达、相关工作、复现与伦理。
- [Semantic Web Journal author guidelines](https://www.semantic-web-journal.net/authors)：强调原创性、意义、表达与可复现实验，并要求尽可能提供稳定可访问的数据/软件与 README。

对当前毕设最实用的写法是使用 **TMLR 的正式单栏版式**，同时保持 ACL/ICLR 式的论证密度；TMLR 没有页数上限，不代表应把开发过程全部写入正文。

## 8. 写初稿前必须建立的 claim–evidence 表

| 候选主张 | 必需证据 | 当前状态 |
|---|---|---|
| ER 防止跨角色错误合并 | 规则定义、单元测试、冻结数据统计或标注样本 | 代码边界已修复并有定向测试；真实数据效果待整理 |
| 代数覆盖组合采购问题 | operator 规范、benchmark 分布、可执行示例 | 代码存在；需从最终实现生成规范与分布 |
| verifier 提高可靠性 | 无 verifier 对照、错误类型、answerable/abstention 指标 | 需统一 evaluator 后复算 |
| exhaustive gate 对 COUNT/SUM 必要 | 局部检索失败例、完整性条件、ablation | 需选择真实样例并跑轻量离线分析 |
| verified traces 改善本地 student | 同 split、同 scorer 的 base/SFT/filtered/DPO 对照 | 部分 artifact 存在；需核对可移植 adapter 与原始预测 |
| 系统答案可追溯 | trace 到 source record 的端到端实例 | 代码链路需选一个样例重放 |
| 系统能可靠拒答 | 严格 answerability 定义、coverage-risk 或 selective accuracy | 有相关代码/数据；结果需重评分 |

摘要中的每个数字、Introduction 的每个贡献和 Conclusion 的每个结论，都应该能回指这张表中的一行。

## 9. 当前建议与下一步

现在可以开始写真正的论文初稿，但不宜先“润色一个最终摘要”。更稳妥的顺序是：

1. 冻结主研究问题和一条 running example；
2. 从最终代码生成 Methodology v0，并让每一步都有输入、变换、保证、失败行为和例子；
3. 统一修正/重跑轻量离线 evaluator，建立可引用结果表；
4. 写 Introduction v0，使每个 gap 都由例子、代码事实或实验事实支撑；
5. 最后把已成立的论证压缩成 Abstract。

换言之，Abstract 虽然最重要，却应该最后定稿：先把它作为研究契约写成占位版，再由 Methodology 和 Results 反向约束每一句话。

## 10. 当前代码与产物能支持的论文事实底稿

这一节是为下一步写初稿准备的“事实边界”，不是从旧稿抽取的故事。

### 10.1 端到端方法链

1. **Ingestion**：流式读取压缩 JSONL，处理坏行/空行，扁平化 release，并按 OCID 保留最新日期版本及来源年份。核心实现为 `src/procurement_graph/ingest/loader.py`。
2. **Entity resolution**：官方 scheme 优先；FTS alias 只有在同一 observation 中名称与角色一致、且跨所有观察只对应唯一官方实体时才建立；其余记录依次经过政府映射、精确 name+region、精确 name 合并，模糊匹配只形成审阅候选。核心实现为 `src/procurement_graph/er/phase1.py`、`phase2.py` 和 `candidates.py`。
3. **Extraction and KG construction**：抽取 tender、lot、award、contract、bid statistics、document/evidence 等事实；金额来源优先级为 award → linked contract → tender fallback，只有 award/contract 来源标为 additive；随后构造 organisation、contract、CPV、evidence 节点及 buyer/supplier/category/evidence 边，并经过主键、引用、coverage、JSON 和 additivity 检查。核心实现为 `src/procurement_graph/extract/tables.py` 与 `src/procurement_graph/kg/*`。
4. **Typed compositional algebra**：程序为有界树，类型包含 records、values、groups、number、value、Boolean 和 ranking；当前实现覆盖 filter、values、count/size/sum/exists/select、argmin/argmax、groupby、top-k、set operations、arithmetic/comparison、semi/anti-join。静态验证与执行分别在 `src/procurement_graph/compose/algebra.py` 和 `eval_runtime.py`。
5. **Program-first training**：`scripts/build_compose_train.py` 生成可执行程序样本，只保留 runtime evaluator 与独立 evaluator 一致且非退化的样本。当前 compose-v3 文件有 12,414 条 train、253 条 validation；训练配置为 `configs/training/qwen3_8b_compose_sft_v3_qlora.yaml`。
6. **PACS evaluation**：七类问题、L1–L3 组合难度，先从 KG 挖 anchor，再经过类型、双 evaluator、非退化、surface 和 split/seal 流程；A 为独立模板表达，B 为受 literal/logic gate 约束的改写。模型单次、temperature 0、无 repair 地生成程序；答案由合法程序执行后与 oracle 比较。核心代码在 `scripts/pacs/*` 与 `scripts/run_compose_probe_eval.py`。
7. **Full CICADA runtime**：Step 1 产生结构化意图并 lint；可确定编译时走 fast path，否则 Step 2 产生 typed graph plan；随后 grounding、穷尽式确定执行、pre/post verification、evidence verdict 和 answer card；只有失败才触发次数有限的 feedback replan，且每次从头经过检查。干净 grounded no-result 终止而不做 answer shopping。核心实现为 `src/procurement_graph/reasoning/*` 与 `scripts/run_compare.py`。

### 10.2 可核对的系统规模

当前 Parquet artifact 的行数为：

- 131,502 organisation nodes；
- 215,221 contract nodes；
- 3,870 CPV nodes；
- 535,731 evidence nodes；
- 215,218 buyer edges；
- 334,063 supplier edges；
- 164,691 category edges；
- 1,326,240 evidence edges。

因此正文不能把 buyer coverage 四舍五入写成绝对 100%；应报告 `215,218 / 215,221`。节点与边来自 `data/kg/nodes/*.parquet` 和 `data/kg/edges/*.parquet`。

### 10.3 目前最适合支撑摘要的结果

PACS 的逐题输出与 summary 都在仓库中，四个同规模 `n=922` 的已存结果为：

| Arm | Correct | Accuracy |
|---|---:|---:|
| Base Qwen3-8B, surface A | 332 / 922 | 36.01% |
| Cloud teacher, surface A | 450 / 922 | 48.81% |
| Compose-v3, surface A | 687 / 922 | 74.51% |
| Compose-v3, surface B | 675 / 922 | 73.21% |

对应文件为 `data/qa/compose_probe_v1/summary_pacstest_{base_a,teacher_a,v3_a,v3_b}.json`。按已存 summary，Compose-v3 相对同底座 A 提高 38.50 percentage points，相对 cloud teacher A 提高 25.70 points；从独立表面 A 到改写表面 B 下降 1.30 points。

这些是当前最可用的 headline numbers，但发表前仍要：固定严格的 bool-aware matcher；完成币种/金额题审计；准确披露 PACS sealing 的 trigram 子采样；报告 family macro、置信区间与不均衡分布。因而现阶段可写进带版本标记的初稿，不应写成最终无条件结论。

### 10.4 可直接放进 Methodology 的真实例子

PACS 第一条 surface-A 样本问：

> Compare two slices: works notices during 2025 on one side, services notices during 2022, on the other. Does the first side hold more notices?

Compose-v3 实际生成的树为：

```text
combine(
  gt,
  count(filter(tender_category = works, release_year = 2025)),
  count(filter(tender_category = services, release_year = 2022))
)
```

执行结果为 `false`，与 oracle 一致。原问题在 `data/qa/pacs_v1/test_channel_a.jsonl`，逐题生成、树、答案和判定在 `data/qa/compose_probe_v1/eval_pacstest_v3_a.jsonl`。这能展示两个独立分支、类型化组合、确定性执行和 Boolean 输出；正式论文的主图最好再选择一个含 `groupby → sum → argmax` 且币种口径经过审计的例子，以进一步展示完整聚合。

### 10.5 暂不能升格为最终主张的结果

- CICADA compare-set 与 final-test 有较强已存结果，但 `outputs` 是指向本机 `/home/uceeh01/migrated_outputs` 的绝对软链接，summary 的 config/category 元数据也不完整；只能称 frozen artifact result，不能说当前 checkout 已完全复现。
- WTQ 当前内部结果不能替代 official evaluator 结果，不进摘要。
- Compose-v3 train 有精确重复，train/val 有两条完全重叠；PACS seal 的 trigram screen 实际对子集取样，不能写成严格穷尽的 near-duplicate proof。
- 当前 KG 的 additive 标记没有同时强制 `currency == GBP`，而问题文本常称 GBP；涉及金额总额的数字和 case study必须先审计。
- 当前工作树包含尚未重建 KG/结果的 correctness 修复，论文必须把每张表绑定到具体 frozen artifact、代码版本和 patch state，而不能统一称为“由当前最终代码重现”。
