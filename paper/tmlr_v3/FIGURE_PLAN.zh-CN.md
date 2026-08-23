# TMLR 初稿图表说明：供手工绘图

这份说明以当前代码和保存产物为准。正文目标是 12 页，图只解释难以用文字迅速看懂的结构；精确比较尽量用表。全文不要使用内部项目昵称，也不要把完整采购运行时、PACS 单次代数规划器和 WTQ 表格适配器画成同一条路径。

## 正文建议保留的图表

### Figure 1（Introduction，通栏）：一个真实完整运行时样例

作用：让读者先看懂系统为什么不是普通 RAG，以及理解器、条件规划器、grounding 和确定性执行分别做了什么。版式借鉴 Chain-of-Table (ICLR 2024) Figure 1 的叙事方法：同一个问题贯穿全图，真实中间状态占主视觉，操作名放在状态之间。不要照搬它的颜色和 table-specific 操作。

在画布顶部放一个真实的多跳采购问题。候选必须来自保存的 `ReasoningPipeline` artifact，而不是 PACS 输出或手工 oracle 执行。当前已经核实的候选是：

> Record check: how many contract notices were published by buyers that have awarded a contract to Dodd Group (Midlands) Limited?

- ID：`L2::bridge_join_0502#L2a`。
- 保存的 Step-1 intent 经过当前确定性代码重放后走 **fast path**，没有调用 Step 2；因此这张图可以展示理解器和条件规划，但不能把它画成“实际使用了 graph planner”。
- 真实中间数量：21 条 source records → 16 个 buyer values → 1,940 条 target records → 答案 1,940。
- 离线 oracle match 为真，没有 repair；在线图中不能把 oracle 画成 verifier。
- release 带有限制：1,940 个匹配记录的前 200 条 evidence sample 中有 8 条没有 supplier field。
- 来源和限定写在 `figs/src/fig01_runtime_bridge.manifest.json`。它是“保存的 Step-1 artifact + 当前确定性重放”，不是一次新跑的 final-test 模型调用。

如果你希望 Figure 1 的实线一定经过 Step 2 planner，就不要使用这个 Dodd Group 候选；应另选 `planner_source` 明确为 Step 2、保存了 graph variable trace 的样例，再填同一版式。不要为了让系统显得更完整而把未调用的 planner 接进本例。

首选形状是：

`anchor buyer -> its supplier set -> other records bound by that set -> target buyer set or count`

画面从左到右：

1. `Question and schema slice`
   - 完整问题，细下划线标 anchor、bridge phrase、target、return operation。
   - 下方只列理解器实际看到的 4--6 个字段，不画数据库图标。
2. `Structured understanding`
   - 用五行窄表展示保存的 Step-1 输出：answer type、source role/filter、bridge role、target filter、return。
   - 若该 trace 的 Step 1 可直接编译，后面写 `fast path`；若被 veto 或无法编译，才进入 Step 2。
3. `Conditional typed planning`
   - 画一个很小的真实分支：`compilable and no veto?`
   - 实际走的分支用实线；未走分支灰色虚线。
   - 如果走 Step 2，展示实际 variable DAG，而不是伪造自然语言思维链：`R0: records -> S: supplier values -> R1: bound records -> return`。
4. `Grounded execution`
   - 这是图中最大的部分。每个变量显示真实 grounded constraint、output size、2--3 个实际值和省略号。
   - 箭头写具体动作：`emit supplier_name`、`bind supplier_name IN S`、`distinct buyer_name` 或实际 operator。
   - 明确 `complete matched population` 与 `evidence sample` 不同。
5. `Release`
   - 四行即可：execution status、population count、evidence verdict、postflight/sanity。
   - 右端写真实 answer card；失败出口统一写 `No verified answer`。

图注必须说明所有实体、计数和分支来自同一 trace；`verified` 只表示运行时检查通过，不表示离线 oracle 证明；evidence IDs 是 contract-level provenance，不是 source-document citation。

这张图只画一个实际分支。未走的 compiler/planner 分支可用细灰线标 `not taken`，但不要放该分支的虚构中间产物。若换候选，只需补跑一个单例并保存 manifest，无需重新跑批量实验。

### Figure 2（Method，通栏）：在线运行时与离线训练构造

作用：给复现者一个没有例子干扰的总算法图。参考 RoG (ICLR 2024) Figure 2 对 planning/retrieval/reasoning 的职责划分，以及 LEVER (ICML 2023) 对 execution result 和 verifier 的分离。

上半部标 `Online inference -- no oracle`：

`Question + schema slice`
→ `Step 1 typed intent`
→ `deterministic compiler` 或 `Step 2 typed graph planner + consistency check`
→ `hard grounding + preflight`
→ `deterministic graph execution`
→ `postflight + answer sanity + evidence verdict`
→ `Answer card` / `No verified answer`

只画一条虚线反馈：从可修复的结构化失败回到 planner，写 `bounded replan; rerun all deterministic stages`。不要让 oracle 连进这条线上。

下半部标 `Offline curation -- hidden oracle available`：

从已执行 trace 向下连接四个条件：runtime checks、answer shape、typed oracle match、exportable graph。再分到：

- direct plan SFT；
- repair SFT（failed plan + feedback → repaired plan）；
- DPO（accepted > rejected）；
- hard negative；
- abstention target。

在学生训练输入前明确写 `oracle value removed`。这张图要回答一个重要问题：硬检测不是一个可微 loss；它通过接受、修复和拒绝样本改变监督分布。

### Figure 3（Method，小型三面板）：数据、KG 与 provenance

作用：覆盖原始项目要求中的 ETL、schema、ER 和 evidence，不让论文看起来只做了模型微调。参考 TheyBuyForYou (Semantic Web 2022) 的 procurement data integration 图，但缩到本项目真正实现的范围。

- `(a) Snapshot`：yearly OCDS JSONL → latest dated release per OCID；旁边显示 exact `(ocid, release_id)` 防旧版本回流。
- `(b) Resolution`：official ID → conservative FTS alias → government/name+region/name-only → unresolved singleton；fuzzy edge只到 review，不直接 merge。
- `(c) Property graph`：buyer → contract-award ← supplier，contract → CPV，contract → evidence；evidence 节点回到 source field/document metadata。

图内可放当前规模：215,221 contracts、131,502 organisations、3,870 CPV、535,731 evidence。注明 Parquet-backed property graph，不写 RDF、SPARQL 或完整 document inspection。

### Results 表格，而不是把精确数值都画成图片

1. `Main procurement table`：主列 n=2,025，旁列完整 n=2,285；行按 closed-book / RAG / teacher / hybrid / fully-local 分组。
2. `Development ladder table`：260 题只标 development diagnostic；Qwen/Llama 分块，列 correct、accuracy、wrong→right/right→wrong 和 McNemar p。
3. `Composition and transfer table`：上半 PACS 922，下半 WTQ 4,344。caption 明确两个 protocol 不可直接比较。

若正文页数允许，只增加一张结果图：按 13 个采购能力 bucket 画 `KG-only vs fully-local Qwen` 的 paired dot plot，并在每行写 n=20。它比总体柱状图更能说明 learned planner 扩展了 bridge/set/top-k 等能力。不要用 pie chart、雷达图或没有样本量的 heatmap。

## Appendix 适合放什么

- 完整递归代数 operator/type 表和 graph-plan JSON schema。
- Step-1、Step-2、repair prompts 与 provider response schema。
- ER 规则、字段缺失、currency/additivity 和 evidence fallback 审计。
- benchmark 13 buckets、split lineage、exact overlap/dedup audit。
- PACS 7 families × L1/L2/L3 × seen/unseen 的精确数表及 v1.1 重评分说明。
- WTQ official evaluator 命令、四个 adapter 的数据来源、官方逐题 McNemar 计数。
- fast path、Step-2 bridge、successful repair、unsupported、ambiguous、no-result 六条完整 trace。
- grounding 固定计划回放 89→103/136、训练目标约束完整性 173/229；它们是可靠性审计，不是主论文故事。
- 全部 source/data/checkpoint/result hash 和命令。

## 可以直接借鉴的论文图，及图里具体画了什么

- **[Chain-of-Table, ICLR 2024, Figure 1](https://openreview.net/pdf?id=4L0xnS4GQM)**：同一问题与真实表放左边；上方压缩展示 generic reasoning 和 SQL 的错误；下方展开 operation selection、arguments、intermediate table、operation history和正确答案。应借它的“共享输入 + 可见中间状态”，不是照搬配色。
- **[CABINET, ICLR 2024, Figures 1--2](https://openreview.net/pdf?id=SQrHpTllXa)**：Figure 1 用同一表对比 hard sub-table 丢信息和 cell relevance 保留信息；Figure 2 的八步方法仍然在每步放真实 table/question artifact。可借局部数据高亮。
- **[KQA Pro, ACL 2022, Figure 1](https://aclanthology.org/2022.acl-long.422.pdf)**：微型 KG、自然语言问题、SPARQL 与 KoPL tree 对齐。可借“语言—图路径—程序”三者对应；完整 operator 清单应留附录。
- **[RoG, ICLR 2024, Figure 2](https://proceedings.iclr.cc/paper_files/paper/2024/file/3e2aeb66481dd63a32421bf032b70384-Paper-Conference.pdf)**：question、relation-path plans、retrieved reasoning paths 和 answer 围绕真实 KG 排布，并区分 training 与 inference 箭头。可借职责边界和真实 relation path。
- **[GrailQAbility, ACL 2023, Figure 1 / Table 3](https://aclanthology.org/2023.acl-long.576.pdf)**：图用实际缺失 schema/fact 定义多种 unanswerability；主表把 answerable 与 unanswerable 单独列组。可借 abstention 的任务定义方式。
- **[PICARD, EMNLP 2021, Figure 2](https://aclanthology.org/2021.emnlp-main.779.pdf)**：用一个 decoding step 显示哪些 candidate token 被 parser 接受或拒绝。可借“硬约束对具体候选做什么”，不必画成大框架图。
- **[QueryAgent, ACL 2024, Figure 2](https://aclanthology.org/2024.acl-long.274.pdf)**：完整 thought/action trace 中标出环境反馈与 correction。可借结构化失败→有限修复，但只有真实发生 repair 的 trace 才能这样画。
- **[TheyBuyForYou, Semantic Web 2022, Figures 3--4](https://doi.org/10.3233/SW-210442)**：多采购数据源、ETL、triple store、API 与每日 ingestion。只适合数据方法图，不适合 Introduction hero figure。

## 统一视觉规范

- 白底、黑/深灰文字；蓝色只表示 learned planning，青色表示 deterministic data/execution，绿色只用于最终 release，琥珀表示 non-answer。
- 不使用产品 logo、机器人头像、盾牌、渐变、阴影、营销口号或大量圆角卡片。
- 最终宽 6.5in 的 PDF 中，正文文字至少 7.5pt；线条至少 0.5pt；所有字体嵌入。
- 图内模块名简短，真实 state/table/set 占面积；caption 放 LaTeX，不塞进图片。
- 每个数字都能回到 trace 或 artifact ledger；无法追溯的内容用占位符，不凭印象填写。
