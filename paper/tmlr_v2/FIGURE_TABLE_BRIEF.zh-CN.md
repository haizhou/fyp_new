# 论文图表绘制说明（供作者自行作图）

这份说明只规定“图要证明什么、里面放哪些真实内容、从哪里取数”。它不要求你照搬某篇论文的美术素材，也不把 direct-PACS 样例伪装成完整运行时。所有数字、实体名和中间状态必须能追溯到同一次保存的运行或同一个冻结 summary。

## 先说结论：正文以表格为主，图片承担结构和趋势

顶刊顶会并不是“结果大多画成图片”。常见分工是：

- **主结果、消融、数据统计：表格。**读者需要精确比较数值、分母和设置。
- **系统机制、单个 running example：图片。**读者需要看清状态如何变化、错误在哪里发生。
- **按深度/规模/K/延迟的有序趋势：折线或柱形图。**
- **family × depth 这样的二维诊断：heatmap；旁边或附录保留精确表。**
- **失败类型：互斥时用水平条形图或 100% stacked bar。**不要为了好看使用饼图或 sunburst；只有真正的层级 taxonomy 才适合 sunburst。

正文建议保留 4–5 张图、3–4 张表。完整 schema、operator、prompt、逐 family×depth 数值和完整 trace 放附录。

---

## Figure 1（Introduction）：一个真实完整运行时的视觉论点

### 它要证明什么

一个复杂采购问题不是直接交给 LLM 生成答案，而是经过：

```text
question
→ structured understanding
→ actual routing decision
→ deterministic compiler OR Step-2 typed planner
→ grounded variable DAG
→ exhaustive intermediate sets
→ evidence/release checks
→ answer or no verified answer
```

### 必须使用什么样例

如果你希望图里同时出现“理解器 + planner”，必须专门选择一个**当前完整运行时确实走 Step-2 planner 分支**的 bridge 问题。筛选条件：

1. `expected_status=answerable`，离线 oracle match 为真；
2. Step 1 是 live structured understanding，不是只剩一段 briefing 文本；
3. `teacher.plan`/planner source 不是 `deterministic_intent_compiler`；
4. graph 中至少有 `source record_set → entity_set → dependent target record_set → terminal`；
5. 每个变量的 grounded constraints、output size 和少量值已保存；
6. preflight、postflight、answer sanity、evidence verdict、AnswerCard 都来自同一 trace；
7. 最好没有 repair；若有，图中必须画真实 failure reason 和一次完整重执行。

当前仓库还没有一条足够自包含、可直接画成该图的“现行代码完整 trace”。这只需单题导出，不是批量重跑。不要用旧 teacher trace 缺失的字段自行补造。

已有的 Dodd Group 问题可以作为备选：

> Record check: how many contract notices were published by buyers that have awarded a contract to Dodd Group (Midlands) Limited?

它的已知确定性重放状态是：供应商过滤得到 21 条记录 → 16 个 buyer-name bridge values → 这些 buyer names 绑定到目标记录得到 1,940 条 → count=1,940。它经过理解器，但实际走 deterministic compiler，Step-2 planner 被 bypass。因此如果用它，图中应把 planner 画成灰色未走分支，不能宣称“使用了 planner”。

### 图内从左到右放什么

1. **Question + mini record excerpt（宽 20%）**
   - 完整问题；细下划线标 anchor、bridge phrase、target scope、return/count。
   - 2–4 行真实 records：`buyer_name | supplier_name | contract_node_id/evidence_id`。
2. **Structured understanding（宽 18%）**
   - `answer signature`
   - `source role + literal`
   - `bridge role / emitted field`
   - `target bind field`
   - `terminal operation`
   - 不要写模型没有保存的 free-form rationale。
3. **Routing and executable graph（宽 22%）**
   - gate 的真实原因；走过的分支实线，未走分支灰虚线。
   - DAG 节点写实际 variable ID、类型、filter、emit/bind field。
4. **Grounded execution（宽 25%，视觉主体）**
   - source rows 的 output size 与 1–2 个例值；
   - bridge set 的完整 count 与 2–4 个例值；
   - target rows 的 output size 与 2–3 个例行；
   - terminal operation 和原始答案。
5. **Release（宽 15%）**
   - preflight / execution / postflight / sanity / evidence verdict；
   - AnswerCard 或 No verified answer；
   - offline oracle match 只放图注或单独金色注释，不能画成在线 verifier。

### 最像哪些论文

- [Chain-of-Table, ICLR 2024, Fig. 1](https://openreview.net/pdf?id=4L0xnS4GQM)：同一个问题贯穿，真实 intermediate table 是主视觉；错误路径短，本文路径展开。
- [KQA Pro, ACL 2022, Fig. 1](https://aclanthology.org/2022.acl-long.422.pdf)：自然语言、微型 KG、程序三者对齐。
- [QueryAgent, ACL 2024, Fig. 2](https://aclanthology.org/2024.acl-long.274.pdf)：真实 thought/action/environment feedback trace；只借信息结构，不借 pastel 卡片美术。

---

## Figure 2（Method）：一般系统控制流

### 图内内容

严格按当前实现画：

```text
Question + retrieved schema context
→ Step-1 structured intent
→ routing gate
   ├─ valid and no narrow veto → deterministic intent compiler
   └─ otherwise → Step-2 typed graph planner → consistency check → compile
→ field/value/entity grounding
→ preflight
→ dependency-ordered deterministic execution
→ EvidenceBundle
→ postflight + answer sanity + EvidenceVerdict
→ AnswerCard / no verified answer
```

只画一条虚线 repair 回路：structured defect → `replan_with_feedback` → 重新 ground、execute、check。旁边写“clean grounded no-result is terminal”。不要把 PACS、oracle 或 17 个 buyer-name 的 direct-compose 结果放进这张一般运行时图。

参考：

- [TIARA, EMNLP 2022, Fig. 1](https://aclanthology.org/2022.emnlp-main.555.pdf)：每个 failure mode 对应一个组件。
- [ReTraCk, ACL Demo 2021, framework figure](https://aclanthology.org/2021.acl-demo.39.pdf)：Retriever–Transducer–Checker 的模块顺序和 checker 分层。
- [RoG, ICLR 2024, Fig. 2](https://proceedings.iclr.cc/paper_files/paper/2024/file/3e2aeb66481dd63a32421bf032b70384-Paper-Conference.pdf)：question、KG、planning、retrieval、reasoning 的数据流。
- [PICARD, EMNLP 2021, Fig. 2](https://aclanthology.org/2021.emnlp-main.779.pdf)：用一个局部放大明确“哪个阶段拒绝什么”；不要把我们的 post-hoc validator 写成 token-level constrained decoding。

---

## Figure 3（Method/Data）：采购 KG、query-record view 与 provenance

三面板即可：

1. **OCDS input**：一个 release snippet，标 `ocid/date/parties/tender/awards/contracts/documents`；同 OCID 多 release，latest-date selection。
2. **Normalised record**：真实字段 `contract_node_id, buyer_name, supplier_name, release_year, tender_cpv_id, tender_category, value_amount, value_currency, value_is_additive`。
3. **Mini graph + evidence backlink**：buyer organisation → contract/notice ← supplier；contract → CPV；evidence/document → contract。用不同线型区分 query edge 与 provenance edge。

图旁或 Table 1 报：215,221 contracts、131,502 org nodes、3,870 CPV nodes、535,731 evidence nodes；buyer/supplier/category/evidence coverage。不要把 stored names 画成已经完全消歧的 legal entities。

参考：[TheyBuyForYou, Semantic Web 2022, Figs. 3–4](https://journals.sagepub.com/doi/pdf/10.3233/SW-210442)。借数据源→ETL/KG→服务和 provenance 的层次，不要照搬它的基础设施全景。

---

## Figure 4（Method/Experiments）：PACS 构建与 split barriers

从左到右画：

```text
KG-anchored templates
→ typed tree + question + answer type
→ static validation
→ runtime evaluator / independent evaluator agreement
→ non-degeneracy
→ status variants
→ shape signature / tree hash
→ cluster split
→ Channel A / independently naturalised Channel B
→ frozen hashes
```

在下方单独列真实 barrier：exact normalised question、tree hash、unseen-shape；near-duplicate 旁必须写 `trigram screen against train_tris[::7]`。不要画不存在的 G3，也不要写“all-pairs zero leakage”。

参考：[KQA Pro, Fig. 2](https://aclanthology.org/2022.acl-long.422.pdf) 的 canonical question/program→uniqueness→paraphrase 构建链；[CFQ](https://openreview.net/pdf?id=SygcCnNKwr) 用来理解更强的 atom/compound split，但 PACS 不能自称 MCD。

---

## Figure 5（Results）：能力 family × model/channel

最合适的是两种选一：

- 两个 heatmap：Untuned A / Typed A，行是七个可读 family，列是 L1/L2/L3 或 seen/unseen；cell 写 `correct/n`，颜色只辅助。
- 如果某些 family-depth cell 缺失，则改成 F1–F7 grouped bar：Base A、Cloud A、Typed A、Typed B；误差线用 Wilson CI。

主要结论应一眼可见：F5 94.49、F4 81.63、F6 67.79；F3 51.11、F7 41.79 是剩余瓶颈。每个 family 的 n 不相同，必须在标签或旁表写分母。不要画无分母的“overall improvement”大数字海报。

参考：

- [Chain-of-Table, Fig. 3 + Table 2](https://openreview.net/pdf?id=4L0xnS4GQM)：柱图展示 chain length，旁表给每段 n。
- [KQA Pro, Table 5](https://aclanthology.org/2022.acl-long.422.pdf)：按 reasoning skill 分解。
- [DIN-SQL, Table 4/5](https://proceedings.neurips.cc/paper_files/paper/2023/file/72223cc66f63ca1aa59edaec1b3670e6-Paper-Conference.pdf)：按 difficulty 和 ablation 分表；不要照搬 sunburst，除非错误类别真有父子层级。

---

## Figure 6（可选 Results）：互斥 outcome 构成

每个方法一根 100% stacked bar：

```text
correct executed answer
correct explicit abstention
faithful empty result
wrong executed answer
invalid tree / unparseable / API error
```

只使用同一 PACS protocol 的 per-item outcomes。Cloud 的 25 个 API errors 要单独颜色/纹理，不能混进 invalid plan。若不能保证类别互斥，就删掉这张图，改用附录表。

---

## 正文表格

### Table 1：Data/KG/PACS statistics

现稿已包含：四类节点、四种 contract coverage、PACS n/status/depth/exposure/family imbalance。补图 Figure 3 后仍保留这张表，因为读者需要精确数字。

### Table 2：PACS main result

保留 exact correct、n、strict accuracy、95% Wilson CI、valid tree diagnostic。正文不要把 safe accuracy 与 exact answer accuracy并列成两个“准确率”；safe outcome 放附录。

### Table 3：Family diagnostic

F1–F7 的 n、Base A、Cloud A、Typed A、Typed B。图 5 如果做 heatmap/柱图，这张精确表可留正文或移到附录，二者至少保留一个在正文。

### Table 4：Full-runtime system result

正文只报 260-item frozen snapshot 的三行总结果，并明确 artifact provenance 弱于 PACS。13 类逐项 correct/20 放附录。不要把缺少 config/category manifest 的 2,285 final-test 数字放进摘要。

### 以后补的 ablation table

只有拿到同一问题集、同一模型、同一预算下的结果再写：compiler-only / always planner / conditional route / no verifier / no repair / full。当前没有完整公平的六方 artifact，不要用跨版本历史 run 拼出消融。

---

## 附录应该放什么

TMLR 没有固定页数上限，但 reviewer 可以不看 appendix，所以核心方法和主结果必须留正文。附录建议：

- A：全部 artifact hash、环境、命令、模型/adapter/prompt/scorer revision；
- B：完整 operator signature、predicate、空值、多答案、tie、去重、失败语义；
- C：PACS family/template、split、hash、duplicate/leakage gate、human audit；
- D：family × depth × exposure × channel 精确表，全部 outcome、CI、260 runtime 13 类结果；
- E：fast compiler、真实 Step-2 planner、一次成功 repair、unsupported、ambiguous、no-results 的完整 trace；
- F（最终再加）：完整 Step1/Step2/repair prompts 和 provider JSON schema；
- G（若补效率实验）：硬件、median/p95 latency、LLM calls、repair rate、executor variables、cache on/off。

---

## 画图时的事实红线

1. `plan parses` ≠ `plan executes` ≠ `runtime verified` ≠ `oracle matches` ≠ `real-world true`。
2. PACS direct-compose row 没有经过 Step1 understander、Step2 graph planner、EvidenceBundle 或 repair。
3. 未走的 planner/fast-path 必须灰色标 bypassed，不得并入实线。
4. `17 stored buyer-name values` 不能写成 `17 resolved organisations`；样例里确实有别名变体。
5. Oracle 是离线 scorer，不是在线 verifier。
6. Sum 类目前不能宣称“GBP-correct”，因为有 1,059 个 additive non-GBP rows。
7. 图中任何 count、实体名、record/evidence id 都必须来自同一 trace 或同一 frozen artifact。
8. 图注中写清模型、数据快照、是否 single pass/repair、灰色分支含义和作者标注（若有）。
