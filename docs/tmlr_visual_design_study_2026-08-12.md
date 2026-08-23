# TMLR 论文图表与系统框架图视觉研究（2026-08-12）

> 本轮只学习官方 TMLR 规则、相关顶会/顶刊论文 PDF 中的实际图表，以及当前代码/结果可支撑的视觉叙事；没有生成新图，也没有把仓库旧论文作为依据。原有 architecture 图仅作为需要被审计的历史参考，不作为新图母版。

## 1. TMLR 版式事实

- 官方指南：[TMLR Author Guidelines](https://jmlr.org/tmlr/author-guide.html)。
- 官方模板：[JmlrOrg/tmlr-style-file](https://github.com/JmlrOrg/tmlr-style-file)。
- TMLR 是单栏 US Letter，正文区域约 `6.5 × 9 in`，正文 10 pt。
- TMLR 没有固定 8 页上限，而是允许 “any length”；正文仍应紧凑，核心证据不能只放 appendix。
- 本项目是署名 FYP，并非已被 TMLR 接收的论文，应使用 `\usepackage[preprint]{tmlr}`；不能使用 `accepted`。
- 不应修改 `tmlr.sty`、字体或页边距。
- figure caption 在图下，table caption 在表上；图在 `\linewidth` 尺寸下必须清晰，彩色图在黑白打印时也要可理解。

TMLR 的 6.5-inch 单栏宽度特别适合一张横向的 visual-thesis 图，也容许在 Methodology 中再放一张更完整的系统图。

## 2. 关键认识：Introduction Figure 1 通常不是完整架构图

对实际论文逐页查看后，最稳定的模式是：

```text
Figure 1 in Introduction
  = visual thesis
  = 一个问题 / 一份数据 / 一个明确失败 / 本文 insight

Figure 2 or later in Methodology
  = full system architecture
  = training / inference / data flow / modules
```

优秀 Figure 1 让读者在十几秒内回答：

1. 问题究竟难在哪里？
2. 普通方法在哪个具体位置失败？
3. 本文改变了哪一个中间表示或计算过程？
4. 为什么这个改变会产生正确、可检查的结果？

因此，图的主角应是 **数据状态和中间产物**，不是模块名称。

## 3. 实际论文 Figure 1 的视觉拆解

### 3.1 Chain-of-Table — ICLR 2024

论文：[Chain-of-Table](https://proceedings.iclr.cc/paper_files/paper/2024/file/f53fd88a4340063ecd258c0ae9948b40-Paper-Conference.pdf)，Figure 1，PDF 第 2 页。

- 左侧固定同一个四行表和问题。
- 右侧分三条横带：generic reasoning 答错、program-aided SQL 失败、本文方法逐步正确转换表格。
- 红色局部放大框准确指出前两种方法为什么错。
- 本文路径展示 operation selection、argument、intermediate table、operation history 和最终答案。
- 颜色有语义：红是错误，蓝是步骤，绿是正确操作与答案；不是装饰。
- 读者可以沿着真实行值自行复核全过程。

这是本项目最值得学习的母版：同一个真实采购问题，上方展示浅层/错误程序，下方展示组合程序怎样得到可验证答案。

### 3.2 CABINET — ICLR 2024

论文：[CABINET](https://proceedings.iclr.cc/paper_files/paper/2024/file/19a42d5885e25e51852aca8144e5af0d-Paper-Conference.pdf)，Figure 1，PDF 第 2 页。

- 左侧是真实 question + table，随后分叉成 baseline 和 ours。
- baseline 裁出不完整 sub-table，得到红色错误答案。
- ours 保留数据，并用 heatmap 高亮相关单元格，得到绿色正确答案。
- 模块框很少，证据表格占据主要视觉空间。
- Figure 2 才展开完整架构，但每个阶段仍露出实际 artifact，而非只写模块名。

对本项目的启示是：可以把错误的 shallow tree 与正确的 nested tree 并排，并直接显示它们各自执行出的集合。

### 3.3 RoG — ICLR 2024

论文：[Reasoning on Graphs](https://proceedings.iclr.cc/paper_files/paper/2024/file/3e2aeb66481dd63a32421bf032b70384-Paper-Conference.pdf)，Figure 1，PDF 第 2 页。

- Figure 1 不是系统图，而是两张问题卡：lack of knowledge 与 hallucination。
- 错误词句用红色标记，卡片下直接给可补救它的 KG triple/relation path。
- Figure 2 才画 planning–retrieval–reasoning 系统，并让同一实体在问题、图和答案中保持相同颜色。

这说明 Introduction 的首图可以先定义矛盾，不必一次塞入 ingestion、training、runtime 和 evaluation。

### 3.4 KQA Pro — ACL 2022

论文：[KQA Pro](https://aclanthology.org/2022.acl-long.422.pdf)，Figure 1，PDF 第 2 页。

- 上部是微型 KB，包括 entity、literal、relation 和 qualifier。
- 下部是两个自然语言问题，同时给 SPARQL 和 KoPL program tree。
- 重点不是“系统有几个组件”，而是让问题、图结构、程序树三者对齐。
- 视觉朴素但学术信息密度高；缺点是字号偏小，不能直接照搬密度。

这最适合学习如何展示本项目的 typed algebra。

### 3.5 GraphQ IR — EMNLP 2022

论文：[GraphQ IR](https://aclanthology.org/2022.emnlp-main.394.pdf)，Figure 1，PDF 第 2 页。

- 一张真实 property graph + 一个问题。
- 同一语义分别显示为 SPARQL、Cypher、KoPL、Lambda-DCS 和 GraphQ IR。
- Figure 1 本质上是表示法对齐示例，不是 pipeline。

本项目可以类似地并排展示 natural question、采购图局部、typed algebra tree 和执行结果，但无需展示多种不相关 query language。

### 3.6 Interactive-KBQA 与 DARA — ACL 2024

- [Interactive-KBQA](https://aclanthology.org/2024.acl-long.569.pdf) Figure 1 使用手绘纸张式 prompt/exemplar 卡片、模型/齿轮/数据库 icon，以及真实 action–observation 片段。
- [DARA](https://aclanthology.org/2024.findings-acl.203.pdf) Figure 1 把实际问题、tool list、KG、LLM、thought/action trace 和最终 logical form 放在同一张小图中。

它们看起来不像传统工程矢量图，是因为使用了 editorial cards、icon 和真实 trace，而不是因为信息更多。可以借鉴卡片语言，但不能照搬手写字体、过密文字或装饰性大脑图标。

### 3.7 QueryAgent — ACL 2024

论文：[QueryAgent](https://aclanthology.org/2024.acl-long.274.pdf)。

- Figure 1 用三条等宽色带比较 ICL、普通 agent 和 QueryAgent。
- 三者共享相同 `Question → Logic Form` 骨架，差别只放在中间过程，因而比较非常直观。
- Figure 2 才显示完整多步运行实例与环境反馈。

这适合学习“保持输入输出不变，只突出本文改变的机制”。

### 3.8 StructGPT — EMNLP 2023

论文：[StructGPT](https://aclanthology.org/2023.emnlp-main.574.pdf)，Figure 1，PDF 第 4 页。

- 底层放 KG、table、DB 三种真实结构化数据。
- 中层放对应 interface 调用与 linearized evidence。
- 上层放问题、LLM 和最终 answer/SQL。
- 绿、黄、桃三种颜色跨全部层保持同一任务语义。

优点是每个阶段都有真实 artifact；缺点是文本太小、路径复杂。新图应保留“实际中间产物”，降低同时出现的任务数量。

### 3.9 GrailQAbility — ACL 2023

论文：[GrailQAbility](https://aclanthology.org/2023.acl-long.576.pdf)，Figure 1，PDF 第 2 页。

- A 层展示 KG schema/facts，红色虚线表示理想 KB 中缺失的元素。
- B 层展示五种 unanswerable 情形，并把缺失实体/关系/类型直接标入 logical form。
- C 层展示 train/test answerability 场景。

这是 verifier/abstention 辅图的好母版：用颜色指出“缺的是什么”，而不是只画一个 `Abstain` 方框。

### 3.10 MAC-SQL、PICARD 与 AutoPrep

- [MAC-SQL](https://arxiv.org/pdf/2312.11242) Figure 1 是 `User Question → Database schema → Evidence → Gold SQL` 的竖向 editorial card；对应 token 在各层保持红/橙/蓝高亮。
- [PICARD](https://aclanthology.org/2021.emnlp-main.779.pdf) 的 Introduction Figure 1 直接放 headline result 曲线；Method Figure 2 用绿色勾、红色叉和灰色未检查节点解释 constrained beam search。
- [AutoPrep](https://www.vldb.org/pvldb/vol18/p3504-fan.pdf) Figure 1 直接用两张 error-distribution donut 证明 data preparation 是真实问题，完整系统放在后文。

这再次说明 Figure 1 可以是例子或证据，不必强制成为全系统框架图。

## 4. 为什么这些图“看起来不像矢量图”

文件格式和视觉语言是两件不同的事。对论文 PDF 的 image resources 检查显示：

- QueryAgent 的 Figure 1 以较大的 raster image 嵌入；
- CABINET Figure 1 的主体是约 `2148 × 916` 的 raster image；
- Chain-of-Table 页面混入多块高分辨率 raster/transparent assets；
- DARA 与 Interactive-KBQA 混用 raster icons、透明蒙版和矢量文字；
- MAC-SQL 的卡片/文字大多保持矢量，但使用小 raster icons；
- GraphQ IR/KQA Pro 更接近纯 LaTeX/vector 风格，所以也更朴素。

因此，顶会常见的真实制作方式更接近：

```text
Figma / Illustrator / PowerPoint / draw.io / SVG canvas
  + 实际数据表或 trace
  + 少量统一风格 icon
  + 高亮、callout、局部放大
  → 高分辨率 PDF/PNG 或 mixed PDF
```

新图不必坚持“所有像素都必须矢量”，但文字、线条和关键结构应尽量保持矢量；若嵌入表格截图或 icon，应保证最终 6.5-inch 宽度下清晰。

## 5. 对原有 architecture 图的判断

原图不应局部美化后继续使用，原因不是颜色不好看，而是论点已与当前主要代码路径不一致：

- 它以 Step 1/Step 2 LoRA、reflector、RSFT/DPO 为视觉中心；
- 当前 compose/PACS 主协议是单次 tree emission、无默认 repair；
- 当前训练主线是 plan-first、类型正确程序、runtime evaluator 与独立 evaluator 同意后保留；
- 原图没有 closed typed algebra、PACS compositional generalization 或 WTQ adapter；
- 图中训练、推理、验证三类箭头交叉，却没有一个可以复核的真实例子。

因此，原图只作为历史架构参考；新 Figure 1 和 Figure 2 都应从空白画布重新设计。

## 6. 已找到的真实 Figure 1 候选样例

当前最强的非金额、可复核样例来自 PACS：

- ID：`PACS::F6:L3:f6_other_buyers_via_suppliers:0011#a`
- 标签：`F6-L3-unseen`
- 原问题：`data/qa/pacs_v1/test_channel_a.jsonl` 第 2 行。
- base 结果：`data/qa/compose_probe_v1/eval_pacstest_base_a.jsonl` 第 2 行。
- Compose-v3 结果：`data/qa/compose_probe_v1/eval_pacstest_v3_a.jsonl` 第 2 行。

问题要求：从 DfI TRAM 的 suppliers 出发，找到这些 suppliers 服务的其他 buyers，并排除 DfI TRAM 本身。

Base Qwen 生成了错误的浅层程序：它把组织当成 supplier 直接过滤，并投影 `contract_node_id`，最后返回空集。

Compose-v3 生成的实际组合树可以压缩为：

```text
S = suppliers(buyer = DfI TRAM)
B = buyers(supplier IN S)
answer = B − {DfI TRAM}
```

完整树使用 nested `values → filter → in_expr → values → set difference`。确定性执行返回 17 个 buyers，与 oracle 完全一致。

这个样例适合 Chain-of-Table/CABINET 式 Figure 1，因为它同时具备：

- 同一个输入下真实的 base failure 与 trained success；
- unseen、depth-3 composition；
- 二跳 semijoin、projection 与 set difference；
- 非金额问题，不受当前混合币种风险影响；
- 可展示 3 个真实 buyer 名称并以 `… 17 buyers` 收束；
- 不需要编造 RAG 漏证据的假案例。

## 7. 暂定的两张核心图，而不是一张塞完

### Figure 1 — Visual thesis in the Introduction

建议采用 Chain-of-Table/CABINET 的对照结构：

```text
同一个真实问题 + 小型 buyer–supplier graph

(a) Base planner
    shallow/wrong program
    wrong projection highlighted in red
    empty result  ×

(b) Compose-trained planner
    Step 1: suppliers of anchor buyer
    Step 2: semi-join to other buyers
    Step 3: set difference
    17 buyers + evidence  ✓
```

图中模块名退居其次；主要视觉对象是问题、graph fragment、程序树、中间集合和答案。

### Figure 2 — Full architecture in Methodology

在主叙事最终冻结后，再展开：

1. **Offline program/data construction**：typed grammar → type-correct sampler → runtime/independent dual execution → non-degenerate agreement filter → rendered questions → SFT。
2. **Online inference**：question + schema → recursive JSON tree → static type checker → deterministic evaluator → answer/structured failure/explicit abstention。
3. **Generalization evaluation**：PACS seen/unseen families and surfaces；WTQ dynamic schema adapter（若最终论文保留跨域迁移为主要贡献）。

每个阶段最多展示一个真实 artifact，不画内部类名或脚本名。

## 8. 结果章节应采用的图表组合

1. **Main result table**：PACS strict accuracy，base、teacher、Compose-v3 A/B；单列 API errors，不能把 safe 与 strict 混成一个指标。
2. **Family/depth decomposition**：F1–F7 × L1–L3 × seen/unseen，用 dot plot、small multiples 或谨慎设计的 heatmap，并显示 `n`。
3. **Outcome/error plot**：wrong denotation、invalid tree、abstain、empty result、API error，用水平条或 stacked bar，不用饼图。
4. **Ablation table**：只使用有真实 artifact 支撑的组件对照。
5. **WTQ transfer table**：只有在 official evaluator aggregate 被持久化后才作为正式结果；当前内部 denotation 不能标成 official WTQ accuracy。
6. **Case studies**：successful composition、correct abstention、remaining failure 各一个紧凑案例。
7. **Method/operator table**：operator、input type、output type、deterministic semantics、failure condition。
8. **Data/isolation table**：训练规模、PACS family/depth/exposure、实际执行的 overlap gates；明确披露抽样 gate。

## 9. 后续制作方式

先不决定“Matplotlib 还是 Figma”，先冻结视觉故事与真实样例。确定后建议：

- editorial Figure 1/2：使用 Figma/Inkscape/PowerPoint 风格的 SVG 画布或可编辑的 HTML/SVG source，混合真实表格/graph/program cards，最终导出 PDF；
- 结果图：用脚本直接从 JSON/CSV artifact 生成 PDF/SVG，并输出 sidecar manifest；
- 表格：从 summary JSON 自动生成 LaTeX `booktabs`/`siunitx`，避免手抄数字；
- 不把模型 logo、brain icon 或渐变当作“精美”的来源；
- 统一语义色：蓝=learned planning，紫/青=typed program/data，绿=verified，红=error，amber=abstain；同时配合文字、形状和线型；
- 最终在 TMLR 的 `6.5 in` 实际宽度、100% 缩放和灰度打印下检查；普通图中文字不低于约 8 pt。

下一步应先做 Figure 1 的低保真 storyboard：只确定各区内容、读者视线和真实 artifact，不急于上色或导出最终图。
