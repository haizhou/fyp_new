# 论文主线与结构地图

## 一句话主线

本项目利用英国公共采购研究验证如何成为可执行推理的学习接口。LLM 负责提出类型化计算，确定性代码负责 grounding、完整数据访问、计算和发布检查。在线检查只支持一次有边界的修正，离线构造才使用答案类型和隐藏 oracle 选择训练数据。采购数据是包含角色、多跳关系、聚合和不完整数据语义的现实测试环境，而不是论文需要放大的唯一研究贡献。

## 为什么这样最符合原始项目要求

- 数据与表示层：OCDS ingestion、版本选择、实体解析、schema/property graph、provenance。
- 检索与推理层：Step-1 typed understanding、条件 Step-2 graph planning、grounding、完整集合执行、聚合与多跳 bridge。
- 可靠性层：type/schema checks、pre/postflight、evidence verdict、bounded repair、unsupported/ambiguous/no-result。
- 评估层：closed-book、RAG、plan-guided RAG、KG-only、cloud 和 local systems；answer correctness、non-answer detection、PACS composition、WTQ transfer。
- 明确未完成：source-document citation accuracy、KG deletion/incompleteness sensitivity、multi-seed training variance。

## 当前正文顺序

1. Abstract + Introduction。定义 execution 和 answer match 为什么都不是语义证明，说明 procurement 作为完整集合推理环境的作用，并提出三项 RQ。
2. Related Work。按 executable reasoning、inference verification、execution selected training、denotation faithfulness、answerability 与 provenance 组织。
3. Task, Data, and Layered Verification Contract。冻结快照、图规模、answerability、在线 runtime acceptance、离线 oracle agreement 与 constraint faithfulness。
4. Method。先讲 online proposal and checking，再讲 offline supervision，随后交代数据/KG、benchmark 和单次代数规划器。
5. Experimental Setup。明确区分 2,025 主采购、260 development、136 grounding replay、229 repair audit、922 PACS 和 4,344 WTQ。
6. Results and Analysis。主系统结果、训练 ladder、development system diagnostic、verification gains and misses、PACS 与 WTQ。
7. Discussion and Limitations。按 learning、data truth、verification、reproducibility 和 responsible use 组织边界。
8. Conclusion。回到 structured but incomplete feedback，不宣称 autonomous self improvement。

当前环境暂时缺少 TeX 可执行文件，新版本仍需在恢复 TeX 后检查分页、float 和 overfull box。

## 从头到尾的论证链

1. **核心问题是可执行不等于语义正确。** 引言先区分模型能否提出计算，以及系统能否确认这个计算在数据上的含义。采购中的角色、多跳关系、完整集合和聚合使这一差异尤其明显。
2. **先固定数据语义，再讨论验证。** Task/Data 章节定义 snapshot、contract-award record、entity alias、amount/additivity 和三类 non-answer。在线依次检查 valid、grounded、executable 和 runtime verified。离线另行审计 oracle correctness 与 constraint faithfulness，两者不是同一条层级链。
3. **Method 先解释反馈接口，再交代数据来源。** Step-1 understander → fast compiler 或 Step-2 planner → hard grounding → exhaustive deterministic execution → release/abstention 是在线路径。offline curation 在同一 trace 之后增加 answer shape 与 typed oracle。随后再解释 OCDS ingestion、version selection、precision-first ER、property graph 和 executable benchmark。
4. **在线和离线权限严格分开。** 在线 runtime 没有 oracle；离线 curator 才用 oracle 和 answer shape 对已执行结果作 SFT/repair/DPO 选择。硬检测通过改变样本的接受、修复和拒绝分布产生监督，不是一个直接反向传播的 loss。
5. **四套实验各回答一个有限问题。** 2,025 是排除开发 ID 后的主 full-runtime 结果；260 只做 checkpoint 与 baseline 诊断；PACS 隔离 single-call compositional planning；WTQ 检验同一递归代数的跨域适配。它们不能合成一条学习曲线。
6. **结果先主系统、后训练与边界。** 主表比较 closed-book/RAG/teacher/hybrid/local；随后说明 SFT 是两个模型族中唯一共同改善的 checkpoint transition；development system diagnostic 单独解释 217 与 224 的配置差异；grounding replay 和 constraint audit 先说明验证改变什么、遗漏什么；PACS 与 WTQ 最后作为较窄协议。
7. **Discussion 明确保证在哪里停止。** Data truth boundary 说明结果只对冻结快照成立。Verification boundary 区分运行时通过、答案相同和约束忠实。Reproducibility boundary 记录单次训练、历史 manifest 和 snapshot 缺失。

## 附录现在承担的复现信息

- A：主 full runtime 的 intent/graph IR、7 步在线算法、evidence cap、真实 trace provenance。
- B：teacher/main/dev/PACS/WTQ 的 plan form、decoder、repair budget、oracle 权限和保存内容。
- C：PACS/WTQ 使用的 recursive algebra operator 与类型。
- D--E：数据/ER/KG、benchmark 生成、13 bucket 和 split/duplicate 审计。
- F--G：PACS v1.1 与 WTQ official evaluator 的修正和 artifact 边界。
- H--I：grounding/constraint audit、训练数据数量、完整 QLoRA continuation 配置和 serving 版本。
- J：入口脚本、hash ledger、teacher prompt/run manifest 与当前 dirty worktree 的复现限制。
- K：原始 project brief 的逐项 implemented/partial/not evaluated 对照。
- L：现有证据究竟是 runtime intervention、checkpoint ladder、system comparison 还是 robustness test。
- M：若以后补做 grounding-supervision 因果实验，所需的 matched ablation 设计。
- N：Step-1、Step-2、repair 的 system prompt、动态 payload、provider schema、重试和信息边界。

`paper/tmlr_v3/artifacts/REPRODUCIBILITY_MANIFEST.md` 另以机器可审计方式记录源码/config
SHA256、教师模型与调用设置、哪些字段来自保存产物、哪些从当前 runner 重建，以及 resume
非事务性等历史限制。

## 结果的主次顺序

1. 主结论：非开发子集 2,025 题的完整 runtime，fully-local Qwen 85.73%。
2. 系统比较：完整 2,285 冻结 artifact 作为透明对照；cloud、hybrid、RAG、plan-RAG、closed-book 均在同表。
3. 开发诊断：260 题只用于 checkpoint/baseline 配对分析，不再冒充独立 test。
4. 组合能力：PACS 922，明确是 single-call recursive algebra protocol；Compose-v3 A/B 为 78.31/77.11。
5. 跨域迁移：WTQ 4,344 official，22.51 → 27.33 → 44.43 → 51.80。
6. 可靠性小证据：fixed-plan grounding 89 → 103/136；详细训练目标约束审计放附录。

## 不应再出现的叙事

- 不把 260 题写成论文全部实验或 independent final test。
- 不把 PACS/WTQ 说成经过完整 understander→graph planner→evidence runtime。
- 不把 executable、runtime verified、oracle correct 和 plan-equivalent 混为一谈。
- 不把在线一次 replan 称为 self learning，也不把离线训练路径包装成 recursive self improvement。
- 不把 1,725 repair 和 390 preference 与 5,598 positive 加总；它们是重叠训练视图。6,860 个 answer-producing runtime-pass traces 只互斥分为 5,598 positives 和 1,262 hard negatives。
- 不把 contract IDs 称为经过验证的 source-document citations。
- 不声称 grounding-derived training rows 单独导致了 SFT 的全部提升；当前没有 matched remove-only ablation。
