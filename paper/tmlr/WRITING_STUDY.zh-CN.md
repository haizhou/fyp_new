# 顶刊顶会写作学习与本稿采用的策略

本笔记只学习官方论文网页/PDF的论证结构；本项目的事实和数字只来自当前代码及冻结 artifact。没有读取或继承 `docs/paper_draft_v1.md`。

## 最值得直接学习的论文

| 论文 | 最值得学习什么 | 本文怎样采用 | 不能照搬什么 |
|---|---|---|---|
| [TIARA, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.555/) | Abstract 把两个精确 failure modes 分别映射到 retrieval 和 constrained decoding；Introduction 先讲缺口再讲组件 | 摘要把 schema/组合/invalid plan/unsupported 分别对应 understander、conditional planner、validation、abstention | 当前系统没有 TIARA 的 multi-grained retrieval 或 prefix-trie decoding，不能使用其术语 |
| [ReTraCk, ACL Demo 2021](https://aclanthology.org/2021.acl-demo.39/) | Method 按 Retriever→Transducer→Checker，再逐层解释 checker | Method 先总体 data contract，再依次讲 understand/plan、ground/execute、release checks | 它的 checker 在候选生成中；我们的检查分散在 compile、grounding、execution、postflight |
| [KQA Pro, ACL 2022](https://aclanthology.org/2022.acl-long.422/) | 先定义 KB 和 typed operators，再讲数据生成；结果按 reasoning skills 分解 | 正文先给 record/types/operators，再讲 program-first supervision；主表按 F1–F7 | 显式程序不自动等于 evidence verified；PACS 也不是 KQA Pro 同规模人工数据集 |
| [Shaw et al., ACL 2021](https://aclanthology.org/2021.acl-long.75/) | 高精度 grammar path + neural fallback；同时测 compositionality 和自然语言变化 | 对应 deterministic intent compiler + conditional graph planner；PACS A/B 对应固定语义下的 surface variation | 没有 compiler-only/always-planner/conditional 三方公平消融前，不能把总增益归因给 router |
| [CFQ, ICLR 2020](https://openreview.net/forum?id=SygcCnNKwr) | 用 atoms、compounds、divergence 严格定义 compositional generalisation | 本文把 PACS 的 shape holdout 说清楚并主动限定 claim | PACS 未做 MCD/DBCA，不能使用“系统性组合泛化已被证明”这类措辞 |
| [COGS, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.731/) | overall 之外按 lexical/structural/recursion 等 case 报告 | 用 family、depth、exposure 三轴诊断，避免 overall 掩盖 F3/F7 | 合成 split 的定义不同，不能直接类比其 generalisation gap |
| [Learning from Executions, NAACL 2021](https://aclanthology.org/2021.naacl-main.219/) | 明确 executability 是免费但弱的信号，错误程序也可以执行 | 单独定义 static-valid、executable、runtime-verified、oracle-match、real-world true | 不能把 verifier pass 命名为 answer accuracy |
| [PICARD, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.779/) | 说清检查发生在 decoding 的哪个时刻、拒绝哪个 token | 本文逐项写 compile/preflight/postflight/evidence 在哪个阶段拒绝什么 | 我们不是 token-level constrained decoding |
| [PER-KBQA, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.720/) | parse→execute→refine 的阶段式 ablation 和 intermediate execution | repair 只生成新计划，并完整重跑 ground→execute→verify | 它的 refinement 可以直接生成答案；我们不能绕过 executor |
| [PullNet, EMNLP 2019](https://aclanthology.org/D19-1242/) | 同时报 subgraph recall、size 和最终 QA；先给算法骨架再展开 primitive | 引言用“相关 evidence ≠ COUNT/SUM complete denotation”建立核心缺口 | neural retrieval graph 不是 deterministic execution DAG |

## Abstract 采用的五个功能句

顶会好摘要不是把所有模块列一遍，而是完成五件事：

1. **任务与现实困难**：采购问题需要角色解析、集合、桥接、聚合和 provenance。
2. **精确缺口**：检索到少量相关记录不足以支撑完整集合；可执行程序仍可能语义错误。
3. **方法主张**：模型输出 typed computation，事实值由 deterministic executor 决定。
4. **评测证据**：同一 922-case PACS protocol 下报 exact numerator/denominator、A/B surface、baseline。
5. **窄结论与边界**：说 typed execution 提高 compositional reliability，同时披露 currency、alias、split、artifact 问题。

初稿摘要没有写“SOTA”“hallucination-free”“guaranteed correct”，也没有把 2,285-item 缺 manifest 的历史结果放进 headline。

## Introduction 采用的六段论证链

1. 公共采购问题为何需要多跳集合和聚合，而不是宏观宣传。
2. 用形式式说明 top-k relevant subset 不能推出 exact count/sum/ranking。
3. 说明 direct program generation 的 role/projection/type 失败。
4. 给本文 insight 和两条严格分离的路径：direct typed-program evaluation 与 full runtime。
5. 给审计后主结果，并解释 strongest/weakest families。
6. 三至四条可验证贡献 + scope boundary。

这样写比“先介绍 KG、再介绍 LLM、再逐个脚本罗列”更像研究论文，因为每个组件都回答前文的某一个 failure mode。

## Methodology 为什么占最大篇幅

Method 的可复现标准不是“脚本名齐全”，而是读者能回答：每一步输入是什么、输出是什么、保证什么、失败后发生什么。当前正文按以下顺序组织：

```text
two protocols and guarantee boundary
→ OCDS ingest/latest-release policy
→ role-aware entity resolution
→ extraction, additivity and provenance
→ graph/query-record contract
→ typed operator signatures and failure semantics
→ deterministic execution
→ program-first data acceptance and QLoRA config
→ structured understanding
→ deterministic compiler / conditional planner routing
→ grounding and dependency execution
→ evidence/release checks
→ bounded repair and clean no-result policy
```

PACS 构建放 Experimental Setup，因为它是“如何检验方法”，不是在线推理步骤。完整 operator、hash、命令、prompt、逐项结果和 trace 放附录；但 guarantee boundary、核心 operator、routing、repair/abstention 和主 protocol 必须留正文，因为 TMLR reviewer 可以不看 appendix。

## 结果怎样写才像论文而不是演示稿

- 主表永远给 exact correct/n、metric definition 和 inference protocol。
- Family 表旁解释主要模式和反例，不逐行复述数字。
- 报 95% CI，但明确 Wilson interval 不是 training-seed variance。
- 把 71.58% valid-tree 与 74.51% strict accuracy并列是为了证明两者不等价，不是说 validity 越高越好。
- A→B 只解释同一个 model 的 surface robustness；没有 base/teacher B 时不能声称三系统跨 surface 的公平比较。
- 260 full-runtime snapshot 是 secondary artifact；没有公平 component ablation 时不把总增益归因给 planner/verifier/repair。
- 把负面结果留在正文：F3/F7 较弱、currency 不安全、alias variants、provider API errors。顶刊顶会论文通常因为边界清楚而可信，不是因为每个数字都漂亮。
