# CICADA Planner 训练流水线（v4 · 最终版）

> 从 L1 数据清理到 Final 验收的端到端实验路线。
> **执行记录**：每次完成/修改的方法与结果记在 [cicada_worklog.md](cicada_worklog.md)。
> **Nano teacher 闭环规范**：见 [trace_first_teacher_pipeline.md](trace_first_teacher_pipeline.md)。
>
> **术语**：本文的 **RSFT = Rejection-sampling fine-tuning**（采样多个 plan → 筛出正确的 → 在通过的输出上做 supervised fine-tune）。它是 SFT 的变体，**不涉及 policy gradient / RL**。
>
> **核心结构**：SFT 建 floor → **RSFT 主升**（把 pass@K 折进 pass@1）→ DPO 精修 + 对照臂。reflector 保持为 prompt 模块（不微调），在 DPO 阶段兼职 pair 生成器；**executor 是最终裁判**，但训练侧收样本时必须叠加 faithfulness guard。
>
> **v4 相对 v3 的改动**（全部已合并进正文，此处为索引）：
> 1. **Abstention 进训练闭环**：unanswerable/ambiguous 题在 SFT mix、RSFT accept 规则、DPO pair 三处都有明确定义（v3 只出现在评估指标里，三条件 filter 会把正确弃答全部筛掉，把模型推向硬答）。
> 2. **Guard 标定实验（新增 Phase C0）**：用 oracle plan × L2 改写题实测 faithfulness guard 的假拒率；训练侧用放宽版 guard，评估侧用严格版。否则 RSFT 数据系统性偏向字面表达，削弱 L2/L3 泛化。
> 3. **RSFT 重训协议钉死**：每轮从 SFT checkpoint 重训（ReST/STaR 惯例），用累积 accepted 数据；不做链式续训。另加 per-question cap。
> 4. **DPO 防退化**：pair 用当前（RSFT 后）policy 重采；加 NLL anchor（RPO 式）或至少监控 chosen log-prob；β 在 dev_tune 上调；`oracle_match is True` 才能进 chosen（堵现有 `log_preference` 里 None 算 good 的口子）。
> 5. **评估增强**：第五臂 oracle-plan 上界；每臂 pass@K 曲线；配对显著性检验（bootstrap / McNemar）；新增 guard 拦截率（"碰巧对"率）指标。
> 6. **数据卫生**：近重复按（template × entity 组合）归组后再切分，组不跨 split；KG snapshot hash + executor/guard 版本冻进数据集元数据；每条样本带 provenance。
> 7. **算力现实**：K 由 dev_smoke 上的 pass@8 vs pass@16 决定；RSFT round 1 先在 2-3k stratified 子集跑通再放全量；oracle replay 按类型接受率定向配比。

---

## 全局图

```
Phase A   数据           L1 清理 → L2 生成/体检 → 近重复归组 → 合并分层 split（含 abstain 样本）
Phase B   SFT            oracle plan + abstain 样本建 floor（off-policy）
Phase C0  Guard 标定     oracle × L2 实测假拒率 → 定训练版/评估版两套 guard        ★新增
Phase C   调通/调 prompt  dev_smoke 调通 → dev_tune 调系统 → dev_select 选型（不混）
Phase D   RSFT ←主升     采样 → 双分支 accept → 混定向 oracle replay → 从 SFT ckpt 重训
Phase E   DPO 精修/对照   三路 pair（reflector / RSFT / abstention），当前 policy 重采
Phase F   对比           5 臂 × {planner-only, full-system} on dev_select + 显著性
Phase G   Final          冻结（数据侧 + 系统侧），只跑一次
```

**核心叙事（method 章节主轴）**：partial verifiability —— deterministic execution 作为 verifiable reward anchor；SFT 教会 schema；RSFT 内化"可执行且忠实"的成功 plan；DPO 在忠实可执行与真实失败之间精修偏好；**知道哪些能验证、哪些不能（并正确弃答），本身就是可靠性的一部分**。

---

## Phase A — 数据准备

### A1. 清理 L1

判定原则（钉死两条）：

```
题目本身有毛病        → 删除或重新生成
题目挑战体系的能力     → 保留（这正是要学的难题，含被杀过风控的）
```

**有毛病包括**：gold answer 无法由 KG 验证；question 与原始 plan 不一致；CPV / year / entity 缺失或错配；count / sum 的目标集合定义混乱；问了 KG 不支持的字段**却没有标 unsupported**。

> ⚠️ v4 明确：问了 KG 不支持字段、且**已正确标注** `unsupported` 的题，**不是脏数据，是一等公民**——它们是 abstention 训练/评估的 ground truth，清理时保留并核对标签。

清理后每题保留：

```
question / question_type / gold_answer（或 abstain_label + abstain_reason_category）
oracle_plan（或 oracle 结构化弃答输出）
evidence / executable trace（如有）
template_family / difficulty / tag
provenance: {source: L1|L2, generator_model, template_id, kg_snapshot_hash}   ★新增
```

### A2. 生成并体检 L2

L2 补稀缺题型：multi-hop / bridge_join、role-bound factoid、comparison、boolean、sum、role-conflict / ambiguous wording，**以及成比例的 unsupported / ambiguous 题**。

生成后逐题带查询走一遍单体检：plan ↔ question 一致；gold answer 可由 executor 复现；role direction 明确；bridge / comparison 有中间集合；unsupported / ambiguous 已标记且理由类别正确。

### A3. 近重复归组 → 合并 → 分层 split

**先归组，再切分，组不跨 split**（★新增，防 iid 虚高）：

```
dedup_group = hash(template_family, 规范化实体组合, year/CPV 组合)
同组题目只能整组落入同一个 split
```

同一 template 换相邻实体填充的题几乎是复写；分层随机切分会让近重复对横跨 train/final_iid，把 iid 结果虚高 3-5 个点都有可能。归组后再按 `question_type × template_family × difficulty × entity group × year/CPV × role pattern` 分层。

建议 split（总量 ~10k）：

```
train pool   70-80%    ~7000-8000（含 5-10% abstain 样本）
dev          10%       ~1000 → 进 Phase C 拆成 dev_smoke / dev_tune / dev_select
final_iid    10%       ~1000
final_ood    5-10%     ~500-1000（新模板 / 新实体 / role-conflict / bridge-heavy / 新型 unsupported）
```

### A4. 数据集版本冻结（★新增）

oracle plan 的正确性依赖生成时的 KG 快照。训练期间 KG 或 executor/guard 任何改动都会让标签悄悄失效。数据集元数据必须写死：

```
kg_snapshot_hash / executor_version / guard_version(训练版+评估版) / generator_versions / seed
```

后续任何 phase 开跑前先校验 hash 一致，不一致就停下来重新验证 oracle plan（全量重跑 executor 过一遍即可，便宜）。

---

## Phase B0 — Nano teacher closed-loop（先产 baseline / verified data）

在正式训练 Qwen 之前，先运行 [trace_first_teacher_pipeline.md](trace_first_teacher_pipeline.md) 定义的
trace-first closed-loop teacher pipeline。Nano 在这里不是最终模型，也不是 gold label 来源；它只负责提出
candidate plan / repair plan。只有经过 grounding、schema check、KG executor、verifier、answer sanity/postflight
接受的输出，才记为 **verifier-accepted plan / verified plan**。

关键边界：
```
Verifier = 裁判，只输出 accepted/rejected、failure_stage、failure_reason、answer_correct/final_verdict，
           可给规则化 repair_hints，但不生成新 plan。
Reflector = repair module，只看 compact structured trace summary，不看 hidden reference answer；
            repaired plan 必须重新执行并重新验证。
```

第一轮只跑 `dev_smoke`，产出：
```
full_traces.jsonl
reflector_inputs.jsonl
reflector_outputs.jsonl
verified_plans.jsonl
repair_sft.jsonl
dpo_pairs.jsonl
failures.jsonl
summary.json
```

这一步的目的：
- 得到未微调闭环 baseline；
- 收集 `question -> verified plan` 的 SFT 候选；
- 收集 `chosen=verified/repaired plan`、`rejected=failed plan` 的 DPO pair；
- 形成 failure taxonomy，决定后续是否需要改 planner prompt / reflector summary / verifier hints。

**Attempt 协议**（详见规范文档 Attempt Protocol 节）：Attempt 0 = 初始 plan，最多 3 次
verifier-guided repair；**每次 repair 必须重跑完整 grounding → executor → verifier**，reflector
永不自证修好；每次 repair 以最近一次失败的 feedback 为输入；首次通过即停。运行时开关
`ReasoningPipeline.max_feedback_replans`（运行时默认 1，teacher 跑 2-3），逐 attempt 记录在
`metadata.feedback_replan.attempts`。闭环指标随之产出：first-pass / Repair@1/2/3 / final
verified accuracy、平均 attempt 数、预算耗尽后的失败题型分布。

**Teacher 混配（可选）**：Step-1 理解与 Step-2 结构化可用不同模型
（`TypedLLMPlanner.understanding_client/understanding_model`，如 nano 理解 + 严格 JSON 模型填壳）。
两条不变量：① 标签权威永远是 executor/verifier，teacher 组合不定义正确性；② 每条工件记
`teacher` provenance 便于消融。混配选择在 dev_smoke 上按形状失败率（嵌套回显/占位符/枚举回显）实测决定。

---

## Phase B — SFT bootstrap（建 floor）

不需要 reflector、不需要 nano。把 train pool 的 oracle plan 转成 runtime 统一格式：

```
question → gold structured plan
question(unsupported/ambiguous) → 结构化弃答输出        ★新增，必须进 mix
```

结构化弃答输出与 typed DSL 对齐（`question_type: unanswerable` + `reason` 槽位），格式和普通 plan 同一套 schema——弃答不是特殊 token，是 DSL 里的一个合法类型。

教 Qwen：输出合法 schema；保留 year / CPV / entity；分清 count / sum / factoid / boolean / bridge；保留 buyer / supplier / publisher 角色；**对 unsupported 题输出结构化弃答而不是硬编**；不生成最终答案。

**定位**：off-policy，学"别人的正确"。作用是从零到会——否则 RSFT 裸采样会挨饿、没正样本可收。此步之后 pass@1 约 ~32% 量级是起点不是终点。

---

## Phase C0 — Guard 标定实验（★新增，Phase D 的前置门）

训练侧 accept 要用 question-plan faithfulness guard，但现有 `plan_consistency_check`（`src/procurement_graph/reasoning/typed_planning.py`）是**字面 surface 检查**（`surface not in question` / `invented_number` / `role_flipped` / 比较方向词）。L2 恰恰是改写题——surface 不再逐字出现，guard 会假拒正确 plan。若直接拿去过滤 RSFT 数据，接受集会系统性偏向字面表达，**恰好削弱 L2/L3 泛化——和引入 L2 的目的自相矛盾**。

标定方法（便宜，半天）：

```
oracle plan（构造上必然 faithful）× 对应 L2 改写题 → 过严格版 guard
按 check 类型统计假拒率（FRR）
```

产出**两套 guard 并在代码里显式命名**：

```
guard_eval（严格版）  ：原样保留，用于评估、trace 审计、论文里的 faithfulness 指标
guard_train（放宽版）：surface 匹配放宽为归一化/别名/子串匹配；
                       保留硬规则：role_flipped、invented_number、invented_threshold、
                       comparison 方向、operation_outside_type
```

决策规则：某 check 在 oracle×L2 上 FRR > ~10% → 该 check 进放宽名单；FRR 可忽略 → 保持严格。标定结果写进论文（这本身是 guard 设计合理性的证据）。

---

## Phase C — 调通链路 + 调系统

### C1. dev_smoke（20-50 条）

只为确认链路能跑通，**不看准确率**：

```
Step1 出 plan → compiler 接住 → verifier 检查 → executor 跑 → reflector 修 → 日志存 good/bad plan
```

顺带在这里做两个便宜测定（★新增）：
- **pass@8 vs pass@16**（当前 SFT 模型、拟用温度）：差距 <2-3 个点就用 K=8，RSFT 成本直接减半；
- 温度扫描：结构化 compiler 从 0.6-0.8 起步，嫌多样性不足再往上抬（高温 schema 崩得多）。

### C2. dev_tune（100-300 条 stratified）

调：nano prompt、compiler prompt、verifier 硬规则、**reflector prompt**、retry 次数、failure logging 格式、**DPO 的 β 也在这里调**。

> ⚠️ 三点注意：
> 1. 这部分**只调系统，不进训练**。
> 2. reflector 是**纯 prompt 模块**（方案 B）——不微调，靠 prompt 工作，后续推理时保留它并出场。
> 3. guard_train 的放宽参数如需微调，也只能用 dev_tune，不许碰 dev_select。

### C3. dev_select（500-1000 条，与 dev_tune 不重叠）

**专门用于选 checkpoint 和 Phase F 的横向对比**，绝不回头调 prompt / reflector。

> **为什么 tune 和 select 必须掰开**：用同一个 dev 池子边调 prompt/reflector 边选 checkpoint，等于拿调过拟合的集合去做模型选择，成色存疑、审稿人必疑。数据不多时，退一步的底线是：**调 prompt 的 subset 和选型对比的 subset 必须分开，并在论文里写明**。

**dev_select 使用登记**（★新增）：它会被 RSFT 停机判据、checkpoint 选择、Phase F 对比反复查看，轻微 peeking 不可避免——每次查看记一条日志（日期/用途/查看的指标），论文 limitation 一句话交代；headline 数字反正在 final 上。

**dev 三层总览**：

```
dev_smoke   20-50      链路通不通 + K/温度测定（不看准确率）
dev_tune    100-300    调 prompt / verifier / reflector / guard_train / β（不选型、不进训练）
dev_select  500-1000   选 checkpoint + 横向对比（不调参、不进训练，查看要登记）
```

---

## Phase D — RSFT（←主升，把 floor 抬起来）

**目的**：把模型 pass@K 里"偶尔能对"的能力，压进 pass@1 的"稳定能对"。

```
迭代循环（1-3 轮）：
  对 train pool 每题：
    当前模型采样 K 个 plan（K 来自 C1 测定，缺省 8；温度来自 C1）
    每个 plan 过双分支 accept（见下）
    留下通过的 plan
  去重：同题多个正确 plan 按结构签名去重
  per-question cap：每题最多保留 2-4 个去重后的正确 plan          ★新增
  重采：按 question_type 平衡；abstain 样本维持在 mix 的 5-10%     ★新增
  构造 RSFT dataset：
    accepted self-sampled + replayed oracle（定向配比，见下）
  从 SFT checkpoint 重训（不链式续训，见下）
  在 dev_select 上盯 execution success + abstention precision，
  两轮涨幅 <1-2% 就停
```

### accept 标准：双分支，缺一不可（★v4 定稿）

```
可答题（有 gold answer）：
  accepted = schema valid
           + question-plan faithful（guard_train）
           + executor answer correct（对 gold，严格相等/容差按类型定义）

不可答题（标注 unsupported / ambiguous）：
  accepted = schema valid
           + 输出为结构化弃答（question_type=unanswerable / ambiguous）
           + abstain 理由类别与标注一致
```

> **为什么不能只看 executor correct**：count/sum 一类约束集合可能漏了 year constraint 但该实体恰好只在那年活动——答案碰巧对、plan 是错的。收进训练集等于教模型"漏约束也没事"，且难题上更频发。faithfulness guard 就是拦这个的。
>
> **为什么必须有弃答分支**：三条件 filter 对 unanswerable 题永远不通过 → 每一轮 RSFT 都在把正确弃答洗出训练集 → 模型被系统性推向硬答，abstention precision 崩掉，而这恰是论文主线指标之一。

### 重训协议（★v4 钉死）

```
每轮：从 SFT checkpoint 重新训练，数据 = 截至本轮的全部 accepted（累积）+ oracle replay
不做 RSFT-ckpt → RSFT-ckpt 的链式续训（1-3 轮就会复利放大分布漂移和过拟合）
学习率取 SFT 的一半量级起步
```

### oracle replay：定向配比（★v4 升级）

纯 self-sampled 有两个риск：**模式收窄**（只在自己已会的分布上强化）和**难题边缘化**（bridge/sum 收不到正样本 → 恶性循环）。所以混 oracle replay，且**不做全局一刀切，按类型接受率定向**：

```
type_ratio(t) = clip( target_accept / max(accept_rate(t), ε), 1.0, 4.0 )
即：self-sample 接受率越低的类型（bridge/sum），oracle 比重越高
全局落点仍控制在 self : oracle ≈ 1:1 ~ 2:1，具体按 dev_select 表现微调
```

这也让 SFT → RSFT 是渐变而非硬切：SFT 纯 oracle，RSFT 是 oracle 掺进自采正样本。

**可选增强（第三轮再上）**：难题裸采 K 次全军覆没时，让采样走一遍 reflector 多捞几个正确 plan，扩大正样本池。首轮别上——会污染"纯裸采样效果"的读数，留作消融。

### 试跑门（★新增）

Round 1 先在 2-3k stratified 子集上跑通：确认 accept 率分布合理（不是 0% 也不是 90%）、双分支都有样本、混合比生效、训练脚本闭环。子集通过后再放全量。K=8 × 7.5k 题 × 1 轮 ≈ 6 万次生成 + 执行（executor 是 parquet 上的确定性查询，便宜；生成是大头），这是全流程算力主项，别一上来就全量翻车。

---

## Phase E — DPO（精修 + 对照臂；reflector 在这里出场）

RSFT 抬完 floor、难题正样本桶不再为空后，DPO 才有干净的 on-policy pair 可用。

### 判定标准——chosen 全环通过，rejected 任一环失败

```
chosen   = schema valid + question-plan faithful（guard_train）+ executor answer correct
           且 oracle_match is True（有 gold 时必须显式对上，None 不算过）      ★收紧
rejected = 采样 plan 在任一环失败：
           schema invalid / compiler failure / executor failure /
           wrong answer / unfaithful plan / answer leakage
```

> ⚠️ chosen 判定与 Phase D 的 RSFT accept **必须同一把尺子**。RSFT 挡了"碰巧对"，DPO 若只看 executor 就等于后门放行。
>
> ⚠️ 代码对接：现有 `trace_reflector.log_preference` 里 `label="good"` 的条件是 `oracle_match in (True, None)`——审计日志无妨，**构造训练数据时必须收紧为 `oracle_match is True`**。

**abstention-pair（★新增第三路）**：

```
路0 abstention-pair（直接服务论文主线）：
    chosen   = 对 unsupported/ambiguous 题的正确结构化弃答
    rejected = 同题上模型硬编出来的 plan（采样中自然产生，无需人工 corruption）
```

### 三路 pair 混合构造

```
路0 abstention-pair：同上（占比 10-20%）

路1 reflector-pair（同源、最小差异，最干净）：
    主 pair：chosen = 首个通过 verifier 的 attempt k，rejected = attempt k-1（最近失败）
             —— 最能训练"按 verifier feedback 修正"的行为
    弱 pair（可选、降权）：chosen = attempt k，rejected = attempt 0
    ⚠️ chosen 必须再过 executor 复验通过才算数
    （reflector 说"修好了"不作数；修完执行仍错的，这对作废）
    ⚠️ 全败题（预算内无任何通过）单独进 hard-case 池，不进 DPO——除非同题后来有
       verified plan（更强 teacher / oracle plan / 再采样成功）补上 chosen 侧

路2 RSFT-pair（补 reflector 摸不到的难题）：
    chosen   = 采样中全环通过的 plan
    rejected = 同题同轮采样中失败的 plan
    （同一道题同轮配对，只差 correctness，不掺人工 corruption）
```

**pair 必须用当前 policy 重采**（★新增）：DPO 开跑前，用 RSFT 最终 checkpoint 重新采样构造全部 pair。不回收 RSFT 早期轮次攒的失败样本——那是 SFT 时代 policy 的错误，模型已经不犯了，学"避开已不犯的错"轻则浪费、重则有害。

rejected 里标注具体失败模式（missing constraint / wrong question_type / role flip / sum 未去重 / count year-scope 错 / target_unreachable / schema violation / answer leakage / bridge anchor missing / comparison direction / **premature abstain**（可答题弃答）/ **hallucinated plan on unsupported**）。

### 两条铁律（保留）

- ❌ **不用 gold plan 当 chosen**（off-policy，会把格式当 spurious 信号学）。gold 只留给 SFT。
- ❌ **不用人工 corruption 造 rejected**（人造伤疤不迁移到真实失败）。

### 归一化 + 训练稳定性（★v4 扩充）

- **长度/格式归一**：chosen/rejected 统一 schema 序列化格式，记 `length_delta` 字段，差异过大的 pair 丢弃或降权。reflector-pair 尤其危险——修复版系统性更长，DPO 会学到"长 = 好"的 length bias。
- **防 chosen log-prob 塌陷**（★新增）：vanilla DPO 的著名退化是 chosen 的 log-prob 跟着 rejected 一起降。两个措施选一（推荐前者）：
  ```
  ① loss = DPO_loss + λ·NLL(chosen)      （RPO 式 anchor，λ~0.1-1.0 在 dev_tune 调）
  ② 至少全程监控 chosen log-prob 曲线，出现同降立即停
  ```
- **β 在 dev_tune 上调**（不许用 dev_select）。
- **配额平衡**：按 `question_type × failure_type` 分层限额；别让 bridge/sum 失败霸屏，也别让 reflector 只覆盖中等难度。

### executor 复验决口（伪代码，保留）

```
原始 plan → executor → 错（记为候选 rejected）
  → reflector 诊断 + 重生成
    → 新 plan → executor 复验
        对 → 这对 (旧错, 新对) 成立，进 DPO
        错 → reflector 没救回来，这对作废
```

**executor 是最终裁判；reflector 只是候选生成器。**

---

## Phase F — 对比（5 臂 × 两套评估）

在 **dev_select** 上跑（不是 dev_tune），可反复跑，**不进训练**；每次查看登记。

### 五个臂（★v4 加第 ⑤）

```
①  prompt-only / no SFT
②  SFT only
③  SFT + RSFT           ← 关键对照：RSFT 抬了多少
④  SFT + RSFT + DPO     ← DPO 在 RSFT 之上还有没有增量
⑤  oracle-plan 上界     ← oracle plan 直接喂进同一套 compiler/executor      ★新增
```

⑤ 给出任何 planner 的天花板，把"planner 错"和"KG/executor/compiler 本身答不了"干净分开——几乎零成本，审稿人必问。

**外部基线阶梯**（引用/补测，撑开对比谱系）：Plain LLM 直答 → Standard RAG（2026-07-02 已有
23/25% 对比数据）→ KG-retrieval + LLM 自答 → planner+executor 无 reflector（≈ planner-only 口径）
→ 本表 ①-④。闭环指标（first-pass / Repair@k / final verified accuracy、平均 attempt 数）
从 trace 的 `feedback_replan.attempts` 直接统计，full-system 口径随表附上。

### 固定系统配置，分两套评估（保留）

```
planner-only：只测 first-pass plan（斩 reflector/retry）
  → 干净衡量 planner 本身变强多少                    【放附录表】
full-system：固定 compiler/verifier/executor/reflector/retry，只换 planner ckpt
  → 系统真实表现                                     【放主表】
```

### pass@K 曲线（★新增，论文核心图）

每臂报 pass@{1,4,8,16}（planner-only 口径）。"RSFT 把 pass@K 折进 pass@1"是本文核心论点，②→③ 两条曲线的形变（K>1 区间塌向 K=1）就是最直接的证据；采样在 RSFT 过程中顺手可得。

### 显著性（★新增）

dev_select 500-1000、final ~1000 的规模下，臂间 2-3% 差异在噪声边缘：
- 逐题配对做 **paired bootstrap**（或二分类指标用 **McNemar**），主表报点估计 + 95% CI；
- 训练多 seed 跑不起就跑采样多 seed；都跑不起，limitation 里明说单次运行。

### 指标

```
schema valid rate
plan exact / partial match          （只当 proxy，不拿它选 checkpoint）
execution success                   ← 主指标
answer accuracy                     ← 主指标
hard constraint preservation
guard 拦截率（"碰巧对"率）           ← executor-correct 中被 guard_eval 拦下的比例   ★新增
reflector recovery rate             （仅 full-system）
role error rate
abstention precision / recall       ← 拒答的准确性 + 该拒有没有拒
unsupported detection accuracy      ← 识别 KG 不支持题的能力
```

> **guard 拦截率为什么值钱**：它把"三条件 filter 比 executor-only 多挡了多少毒样本"从设计论证变成测量结果——直接量化 partial verifiability 主线的一个 contribution。
>
> **abstention 指标为什么必须在**：只盯 execution success / answer accuracy 会激励对不支持题硬答（硬答有概率蒙对；正确弃答反而"不得分"）。abstention precision + unsupported detection 才能压住"为了刷分而答"。ground truth 在 A1/A2 已标好，直接能测。
>
> **关键交卷点**：③↔④ 的 gap 回答"在可验证 reward、RSFT 已抬 floor 之后，DPO 到底还加了什么"——这正是对 DPO 线（Aletras、trajectory-scoring 批评）做出 contribution 的地方。没有 ③ 这个对照就回答不了。

---

## Phase G — Final（封盘）

最后固定后**只跑一次**（或极少次）：

```
final_iid：同分布测试
final_ood：新模板 / 新实体 / role-conflict / bridge-heavy / 新型 unsupported
```

### 冻结清单（★v4 扩到系统侧）

```
数据侧：generator version / random seed / template distribution /
        entity holdout rule / question_type distribution / kg_snapshot_hash
系统侧：executor version / compiler version / guard_eval + guard_train version /
        reflector prompt version / retry 次数 / 温度与采样参数                ★新增
```

### 完全隔离清单（论文里也要放）

```
final_iid / final_ood must NEVER be used for:
  · prompt tuning          · reflector tuning
  · guard calibration      · RSFT filtering
  · DPO pair construction  · checkpoint selection
```

主表报 final 数字 + CI；预先指定一个 headline metric（建议 full-system answer accuracy on final_iid，final_ood 作泛化叙事）。

---

## 与现有代码的对接点

| 事项 | 位置 | 动作 |
|---|---|---|
| guard 拆两版 | `src/procurement_graph/reasoning/typed_planning.py` 的 `plan_consistency_check` | 加 `mode: "eval"\|"train"` 参数或拆两个函数；train 版放宽 surface 匹配、保留 role/number/direction 硬规则 |
| preference 日志收紧 | `src/procurement_graph/reasoning/trace_reflector.py` 的 `log_preference` | 训练数据构造侧过滤 `oracle_match is True`；日志本身可不动 |
| 弃答输出格式 | typed DSL 已有 `unanswerable` 类型 | SFT/RSFT 目标格式直接复用，补 `abstain_reason_category` 槽位对齐 A1/A2 标注 |
| 新脚本 | `scripts/` | ① oracle→SFT 格式转换 + 全量 executor 复验；② RSFT 采样-过滤-混合循环（可复用 L2 并行生成的 resume 机制）；③ DPO pair 构造 + 归一化 + length_delta；④ 5 臂 × 2 口径评估 + bootstrap |
| guard 标定 | 新脚本 | oracle plan × L2 题过 guard_eval，按 check 类型出 FRR 报告 |

---

## 执行顺序与 go/no-go 里程碑

```
M0  A1-A4 完成        gate：oracle plan 全量 executor 复验通过率 >99%；
                            abstain 样本占比 5-10%；split 组不跨界抽查通过
M1  SFT 完成          gate：schema valid >95%；dev_smoke 上 pass@1 明显高于 prompt-only
M2  C0 guard 标定     gate：guard_train 在 oracle×L2 上 FRR <2%；硬规则拦截率不降
M3  C1-C3 调通        gate：K/温度定稿；dev_tune 上 reflector recovery 有非零读数
M4  RSFT 子集试跑     gate：accept 率在 10-60% 区间；双分支均有样本；混合比生效
M5  RSFT 全量 1-3 轮  gate：dev_select execution success 两轮涨幅 <1-2% 停；
                            abstention precision 不下降（下降 = 弃答分支配比出了问题，回 M4）
M6  DPO               gate：chosen log-prob 不塌；dev_select 上 ④≥③（至少不显著变差）
M7  Phase F 对比      gate：5 臂 × 2 口径 + pass@K + CI 全部落表
M8  Final             一次性跑，封盘
```

任何 gate 不过，回上一级修，**不许带病进下一 phase**（尤其 M2 和 M5 的 abstention 检查——这两处翻车会无声污染后面所有结果）。

---

## 一句话路线总结

```
Clean L1(+abstain) → Generate/check L2 → dedup-group & stratify
  → SFT on oracle plans + structured abstains（off-policy floor）
  → Calibrate guard_train vs guard_eval on oracle × L2      ★
  → Tune on dev_smoke / dev_tune（reflector = frozen prompt）
  → RSFT: sample K → dual-branch filter → targeted oracle replay
          → retrain from SFT ckpt（lift pass@1）             ←主升
  → DPO: 3-way pairs from CURRENT policy, normalized,
         NLL-anchored, executor as judge（精修/对照）
  → Compare 5 arms × {planner-only, full-system} + CI on dev_select
  → Freeze（data + system）→ Final, once
```

### Thesis 主轴（method 章开头可直接用）

```
We use deterministic execution as a verifiable reward anchor.
SFT teaches the planner to speak the schema — including how to abstain.
RSFT internalizes executable, faithful successful plans,
  with calibrated faithfulness guards that survive paraphrase.
DPO then refines preferences between faithful executable plans
  and realistic failures — including the preference to abstain
  over hallucinating a plan the KG cannot support.

Core claim: a verifier-filtered LLM pipeline can bootstrap its own
  training data; the resulting trace-tuned model improves executable,
  graph-grounded reasoning beyond RAG and prompting-only baselines.
```

---

## 附：三种训练方式定位速查

| | policy | 用样本 | 学什么 | 何时用 |
|---|--------|--------|--------|--------|
| **SFT** | off-policy | gold plan + 结构化弃答（正） | 别人的正确 | 建 floor，从零到会 |
| **RSFT** | on-policy | 自筛正确 plan + 正确弃答（净正，双分支 guard） | 自己的正确 | 抬 capability，偶对变稳对 |
| **DPO** | on-policy | 同源 chosen/rejected（正+负，归一化后，当前 policy 重采） | 对 > 错 & 弃答 > 硬编 | 精修 + 对照臂 |

- **accept 标尺**（RSFT 收样本 / DPO chosen 通用）：可答题 `schema + faithful(guard_train) + executor correct(oracle_match is True)`；不可答题 `schema + 结构化弃答 + 理由类别对`。
- **reflector**：独立 prompt 模块，不微调，推理时出场；DPO 阶段兼职 pair 生成器（chosen 须 executor 复验）。
- **executor**：最终裁判；但收训练样本时必须叠 faithfulness guard，不能单独放行。
- **abstention**：贯穿 SFT mix → RSFT 分支 → DPO 路0 → 评估指标 → M5 gate，全链路一等公民。
- **GRPO**：归 future work（除非 HPC 可用）。
