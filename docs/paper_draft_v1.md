# 论文重写稿 v1 — Abstract / Introduction / Methodology

> 按 ACL·EMNLP·TMLR 的实际写法重写,不参考原稿。
> 英文是可直接使用的正文;`> 注:` 开头的中文是写作说明,交稿前删除。
> 所有数字都经过代码或论文核实,核实状态见文末「证据台账」。

---

## 0. 先解决三处必须改的事实问题

> 注:这三条会直接影响下面的措辞,先看。

**(1) 「12 个规范化变换」不成立,实际是 10 个。**
`graph_planning.py :: normalise_graph_plan` 里出现的标号是 `T1 T2 T3 T4 T7 T8 T9 T10 T11 T12` —— **T5 与 T6 不存在**。标号最大到 12,但实际变换是 10 个。下文一律写 **ten**。

**(2) 「三道编译闸门」成立,已逐一定位。**
- `ungroundable_variable:{var_id}:{reason}` — 全程序可接地(graph_planning.py:461)
- cycle 检测经 `level_reason` — DAG 合法性(:463–467)
- `unconstrained_bind_source:{var_id}` — 无未受约束的绑定源(:484)

**(3) 头条 99.88% 是 `flat` 约定,不是 `edge`。**
工作日志 2026-07-04:「v2 flat 约定一致率 99.88%(14,752/14,770)」。而 `--convention flat` 下 `load_frames_flat()` 会 `import ParquetKGQueryBackend` 构造扁平视图。
**因此不能再写 “written without shared execution code”。**正确表述是:第二实现独立重写了**全部算子与聚合逻辑**,并共享记录全域的物化(源码注释:*"op implementations remain this module's own"*)。下文已按此改写。

---

## 1. Abstract

> 注:目标 190 词上下,五步走(任务 → 前人不足 → 提出 → 怎么做 → 一个结果)。
> 摘要里**不出现**置信区间、p 值、replicate 明细 —— 那些是 Results 的事。

Question answering over structured public records demands two capabilities that
rarely coexist: reading an underspecified natural-language question, and
producing an answer a reviewer can re-derive. Language models supply the first
and cannot guarantee the second; query engines supply the second but cannot
read. Neuro-symbolic systems bridge the two, but are typically evaluated over
curated schemas whose field semantics and entity identities are already clean —
an assumption that fails on administrative data. We present a framework in which
a language model writes a program in a closed typed algebra and deterministic
code alone produces the answer. We instantiate it over 166,277 UK
public-procurement releases, where a single organisation appears under as many
as 77 platform identifiers in one year and monetary fields reported at four
levels are not mutually additive, so that a program can execute cleanly and
still compute the wrong quantity. No human-written programs are used for
training: candidate programs are sampled from a model, and only those that
compile, ground, execute, and reproduce an independently recomputed oracle are
retained. An 8B student trained solely on this verified region reaches 85.65% on
a sealed 2,285-question test, above the 69.76% of the cloud model that proposed
its training candidates. A controlled access ablation locates the mechanism:
retrieval answers single-record questions at 66.5% but falls below 4% on
questions requiring exhaustive aggregation, even when the gold program is
supplied as the retrieval query.

> 注:为什么这样排。
> 第 1–2 句立任务并制造张力;第 3 句点前人不足(**curated schemas** 是你真正的差异点);
> 第 4 句一句话提出;第 5 句给领域,并且**把「可执行但算错」这个核心难点放进摘要** —— 这是你区别于一般 KGQA 的地方;
> 第 6 句给训练规则;最后两句给两个数字,一个是主结果,一个是机制。
> 全文只有 5 个数字:166,277 / 77 / 85.65 / 69.76 / 66.5 与 4。
> 对照 PGR(EMNLP 2025)只有 1 个数字、153 词,你这个密度已经偏高但仍可接受,因为你有两条独立主张(蒸馏结果 + 访问消融)。

---

## 2. Introduction

> 注:五段,每段只干一件事。第三段是全文最重要的一段 —— 它必须在你提出任何东西**之前**,
> 论证「为什么必须是可执行程序」。PGR 就是这么写的,而你原稿缺这一段。

### ¶1 — 问题

Large language models can read a question and produce a fluent answer, but
neither the answer nor the reasoning that accompanies it is guaranteed to be
executable or checkable. When a task requires several linked operations — a
join, a filter, an aggregation — a model may retrieve the right records while
applying the wrong relation, counting the wrong unit, or summing values that
should have remained separate. The failure is not merely that the answer is
wrong. It is that nothing in the output tells a reader *where* it went wrong.

### ¶2 — 前人工作及其不足

Three lines of work address parts of this. Retrieval-augmented generation
places relevant records in context, but supplies no guarantee about the joins,
counts, or aggregations performed over them. Neuro-symbolic semantic parsing
maps language to executable programs and recovers checkability, but is
typically evaluated over curated benchmarks whose field semantics and entity
identities have already been resolved. Bootstrapped training methods such as
STaR and rejection-sampling fine-tuning learn from a model's own selected
outputs, but presuppose an acceptance test that is cheap and trustworthy. On
real administrative data none of these presuppositions holds: identity is
fragmented, field semantics are contested, and successful execution does not
establish that the program answered the question that was asked.

### ¶3 — 为什么必须是可执行程序(核心论证段)

Consider what it takes to answer *how much has this buyer spent under this
category*. The answer is not contained in any bounded set of retrieved records;
it is a property of the entire matching population. A system that inspects the
top *k* records can restate them fluently but cannot, even in principle,
enumerate a set it never saw. Exhaustiveness is therefore not a matter of
retrieval quality but of representation: the question must be turned into an
object that *ranges over* the data rather than one that *reads* a sample of it.
A program is such an object. It also has a second property that matters more
here than expressiveness: if the program is drawn from a closed, typed language,
then membership in that language is decidable, the language compiles to a
grammar that constrains decoding, and — critically — the language is small
enough that a second party can reimplement it and check the first
implementation's answers. Correctness over dirty administrative data cannot be
established by the executing system alone, because the executing system also
encodes the conventions under test.

### ¶4 — 我们做了什么

We build the task and the system end to end. Records are normalised into a
canonical layer of 215,221 award records and 131,502 resolved organisations
under precision-first identity rules, in which fuzzy similarity never writes a
merge. Questions are answered in two model stages followed by five
deterministic ones. A *reader* turns the question into a briefing that names the
answer signature, the literal constraints, and a dependency-ordered program
skeleton whose fields are still natural language. A *planner* expands that
skeleton into a program over seven value types and seventeen operators, emitted
under grammar-constrained decoding. Deterministic code then grounds the program
to the data layer, compiles it under three admission gates, executes it, and
verifies properties of the result that the executor has no reason to doubt. The
answer is always produced by execution, never by generation. For training, no
human writes a program: sampled candidates are retained only when they compile,
ground, execute, and reproduce the oracle of a 12,828-question benchmark whose
answers a second, independently written evaluator recomputes at 99.88%
agreement.

### ¶5 — 贡献与结果预览

> 注:顶会惯例是 itemize。三到五条,每条一句话,**动词开头**。

Our contributions are:

1. **A verified-supervision training regime for program-writing models.**
   Acceptance is oracle match rather than execution success, on the grounds that
   a program can compile, ground, and execute perfectly while computing the
   wrong quantity. An 8B student trained only on the accepted region reaches
   85.65% on a sealed 2,285-question test against 69.76% for the cloud model
   that proposed its candidates, and the ordering replicates on a second base
   model.
2. **A controlled decomposition of where the performance comes from.** Access
   baselines separate the contribution of the executable scaffold from that of
   model training. Retrieval answers single-record questions at 66.5% and
   collapses below 4% on exhaustive aggregation; supplying the gold program as
   the retrieval query does not recover it, which rules out query formulation as
   the cause and isolates enumeration.
3. **A benchmark whose oracles are independently audited.** 12,828 questions
   carry executable gold programs; a second evaluator that shares no operator
   implementation with the system recomputes 14,752 of 14,770 audited cases.
   The 18 residual disagreements are attributable to two declared heuristics in
   the auditor and are reported rather than absorbed.
4. **Two negative results we consider load-bearing.** Acceptance rate rises
   across self-training rounds while the evaluation gain of the second round is
   0.4 points and not significant, so acceptance rate is not a proxy for
   capability gain. And an LLM selector over accepted candidates does not
   improve on taking the first accepted candidate, while a small deterministic
   scorer over execution features does.

> 注:第 4 条是我建议你加的。顶会评审很吃「作者主动报告负结果」这一套,
> 而你恰好有两个真负结果。原稿把它们埋在附录里,浪费了。

---

## 3. Methodology

> 注:七小节,严格对标 ACL/EMNLP 的 method 结构。
> 最重要的改动:**新增 §3.1 Problem Definition(你原稿完全没有)**,
> 以及**把训练数据构造搬进 Method(原稿在 Evaluation 里)**。

### 3.1 Problem Definition

> 注:这一节是「methodology 不清楚」的头号解药。形式化不必花哨,但必须有。

Let `D` be a record universe derived from a corpus of administrative releases,
and let `S` be its schema: a finite set of typed fields together with a set of
typed relations among entity classes. A question `q` is a natural-language
utterance whose answer, if it exists, is a function of `D`.

We seek a mapping

```
    M : q  ↦  (a, E, π)
```

where `a` is the answer, `E ⊆ D` is the set of records that produced it, and
`π` is a program in a language `L` such that executing `π` over `D` yields
exactly `(a, E)`. Three requirements distinguish this from standard semantic
parsing.

**R1 — Execution is the sole answer authority.** `a` is defined as
`exec(π, D)`. No component may emit `a` directly. A model that produces a
fluent answer without a corresponding `π` has not solved the task.

**R2 — Abstention is in the codomain.** `M` may return `⊥` with a reason drawn
from a fixed set: the question is ambiguous, it is unsupported by `S`, or it is
supported but matched by no records. These are answers, not errors, and are
scored as such.

**R3 — Membership in `L` is decidable.** For any candidate `π̂`, a type
checker decides `π̂ ∈ L` and, when it does not, returns a typed diagnostic and a
path into `π̂`. This is what makes both constrained decoding and independent
re-implementation possible.

Under R1 the learning problem is not "produce the answer" but "produce a
program whose execution is the answer", and the training signal is therefore
available without any human writing a program — provided an oracle exists for
`q`. §3.7 constructs that oracle and §3.6 states the acceptance rule.

### 3.2 Framework Overview

> 注:一段话 + 一张表。表里最重要的一列是「谁负责」—— 那是你整篇论文的卖点。

`M` factorises into two model stages and five deterministic stages. The
factorisation is not merely engineering convenience: it places a typed,
inspectable interface between language understanding and computation, so that a
failure can be attributed to one side or the other.

| # | Stage | Owner | Input → Output |
|---|-------|-------|----------------|
| 1 | Reader | model | `q` → briefing `z` |
| 2 | Planner | model | `(q, z)` → candidate program `π̂` |
| 3 | Grounding | deterministic | `π̂` → `π̂ᵍ` with fields and entities bound to `S` |
| 4 | Compilation | deterministic | `π̂ᵍ` → executable plan, or a typed rejection |
| 5 | Execution | deterministic | plan → `(a, E)` |
| 6 | Verification | deterministic | `(π, a, E)` → check reports |
| 7 | Repair | deterministic control | failure diagnosis → bounded retry |

Stages 3–7 contain no model. This is the property we rely on throughout: the
model proposes the structure of the reasoning, but has no authority over whether
that reasoning is correct.

### 3.3 The Program Language `L`

Programs are trees over seven value types: `records` (a subset of `D`),
`values` (a distinct set of scalars), `groups` (group key → number), `ranking`
(an ordered list of key–value pairs), and the scalars `number`, `value`, and
`bool`. Seventeen node types are defined over these; depth is capped at 16 and
node count at 64.

| Node | Signature | Function |
|------|-----------|----------|
| `filter` | predicates → records | filter records |
| `values` | records → values | project distinct values |
| `count` | records → number | count records |
| `size` | values → number | set cardinality |
| `sum` | records → number | guarded money sum |
| `exists` | records → bool | test non-emptiness |
| `select` | records → value | unique field lookup |
| `extreme` | records → value | record argmin/argmax |
| `groupby` | records → groups | grouped count or sum |
| `argext` | groups → value | extremum group key |
| `top` | groups → ranking | top-*k* groups |
| `num` | literal → number | numeric literal |
| `combine` | number² → scalar | arithmetic or comparison |
| `vcompare` | value → bool | scalar comparison |
| `setop` | values² → values | set operation |
| `gcombine` | groups² → groups | key-wise group combination |
| `keys_where` | groups → values | thresholded group keys |

The language is deliberately narrow. Narrowness buys the three properties
argued for in §2¶3: decidable membership (R3), compilation of the grammar to a
JSON schema for constrained decoding, and an implementation small enough to be
independently rewritten — which §3.7 relies on.

> 注:表格直接来自论文附录 Table `tab:algebra`,已核实为 17 个;
> WTQ 迁移新增的第 18 个 `extreme_rows` 放到 portability 一节再提,别在这里混进来。

### 3.4 Stage 1 — The Reader and the Briefing

The reader maps `q` to a briefing `z` without touching `D`. `z` is a *program
skeleton with unbound fields*: it commits to the shape of the computation while
deferring every schema decision.

Concretely `z` carries an answer signature (operation, value type, and the text
naming the requested field) and an ordered node list in which each node declares
its inputs. A verified training example, taken unmodified from the corpus:

```jsonc
// q: "...for Powys County Council, under CPV 85000000, which supplier is
//     listed on the contract notice awarded to ABICARE SERVICES LIMITED?"

answer_signature: { operation: "select", value_type: "entity",
                    answer_field_text: "supplier" }
program: [
  { id: "A", op: "filter_records", inputs: [],
    args: { filters: [
      { field_text: "release_year",       operator: "eq", value: "2025" },
      { field_text: "buyer organisation", operator: "eq", value: "Powys County Council" },
      { field_text: "CPV code",           operator: "eq", value: "85000000" } ] } },
  { id: "B", op: "filter_records", inputs: ["A"],
    args: { filters: [
      { field_text: "supplier_name", operator: "eq", value: "ABICARE SERVICES LIMITED" } ] } }
]
```

Two features carry the design. First, `inputs: ["A"]` makes the dependency
between the two steps explicit, so the intermediate set the answer depends on is
named before any data is touched. Second, the field names are still natural
language — `"buyer organisation"`, `"CPV code"` — and are bound to `S` only in
§3.5. The briefing therefore commits to *what the question needs*, not to *how
this schema stores it*.

The briefing also carries a slot for information the question requires but the
schema cannot supply. Populating it is how the system reaches R2's `unsupported`
outcome before planning begins rather than after execution fails.

> 注:这一段是你和 UPFT 那条线接得上的地方,但**在论文里不要提 UPFT**(那是交流用的)。
> 论文里只要把「语义前缀」这个性质说清楚即可,让读者自己联想。

### 3.5 Stages 3–4 — Grounding and Compilation

**Grounding** binds each `field_text` to a canonical slot of `S` and each entity
mention to the resolved organisation registry. Both use the same discipline:
candidate generation, a type gate, a confidence threshold, and a margin between
the top two candidates. The top candidate is never taken merely for being top.
Where a relation phrase and a role noun conflict, the relation phrase wins, so
*awarded to X* binds X as supplier irrespective of surface order.

**Compilation** first applies ten normalisation transforms that rewrite
mechanically recognisable planner idioms into the executable structure they
denote, without guessing semantics. Representative cases: a filter whose value
names another variable is a dependency edge rather than a constraint (T1); an
empty organisation filter is dropped because a bind, not a literal, carries it
(T2); a comparison missing its left operand compares its own input (T4); a
same-role entity set chained onto another is a re-filter of one set rather than
a second hop, and its record-level filters fold into the root (T7).

Dependency recovery is the subtle part. Variable-naming conventions imply an
ordering, but planners routinely reuse a suffix to mean the opposite relation —
`b1a1` may mean *feeds a1* or *derived from a1*. When the naming-derived edge
contradicts an explicit `depends_on`, the explicit declaration wins and the
inferred reverse edge is discarded, so the compiler never emits a two-cycle.
Separately, a declared relation *from S to T* is treated as a genuine dataflow
edge, because binds are applied per dependency and a relation without a matching
edge would silently execute unbound.

The compiled plan must then clear three admission gates. Failure at any gate
returns a typed reason rather than a degraded answer:

| Gate | Rejection reason | Meaning |
|------|------------------|---------|
| Groundability | `ungroundable_variable:<id>:<why>` | some variable carries a literal that binds to nothing |
| Acyclicity | cycle detected during levelling | the dependency graph is not a DAG |
| Bind safety | `unconstrained_bind_source:<id>` | a bind source is not itself constrained |

### 3.6 Stages 5–7 — Execution, Verification, Repair

**Execution** proceeds by dependency level with bind propagation, exact decimal
arithmetic, contract-level deduplication, and date-aware comparison. Two details
are worth stating. Preflight failures are mapped to distinct statuses — a
missing field yields `schema_error`, an unmet exhaustiveness requirement yields
`incomplete_evidence`, and the remainder yield `constraint_conflict` — so that
repair receives a localised diagnosis rather than a generic failure. And
`count`/`exists` take a fast path through a vectorised backend count with a
capped evidence sample, so that the second hop of a bridge query is never
materialised in full.

**Verification** is separate from execution by design: it tests properties of
the result that the executor has no reason to doubt.

| Phase | Check | Condition |
|-------|-------|-----------|
| pre | `schema_fields` | every referenced field exists in the backend |
| pre | `constraint_conflicts` | no two constraints are mutually unsatisfiable |
| post | `answer_uniqueness` | a claimed unique answer matches ≤ 1 record |
| post | `sum_additive_safety` | every summed value is flagged additive |
| post | `population_coverage` | an enumerated set covers its population |

`sum_additive_safety` reports explicitly when it cannot check — no rows, or a
backend that does not expose the additivity flag — rather than passing silently.
This matters because non-additivity is the failure mode that motivates the whole
data layer: a framework ceiling repeated across its call-offs is the canonical
way for a valid program to compute a wrong total.

**Repair** is gated. It fires only on a diagnosable defect and is bounded at
three rounds. Refusals and meaningful empty results are terminal; an earlier
ungated version converted six correct abstentions into confident wrong answers,
which is why the gate exists. The controller emits one of four actions:
`no_repair_needed`, `mark_unsupported`, `ask_clarifying_question`, or
`report_insufficient_evidence`.

### 3.7 Benchmark Construction and Oracle Audit

> 注:如果投会议,这一节可以拆成独立的 §4 Benchmark;若是毕设,留在 Method 里更连贯。

**Plan-first generation.** Questions are not written and then annotated. A
parameterised program is instantiated, executed against `D`, and only then given
a natural-language surface. Every row therefore ships with an executable gold
program and a computed answer.

Programs are instantiated by sampling *constraint groups* over the flattened
record universe. For a chosen column combination — for instance
`(release_year, tender_category)` — the universe is grouped, and groups are
admitted only when their cardinality falls inside a band, when no grouping key
is null or empty, and after a deterministic sort. The band is the substantive
choice: its floor (three records for aggregation families) excludes degenerate
questions whose answer is trivially recoverable, and its ceiling (20,000)
excludes groups too large to audit. The admitted group key becomes the
constraint set and its aggregate becomes the oracle.

> 注:**这里必须写实话** —— 是「扁平记录上的约束组采样」,不是「图上的子图采样」。
> 代码在 `samplers.py :: _moderate_groups`,审稿人一看就知道。而这个做法本身完全站得住。

Multi-hop questions are generated by seven bridge families, in which the first
hop resolves an intermediate entity set and the second aggregates over it. The
constraint is expressed as an explicit subquery, and degenerate instances are
discarded — a bridge over fewer than two intermediate entities is not a bridge:

```python
suppliers = {r["supplier_name"] for r in query(backend,
              [{"field": "buyer_name", "op": "eq", "value": buyer}])}
if len(suppliers) < 2:      # degenerate: not a genuine second hop
    continue
constraints = [{"field": "supplier_name", "op": "in_subquery",
                "value": {"resolve": "suppliers_of_buyer", "buyer": buyer}}]
```

**Surface levels.** Each plan receives surfaces at three levels: a template
rendering, a model paraphrase, and an adversarial reformulation. *All surfaces
of a plan share one oracle*, so accuracy differences between levels isolate
language-to-program generalisation from answer difficulty. Model-written
surfaces pass a deterministic gate before admission — required atoms preserved,
no new numerals, no new organisation names, unanswerability triggers retained,
and the surface actually rewritten — with rejection sampling on failure.

**Oracle audit.** Because the oracles were produced by the generator and
executor family, "correct" is circular until an independent implementation
recomputes them. We therefore wrote a second evaluator in pandas that
**reimplements every operator and aggregation without reference to the system's
executor**, and recomputed the audited set. Agreement is 99.88%, or 14,752 of
14,770 cases.

Two qualifications belong in the record. First, the reported figure uses the
flat record convention, under which the second evaluator shares the
materialisation of the record universe with the system while implementing all
operator semantics independently; an edge-level convention that also rebuilds
the universe from the raw node and edge tables differs on bridge families, and
that divergence is a documented convention difference rather than a defect.
Second, five question families do not serialise every parameter into the gold
program — thresholds and comparison sides survive only in the surface — and the
auditor recovers these by parsing the surface. These families are declared in
the auditor and reported separately. All 18 residual disagreements fall inside
the two declared heuristics: fifteen are top-*k* parameter inference, where the
auditor assumes rank-three-by-count, and three are boolean surface parses.

> 注:这段是整篇论文最能体现方法学水准的地方。原稿把 99.88% 当成一个成绩来报,
> 我改成把**它的边界**也一起报。审稿人对后者的评价远高于前者。

### 3.8 Training Data Construction

> 注:**这一节原稿在 Evaluation 里,必须搬到 Method。**
> 理由:它是方法的一部分,不是评测的一部分。V-STaR 就是把整个迭代过程放在 §3。

No human writes a program. Training data is produced by sampling candidates
from a model and filtering them through the pipeline of §3.5–3.6.

**Acceptance rule.** A candidate is retained iff it (i) type-checks under `L`,
(ii) grounds completely against `S`, (iii) executes without a verification
failure, and (iv) reproduces the benchmark oracle — where reproducing the oracle
includes *correctly abstaining* on a row whose gold status is abstention.
Acceptance is oracle match, not execution success. The distinction is the
methodological core of this section: a program that runs cleanly proves only
that it was well-formed, not that it computed the quantity the question asked
for.

The same rule governs the reader. Because the briefing is upstream of every
later stage, a briefing is accepted exactly when the whole pipeline succeeded
beneath it; no separate judgement of briefing quality is made or needed.

**Yield.** Over 8,555 answerable training questions, the teacher pass accepts
5,605 traces. Self-harvest rounds, in which candidates are sampled from the
student rather than the teacher, accept 6,556 and 6,685 on the two base models.

**A negative result on iteration.** Acceptance rises again in a second
self-harvest round, to 86.3%, while the evaluation gain of that round is 0.4
points and not statistically significant. We therefore report the loop at its
measured ceiling rather than iterating further, and note the general point:
**acceptance rate is not a proxy for capability gain**, and a self-training loop
that is monitored only by its acceptance rate will not notice when it has
stopped learning.

Accepted traces are exported with family caps, an upper bound on the share of
abstention demonstrations relative to the accepted plan pool, and a
hash-ordered shuffle for determinism.

---

## 4. 证据台账

> 注:这张表是给你自己用的,交稿前删。每一行都标了我核实到哪一步。

| 主张 | 状态 | 依据 |
|---|---|---|
| 7 值类型 / 17 算子 | 代码+论文核实 | `main.tex` 附录 Table `tab:algebra`,逐行数过 |
| **10 个规范化变换(非 12)** | **代码核实,与原稿冲突** | `graph_planning.py` 中标号为 T1–T4, T7–T12,**无 T5/T6** |
| 3 道编译闸门 | 代码核实 | graph_planning.py:461 / :463–467 / :484 |
| 5 项验证检查 | 代码核实 | `verifier.py` 中 `_check("...")` 具名调用 |
| 修复上限 3 轮 | 代码核实 | `reflector.py:41` `max_rounds: int = 3`;`pipeline.py:66` 同 |
| 4 类反思动作 | 代码核实 | `reflector.py` `ReflectionAction(...)` |
| 执行器支持 10 个操作 | 代码核实 | `executor.py` `_SUPPORTED` 集合 |
| 约束组采样 + 基数带 | 代码核实 | `samplers.py :: _moderate_groups` |
| 聚合题证据下限 3 / 上限 20,000 | 代码核实 | `SamplerConfig` |
| 7 个 bridge 家族 | 代码核实 | `targeted_v2/builder.py:550-556` |
| L1/L2/L3 共享同一 oracle | 代码核实 | `build_multilevel_qa.py` 文档字符串 |
| L2/L3 五项表面门控 | 代码核实 | 同上 + `qa.multilevel.check_surface` |
| **99.88% 用的是 flat 约定** | **代码+日志核实,与原稿措辞冲突** | 工作日志 2026-07-04;`load_frames_flat` 会 import `ParquetKGQueryBackend` |
| 残余 18 = 15 top_k + 3 boolean | 代码+日志核实 | `rank_top_k` 硬编码 top-3;`UNDERSPECIFIED_FAMILIES` |
| 8,555 → 5,605 / 6,556 / 6,685 | 论文核实 | `main.tex` ladder 表 |
| 第二轮 +0.4 分不显著 | 论文核实 | `main.tex` |
| 验收 = oracle match 含正确弃答 | 代码核实 | `export_step1_sft.py` 文档字符串 |
| 166,277 releases → 215,221 awards | 论文核实 | `main.tex` §1 |
| 85.65 / 69.76 / 78.31 / 50.33 / 66.5 / 3.7 / 3.3 | 论文+结果文件核实 | `outputs/eval/final_test/*` |

### 仍未解决的一项

**训练文件行数与论文数字有个位数差异**:`verified_sft.jsonl` 实际 5,598 行 vs 论文 5,605;
`rsft_qwen_r1/verified_sft.jsonl` 实际 6,550 vs 论文 6,556。
差值分别是 7 和 6。可能是导出时的去重或后处理,但我没有找到确证。
**建议**:要么定位到差异原因并在脚注说明,要么统一改用文件实际行数。这个数字级别不影响任何结论,但被问到时要答得出。
