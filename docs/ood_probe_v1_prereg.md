# ood_probe_v1 — 预注册(生成前 commit;跑一次,报什么算什么)

日期:2026-07-06。状态:**设计冻结**。所有学生 adapter 已训练完毕且冻结,概率上不存在泄漏:
本 probe 的任何行都在全部训练之后生成,且结构签名在全部既有数据中零出现(下述断言)。

## 0. 目的与对称性优势

测量**组合泛化**:已支持算子的未见组合(compositional OOD;非 domain OOD,正文命名遵此)。
天然优点:这是全链路第一个**学生与教师对称**的比较——双方对这些组合同为 zero-shot,
"教师无域内示例"的不对称反驳在本切片上不存在。论文明说。

## 1. 真结构签名(断言的执行层)

签名 = `(answer_operation, bridged := ∃constraint.op=in_subquery, group_by 轴, 比较边类型, 过滤槽组合)`,
由 `gold_plan` 扁平 spec 程序化重建(非 family 名——family 名断言是反身的,已弃用)。

普查(train ∪ final_test ∪ dev_tune ∪ dev_select,12,828 行):**50 个结构签名**;粗胞
`(op, bridged, group_by)` 仅 12 个:桥接只存在 count/sum 两种;分组只存在非桥接 rank_top_k(buyer_name);
compare 的边类型只有 4 种:实体计数对、日期对轴、字段相等、数值阈值(逐家族核对表见 worklog)。

## 2. 五模板与新颖性断言(3 桥接 + 2 非桥接,衰减归因可分离)

| # | 模板 | 结构签名 | 断言 | 最近 train 家族(人工比对差异) |
|---|---|---|---|---|
| B1 | bridge_argmax | (argmax, **bridged**, group=tender_cpv_id) | **NOVEL**✓ | min_max(argmax,直连无分组无桥)/ buyer_cpv_set_count(桥接但 count) |
| B2 | bridge_top_k | (rank_top_k, **bridged**, group=supplier_name) | **NOVEL**✓ | top_k_buyers_cpv:**直连 eq 过滤**+group=buyer_name,无 in_subquery(已核 gold_plan) |
| B3 | bridge_compare_2x | (compare, **bridged**, 边=桥接计数×2) | **NOVEL**✓ | comparison:边为实体(直连计数),非桥接子计划 |
| N1 | temporal_argmax | (argmax, 非桥, group=release_year) | **NOVEL**✓ | temporal_count(count 无 argmax)/ min_max(无分组) |
| N2 | filtered_sum_compare | (compare, 非桥, 边=**过滤聚合 sum**×2) | **NOVEL**✓(边类型轴,非空字段反身) | comparison(边=实体计数)/ additive_sum(sum 无比较) |

断言验证脚本:worklog 2026-07-06 记录的普查代码;生成器将内置同一断言,任何生成行签名命中既有集合即拒绝。

## 3. 成功判据(三分支,先于数据写死)

主指标:各系统 probe 准确率;辅指标 retention := probe_acc / 该系统 compare_v4 可答题准确率。
- **(i) 学生 retention ≥ 教师 retention − 5pt** → 组合泛化随蒸馏保留,主张加强;
- **(ii) 学生 retention < 教师 retention − 5pt(配对显著)** → 蒸馏牺牲组合泛化,进 limitation 如实报;
- **(iii) 全体 probe 绝对准确率 < 30%** → 只证明任务难,不区分假说;报数字,不做主张。
桥接/非桥接分开报(归因分离);任何分支的数字都入总表。

## 4. 可执行性试点(生成前的 go/no-go)

每模板 5 题(共 25)人工构造,过 **fully-local 全管线**:编译成功 + 执行 + oracle 一致。
- 模板通过 ≥4/5 → 纳入;
- 降级路径:5 → 4 → 3 模板,**保持桥/非桥平衡**(若桥接模板落一个:2+2;再落:降为 B1+N1+N2 或 2+1 并如实记录);
- B3(compare 接桥接子计划)与 N1(按年分组 argmax)为已知最高风险,试点结果原样入 worklog。

## 5. 生成规格

- 规模:5 模板 × 80 L1 = 400;其中 50% 过 L2 改写(同一漂移检测门)→ 共 **600 行**,目录 `data/qa/ood_probe_v1/`。
- Oracle:生成器 pandas 直算 + **独立第二实现重算**,不一致即弃行(与 v4 双重验证同纪律);实体支持度门(桥接锚实体 ≥30 条记录;分组胞 ≥3);去重键 = (模板, 锚实体, 过滤组合)。
- 并入现有 test-only 族 `(top_k, rank_top_k)` 行,单独标 `origin=legacy_test_only`。
- final_test **不动**;probe 独立报数。

## 6. 评测矩阵与成本

系统 = **论文 headline 表中出现的全部系统 + 教师 ×3 副本**(外延定义,防真空洞:r2 学生
届时已存在且大概率进 headline 表,矩阵按此定义自动包含它;任何进入 headline 表的后续系统同理)。
当前已知成员:fully-local-qwen、hybrid-qwen(dpo-v1)、fully-local-llama、hybrid-llama(sft)、
r2 学生(训练完成后)、教师 ×3。
成本:学生本地免费;教师 3×600 ≈ 2,300 次 grok + nano ≈ 原收割成本的 ~1/5,可接受。
队列:**hybrid final_test(裁决,最优先,已在跑)→ r2 恢复+训练 → 本 probe 试点→生成→评测**。

**修订记录**:2026-07-06 数据生成前 amend——矩阵从枚举改为外延定义(纳入 r2);此后不再修订。

## 6b. 生成器断言的实现约束(防"因空而过")

N2 的新颖性断言**必须复用第 1 节普查的同一套"比较边类型重建"逻辑**(从 compare_params 的
key 结构分类:sides/direction+pivot/threshold/空),禁止直接读 metric 字段——该字段全语料为空,
读它会对任何值恒判 NOVEL(形式通过、实质未检)。生成器代码 import 普查模块共享实现,不另写捷径。

## 7. 报告承诺

无论落在哪个分支:五模板逐一报绝对值+CI;桥/非桥分开;与 hard-composite 切片并排;
不合并、不挑选、不重跑。若试点导致模板降级,降级本身入论文 limitation。


## 8. Gate self-audit (MANDATORY, added 2026-07-07 after 3 consecutive diagnostic gate bugs)

Track record: the schema diagnostic had three opposite-signed gate flaws (extractor over-counted
failures, shape gate over-counted successes, content gate under-counted the base) — each caught
only by after-the-fact spot-check. This probe has the DENSEST gate logic in the project
(novelty-assertion gate, oracle gate, executability gate), so by that base rate it will have at
least one gate bug. Pre-empt it:

- **Before any full run**, each gate is applied to 10 hand-labelled examples (known expected
  verdict) and the gate's decision must match human judgment on all 10; mismatch -> fix gate,
  re-audit, THEN full run.
- **Random-10 raw dump for every rate**: `scripts/dump_raws.py` prints 10 DETERMINISTIC-random
  decisions (passes AND failures, not failures-only) with full raw + verdict fields. Any
  rate-type number entering a table/figure requires a human to have scanned its random-10 first.
- Novelty-assertion gate specifically: audit 10 by confirming each asserted-NOVEL template's
  reconstructed structural signature is genuinely absent from the printed corpus inventory, and
  10 asserted-PRESENT (existing families) correctly flag as present — both directions.


## 9. Amendments 2026-07-07 (pre-full-run, user-mandated)

**9a. N2 is the weakest novelty — annotate, do not silently rely on it.** B1/B2/B3/N1 are novel at
the COARSE-CELL level (op,bridged,group_by) — strong. N2 (filtered_sum_compare) collides with the
existing comparison coarse-cell AND the `sides` compare-side-type; its novelty rests ONLY on the
filter-slot axis ("filtered-aggregate-sum edge not previously seen"). Rules: (a) keep N2 but the
thesis must state "N2's novelty is filter-slot-axis only, weaker than the other four; read its
decay separately"; (b) if the Part-B production-entry pilot shows N2 problematic, drop to 4
templates (3 bridge + temporal_argmax) per the §4 degradation path. Decision deferred to Part B.

**9b. Executability pilot MUST use the production entry, not a hand-wired path.** The executor's
native op set does NOT include `compare` (grounding.py _SUPPORTED): B3's compare compiles via the
decomposition combine="compare_gt" two-subplan path; B2's rank_top_k compiles to top_k; B1/N1's
group-by argmax compiles to top_k k=1. So §4's answer is not "executes Y/N" but "executes THROUGH A
NON-TRIVIAL COMPILE REWRITE" — and whether that rewrite is faithful cannot be judged by an agent
that assembles its own executor call. MANDATE: Part B runs each template as a hand-written natural-
language QUESTION through the EXACT fully-local pipeline entry used by final_test
(ReasoningPipeline.run / run_compare.py path, local Step-1 + Step-2), NOT a direct
executor/decomposition call. A bypass "executable" does not count. This pre-empts a 4th gate bug:
proving the PRODUCTION path supports a plan shape using a TEST-only path.


## 9c. Oracle-consistency requirement (surfaced by the VOIDED bypass Part B; affects the 600-row double-verification)

The abandoned hand-wired Part B (its 3/5-executable table is VOID — a test-only path, not the
production entry; do not cite it) nonetheless surfaced a real oracle-consistency risk that WILL
systematically mis-verify a batch of bridge/group questions in the 600-row generation unless
fixed. Diagnosis-from-bypass is legitimate; VERDICT still requires the production entry.

Two conventions where the second-implementation oracle (naive pandas) diverges from the runtime,
observed live:
- **Empty-string group handling** (B2): runtime correctly EXCLUDES the empty-supplier group
  ['', 11035]; the naive pandas oracle counted it into top-1 -> false "not-sensible". The 600-row
  oracle MUST replicate runtime's empty-string-group exclusion.
- **Dedup key / multi-value expansion** (B3): boolean answer agreed (False) but the two sides'
  raw counts differ 2.4x (runtime 7749/8683 vs oracle 18784/19718) — a coincidental
  magnitude-ordering match, NOT logical agreement. Likely dedup-key difference (runtime dedups by
  contract_node_id; oracle may row-count) or multi-value supplier_names list expansion. The 600-row
  oracle MUST align its dedup key (contract_node_id) and multi-value handling with runtime.

**Verdict rule for Part B production-entry rerun**: B2/B3 "not-sensible" from the bypass = "oracle
convention PENDING ALIGNMENT", NOT "not executable". Re-judge sensible only AFTER the oracle's
empty-string-group exclusion + dedup key are aligned with runtime. If, after alignment, B3's
compare_gt two-subplan path STILL shows a genuine count discrepancy -> that is a real runtime bug
-> drop B3 to a 4-template probe per §9a. If it was oracle convention -> fix oracle, B3 stays.
This oracle-alignment step is a prerequisite for the whole 600-row double-verification, not just B3.
