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

系统:fully-local-qwen、hybrid-qwen(dpo-v1)、fully-local-llama、hybrid-llama(sft)、教师 ×3 副本。
成本:学生本地免费;教师 3×600 ≈ 2,300 次 grok + nano ≈ 原收割成本的 ~1/5,可接受。
队列:**hybrid final_test(裁决,最优先,已在跑)→ r2 恢复+训练 → 本 probe 试点→生成→评测**。

## 7. 报告承诺

无论落在哪个分支:五模板逐一报绝对值+CI;桥/非桥分开;与 hard-composite 切片并排;
不合并、不挑选、不重跑。若试点导致模板降级,降级本身入论文 limitation。
