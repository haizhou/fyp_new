# 交接文档:CICADA compose 轨道 + PACS 建设(写于 2026-07-18,GPU 迁移前)

> 新会话第一步:通读本文档。第二步:读 `docs/paper_compose/pacs_spec.md`(冻结规格,一切建设以它为准)。
> 第三步:读 `docs/cicada_worklog.md` 的 2026-07-10 之后条目(全部实验数字与决策的唯一可信来源)。

---

## 0. 一句话现状

学位论文已完成(独立冻结,勿动);当前工作是**第二篇独立论文**(compose 轨道:教 8B 本地模型在可验证算子代数上自由组合)——论文初稿已写完,主基准 PACS 的规格已冻结,建设做到第 2 步,**所有 GPU 实验被 malmo 节点的 NVIDIA 驱动锁死阻塞**,正在迁移 GPU。

---

## 1. 必读文件(按序)

| 顺序 | 文件 | 作用 |
|---|---|---|
| 1 | 本文档 | 总览 |
| 2 | `docs/paper_compose/pacs_spec.md` | **PACS 规格 v2.2,已冻结**。所有生成/评测/统计规则以此为准,不得偏离 |
| 3 | `docs/cicada_worklog.md`(2026-07-10 起) | 全部实验方法+结果+决策,含 compose v1/v2/v3 三轮、边界实验、审计 |
| 4 | `docs/paper_compose/paper.md` | 论文全文初稿(已过两轮用户审稿修正) |
| 5 | `docs/paper_compose/outline.md` | 论文结构依据 |
| 6 | `docs/thesis_draft/09_compose.md` | 学位论文版章节(与 paper.md 独立,勿混) |

---

## 2. 关键环境知识(踩过的坑,勿重踩)

1. **嵌套双仓库**:真仓库是 `~/fyp_new/fyp_new`(内层)。外层 `~/fyp_new` 是废弃旧历史,remote 已摘除。**所有命令先 `cd /home/uceeh01/fyp_new/fyp_new`**;后台命令用绝对路径(shell cwd 会被重置,已因此炸过 4 次)。
2. **两个 HF 缓存**:`/var/tmp/cicada/hf` 完整可用;`~/.cache/huggingface` **不完整会静默炸**。任何 serve/训练必须 `HF_HOME=/var/tmp/cicada/hf`。
3. **outputs 是符号链接** → `/var/tmp/cicada/outputs_store/`。**迁移新机器时 /var/tmp 不会跟着走**——适配器(每个 698MB:`qwen3_8b_compose_sft_v1/v2/v3`、旧阶梯的 dpo/sft/step1 等)必须手工拷到新机器,否则一切评测无从谈起。数据 `data/`(KG parquet、基准、compose 池)在 git 内,clone 即得(大文件已 committed)。
4. **malmo 节点驱动锁死**(2026-07-17 起):CUDA 重型子进程一律 SIGSEGV,GPU1 卡死,仅真重启可解。这是迁移 GPU 的原因。诊断台账在 worklog 2026-07-17 三条。
5. serve 命令模板:`HF_HOME=/var/tmp/cicada/hf bash scripts/serve_student.sh Qwen/Qwen3-8B cicada-qwen3-composev3 <GPU_UUID> outputs/qwen3_8b_compose_sft_v3`(GPU 用 UUID 不用序号)。评测驱动:`scripts/run_compose_probe_eval.py`(有 --guided/--resume/--old-benchmark-abstention 双指标)。

---

## 3. 已完成(全部 commit,数字勿再算)

- **compose v1→v2→v3 三轮**:B_clean 91.72→98.99→99.17%;构造扣留 keys_where 64.8%(v1)、intersect 39/39(v2)、19/19(v3,三轮零演示);顺序缺口 44.7→24.0→8.7;旧题 400 样本 87.0/87.5/87.25%;弃答 12/12。检查电池:泄漏审计(抓过 196 行旧题通道泄漏)、求值器审计 300/300、遮蔽负对照、few-shot 边界实验(C3/C5 两演示→100%)、pass@16。
- **论文初稿** + 17 处审稿修正(num 节点补齐=17 节点、口径修正、措辞降级、Wilson CI)。
- **PACS 规格 v2.2 冻结**:7 任务族×3 深度、状态轴横跨、depth/exposure 正交、三配对表面通道、intent 簇统计、dev20/test80、~1,100–1,350 簇、隔离账本(七标识符+五道零重叠硬门)、模型冻结规则(首轮结果必须来自现 compose-v3,PACS 不得反哺 v4)、test 单评+技术重跑条款。
- **PACS 建设 ①②**:`scripts/pacs/identifiers.py`(七 ID+逻辑签名导出,已验证)、`scripts/pacs/templates.py`(21 单元金树模板,63/63 冒烟通过)。

## 4. 待做(按序)

### 无需 GPU(当前主线)
- **③ 生成驱动器** `scripts/pacs/generate.py`:按规格配额(每 family×depth 单元 40–50 可答意图,unseen 15–20,用模板的 decoration 钩子把形状引导出训练签名集——训练签名从 `data/training/llamafactory_compose_v3/` 的树重算,方法见 `scripts/export_compose_sft.py` 的审计段);逐意图过双求值器(`compose/eval_runtime.py` + `scripts/compose_independent_eval.py`),不一致即弃;族级敏感性门(弃空/零/刀刃/超大集);锚点复用 ≤3;状态变体派生(ambiguous/empty_result/unsupported_field/requires_missing_operator,每族 ≥20% 行、≥2 类);七标识符齐全;**五道零重叠构建门失败则拒绝封存**(规格 Isolation ledger 节)。
- **④ 独立 surface grammar** `scripts/pacs/surface_independent.py`:第二套题面文法,**不得共享**训练渲染器(`build_compose_train.py` 里的句干/连接词/排列策略)的任何资产。
- **⑤ 自然化门**:LLM 改写 + 字面量门(实体/数字/弃答线索 verbatim)+ **逻辑签名门**(操作/否定/比较方向/量词/左右作用域五检,identifiers.logical_signature 就是靶子)。
- **⑥ 生成 + intent 级 dev/test 切分(20/80,整簇移动)+ 审计**(每单元 ≥5 自然化实例人读,高危构造与全部状态类型加抽,用户参读;十样本铁律)。**PACS-test 生成后封存,冻结前不逐行查看。**

### 需 GPU(新卡就绪后,按序)
1. 起 serve(见 §2.5)→ 跑通一次 probe 冒烟确认环境。
2. **PACS 评测**:冻结的 compose-v3 + 裸 Qwen3-8B 对照;先 PACS-dev,后 **PACS-test 一次**(单评规则+技术重跑条款见规格)。
3. **Companion A 配对**:compose-v3 全量 final_test 2,285(`data/qa/compose_train_v3/final_test_full.jsonl` 已备好),`--guided --old-benchmark-abstention`,逐题保存,与旧冠军逐题 McNemar(旧结果在 `outputs/eval/final_test/`,注意 outputs 迁移),分桶+成本指标。
4. 补格:v3-base 与 discordants(同 B_clean_v3 行);B_anchor 固定集三版本重跑。
5. 结果回填 `paper.md`:§5.2/§5.5 两处 [Pending] 方括号 + 主表换 PACS-test 任务族表(格式见规格 Main table 节)。

---

## 5. 约束纪律(违反=返工,全部有据)

- **模型冻结**:PACS 首轮官方结果必须来自现在的 `outputs/qwen3_8b_compose_sft_v3`,PACS 暴露的问题只进 Discussion/future work,不训 v4。
- **主张纪律**:训过的构造=覆盖修复,不算泛化;泛化只引"测量时零演示"格子与逐轮重扣的 B 集。
- **统计单位** = intent 簇;三表面是配对测量,绝不当三倍样本;意图级 cluster bootstrap。
- **铁律**:任何 gate 出的数字,先人读 10 个确定性随机 raw 再引用。
- 求值器 v1.1(gte/lte 数字字符串煤准)是现行约定,两实现同步。
- 学位论文(`docs/thesis_draft/`、`main (2).tex`)与主线 85.65% 结果**全部冻结勿动**。

## 6. 当前 todo 快照

见会话 todo(若丢失按 §4 顺序重建)。用户侧待办:①新 GPU/机器就绪通知;②适配器从 malmo `/var/tmp/cicada/outputs_store` 迁出;③PACS 审计排期;④(若仍要修 malmo)与朋友协调后请管理员重启节点。
