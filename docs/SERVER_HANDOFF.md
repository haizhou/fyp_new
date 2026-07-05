# SERVER HANDOFF — 给服务器端 Claude 的任务书

你在一台带 A100 的服务器上,刚 clone 下 CICADA 仓库。CICADA 是一个 UK 公共采购 KGQA
系统(FYP 项目):两步 LLM 规划(nano Step-1 简报 → Step-2 图计划)→ 确定性编译/执行/
验证 → 门控修复。论文主张 **partial verifiability**:验证器过滤的管线自举出本地学生
模型的训练数据。你的任务:**完成 8B 学生阶梯的训练与评测**。

先读(按顺序):
1. `docs/training_runbook.md` — 操作手册,你的主线剧本
2. `docs/cicada_worklog.md` 最后几条(2026-07-05)— 数据是怎么冻结的、修过哪些坑
3. `docs/PROJECT_STRUCTURE.md` — 代码地图(只看第 8-10 节即可)

## 已经完成、你不要重做的

- 教师收割 9,267/9,267(`data/qa/teacher_full_v1/`),Decimal 形状门 bug 已修+打捞
- 正式导出 `data/training/llamafactory_v1/`(plan_sft 2,787 / repair_sft 1,679 / dpo 390,
  族帽 150 + 桶帽 400 + 弃答 10% 帽都已生效)——**不要重新导出**,除非有记录在案的理由
- 六份训练配置 `configs/training/*.yaml` 已按 A100 校准(bf16/fa2 可用)

## 你的任务清单(按序)

0. 环境:`pip install -r requirements.txt` + `pip install llamafactory[torch,metrics] vllm`;
   用户会提供 `.env`(AZURE_OPENAI_API_KEY,Step-1 nano 要用);`huggingface-cli login`
   (Llama-3.1 是 gated,用户需先在网页接受 license);`nvidia-smi` 确认卡型。
1. SFT×2:`llamafactory-cli train configs/training/qwen3_8b_sft_qlora.yaml`,llama 同理。
   观察 eval loss 收敛;两条各约数小时。
2. 起服务冒烟:vLLM 挂 SFT adapter(命令在 runbook §2,**别忘 --max-lora-rank 64**),
   用 `scripts/run_compare.py --system cicada --plan-base-url http://localhost:8000/v1
   --plan-model <served-name> --questions data/qa/cicada_core_v4/train_strat50.jsonl`
   跑 50 题 sanity(这是 train 的分层子集,当冒烟集用,不碰任何测试集)。
3. RSFT 环(runbook §3):自收割**必须** `--plan-temperature 0.7 --plan-samples 4`
   (贪心解码会让拒绝采样失去意义);round-2 导出到独立目录;从 SFT adapter 续训。
4. DPO×2(runbook §4):配置默认接 RSFT adapter;rank/alpha 64/128 **不许改**
   (QLoRA 量化基座不能合并 adapter,续训必须同秩)。
5. 评测矩阵(runbook §5):
   - 阶梯每档 × `data/qa/eval/compare_set_v4.jsonl`(260 题):zero-shot、SFT、
     SFT+RSFT、SFT+RSFT+DPO,两基座
   - 最好的学生 × `data/qa/cicada_core_v4/final_test.jsonl`(2,285 题,本地免费)
   - 诊断副表:100 题切片关 guided decoding,量各档 schema-valid 率
   - 结果 JSON 存 `outputs/eval/`,提交回仓库

## 铁律(违反任何一条都会毁实验)

- **训练/收割只许碰 `data/qa/cicada_core_v4/train.jsonl`**。final_test、compare_set_v4、
  dev_* 是评测专用——出现在任何训练/收割命令里即为事故。
- **阶梯纪律**:相邻两档只允许一个变量不同。有效 batch 恒 16(A100 上加大
  per_device 就等比减 accumulation);所有本地档(含 zero-shot)统一走运行时的
  guided-json(`complete_schema` 自动触发,不需要你做什么,但别绕过它)。
- Qwen3 三份配置里 `enable_thinking: false` 是训推一致性要求,不许删。
- `.env` 内容不打印、不提交、不进日志。
- **诚实汇报**:每步跑完把方法+结果(含确切命令、数据路径、数字)按惯例追加到
  `docs/cicada_worklog.md`;失败就写失败,不要美化;数字必须来自真实产物文件。
- 不确定要不要偏离剧本时:先做剧本内的,把偏离建议写进 worklog 的 Next 供用户决定。

## 交付物(用户要拿回本地写论文的)

- `outputs/*/`(各档 adapter)+ `outputs/eval/*.json`(阶梯×compare_v4 矩阵、
  final_test 结果、schema-valid 诊断)
- 更新后的 `docs/cicada_worklog.md`
- git commit(信息里写清楚每档的关键数字),等用户审后 push
