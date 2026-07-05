# CICADA 学生训练 Runbook(服务器日操作手册)

目标:在 GPU 服务器上完成 Qwen3-8B / Llama-3.1-8B 学生阶梯
(zero-shot → SFT → RSFT → DPO),并把每一档接回 CICADA 管线评测。

前置:`pip install llamafactory[torch,metrics] vllm`;本仓库代码 + `data/` 同步到服务器;
Step-1 仍走 Azure(需要 `.env` 里的 `AZURE_OPENAI_API_KEY`,`set -a && . ./.env && set +a`);
Llama-3.1 是 gated 模型——先在 HF 上接受 license,服务器上 `huggingface-cli login`。
硬件:学校 A100(Ampere)——bf16 / flash-attn fa2 配置原样可用。

## 0. 正式导出(收割完成后,本机或服务器一次)

```bash
python scripts/export_llamafactory.py \
  --teacher-dir data/qa/teacher_full_v1 --out-dir data/training/llamafactory_v1
```

产物:`cicada_plan_sft(.val)/cicada_repair_sft(.val)/cicada_dpo` + `dataset_info.json`。
配置里 `dataset_dir` 直接指向该目录,无需合并进 LLaMA-Factory 全局注册表。

## 1. SFT(两基座,配置除基座/模板外逐参相同)

```bash
llamafactory-cli train configs/training/qwen3_8b_sft_qlora.yaml
llamafactory-cli train configs/training/llama31_8b_sft_qlora.yaml
```

## 2. 起服务(每一档同一套命令,只换 adapter/名字)

```bash
# zero-shot 档(阶梯下界)——同样走 guided-json,见下方"格式强制"
vllm serve Qwen/Qwen3-8B --served-model-name cicada-qwen3-zeroshot --port 8000

# SFT 档(LoRA 直挂,免合并)。--max-lora-rank 必须给:vLLM 默认 16,我们的 adapter 是 rank 64
vllm serve Qwen/Qwen3-8B --enable-lora --max-lora-rank 64 \
  --lora-modules cicada-qwen3-sft=outputs/qwen3_8b_cicada_sft_v1 \
  --served-model-name cicada-qwen3-sft --port 8000
```

**格式强制(阶梯同构的关键)**:运行时 `ChatClient.complete_schema` 发送标准
`response_format={"type":"json_schema",...}`,vLLM OpenAI server 原生映射为 guided-json。
所有本地档(含 zero-shot)自动同享格式强制——阶梯每档分差只反映规划能力。
Qwen3 注意加 `--chat-template-content-format string` 且推理关闭 thinking
(serve 时 `--reasoning-parser` 不配即可,或请求侧 `chat_template_kwargs={"enable_thinking": false}`)。

## 3. RSFT 自举轮(论文核心主张的实验证据;教师不参与)

```bash
# 3a. SFT 学生在 train 池上自收割(Step-1 仍 nano/Azure,Step-2 指向本地学生)
# --plan-temperature 必须 >0:贪心解码只能收到"本来就会做的题",探索不了新解;
# 温度 0.7 + 4 次结构重采样 = pass@k 拒绝采样,验证器负责守门
python scripts/run_teacher.py --out-dir data/qa/rsft_qwen_r1 \
  --plan-base-url http://localhost:8000/v1 --plan-model cicada-qwen3-sft \
  --plan-temperature 0.7 --plan-samples 4 \
  --limit 0 --workers 8 --resume

# 3b. round-2 导出(同一导出器,换目录)
python scripts/export_llamafactory.py \
  --teacher-dir data/qa/rsft_qwen_r1 --out-dir data/training/llamafactory_rsft_qwen_r1

# 3c. 从 SFT adapter 继续训
llamafactory-cli train configs/training/qwen3_8b_rsft_qlora.yaml
```

Llama 同理(`rsft_llama_r1` / `llama31_8b_rsft_qlora.yaml`)。

## 4. DPO(在 RSFT adapter 之上;配置默认指向 SFT adapter,跑完 RSFT 后把
`adapter_name_or_path` 改成 `*_cicada_rsft_v1`)

```bash
llamafactory-cli train configs/training/qwen3_8b_dpo_qlora.yaml
llamafactory-cli train configs/training/llama31_8b_dpo_qlora.yaml
```

DPO 池 = 全量收割的 dpo_pairs(hard negatives:验证器通过但 oracle 错)+ RSFT 轮新增对。

## 5. 评测(每一档一条命令)

```bash
# 主对比表:compare_set_v4(260 题,13 桶 × 20)
python scripts/run_compare.py --system cicada --questions data/qa/eval/compare_set_v4.jsonl \
  --plan-base-url http://localhost:8000/v1 --plan-model cicada-qwen3-sft --workers 8

# 全量:final_test(2,285 题,学生本地免费,一定跑)
python scripts/run_compare.py --system cicada --questions data/qa/cicada_core_v4/final_test.jsonl \
  --plan-base-url http://localhost:8000/v1 --plan-model <rung-name> --workers 8
```

评测矩阵:RAG naive / RAG strong / 教师(nano+grok) / 每基座 zero-shot、SFT、SFT+RSFT、
SFT+RSFT+DPO。诊断副表:100 题切片关 guided decoding 测 schema-valid 率
(证明 SFT 内化了输出契约,而非解码器兜底)。

## 显存参考(A100 40G/80G 都很宽裕)

QLoRA 4-bit r=64 训练 ~14-18GB;vLLM 8B bf16 推理 ~17GB。训练与推理不同时占卡。
A100 上可把 `per_device_train_batch_size` 提到 4(40G)或 8(80G),同时把
`gradient_accumulation_steps` 降到 4 或 2 —— **保持有效 batch = 16 不变**(两基座
逐参对照的实验纪律)。其余参数不动。
