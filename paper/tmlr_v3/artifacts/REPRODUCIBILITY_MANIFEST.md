# Reproducibility manifest: teacher, training, and evaluation

This document binds the paper's reproducibility claims to the artifacts that are actually
available. It distinguishes settings stored in experiment outputs from settings reconstructed
from the current runner. The audit base is commit
`d15ec0a3bd5b5f295f9828e44067094b73ab2cd7`; the working tree is not clean, so the hashes below
identify the exact audited files.

## 1. Teacher harvest

### Artifact-confirmed facts

- Input: `data/qa/cicada_core_v4/train.jsonl`, 9,267 questions.
- Stored teacher models: Step 1 `gpt-5.4-nano`; Step 2
  `grok-4-1-fast-non-reasoning`.
- Stored outcomes: 6,860 runtime-verified traces; 5,605 answerable traces that also match the
  answer oracle; 5,598 direct graph targets; 1,725 repair targets; 390 preference pairs; and 590
  abstention targets.
- The summary was recomputed from `traces.jsonl` after an earlier smoke invocation replaced the
  original summary. It does not preserve a complete command line or prompt version.

### Settings reconstructed from the audited runner

These values are the defaults of the audited `scripts/run_teacher.py` and its prompt/client code.
They are consistent with the stored model names, but most are not repeated in the historical
summary and should not be described as independently artifact-confirmed.

| Setting | Reconstructed value |
|---|---|
| Step-1 temperature | 0.0 |
| Step-2 temperature | 0.0 |
| Step-2 prompt/schema profile | `lean` / `optional` (selected from the Grok model name) |
| Structural plan samples | 2; a new sample is requested only after a detectable compile or consistency failure |
| Feedback repair budget | at most 2 |
| Output constraints | strict provider JSON Schema for Step 1, Step 2, and graph repair; recoverable schema errors may fall back to JSON extraction |
| API client timeout | 60 seconds |
| API retries | 3 retries after the first call (at most 4 attempts), with 2/4/6-second waits; rate limits remain retryable and most other 4xx errors do not |
| Token limit | no explicit `max_tokens` in this client; provider/model defaults apply |
| Credentials | environment variables; keys are not stored |

The worklog describes an eight-worker resumed full job, whereas the current CLI default is four.
Because no authoritative command manifest was stored, worker count is not used as a scientific
setting in the paper.

## 2. Exact prompt contract

The canonical prompt builders are in
`src/procurement_graph/reasoning/typed_planning.py`. The exact whitespace and dynamic JSON are
reproduced by those functions; the paper appendix prints the system messages and enumerates every
dynamic field.

### Step 1: typed understanding

Builder: `question_intent_program_messages(question, schema_context)`.

System message:

> You are the typed intent programmer for a UK public-procurement KGQA system. Convert the question
> into a small computation program. Do not answer the question. Use only information explicitly
> present in the question. The output shape is enforced by a strict JSON schema.

The user JSON contains `question`, `task`, `allowed_slots`, `operation_guidance`, `hard_rules`, and
`retrieved_schema_context`. The response schema requires an answer signature, ordered typed
program steps, an answer-step identifier, and an unsupported/ambiguous reason. The model receives
neither records nor an oracle answer.

### Step 2: typed graph plan

Builder: `typed_plan_messages(question, understanding, schema_context, variant="lean")`.

System message:

> You are the structured graph planner for a UK public-procurement KGQA system. Turn the question
> and the Step-1 briefing into an executable graph_plan (variables + return). A strict schema fixes
> the output shape; fill the fields faithfully and never answer the question.

The teacher's `lean` user JSON contains `question`, `step1_understanding_briefing`,
`retrieved_schema_context`, and `instructions`. The instructions define flat variable identifiers,
conjunctive filters, source-set-target bridge bindings, comparison returns, literal preservation,
signed-date handling, additive guards for money sums, and abstention. The larger capability card
used by the `card` variant is not part of the reconstructed Grok teacher profile.

### Repair understanding

Builder: `repair_understanding_messages(question, feedback)`.

System message:

> You are the repair reading layer for a verifier-guided UK procurement KGQA system. Revise the
> question-understanding scaffold using the failed Stage-2 graph plan and failure feedback. Do not
> answer the question and do not infer hidden reference answers.

The feedback sanitizer removes `oracle_answer`, `reference_answer`, `gold_answer`,
`expected_answer`, and any key containing `oracle`. It retains the previous understanding, failed
plan, failure stage/reason, execution status, grounding/schema/compiler issues, failed variable or
operation, and permitted repair hints.

### Repair graph plan

Builder: `typed_replan_messages(question, feedback)`.

System message:

> You are the repair reflector inside a verifier-guided KGQA graph planner. Diagnose the failed
> graph plan from the structured failure feedback, choose one allowed repair action, and return a
> repaired understanding network plus repaired graph plan using the SAME schema as the planning
> step. Never answer the question and never use hidden reference answers. Return strict JSON.

For an offline answer mismatch, visible feedback says only that the submitted answer failed
external validation. It does not include the expected value. The repaired plan becomes a positive
target only if deterministic re-execution subsequently matches the hidden oracle and answer shape.

## 3. Selection authority and exported supervision

The online runtime has no oracle. During offline harvest, candidate routing is:

1. Run typed planning, grounding, deterministic execution, and release checks.
2. Record runtime verification (`execution.passed` and a non-null answer).
3. For answerable items, compare the released answer with the hidden oracle and expected answer
   shape.
4. Route runtime-passing but externally wrong candidates to `hard_negatives`; optionally attempt a
   repair with oracle-hidden feedback.
5. Export direct SFT only from candidates that survive this routing. Export repair SFT and DPO only
   when the repaired answer also passes the oracle and shape gates. Correct non-answer decisions
   enter the abstention pool.

The emitted direct-SFT metadata says `acceptance: executor_verifier`, but this label is stale: the
actual control flow also applies the offline oracle/shape gate before that branch. The paper uses
the control flow, not this metadata string, as the source of truth.

Resume is not transactional in the audited teacher runner. It writes `traces.jsonl` before the
downstream training sinks, and `--resume` treats an ID in `traces.jsonl` as complete. A crash
between writes can therefore leave a trace without its SFT/DPO row. This is a reproducibility
limitation; it is not assumed to explain every difference between aggregate counts.

## 4. Training configurations

All reported graph and compose adapters use 4-bit QLoRA on all target modules, rank 64,
alpha 128, dropout 0.05, bfloat16 compute, and effective batch 16.

| Stage | Initialization and data | Objective / schedule |
|---|---|---|
| Step-2 SFT (Qwen/Llama) | base model; 2,787 direct + 1,679 repair rows; plan validation 57 | SFT, 3 epochs, LR 1e-4, cutoff 6,144 |
| Step-1 SFT (Qwen/Llama) | base model; 5,091/96 intent programs | SFT, 3 epochs, LR 1e-4, cutoff 6,144 |
| Qwen RSFT | Qwen SFT adapter; 3,019 direct + 3,169 repair rows; plan validation 65 | SFT, 2 epochs, LR 5e-5, cutoff 6,144 |
| Llama RSFT | Llama SFT adapter; 3,026 direct + 3,190 repair rows; plan validation 66 | SFT, 2 epochs, LR 5e-5, cutoff 6,144 |
| Qwen DPO | Qwen RSFT adapter; 390 teacher + 689 on-policy pairs | sigmoid DPO, beta 0.1, 1 epoch, LR 5e-6, cutoff 6,144 |
| Llama DPO | Llama RSFT adapter; 390 teacher + 693 on-policy pairs | sigmoid DPO, beta 0.1, 1 epoch, LR 5e-6, cutoff 6,144 |
| Compose-v3 | fresh Qwen3-8B; 12,414/253 recursive trees | SFT, 2 epochs, LR 1e-4, cutoff 4,096 |
| WTQ A | fresh Qwen3-8B; 4,615/94 answer-matched examples | SFT, 2 epochs, LR 1e-4, cutoff 4,096 |
| WTQ C | fresh Qwen3-8B; approximately 5,423 translated gold-program examples | same recipe; exact frozen training split is missing |

LLaMA-Factory training logs record one CUDA process, bfloat16, Transformers 5.6.0, and the expected
example/step counts. They do not preserve a formal hardware-model manifest. Evaluation server logs
record vLLM 0.24.0, seed 0, bfloat16, maximum sequence length 8,192, LoRA rank 64, and Qwen thinking
disabled. The final procurement run directories do not provide a single complete mapping from every
reported arm to checkpoint hash and command.

## 5. What is and is not an ablation

| Evidence | Controlled factors | Changed factors | Supported interpretation |
|---|---|---|---|
| 136-plan pre/post grounding replay | saved plans, questions, oracle, KG, executor | deterministic grounding transformation | narrow runtime intervention; mostly tests the additive-sum guard |
| 260-question untuned/SFT/RSFT/DPO ladder | questions, scorer, runtime budget, base within family | training corpus, initialization, and objective across rungs | sequential checkpoint comparison, not a matched component ablation |
| hybrid vs fully local runtime | test items and downstream runtime | Step-1 understander and historical deployment configuration | system-configuration comparison; incomplete run manifest limits causal attribution |
| PACS A vs B | programs/oracles and model | surface rendering | paraphrase robustness, not a method ablation |
| WTQ base/v3/A/C | test set and official scorer | pretraining/adaptation source and quantity | supervision ladder, not a single-variable ablation |

The current artifacts do not support a causal claim that grounding-derived examples alone produced
the SFT gain. A matched study would hold unique questions, family, program depth, output type,
target length, optimiser tokens, base checkpoint, and decoding fixed while comparing at least:
plan-only SFT; plan plus ordinary repair; plan plus grounding-derived repair; and the latter plus
grounding-derived preference pairs. Other missing component tests are guided versus unconstrained
decoding, online grounding on/off, verification/repair on/off, and fast-path versus forced Step 2.

## 6. Audited source hashes

```text
adb9c5ef58dfc3efa775f751c80f98f5885181be0f0ad51db323fbfb33a82980  typed_planning.py
001f0c5415c6b0dee237dbb3b210f41b52f20a2a4078f43632f3b05454eee81c  chat.py
ae0278f23b130ebe4065b6e717acf8ba4209e37bf53b8875ebd3e1731c3a4b23  run_teacher.py
d5c946d788f2f72e556c04abc605e7a402178bd583fa4bfe688303a45d8942a1  export_llamafactory.py
fb30c2429ec8d1ff810c2905537c012f4e3a23285c88df83ea3ebfca10190e24  export_step1_sft.py
5a3c90e74c8e8232e0f40cbc1bbf99b3a32b78bafb9db00b723012f9299b834c  run_compare.py
a49a05fc69bca068c9006161a48065afd541b089e965c8d31c72b582a66dbc42  eval_ladder.sh
```

The seven training configuration hashes are listed in `ARTIFACT_LEDGER.md` so that later edits to
comments or paths cannot be confused with the audited experiment configurations.
