# Paper artifact ledger

This ledger separates stored experimental outputs, deterministic re-scores, and limitations. It is the source of the numbers in `main.tex` and `sections/06_results.tex`.

The teacher prompt, call, training, and ablation audit is recorded in
`REPRODUCIBILITY_MANIFEST.md`. Audited training-configuration hashes are:

```text
832802774952da5c691dc8c154627f8f60d10fc0c844dccf500a405a9df2a1bc  qwen3_8b_sft_qlora.yaml
6bbf69e6c5d2782764255e0c3fc8c6320bd47af504d97a56f529fad79f2d1171  llama31_8b_sft_qlora.yaml
a2c564579828d4056bb3637e7dbbd8cc2c5878f52aea53d45c0d4f56f29d9f09  qwen3_8b_rsft_qlora.yaml
4c95235ecf845f2e24406d342b61e7d995b740aa20ef41b8541496e444068a65  llama31_8b_rsft_qlora.yaml
5b7a115491b98f03d3a7245b7956ec810ffbc9392c90e0658aae8601ac715dbe  qwen3_8b_dpo_qlora.yaml
2119cde4f58fd5738913d30e841cc8c8f2e0319b53d5292c3c37afa312da0cef  llama31_8b_dpo_qlora.yaml
1d7ada7597ac8395a252e8a6872ed6c8c5e68098caa06c3bc61827367646ec08  qwen3_8b_compose_sft_v3_qlora.yaml
```

Paper audit base commit: `d15ec0a3bd5b5f295f9828e44067094b73ab2cd7` on branch `main`. The working tree also contains staged data artifacts and unstaged correctness fixes, so this hash is a reference point rather than a clean release identifier.

## Procurement runtime

- Frozen final artifact: 2,285 items, including the 260-item balanced set later used for checkpoint selection and pipeline diagnosis.
- Primary non-development re-score: remove all 260 IDs, leaving 2,025 items (1,725 answerable; 100 ambiguous, 100 unsupported, 100 no-result).
- The re-score uses stored per-item predictions only; no model was called.
- Primary fully local Qwen result: 1,736/2,025 = 85.73%. It is 1,436/1,725 on answerable items and 300/300 on declared non-answerable items.
- Inclusive fully local Qwen result: 1,957/2,285 = 85.65%.
- Paired Qwen-local versus teacher on 2,025: 372 wrong-to-right and 37 right-to-wrong, exact McNemar p = 9.79e-71.
- Paired Qwen-local versus Qwen-hybrid: 222 wrong-to-right and 60 right-to-wrong, p = 5.005e-23.
- Limitation: the frozen run directories do not contain one complete command/checkpoint manifest. The 2,285 column is therefore a stored-artifact result, not a clean current-HEAD reproduction.

## Development diagnostic

- Input: 260 items, 20 from each of 13 buckets. This is development/model-selection data, not an independent test.
- Deterministic KG-only control: 155/260 = 59.62%.
- The compact summary is `kg_only_260_summary.json`; it records hashes of the input, runner, full result, and source summary.
- Qwen untuned/SFT/RSFT/DPO: 183/211/212/217. Llama: 156/216/215/201.
- SFT is the only stable gain across both bases. RSFT is neutral, and Llama DPO is significantly worse than Llama RSFT.

## PACS v1.1

- Current status manifest: 922 = 730 answerable + 138 unsupported + 54 no-result.
- The original evaluated channels labelled 36 false or zero-valued answerable denotations as no-result.
- Saved trees were joined by ID and deterministically re-executed on the frozen backend. No model output was regenerated.
- Corrected strict counts: base A 312, teacher A 470, Compose-v3 A 722, Compose-v3 B 711.
- Limitations: the original status transformation script is absent; the seal omits one claimed gate and performs a sampled rather than exhaustive trigram screen.

## WikiTableQuestions

- Official pristine-unseen evaluation: 4,344 questions over 421 unseen tables.
- Predictions are serialised exactly as the frozen runner did: replace tab/newline characters inside answer items with spaces, join fields by tabs, then run `scripts/wtq/official_evaluator.py` v1.0.2 against the CoreNLP-tagged targets.
- Official counts: base 978; procurement Compose-v3 1,187; answer-only A 1,930; translated gold-program C 2,250.
- Official paired transitions: base→v3 387/178 (p=8.498e-19); v3→A 833/90 (p=1.768e-151); A→C 628/308 (p=6.198e-26).
- Do not use the internal JSON `correct` fields for official accuracy.
- Limitation: the exact approximately 5,423-row C-final training snapshot is not present in the current repository state.

## Grounding audit

- Fixed saved plans, same frozen backend, 136 items with oracles: 89 correct before deterministic grounding and 103 after; 14 are rescued, none degraded, one becomes executable but remains wrong.
- All 14 rescues are additive-sum guard interventions. This is a runtime counterfactual, not a causal estimate of training-data effects.
- Of 229 mechanically auditable grounding-related repair targets, 173 retain every mapped question constraint and 56 are flagged. This shows why denotation equality alone cannot prove plan equivalence.
