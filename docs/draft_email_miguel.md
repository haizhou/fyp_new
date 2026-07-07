# 草稿:给 Miguel 的进度邮件(你过目后自己发;此环境不发外发邮件)

Subject: FYP results complete — headline numbers + request for a 30-min slot this week

Hi Miguel,

Quick update ahead of the write-up: the experimental line is complete and the numbers are
stronger than we expected.

Headline (held-out test set, n=2,285, real UK procurement data):
- Our fully-local system (both LLM stages are 8B LoRA students distilled through the
  pipeline's own verifier; zero API calls) reaches **85.7%**, exceeding its cloud teacher
  (69.8%) by **+15.9 points** (McNemar p<1e-15) and the cloud-hybrid variant by +7.6.
- The result replicates on a second base model (Llama-3.1-8B: 83.3%, +13.6 over teacher).
- Statistics are paired throughout; the teacher's run-to-run noise floor was measured with
  three replicates; dev/confirmatory sets are separated, with pre-registered follow-ups
  (compositional-generalization probe) still running.

The thesis frame is one sentence: supervision flows only from the verifiable region, yet
the learned competence extends beyond it — the verifier can't certify correctness (it has
a blind region we characterise), but one-sided filtering is enough to bootstrap the whole
stack past its teachers, with abstention closing the blind region.

Three specific questions where your steer would help most:
1. Chapters 3-4 (data engineering + benchmark construction) currently carry substantial weight. Under the UCL marking scheme, is that investment proportionate, or should they be compressed in favour of the results/analysis chapters?
2. The whole dissertation is organised around one principle (partial verifiability). Does the introduction's build-up of that idea read clearly, or is it too indirect for a first-time reader?
3. The main limitation is a single self-built benchmark in a single domain (Chapter 8 lists it first). Where should this version draw the line: is an honest limitation statement enough, or does the thesis need a cross-schema transfer experiment (e.g. BIRD) to stand?

Draft status, for transparency: all headline figures are machine-verified against a master results table; a full per-sentence provenance audit of secondary figures is still in progress.

Could I grab 30 minutes this week to walk you through the results table and the chapter
plan? I'm starting the draft tonight; any early steer on emphasis would be valuable.

Best,
[你的名字]

---
备注(不进邮件):数字出处 docs/results_master_table.md;若 Miguel 要看图,
outputs/figures/ 的 F2/F4 已可发,四点衰减图今晚终渲后补。
