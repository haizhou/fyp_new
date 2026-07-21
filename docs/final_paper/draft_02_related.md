# §2 Related Work (draft 1)

The loop we use, generate candidates, filter them, retrain on the
survivors, is standard. We do not claim it. The difference against each
neighbour below lies on two axes. The first is what a candidate must prove
before it becomes supervision. The second is what happens to candidates
that pass wrongly. We state both for every cluster.

**Filtered self-training.** STaR keeps sampled rationales whose final
answer matches a gold label and retrains on them. Rejection-sampling
fine-tuning and ReST generalise the pattern to reward thresholds and
iterated grow-filter-train cycles. ExeSQL bootstraps text-to-SQL with
execution success as the filter, and SCD distils structured QA through
self-consistency votes. All of these are relevant because they train small
models from filtered model outputs, exactly our setting. None of them is
enough for our task, for one measured reason. Their pass criteria are
proxies. Execution success accepts queries that run and compute the wrong
thing. Agreement accepts consistent errors. Gold-answer matching accepts
right answers reached by invalid programs. During our own harvest, 86.7
percent of generated bridge plans passed every deterministic check while
only 52.3 percent matched the oracle. A filter blind to that stratum feeds
a third of its pool with structurally valid wrong programs and never knows.
We route that stratum into labelled hard negatives instead, and we report
where it leaks.

**Agents over structured knowledge.** Pangu constrains an LLM to
discriminate among logical-form extensions enumerated from the graph.
ChatKBQA generates a form and grounds it by retrieval. RoG plans relation
paths and reasons over them. StructGPT gives a frozen model iterative read
access through fixed interfaces. These systems share our premise that the
structure, not the model, is the ground. They are not enough for two
reasons. None treats verifier rejections as a supervision source, so the
training value of failure is discarded. And none scores refusal. Questions
that are ambiguous, unsupported by the schema, or matched by no records are
first-class citizens of our benchmarks, with three trap families and a
faithfulness-gated metric.

**Executable table question answering.** Classical semantic parsers on
WikiTableQuestions reach 37 to 46 percent with weak supervision. TAPAS and
TaBERT pre-train table encoders and reach the high forties to low fifties.
Binder and later LLM-with-code systems pass 65 percent by generating SQL or
Python against the table. These works calibrate our transfer numbers, and
our answer-only result of 44.4 percent sits inside the classical band while
our gold-program result of 51.8 exceeds TAPAS-large. They are not enough
for our purpose because the program language is chosen for coverage, not
for verifiability. Free Python has no type checker that a second
implementation can replay, no closed grammar that decoding can enforce, and
no abstention semantics. Our algebra buys those properties and pays for
them in coverage, and Section 5.3 measures the price and then decomposes
it.

**Preference optimisation.** Direct preference optimisation and its
variants are the standard way to use negatives. Likelihood displacement
work shows the objective is most destructive when chosen and rejected
responses are similar. Our verifier produces exactly that regime, near-miss
negatives one slot away from correct plans, and Section 6 reports the
resulting hazard as a bounded negative finding rather than a method.

---
## Reviewer pass (top-venue lens)
- Attack "you cite a 2015-2020 table QA line but not 2024-2026 systems."
  Need 2-3 recent specialised WTQ systems named in the LaTeX pass with
  their scores, cited exactly where we say 65 percent plus. TODO citation
  list: Binder, Dater or similar, one 2025 LLM-SQL system.
- Attack "86.7 vs 52.3 is your own number used against others." The
  sentence already frames it as our harvest measurement, not their systems.
  Keep.
- Attack "abstention claim overreach." Softened to "none scores refusal"
  which is checkable against the cited papers' evaluation sections.
- Style: no em-dash, colons zero, register held. One long sentence in
  cluster C split candidate if page pressure.
