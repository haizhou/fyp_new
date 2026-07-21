# §1 Introduction (draft 1, style-contract compliant)

Public bodies in the United Kingdom publish hundreds of thousands of
contract award notices every year. Journalists and auditors ask analytical
questions of this record. How much did a council spend on cleaning in 2023.
Which supplier received the most awards under a procurement category. Did
spending rise after a policy change. An answer to such a question carries
the authority of official data, so a fluent but wrong answer is worse than
no answer. A system for this task must produce answers that a reader can
trace to specific records, and it must refuse when the data cannot support
one.

Large language models read these questions well and answer them
unverifiably. Nothing in generated text separates a genuine aggregation
over all matching records from a plausible guess over a retrieved sample.
Retrieval augmentation does not close this gap, because the questions are
dominated by exhaustive computation. A count over every contract of every
London borough does not fit in a context window. On our development set a
retrieval baseline answers 31.5 percent of questions while an untrained 8B
model that is forced to emit an executable plan answers 61.5 percent. The
scaffolding is worth more than the retrieval.

The standard route to a competent small model is distillation from a large
one. That route assumes the teacher can be trusted, and on this task it
cannot. The strongest cloud pipeline we evaluate answers 69.76 percent of
held-out questions. Imitating it transfers its errors together with its
competence. Filtered self-training methods replace trust with a pass
criterion, but the criteria in use are proxies. A SQL query can execute
cleanly and compute the wrong thing. A model can agree with itself and be
wrong. Outputs that pass for the wrong reason enter the training pool
silently, and none of these methods measures how often that happens.

This paper takes a different position. The task itself supplies a filter
that is stronger than any proxy. Whether a plan compiles, whether its
entities ground in the graph, whether it executes, and whether its answer
matches an independently computed oracle are all deterministic checks. We
train only on outputs that pass all of them, and we discard everything
else, whichever model produced it. We show that this verified bootstrap is
enough. An 8-billion-parameter local model trained this way reaches 85.65
percent on a held-out procurement benchmark of 2,285 questions, against
69.76 percent for the cloud teacher that generated its training data. The
gap is significant at p below 1e-15 under a paired McNemar test, and it
survives a measured teacher noise floor of one point.

The recipe travels. We rebuild only the data layer and move the system to
WikiTableQuestions, a benchmark of crowd-written questions over web tables
that we did not construct. The procurement checkpoint alone moves the
zero-shot floor from 22.5 to 27.3 percent. The recipe moves it to 44.4
percent when supervision comes from final answers only, and to 51.8 percent
when gold programs exist, both on the official sealed test under a single
frozen-configuration run. The checkpoint contributes a few points. The
recipe contributes the rest.

We make four contributions.

First, a verified-bootstrap training recipe in which deterministic
execution checks, not teacher trust and not proxy rewards, decide what a
model may learn from. Section 5.1 gives the evidence, including the
scoreboard on two base models and the noise-floor discipline behind every
teacher comparison.

Second, a typed compositional operator algebra whose programs are checked
by two independently implemented evaluators, so that benchmark oracles are
measured rather than assumed. Section 4 reports 99.88 percent dual
implementation agreement over 14,770 audited answers.

Third, PACS, a sealed procurement analytics challenge set with an
abstention axis and a constraint-necessity audit. Section 5.2 shows the
frozen planner at 78.31 percent, 28 points above its cloud teacher, and
shows that the headline survives restriction to the audit's irreducible
subset.

Fourth, a transfer study with migration accounting. Section 5.3 reports the
WikiTableQuestions results above, the cost of the move in code and
operators, and an attribution audit showing that a seventh of the apparent
expressiveness gap was compiler and representation debt rather than missing
reasoning capability.

We claim no state of the art. Recent specialised table systems score higher
on WikiTableQuestions than we do. The claim is that a deterministic filter
over an executable language turns weak local models into systems that beat
their own teachers, at auditable cost, twice.

---
## Self-review notes (referee pass on this draft)
- R2 guard present in final paragraph (scoped claim, no SOTA).
- Numbers used: 31.5, 61.5, 69.76, 85.65, 2,285, 1e-15, 22.5, 27.3, 44.4,
  51.8, 99.88, 14,770, 78.31, 28. All ledger-backed. "noise floor of one
  point" rounds 1.1, acceptable or change to "1.1 points".
- Style check: zero em-dashes, zero colons outside none, active voice
  throughout, one idea per paragraph, no judgement words. "The scaffolding
  is worth more than the retrieval" is a factual comparison of two stated
  numbers.
- Open wording question for user: "The checkpoint contributes a few points.
  The recipe contributes the rest." Punchy but informal. Alternative:
  "The checkpoint accounts for 4.8 points of the lift. The recipe accounts
  for the remainder."
- Missing citations to be added in LaTeX pass: RAG, STaR/ReST, ExeSQL/SCD,
  WTQ, TAPAS.
