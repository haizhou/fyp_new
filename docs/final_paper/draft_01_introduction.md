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
model that is forced to emit an executable plan answers 61.5 percent. Forcing an executable plan is worth more than
retrieval before any training takes place.

The standard route to a competent small model is distillation from a large
one. That route assumes the teacher can be trusted, and on this task it
cannot. The strongest cloud pipeline we evaluate answers 69.76 percent of
held-out questions. Imitating it transfers its errors together with its
competence. Filtered self-training methods replace trust with a pass
criterion, but the criteria in use are proxies. A SQL query can execute
cleanly and compute the wrong thing. A model can agree with itself and be
wrong. Outputs that pass for the wrong reason enter the training pool
silently, and none of these methods measures how often that happens.

We introduce, to the best of our knowledge, the first systematic benchmark
and executable framework for auditable compositional question answering
over UK public procurement data. Procurement knowledge graphs exist, and
TheyBuyForYou built one across EU procurement for integration, anomaly
detection and search. What has not existed is the task itself in usable
form. That means a knowledge graph whose money and identity conventions
are explicit enough to verify against, a benchmark whose every question
carries an executable gold program, a typed program language that
expresses real analytical demands, refusal as a scored behaviour, and an
evaluation whose oracles are measured rather than assumed. We build all of
it, and we train a system on it without large-scale manual annotation.

The training method follows from the setting. No annotator pool can label
tens of thousands of compositional procurement programs, so supervision
must come from models, and the task itself supplies a filter stronger than
any proxy. Whether a plan compiles, grounds, executes, and matches an
independently computed oracle are deterministic checks. We train only on
outputs that pass all of them, whichever model produced them. An
8-billion-parameter local model trained this way reaches 85.65 percent on
the held-out benchmark of 2,285 questions, against 69.76 percent for the
cloud teacher that generated its training data. The gap is significant at
p below 1e-15 under a paired McNemar test and survives a measured teacher
noise floor of 1.1 points.

The recipe transfers. We rebuild only the data layer and move the system to
WikiTableQuestions, a benchmark of crowd-written questions over web tables
that we did not construct. The procurement checkpoint alone moves the
zero-shot floor from 22.5 to 27.3 percent. The recipe moves it to 44.4
percent when supervision comes from final answers only, and to 51.8 percent
when gold programs exist, both on the official sealed test under a single
frozen-configuration run. The checkpoint accounts for 4.8 points of the
transfer gain, while verified training accounts for the larger subsequent
improvement.

We make five contributions, ordered from the task outward.

First, the task. We define auditable compositional question answering over
UK public procurement, from raw releases to a convention-explicit
knowledge graph of 215,221 contract awards, with refusal on ambiguous,
unsupported and empty questions as first-class behaviour (Sections 3, 4).

Second, the representation. A typed compositional operator algebra
formalises real procurement analytics, set composition, guarded money
aggregation, grouped comparison and multi-stage analysis, as closed but
recursively compositional executable programs (Section 3).

Third, the evaluation foundation. A program-first benchmark of 12,828
questions with executable gold programs and plan-level split isolation, a
sealed challenge set with an independent surface grammar and a
constraint-necessity audit, and oracles validated by a second independent
implementation at 99.88 percent over 14,770 audited answers (Sections 4,
5.2).

Fourth, the training method the setting demands. A verified bootstrap in
which deterministic execution checks, not teacher trust and not proxy
rewards, decide what the model may learn from, requiring no manual program
annotation (Section 5.1).

Fifth, the empirical case. The trained local system reaches 85.65 percent
at home and 78.31 on the sealed challenge set, 28 points above its cloud
teacher there, and the recipe transfers to WikiTableQuestions at the cost
of a data layer, where an attribution audit separates representation debt
from genuine language limits (Section 5.3).

We claim no state of the art. Recent specialised table systems score higher
on WikiTableQuestions than we do. The claim is that a deterministic filter
over an executable language turns weak local models into systems that beat
their own teachers, across both the home benchmark and an independently
constructed challenge set.

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
- RESOLVED per user review 2026-07-22: formal wording adopted (4.8 points
  of transfer gain attributed to checkpoint, remainder to verified
  training).
- Missing citations to be added in LaTeX pass: RAG, STaR/ReST, ExeSQL/SCD,
  WTQ, TAPAS.
