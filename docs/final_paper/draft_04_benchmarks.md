# §4 Benchmarks and evaluation protocol (draft 1)

**Procurement benchmark.** The main benchmark contains 12,828 questions
over a knowledge graph of 215,221 UK contract awards, built plan-first. A
parameterised program is instantiated and executed, and only then is a
question surface written, so every question carries an executable gold
program. Splits separate plans, not surfaces. Train and test share zero
underlying programs. Oracle correctness is measured, not assumed. A second
evaluator, implemented from the raw files with no shared code, reproduces
99.88 percent of 14,770 audited answers, and the 18 residual disagreements
are characterised individually. Three families of refusal questions,
ambiguous, schema-unsupported and empty-result, are built in as traps with
their cues preserved under paraphrase.

**PACS.** The procurement analytics challenge set evaluates the
compositional planner on questions its training machinery never rendered.
It contains 922 sealed test rows over 694 intent clusters, organised by
seven task families, three depth levels, a seen and unseen exposure axis,
and an answerability axis. Surfaces come from an independently written
grammar. Isolation is enforced by five zero-overlap gates between training
and test at the level of question text, logical signature, surface
template, entity anchors, and program hash. The test split is evaluated
once per frozen configuration. A 93-row human audit preceded unsealing and
its one finding, 46 mislabelled empty-result rows, was corrected by offline
re-execution before any number was read.

**WikiTableQuestions.** The transfer target is the standard benchmark of
22,033 crowd-written questions over 2,108 web tables. We use the official
test split of 4,344 questions and the official evaluator. The test set was
touched exactly once, under a frozen configuration whose commit hash and
file checksums were recorded before launch, and it is treated as consumed
thereafter. All development ran on folds of the training split that share
no tables with any training pool.

**Protocol.** Every evaluation is a single model call at temperature zero
under constrained decoding, with no repair unless a variant is named
explicitly. Answerable questions score by type-aware match against the
dual-verified oracle, or by the official evaluator on WikiTableQuestions.
Refusal questions score under two declared metrics, an exact-status match
and a faithfulness-gated variant in which an empty or multi-valued outcome
counts only if every literal of the emitted program traces to the question.
Paired systems are compared by exact McNemar tests on discordant pairs.
Teacher comparisons respect a measured provider noise floor of 1.1 points,
and no single-run teacher delta below three points is claimed anywhere.

---
## Reviewer pass
- Attack "you grade your own homework." First paragraph leads with the
  independent second implementation and the residual characterisation.
  PACS paragraph leads with renderer independence and sealed one-run.
- Attack "consumed test is unusual language." It is the honest state and
  reads as rigour. Keep.
- Attack "why single call." Protocol paragraph names it and §6 discusses
  the repair variant separately. Consistent with ledger E5/E11.
- Numbers check vs ledger: 12,828; 215,221; 99.88; 14,770; 18; 922; 694;
  93; 46; 22,033; 2,108; 4,344; 1.1. All verified.
- Style contract held.
