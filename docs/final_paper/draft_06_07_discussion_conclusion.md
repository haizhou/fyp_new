# §6 Discussion and limitations (draft 1)

The primary claim is about a task and the system that solves it. The
teacher comparison inside that claim is scoped carefully. The teacher
operates zero-shot while the students are tuned in domain, so the
scoreboard shows that a verified bootstrap beats deploying the teacher it
was distilled from, not that an 8B model outranks its teacher in general
ability. PACS, where both sides face questions neither authored, shows the
gap survives on independent surfaces.

Refusal did not transfer free. On a paired legacy comparison the
compositional planner wins the answerable subset by 4.4 points yet loses
explicit abstention badly, because its training mix carried 37 refusal
demonstrations. Under our pre-registered bar of no key-bucket regression,
we make no retirement claim for the older system. Competence follows
demonstrations, and refusal is a competence.

Preference optimisation was hazardous in exactly the regime our filter
creates. Verifier rejects are near-miss negatives, one slot away from
correct plans, and suppressing them dragged down adjacent correct modes
on one base while cleaner pools made it worse. We report the mechanism
and use additive distillation instead.

The transfer numbers are coverage limited. The algebra expresses roughly
half of gold programs, the planner reaches 85.3 percent of what the gold
subset can express, and the closure pass shows a seventh of the missing
half was compiler debt. The two largest remaining gaps, general
expression predicates and ordered-relation navigation, are specified as
typed extensions and deliberately not built, because widening a language
mid-evaluation would dissolve the boundary that makes the measurements
interpretable.

Four further limits bound the claims. The home benchmark is self-built,
with its integrity measured rather than assumed, and its question
surfaces are model-generated under style controls, which is not user
language. The WikiTableQuestions test set is consumed, so future work on
that benchmark reports development folds or a new holdout. Probe evidence
of composition beyond demonstrated constructs is bounded, 11.5 points of
unseen-exposure cost. The evidence remains confined to executable structured-data analytics
over one procurement graph and one table benchmark.

# §7 Conclusion (draft 2, domain-first)

UK public procurement is public data that the public cannot reliably
analyse in natural language. We defined that task end to end, a
convention-explicit knowledge graph, a program-first benchmark with
refusal as scored behaviour, a typed compositional program language, and
oracles whose correctness is measured by an independent second
implementation. On that foundation a verified bootstrap, deterministic
checks deciding what a model may learn from, trains an 8B local system to
85.65 percent, past the cloud teacher that generated its training data,
and to 78.31 on a sealed challenge set neither system authored. Rebuilt
on web tables at the cost of a data layer, the same recipe turns a 22.5
percent floor into 44.4 without a single gold program and 51.8 with them.
Language coverage is now the dominant measured bottleneck, while program
discovery remains incomplete, and the audits in this paper are what make
that distinction measurable.

---
## Reviewer pass
- §6 opens by conceding R2 (teacher asymmetry) before any reviewer can
  raise it, then routes to PACS symmetry. Order deliberate.
- 4.4 points check: 87.38 minus 83.01 = 4.37, stated 4.4. OK.
- "one slot away" concrete image kept over abstract phrasing.
- Conclusion introduces zero new numbers except restatements. Verified.
- Style contract held throughout.
