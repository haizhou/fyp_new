# §6 Discussion and limitations (draft 1)

The evidence supports a claim about a recipe, not about model quality. The
teacher operates zero-shot while the students are tuned in domain, so the
scoreboard shows that a verified bootstrap beats deploying the teacher it
was distilled from, and PACS, where both sides face questions neither
authored, shows the gap survives on independent surfaces.

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
unseen-exposure cost. And every system here serves one narrow task
family, executable analytics over one graph and one table corpus.

# §7 Conclusion (draft 1)

A deterministic filter over an executable language replaces teacher trust.
Trained only on outputs that compile, ground, execute, and match an
independent oracle, an 8B local model overtakes the cloud model that
generated its training data, by 15.89 points at home and by 27.98 on a
sealed benchmark neither system authored. The recipe moves. Rebuilt on
web tables at the cost of a loader, a linker, and one operator, it turns
a 22.5 percent floor into 44.4 without a single gold program and 51.8
with them. What limits it now is not learning but language coverage, and
the audits in this paper are what make that distinction measurable.

---
## Reviewer pass
- §6 opens by conceding R2 (teacher asymmetry) before any reviewer can
  raise it, then routes to PACS symmetry. Order deliberate.
- 4.4 points check: 87.38 minus 83.01 = 4.37, stated 4.4. OK.
- "one slot away" concrete image kept over abstract phrasing.
- Conclusion introduces zero new numbers except restatements. Verified.
- Style contract held throughout.
