# §5 Experiments (draft 1)

## 5.1 Verified bootstrap on the home benchmark

Table 1 reports the five-system comparison on the held-out test set of
2,285 questions. The teacher is the strongest cloud configuration we could
deploy. The hybrid systems use the cloud model for question understanding
and a local 8B adapter for planning. The fully local systems use 8B LoRA
adapters for both stages, trained only on harvest survivors of the
deterministic filter.

| System | Accuracy |
|---|---|
| Cloud teacher | 69.76 |
| Llama-3.1-8B hybrid | 75.97 |
| Qwen3-8B hybrid | 78.03 |
| Llama-3.1-8B fully local | 83.33 |
| Qwen3-8B fully local | 85.65 |

Every pairing in the table is significant at McNemar p below 1e-14. The
fully local Qwen system exceeds its teacher by 15.89 points with a 95
percent confidence interval of 14.07 to 17.70. The result replicates on
the second base model at 13.57 points. Because the teacher is served by a
provider whose decoding is not seeded, we measured its variability with
three identical development runs and obtained 72.6 plus or minus 1.1
points. The student beats each replicate separately, and no teacher
comparison in this paper rests on a delta inside that floor.

Two controls locate where the gain comes from. Two retrieval baselines answer 31.5 and 28.8 percent of the development
set, while an untrained 8B model forced through the executable
scaffolding answers 61.5 percent, rising to 70.4 under the revised
pipeline. Scaffolding, not retrieval, sets the floor, and training adds
its gains on top of that floor. A format diagnostic separates the two
remaining explanations. Decoded without constraints, the untrained base
produces question-grounded plans in an invented schema on 98 of 100
questions and matches the executable contract on 3. The tuned models match
it on 98 and 99. Fine-tuning contributes contract conformance and planning
quality within it, and constrained decoding grants the contract to every
system equally, so the table measures planning.

## 5.2 A sealed benchmark the system did not author

The compositional planner was frozen and evaluated once on the 922 sealed
PACS test rows. It scores 78.31 percent strict overall and 80.00 on the
answerable subset, with a cluster bootstrap interval of 77.06 to 82.92. On
the same rows the untrained base scores 36.01, a retrieval baseline 21.90,
and the cloud teacher 50.33. The verified-bootstrap student exceeds its teacher by 27.98 points here.
The comparison is a system-level one between deployable configurations,
not a model comparison under equal decoding, because the teacher's API
offers no constrained decoding and 23.1 percent of its rows fail on
format alone.

Two audits attack this headline from opposite directions. Deleting each
oracle-program predicate in turn and re-executing shows that 42.7 percent
of answerable test rows are irreducible, meaning every predicate is load
bearing. On that shortcut-resistant subset the planner scores 78.7,
within 1.3 points of its answerable headline, so the number is not
inflated by questions a partial program could answer by accident. In the
other direction, a typed-feedback repair loop that only ever acts on hard
failures adds 1.7 points on development rows while leaving every refusal
untouched, and we report the single-call number as primary.

The unseen-exposure axis bounds generalisation honestly. Intent shapes
absent from training cost 11.5 points, with an interval of 5.6 to 17.2.
Composition over demonstrated constructs transfers. Constructions never
demonstrated remain measurably harder, and the two weakest families,
scope binding and universal quantification, are named rather than
averaged away.

## 5.3 Transfer to WikiTableQuestions

Moving the system to web tables required a new data loader with typed
views, a column-aware value linker, one generic operator, and nothing
else. All seventeen original operators, the type checker, and both
evaluators transferred unchanged. The linker was admitted by a four-arm
ablation in which real cell candidates gain 4.0 points over the schema
baseline while random cells lose 1.7, so the gain is grounding rather
than added context.

On the official test set, evaluated once under a recorded manifest, the
untrained base scores 22.51 percent, the procurement checkpoint 27.33,
the answer-only bootstrapped model 44.43, and the gold-program model
51.80, each rung significant at p below 5e-18. The checkpoint therefore
carries a 4.8-point zero-shot prior, and the recipe carries 17 to 24.5
points beyond it. The answer-only figure matches classical
weakly-supervised parsers on this benchmark, the gold-program figure
exceeds early table-pretrained baselines such as TAPAS, and neither
approaches recent specialised systems, a gap the next paragraph explains
rather than excuses.

A differential audit against 11,276 gold SQL programs separates three
error sources. The algebra expresses 53 to 55 percent of gold programs.
Translated programs match the reference denotation 92 to 93 percent of
the time. Executed class-A programs recover the target answer in 94.1 percent of
cases. By comparison, the gold SQL annotations themselves reproduce the
benchmark answer in 88.9 percent, so the annotation layer is noisy and is
not treated as an absolute oracle.
Performance is coverage limited, not execution limited. A closure pass
that lowers SQL constructs onto existing nodes and adds typed loader
views raises coverage to 61.0 percent with zero regressions, and its
attribution matrix shows 562 questions recovered by the translator alone,
154 by the loader alone, 14 jointly, and 4,466 still outside the
language. One seventh of the apparent expressiveness gap was compiler and
representation debt. Retraining after representation and compilation closure lifts the
gold-program model from 40.0 to 51.0 on the development fold and the
answer-only model from 31.7 to 40.0, so capability tracks the coverage
boundary in both supervision regimes.

---
## Reviewer pass
- Attack "dev numbers mixed into a test-set section." 5.1 labels the
  controls as development measurements explicitly. 5.3 separates official
  test (one shot) from development-fold retraining. Wording verified.
- Attack "irreducible-subset analysis is post hoc." It is labelled an
  audit of the benchmark, not a training intervention, and the headline
  it defends was frozen first. Sentence order enforces this.
- Attack "why is teacher only one replicate on test." Noise-floor
  paragraph carries the answer with the three dev replicates.
- Numbers cross-checked against ledger: all present in E-entries. 78.7 vs
  80.00 stated as 1.3 points, matches.
- Style contract held. Zero em-dashes, zero prose colons.
