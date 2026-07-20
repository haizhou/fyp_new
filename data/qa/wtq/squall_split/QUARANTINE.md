# Squall / WTQ split quarantine (adopted 2026-07-20)

## Leak audit (run before adoption)
- All 11,276 Squall entries are WTQ **training-set** questions.
- Squall tables ∩ WTQ pristine-unseen (TEST) tables: **0** of 1,617.
- Squall nt ids ∩ WTQ pristine-unseen question ids: **0**.
- `wtq-test.json` in the Squall repo has **never been opened** in this project
  and is on the do-not-read list until final system freeze.

## Disclosure
An aggregate SQL shape census and the first-cut translator (including its
800-entry error-analysis sample) were computed over the FULL 11,276
training-set Squall file before the internal train/dev fold was adopted.
No pristine-test SQL was inspected or used at any point (overlap is zero by
construction). All subsequent development follows the fold rules below.

## Fold rules (from adoption onward)
- `squall_train_ids.json` (9,030 q / 1,290 tbl): translator development,
  self-harvest filtering, any training.
- `squall_dev_ids.json` (2,246 q / 327 tbl, Squall fold-0, table-disjoint):
  error analysis, coverage census, generic-operator design decisions.
- WTQ `pristine-unseen-tables.tsv` (TEST): frozen. No structure inspection,
  no operator additions, no prompt tuning informed by it. One confirmatory
  run per frozen config after full freeze; post-hoc audit only.

## Claim boundary (model freeze discipline)
- `frozen-17-adapter-only` (git tag, commit 89d9de2): 17-operator algebra
  untouched; WTQ needed only loader + driver + schema field parameterization.
  This measures adapter-only portability.
- Extended algebra (commit 4bdefc2+): adds `extreme_rows` (row-preserving
  arg-extremum), a generic relational capability motivated by TRAIN-split
  census. Migration cost: 1 new generic operator; 17/17 original operators
  reused unchanged.
