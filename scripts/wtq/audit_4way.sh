#!/bin/bash
set -e
cd /home/uceeh01/fyp_new/fyp_new
PY=.venv/bin/python
run_arm () {
  export WTQ_LOWERING=$1 WTQ_VIEWS=$2
  echo "=== arm $3 (lowering=$1 views=$2) ==="
  $PY scripts/wtq/differential_audit.py 2>/dev/null | grep -E "coverage|fidelity|executor"
  cp data/qa/wtq/differential_audit.jsonl data/qa/wtq/da_$3.jsonl
}
run_arm v2 v2 baseline
run_arm v3 v2 translator_only
run_arm v2 v3 loader_only
run_arm v3 v3 both
$PY - << 'PYEOF'
import json
arms = {a: {r['nt']: r['class'] for r in map(json.loads, open(f'data/qa/wtq/da_{a}.jsonl'))}
        for a in ('baseline','translator_only','loader_only','both')}
ids = list(arms['baseline'])
def ok(c): return c in ('A','B')
base_in = [i for i in ids if not ok(arms['baseline'][i])]
base_ok = [i for i in ids if ok(arms['baseline'][i])]
m = {'translator_only':0,'loader_only':0,'both_only':0,'still':0}
for i in base_in:
    t, l, b = ok(arms['translator_only'][i]), ok(arms['loader_only'][i]), ok(arms['both'][i])
    if t and not l: m['translator_only'] += 1
    elif l and not t: m['loader_only'] += 1
    elif b and not t and not l: m['both_only'] += 1
    elif not b: m['still'] += 1
    else: m['translator_only'] += 1  # both individually recover it (overlap)
overlap = sum(1 for i in base_in if ok(arms['translator_only'][i]) and ok(arms['loader_only'][i]))
regress = sum(1 for i in base_ok if not ok(arms['both'][i]))
print(f"\nMIGRATION MATRIX (originally inexpressible n={len(base_in)}):")
for k,v in m.items(): print(f"  {k:16s} {v}")
print(f"  overlap(recovered by either alone): {overlap}")
print(f"REGRESSIONS (was expressible, now not): {regress} / {len(base_ok)}")
PYEOF
echo "4WAY AUDIT COMPLETE"
