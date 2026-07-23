import json, re, sys
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts/wtq')
from loader import load_universe, catalog_text
from linker import link, render
from wtq_eval import WTQEvaluator
from zero_shot import PROMPT, denotation_match
from tree_repair import repair_variants
from procurement_graph.compose.algebra import AlgebraError, validate_tree
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

env = dict(l.strip().split('=', 1) for l in open('.env') if '=' in l)
client = OpenAI(base_url=env["AZURE_OPENAI_BASE_URL"], api_key=env["AZURE_OPENAI_API_KEY"], timeout=120)
_J = re.compile(r"\{.*\}", re.S)

briefs = {r['id']: r for r in map(json.loads, open('data/qa/wtq/pilot_brief_v2.jsonl')) if r.get('brief')}
rows = [json.loads(l) for l in open('data/qa/wtq/harvest_grok_pilot.jsonl')]
failed = [r for r in rows if r.get('status') == 'ok' and int(r.get('hits', 0) or 0) == 0][:20]

GUIDE = """

ANALYST BRIEF (advisory; analyst could NOT see the table):
{brief}

HARD RULES: pred ops ONLY eq/in/contains/gte/lte; root must produce the ANSWER VALUE
(select for unique, values for lists); use __num twin columns for extremes/arithmetic;
copy cell spellings EXACTLY from catalog samples."""

def diagnose(tree, ev, targets):
    """Return (verified, diagnostic-string) with typed feedback."""
    try:
        t = validate_tree(tree)
    except AlgebraError as e:
        return False, f"TYPE ERROR: {e}. Fix that node only."
    if t == 'RECORDS':
        return False, "SHAPE ERROR: root returns a record set, not the answer. Wrap it with select (unique) or values (list) on the answer column."
    res = ev.run(tree)
    if res.get('status') != 'ok':
        return False, f"EXEC ERROR: {str(res.get('detail', res.get('status')))[:100]}"
    ans = res['answer']
    if denotation_match(ans, targets):
        return True, ""
    if ans is None or ans == [] or (isinstance(ans, str) and not ans.strip()):
        return False, ("EMPTY RESULT: a filter matched nothing. Re-check each filter value against the "
                       "catalog sample cells (exact spelling, or use contains); consider the __num twin column for numeric comparisons.")
    return False, (f"WRONG ANSWER: plan executed and returned {str(ans)[:60]!r}. The logic or column "
                   "choice is off. Re-read the brief's Reverse Tree and try a different column mapping "
                   "or operation; if a diff came out negative, swap operands.")

def one(rec):
    b = briefs.get(rec['id'])
    try:
        shim, cat = load_universe(rec['context'])
    except Exception:
        return {"id": rec['id'], "status": "table_error"}
    fields = [c[0] for c in cat]
    links = link(rec['question'], shim.raw_df, cat)
    hint = render(links)
    cat_txt = catalog_text(cat) + (('\n\n' + hint) if hint else '')
    cat_txt += GUIDE.format(brief=b['brief'])
    prompt = PROMPT.format(catalog=cat_txt, q=rec['question'])
    ev = WTQEvaluator(shim)
    targets = rec['target'].split('|')
    messages = [{"role": "user", "content": prompt}]
    # k=3 independent samples first
    for _ in range(3):
        try:
            r = client.chat.completions.create(model="grok-4-1-fast-non-reasoning", temperature=1.0,
                max_tokens=900, messages=messages)
        except Exception:
            continue
        m = _J.search(r.choices[0].message.content or "")
        if not m:
            continue
        try:
            tree = json.loads(m.group(0)).get('tree')
        except json.JSONDecodeError:
            continue
        if not isinstance(tree, dict):
            continue
        ok, diag = diagnose(tree, ev, targets)
        if ok:
            return {"id": rec['id'], "status": "verified", "via": "sample", "tree": tree}
        for v in repair_variants(tree, fields, [c for c, _ in links]):
            ok2, _ = diagnose(v, ev, targets)
            if ok2:
                return {"id": rec['id'], "status": "verified", "via": "mech_repair", "tree": v}
        best = (tree, diag)
    # 2 feedback rounds on the last failed tree
    tree, diag = best if 'best' in dir() else (None, None)
    if tree is None:
        return {"id": rec['id'], "status": "unrescued"}
    for rnd in range(2):
        fb = messages + [
            {"role": "assistant", "content": json.dumps({"tree": tree})},
            {"role": "user", "content": f"Your plan failed verification. {diag} Reply with ONE corrected JSON plan."}]
        try:
            r = client.chat.completions.create(model="grok-4-1-fast-non-reasoning", temperature=0.3,
                max_tokens=900, messages=fb)
        except Exception:
            break
        m = _J.search(r.choices[0].message.content or "")
        if not m:
            break
        try:
            tree2 = json.loads(m.group(0)).get('tree')
        except json.JSONDecodeError:
            break
        if not isinstance(tree2, dict):
            break
        ok, diag2 = diagnose(tree2, ev, targets)
        if ok:
            return {"id": rec['id'], "status": "verified", "via": f"feedback_r{rnd+1}", "tree": tree2}
        for v in repair_variants(tree2, fields, [c for c, _ in links]):
            ok2, _ = diagnose(v, ev, targets)
            if ok2:
                return {"id": rec['id'], "status": "verified", "via": f"feedback_r{rnd+1}+mech", "tree": v}
        tree, diag = tree2, diag2
    return {"id": rec['id'], "status": "unrescued"}

with ThreadPoolExecutor(max_workers=8) as pool:
    res = list(pool.map(one, failed))

ver = sum(1 for r in res if r['status'] == 'verified')
vias = {}
for r in res:
    if r['status'] == 'verified':
        vias[r['via']] = vias.get(r['via'], 0) + 1
print(f"two-step v5 (brief + k=3 + typed-feedback loop + mech repair): {ver}/20")
print("via:", vias)
print("\nSCOREBOARD same 20:")
print("  fast alone:              0/20")
print("  v1-v4 handoffs:          1-3/20")
print(f"  v5 feedback loop:        {ver}/20")
print("  reasoning direct:        7/20")
print("  local C-v5 bare:         6/20")
json.dump(res, open('data/qa/wtq/pilot_twostep_v5.json', 'w'), indent=1, default=str)
