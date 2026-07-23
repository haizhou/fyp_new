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
client = OpenAI(base_url=env["AZURE_OPENAI_BASE_URL"], api_key=env["AZURE_OPENAI_API_KEY"], timeout=300)
_J = re.compile(r"\{.*\}", re.S)

rows = [json.loads(l) for l in open('data/qa/wtq/harvest_grok_pilot.jsonl')]
failed = [r for r in rows if r.get('status') == 'ok' and int(r.get('hits', 0) or 0) == 0][:20]

REWRITE_SYS = ("You rewrite table questions into DIRECT, table-grounded form. You see the table's "
    "columns and sample cells. Your rewrite must: use EXACT column names and cell spellings; "
    "eliminate ALL ordinal/positional/relative phrasing (last, first, next, after, above, previous) "
    "by re-expressing it through a sortable column (e.g. 'last X' -> 'the row with the largest year "
    "among rows where ...'); state exactly what value to report. One or two sentences. Do NOT answer "
    "the question. Do NOT write a program. If something truly cannot be expressed without row order "
    "and no sortable column exists, say REWRITE_IMPOSSIBLE and why in one line.")

REWRITE_USER = """Table columns and sample cells:
{catalog}

Original question: {q}

Rewritten question:"""

def one(rec):
    try:
        shim, cat = load_universe(rec['context'])
    except Exception:
        return {"id": rec['id'], "status": "table_error"}
    fields = [c[0] for c in cat]
    links = link(rec['question'], shim.raw_df, cat)
    hint = render(links)
    cat_txt = catalog_text(cat) + (('\n\n' + hint) if hint else '')
    # Step-1: table-grounded rewrite by reasoning
    try:
        r1 = client.chat.completions.create(model="grok-4-20-reasoning", max_tokens=4000,
            messages=[{"role": "system", "content": REWRITE_SYS},
                      {"role": "user", "content": REWRITE_USER.format(catalog=cat_txt, q=rec['question'])}])
    except Exception as e:
        return {"id": rec['id'], "status": "rewrite_api_error", "detail": str(e)[:80]}
    rewrite = (r1.choices[0].message.content or "").strip()
    out = {"id": rec['id'], "original": rec['question'], "rewrite": rewrite}
    if "REWRITE_IMPOSSIBLE" in rewrite:
        out["status"] = "rewrite_impossible"
        return out
    # Step-2: fast answers the REWRITTEN question, standard prompt
    prompt = PROMPT.format(catalog=cat_txt, q=rewrite)
    ev = WTQEvaluator(shim)
    targets = rec['target'].split('|')
    for _ in range(2):
        try:
            r2 = client.chat.completions.create(model="grok-4-1-fast-non-reasoning", temperature=1.0,
                max_tokens=900, messages=[{"role": "user", "content": prompt}])
        except Exception:
            continue
        m = _J.search(r2.choices[0].message.content or "")
        if not m:
            continue
        try:
            tree = json.loads(m.group(0)).get('tree')
        except json.JSONDecodeError:
            continue
        if not isinstance(tree, dict):
            continue
        def ok(t):
            try:
                if validate_tree(t) == 'RECORDS':
                    return False
            except AlgebraError:
                return False
            res = ev.run(t)
            return res.get('status') == 'ok' and denotation_match(res['answer'], targets)
        if ok(tree):
            out.update(status="verified", via="direct", tree=tree)
            return out
        for v in repair_variants(tree, fields, [c for c, _ in links]):
            if ok(v):
                out.update(status="verified", via="repair", tree=v)
                return out
    out["status"] = "unrescued"
    return out

with ThreadPoolExecutor(max_workers=8) as pool:
    res = list(pool.map(one, failed))

ver = sum(1 for r in res if r['status'] == 'verified')
mix = {s: sum(1 for r in res if r['status'] == s) for s in set(r['status'] for r in res)}
print(f"v6.1 table-grounded question REWRITE -> fast: {ver}/20")
print("mix:", mix)
print("\nSCOREBOARD same 20:")
print("  v1-v5 handoffs:      1-4/20")
print(f"  v6 rewrite handoff:  {ver}/20")
print("  reasoning direct:    7/20")
print("  local C-v5:          6/20")
print("\n=== sample rewrites ===")
for r in res[:4]:
    print(f"[{r['id']}] {r.get('status')}")
    print(f"  orig: {r.get('original','')[:70]}")
    print(f"  rewr: {r.get('rewrite','')[:150]}")
json.dump(res, open('data/qa/wtq/pilot_twostep_v61.json', 'w'), indent=1, default=str)
