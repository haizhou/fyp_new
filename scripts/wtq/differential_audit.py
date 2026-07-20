#!/usr/bin/env python3
"""Rung 3: differential oracle audit — reference SQL vs translated algebra.

For every Squall entry (train/dev folds reported separately):
  reference:  gold SQL executed on Squall's own sqlite (tables/db/{tbl}.db)
  translated: gold-derived algebra tree executed on OUR loader+evaluator

Classes:
  A  translated, tree denotation == reference SQL denotation
  B  translated, denotations disagree (translator/loader/executor divergence)
  C  not expressible in the current algebra (skip census)
  D  infrastructure failure (table load, sqlite error, tree exec failure)

Separated metrics (per fold):
  syntactic coverage        = (A+B+D_exec) / total
  translation fidelity      = A / (A+B)
  executor accuracy | A     = class-A trees whose answer matches the WTQ target
  reference ceiling         = reference SQL matching the WTQ target (annotation+norm bound)
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/wtq"))

from loader import load_universe  # noqa: E402
from squall_translate import Skip, translate, norm, num_of  # noqa: E402
from wtq_eval import WTQEvaluator  # noqa: E402

from procurement_graph.compose.algebra import AlgebraError, validate_tree  # noqa: E402

SQ = Path("/var/tmp/cicada/squall/squall-main")


def sql_text(tokens) -> str:
    return " ".join(t[1] for t in tokens)


def ref_execute(tbl: str, sql: str):
    db = SQ / "tables/db" / f"{tbl}.db"
    if not db.exists():
        return None, "no_db"
    try:
        con = sqlite3.connect(str(db))
        con.text_factory = lambda b: b.decode(errors="replace")
        cur = con.execute(sql)
        rows = cur.fetchmany(1000)
        con.close()
    except Exception as exc:
        return None, f"sql_error:{str(exc)[:60]}"
    vals = [r[0] for r in rows if r and r[0] is not None]
    return vals, None


def _fold(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def denot_eq(ours, ref) -> bool:
    """Compare our answer envelope value with reference SQL first-column values."""
    if isinstance(ours, list) and ours and isinstance(ours[0], list):
        ours = [p[0] for p in ours]
    if isinstance(ours, bool):
        ours = [1 if ours else 0]
    a = sorted({_fold(norm(x)) for x in (ours if isinstance(ours, list) else [ours])})
    b = sorted({_fold(norm(x)) for x in ref})
    if a == b:
        return True
    # single numeric content comparison (display commas, units, float repr)
    if len(a) == 1 and len(b) == 1:
        na, nb = num_of(a[0]), num_of(b[0])
        return na is not None and nb is not None and abs(na - nb) < 1e-6
    return False


def target_eq(ours, targets) -> bool:
    if isinstance(ours, list) and ours and isinstance(ours[0], list):
        ours = [p[0] for p in ours]
    if isinstance(ours, bool):
        ours = [1 if ours else 0]
    a = sorted(norm(x) for x in (ours if isinstance(ours, list) else [ours]))
    b = sorted(norm(x) for x in targets)
    if a == b:
        return True
    if len(a) == 1 and len(b) == 1:
        na, nb = num_of(a[0]), num_of(b[0])
        return na is not None and nb is not None and abs(na - nb) < 1e-6
    return False


def main() -> None:
    squall = json.loads((SQ / "data/squall.json").read_text())
    dev_tbls = set(json.load(open(SQ / "data/dev-0.ids")))

    universes: dict = {}
    stats = {f: Counter() for f in ("train", "dev")}
    b_examples = []
    out_rows = []

    for e in squall:
        fold = "dev" if e["tbl"] in dev_tbls else "train"
        st = stats[fold]
        st["total"] += 1
        row = {"nt": e["nt"], "tbl": e["tbl"], "fold": fold}

        num, grp = e["tbl"].split("_", 1)
        csv_rel = f"csv/{num}-csv/{grp}.csv"
        if csv_rel not in universes:
            try:
                universes[csv_rel] = load_universe(csv_rel)
            except Exception:
                universes[csv_rel] = None
        if universes[csv_rel] is None:
            st["D_table_load"] += 1
            row["class"] = "D:table_load"
            out_rows.append(row)
            continue
        shim, catalog = universes[csv_rel]
        colmap = {i + 1: (c[0], c[1]) for i, c in enumerate(catalog)}

        ref, ref_err = ref_execute(e["tbl"], sql_text(e["sql"]))
        targets = e["tgt"].split("|")
        if ref is not None and ref:
            st["ref_ok"] += 1
            if target_eq(ref, targets):
                st["ref_matches_target"] += 1

        try:
            tree = translate(e["sql"], colmap, shim.records_df)
            validate_tree(tree)
        except Skip as exc:
            st["C"] += 1
            st[f"C:{exc.reason.split(':')[0]}"] += 1
            row["class"] = f"C:{exc.reason}"
            out_rows.append(row)
            continue
        except AlgebraError as exc:
            st["C"] += 1
            st["C:translator_invalid"] += 1
            row["class"] = f"C:translator_invalid:{exc.reason}"
            out_rows.append(row)
            continue

        res = WTQEvaluator(shim).run(tree)
        if res.get("status") != "ok":
            st["D_exec"] += 1
            row["class"] = f"D:exec:{res.get('reason')}"
            out_rows.append(row)
            continue

        if ref is None or not ref:
            st["D_ref"] += 1
            row["class"] = f"D:ref:{ref_err or 'empty'}"
            out_rows.append(row)
            continue

        agree = denot_eq(res["answer"], ref)
        hit = target_eq(res["answer"], targets)
        row.update({"class": "A" if agree else "B",
                    "target_hit": bool(hit), "answer": str(res["answer"])[:120],
                    "ref": str(ref[:5])[:120]})
        out_rows.append(row)
        if agree:
            st["A"] += 1
            st["A_target_hit"] += hit
        else:
            st["B"] += 1
            if len(b_examples) < 30 and fold == "dev":
                b_examples.append(row | {"sql": sql_text(e["sql"])[:150],
                                         "tree": json.dumps(tree)[:200]})

    out = ROOT / "data/qa/wtq/differential_audit.jsonl"
    with out.open("w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, default=str) + "\n")
    (ROOT / "data/qa/wtq/differential_audit_Bexamples.json").write_text(
        json.dumps(b_examples, indent=1, default=str))

    for fold in ("train", "dev"):
        st = stats[fold]
        tot, A, B = st["total"], st["A"], st["B"]
        cov = (A + B + st["D_exec"]) / tot
        fid = A / (A + B) if A + B else 0
        exa = st["A_target_hit"] / A if A else 0
        ref_ceiling = st["ref_matches_target"] / st["ref_ok"] if st["ref_ok"] else 0
        print(f"\n=== fold {fold} (n={tot}) ===")
        print(f"syntactic coverage:        {A+B+st['D_exec']}/{tot} = {100*cov:.2f}%")
        print(f"translation fidelity A/(A+B): {A}/{A+B} = {100*fid:.2f}%")
        print(f"executor acc | class A:    {st['A_target_hit']}/{A} = {100*exa:.2f}%")
        print(f"reference ceiling (SQL vs target): {st['ref_matches_target']}/{st['ref_ok']} = {100*ref_ceiling:.2f}%")
        print(f"D_exec {st['D_exec']}, D_ref {st['D_ref']}, D_table {st['D_table_load']}")
        cs = {k: v for k, v in st.items() if k.startswith('C:')}
        print("C census:", dict(sorted(cs.items(), key=lambda kv: -kv[1])[:8]))


if __name__ == "__main__":
    main()
