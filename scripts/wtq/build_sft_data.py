#!/usr/bin/env python3
"""Build WTQ SFT datasets for the three supervision arms (pools NEVER mixed).

C (gold-program ceiling): Squall TRAIN-fold gold SQL -> translator -> keep only
   trees that execute AND match the WTQ target (no wrong-tree teaching).
A (denotation-only main): harvest_A_*.jsonl verified trees; per question keep
   the SHORTEST tree (node count) to bias against spurious complexity, cap 2.

Output format mirrors the zero-shot inference prompt EXACTLY (format-channel
lock-in lesson): user = PROMPT(catalog, q), assistant = {"tree": ...} JSON.

Usage:
  .venv/bin/python scripts/wtq/build_sft_data.py --arm C
  .venv/bin/python scripts/wtq/build_sft_data.py --arm A
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/wtq"))

from loader import catalog_text, load_universe  # noqa: E402
from zero_shot import PROMPT  # noqa: E402

DEV_TBLS = set(json.load(open("/var/tmp/cicada/squall/squall-main/data/dev-0.ids")))


def nodes(tree) -> int:
    if isinstance(tree, dict):
        return 1 + sum(nodes(v) for v in tree.values())
    if isinstance(tree, list):
        return sum(nodes(v) for v in tree)
    return 0


def build_c():
    from squall_translate import Skip, translate  # noqa: E402
    from wtq_eval import WTQEvaluator  # noqa: E402
    from procurement_graph.compose.algebra import AlgebraError, validate_tree  # noqa: E402
    from squall_translate import match  # noqa: E402

    squall = json.load(open("/var/tmp/cicada/squall/squall-main/data/squall.json"))
    out, universes = [], {}
    for e in squall:
        import os as _os
        if _os.environ.get("WTQ_FINAL") != "1" and e["tbl"] in DEV_TBLS:
            continue  # train fold only (final pools include dev per amendment)
        num, grp = e["tbl"].split("_", 1)
        csv_rel = f"csv/{num}-csv/{grp}.csv"
        if csv_rel not in universes:
            try:
                universes[csv_rel] = load_universe(csv_rel)
            except Exception:
                universes[csv_rel] = None
        if universes[csv_rel] is None:
            continue
        shim, catalog = universes[csv_rel]
        orig = [c for c in catalog if "__" not in c[0]]
        colmap = {i + 1: (c[0], c[1]) for i, c in enumerate(orig)}
        colmap[0] = {c[0] for c in catalog if "__" in c[0] or c[0] == "row_index"}
        try:
            tree = translate(e["sql"], colmap, shim.records_df)
            if validate_tree(tree) == "RECORDS":
                continue
        except (Skip, AlgebraError):
            continue
        res = WTQEvaluator(shim).run(tree)
        if res.get("status") != "ok":
            continue
        strict, tolerant = match(res["answer"], e["tgt"].split("|"))
        if not tolerant:
            continue
        q = " ".join(e["nl"]) if isinstance(e["nl"], list) else e["nl"]
        from linker import link, render
        hint = render(link(q, shim.raw_df, catalog))
        cat_txt = catalog_text(catalog) + (("\n\n" + hint) if hint else "")
        out.append({"context": csv_rel, "question": q, "tree": tree,
                    "catalog": cat_txt})
    return out


def build_a():
    out, universes = [], {}
    import os as _os
    import os as _os2
    pats = [_os2.environ.get("WTQ_HARVEST_GLOB", "data/qa/wtq/harvest_A_*.jsonl")]
    if _os.environ.get("WTQ_FINAL") == "1":
        pats.append("data/qa/wtq/harvest_dev_*.jsonl")
    files = []
    for pat in pats:
        files += list(ROOT.glob(pat))
    for f in sorted(files):
        for line in f.open():
            r = json.loads(line)
            if r.get("status") != "ok" or not r.get("trees"):
                continue
            if r["context"] not in universes:
                try:
                    universes[r["context"]] = load_universe(r["context"])
                except Exception:
                    universes[r["context"]] = None
            if universes[r["context"]] is None:
                continue
            shim, catalog = universes[r["context"]]
            from linker import link, render
            hint = render(link(r["question"], shim.raw_df, catalog))
            cat_txt = catalog_text(catalog) + (("\n\n" + hint) if hint else "")
            trees = sorted(r["trees"], key=nodes)[:2]  # shortest first, cap 2
            for t in trees:
                out.append({"context": r["context"], "question": r["question"],
                            "tree": t, "catalog": cat_txt})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "C"], required=True)
    args = ap.parse_args()
    rows = build_c() if args.arm == "C" else build_a()
    out_dir = ROOT / f"data/training/wtq_sft_{args.arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "sft.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps({
                "messages": [
                    {"role": "user",
                     "content": PROMPT.format(catalog=r["catalog"], q=r["question"])},
                    {"role": "assistant",
                     "content": json.dumps({"tree": r["tree"]}, separators=(",", ":"))},
                ]}, default=str) + "\n")
    qs = len({(r["context"], r["question"]) for r in rows})
    print(f"arm {args.arm}: {len(rows)} examples over {qs} distinct questions -> {out_dir/'sft.jsonl'}")


if __name__ == "__main__":
    main()
