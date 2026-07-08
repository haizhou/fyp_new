#!/usr/bin/env python3
"""Deterministic Markdown -> TMLR LaTeX converter for the thesis draft.

Prose is FROZEN: this performs format conversion only. Handles headings, bold/italic/code,
LaTeX escaping, unicode math symbols, pipe tables -> booktabs floats, images -> figure floats,
[Author Year] -> \\citep{key} via a fixed map. Anything unmappable gets a % TODO comment.
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs/thesis_draft"
DST = SRC / "latex/sections"

CITE = {  # substring (lowercased) -> bibkey
    "zelikman": "zelikman2022star", "star": "zelikman2022star",
    "gulcehre": "gulcehre2023rest", "rest-em": "singh2024restem", "singh": "singh2024restem",
    "yuan": "yuan2023rft", "raft": "yuan2023rft",
    "rafailov": "rafailov2023dpo", "azar": "azar2024ipo", "pal": "pal2024smaug",
    "razin": "razin2024displacement", "luo, linhao": "luo2024rog", "rog": "luo2024rog",
    "chatkbqa": "luo2024chatkbqa", "pangu": "gu2023pangu", "gu et al": "gu2023pangu",
    "structgpt": "jiang2023structgpt", "jiang": "jiang2023structgpt",
    "din-sql": "pourreza2023dinsql", "pourreza": "pourreza2023dinsql",
    "mac-sql": "wang2024macsql", "exesql": "exesql2025", "scd": "scd2025",
    "hu et al": "hu2022lora", "lora": "hu2022lora", "dettmers": "dettmers2023qlora",
    "qlora": "dettmers2023qlora", "kwon": "kwon2023vllm", "vllm": "kwon2023vllm",
    "qwen": "qwen3_2025", "llama 3": "grattafiori2024llama3", "grattafiori": "grattafiori2024llama3",
    "burns": "burns2023weaktostrong", "furlanello": "furlanello2018born",
    "hsieh": "hsieh2023distilling", "lightman": "lightman2024verify",
    "deepseek": "guo2025deepseekr1", "lambert": "lambert2024tulu3", "tulu": "lambert2024tulu3",
    "li et al": "li2024bird", "bird": "li2024bird",
}

UNI = {"±": r"$\pm$", "→": r"$\rightarrow$", "←": r"$\leftarrow$", "↔": r"$\leftrightarrow$",
       "Δ": r"$\Delta$", "×": r"$\times$", "≥": r"$\geq$", "≤": r"$\leq$", "⊂": r"$\subset$",
       "∈": r"$\in$", "≈": r"$\approx$", "−": "--", "–": "--", "—": "---",
       "·": r"$\cdot$", "≠": r"$\neq$", "∪": r"$\cup$", "∩": r"$\cap$", "β": r"$\beta$",
       "α": r"$\alpha$", "σ": r"$\sigma$", "①": "(1)", "②": "(2)", "③": "(3)", "④": "(4)",
       "…": r"\ldots{}", "'": "'", "'": "'", """: "``", """: "''", "≫": r"$\gg$"}


def esc(t: str) -> str:
    t = t.replace("\\", r"\textbackslash{}")
    for a, b in UNI.items():
        t = t.replace(a, b)
    t = t.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    t = t.replace("_", r"\_").replace("^", r"\^{}").replace("~", r"\textasciitilde{}")
    t = t.replace("$", r"\$")
    return t


def unesc_math(t: str) -> str:
    # UNI replacements insert $...$; esc() then escaped the $ signs. Restore them.
    return t.replace(r"\$\pm\$", "$\\pm$")  # handled generically below


def inline(t: str) -> str:
    """Escape + convert inline markdown, protecting code/math snippets."""
    out, pos, chunks = [], 0, []
    # protect `code`
    for m in re.finditer(r"`([^`]+)`", t):
        chunks.append(("txt", t[pos:m.start()]))
        chunks.append(("code", m.group(1)))
        pos = m.end()
    chunks.append(("txt", t[pos:]))
    res = []
    for kind, seg in chunks:
        if kind == "code":
            body = seg.replace("\\", r"\textbackslash{}").replace("_", r"\_")
            body = body.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
            body = body.replace("$", r"\$").replace("^", r"\^{}").replace("~", r"\textasciitilde{}")
            res.append(r"\texttt{" + body + "}")
        else:
            e = esc(seg)
            # fix $ inside UNI-inserted math that esc broke
            e = re.sub(r"\\\$\\([A-Za-z]+)\\\$", r"$\\\1$", e)
            e = e.replace(r"\$\pm\$", r"$\pm$")
            for sym in ["pm","rightarrow","leftarrow","leftrightarrow","Delta","times","geq",
                        "leq","subset","in","approx","cdot","neq","cup","cap","beta","alpha",
                        "sigma","gg"]:
                e = e.replace(r"\$\%s\$" % sym if False else "\\$\\" + sym + "\\$", "$\\" + sym + "$")
            # p-values to math
            e = re.sub(r"p\s*<\s*1e-(\d+)", r"$p<10^{-\1}$", e)
            e = re.sub(r"p\s*=\s*([0-9.]+)e-(\d+)", r"$p=\1\\times10^{-\2}$", e)
            # bold/italic
            e = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", e)
            e = re.sub(r"\*([^*\n]+)\*", r"\\emph{\1}", e)
            # [Author Year] citations
            def cite(m):
                inner = m.group(1)
                keys = []
                for part in re.split(r"[;,]", inner):
                    p = part.strip().lower()
                    for sub, key in CITE.items():
                        if sub in p and key not in keys:
                            keys.append(key); break
                if keys:
                    return r"\citep{" + ",".join(keys) + "}"
                return m.group(0) + " % TODO cite"
            e = re.sub(r"\[([A-Z][^\[\]]{2,90}?(?:19|20)\d\d[a-z]?)\]", cite, e)
            res.append(e)
    return "".join(res)


def table_block(lines: list[str], label: str) -> str:
    rows = [[c.strip() for c in re.split(r"(?<!\\)\|", ln)[1:-1]] for ln in lines]
    rows = [r for r in rows if r and not all(set(c) <= set(":- ") for c in r)]
    ncol = max(len(r) for r in rows)
    out = ["\\begin{table}[t]", "\\centering", "\\small",
           "\\caption{TODO caption}", f"\\label{{tab:{label}}}",
           "\\begin{tabular}{" + "l" * ncol + "}", "\\toprule"]
    for i, r in enumerate(rows):
        cells = [inline(c) for c in r] + [""] * (ncol - len(r))
        out.append(" & ".join(cells) + r" \\")
        if i == 0:
            out.append("\\midrule")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def convert(src: Path, dst: Path, bare: bool = False):
    lines = src.read_text(encoding="utf-8").splitlines()
    out, i, tcount, fcount = [], 0, 0, 0
    slug = dst.stem
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|") and i + 1 < len(lines) and set(lines[i+1].replace("|", "").strip()) <= set(":- "):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            tcount += 1
            out.append(table_block(lines[i:j], f"{slug}_{tcount}"))
            i = j
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            if bare:
                i += 1
                continue
            depth = len(m.group(1))
            title = re.sub(r"^\d+(\.\d+)*\s*", "", m.group(2)).strip()
            cmd = {1: "section", 2: "section", 3: "subsection", 4: "subsubsection"}[depth]
            sl = re.sub(r"[^a-z0-9]+", "-", title.lower())[:40].strip("-")
            out.append(f"\\{cmd}{{{inline(title)}}}\\label{{sec:{sl}}}")
            i += 1
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", ln)
        if m:
            alt, path = m.group(1), m.group(2)
            stem = Path(path).stem
            cap = re.sub(r"Render: .*$", "", alt).strip()
            cap = re.sub(r"^Figure [\d.]+ ?[—-] ?", "", cap)
            fcount += 1
            out += ["\\begin{figure}[t]", "\\centering",
                    f"\\includegraphics[width=\\linewidth]{{{stem}}}",
                    f"\\caption{{{inline(cap)}}}", f"\\label{{fig:{slug}-{fcount}}}",
                    "\\end{figure}", ""]
            i += 1
            continue
        if ln.startswith("> "):
            out.append("\\begin{quote}")
            while i < len(lines) and lines[i].startswith(">"):
                out.append(inline(lines[i][1:].strip()))
                i += 1
            out.append("\\end{quote}")
            continue
        if re.match(r"^\s*[-*]\s+", ln):
            out.append("\\begin{itemize}")
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                out.append("\\item " + inline(re.sub(r"^\s*[-*]\s+", "", lines[i])))
                i += 1
            out.append("\\end{itemize}")
            continue
        if re.match(r"^\s*\d+\.\s+", ln):
            out.append("\\begin{enumerate}")
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                out.append("\\item " + inline(re.sub(r"^\s*\d+\.\s+", "", lines[i])))
                i += 1
            out.append("\\end{enumerate}")
            continue
        out.append(inline(ln))
        i += 1
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{src.name} -> {dst.name}: {tcount} tables, {fcount} figures")


MAP = [("00_abstract.md", "00_abstract.tex", True),
       ("01_introduction.md", "01_introduction.tex", False),
       ("02_background_related_work.md", "02_background.tex", False),
       ("03_data_and_kg.md", "03_data_kg.tex", False),
       ("04_benchmark.md", "04_benchmark.tex", False),
       ("05_system.md", "05_system.tex", False),
       ("06_bootstrapping_experiments.md", "06_experiments.tex", False),
       ("07_analysis.md", "07_analysis.tex", False),
       ("08_conclusion.md", "08_conclusion.tex", False)]

if __name__ == "__main__":
    DST.mkdir(parents=True, exist_ok=True)
    for s, d, bare in MAP:
        convert(SRC / s, DST / d, bare)
