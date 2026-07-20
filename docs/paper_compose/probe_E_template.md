# E — Human-written mini-probe (BLOCKS Layer-2 self-harvest / SFT)

Per protocol: this probe must be authored, frozen, and RUN before any WTQ
self-harvest or SFT, so it stands as independent zero-shot/OOD evidence.
30–50 questions, written by a human (not generated), over the procurement KG
and/or a WTQ dev table of your choosing.

One row per question — fill and return; I run it against the frozen
checkpoints and freeze results alongside.

| # | question (natural language) | author | date | expected operators (your guess) | seen-in-training-distribution? (yes/no/unsure) |
|---|---|---|---|---|---|
| 1 | | uceeh01 | 2026-07-__ | | |

Notes
- Write questions the way YOU would ask them — do not imitate the training
  phrasing; idiomatic Chinese-English mixture, typos, and underspecification
  are all valid and valuable.
- Include a few deliberately unanswerable ones if you like (abstention check).
- I will record: checkpoint ids (composev3 frozen + base), guided-decode
  config, per-question trees/answers, and a config hash at run time.
