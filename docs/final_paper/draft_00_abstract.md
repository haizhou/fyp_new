# Abstract (draft 1, written last per lecture discipline)

The United Kingdom publishes hundreds of thousands of contract award
notices, yet the public cannot reliably analyse them in natural language,
because the questions that matter are exhaustive aggregations where a
fluent wrong answer is worse than none. We introduce, to the best of our
knowledge, the first systematic benchmark and executable framework for
auditable compositional question answering over UK public procurement. The
task ships complete. A convention-explicit knowledge graph of 215,221
contract awards, a benchmark of 12,828 questions each carrying an
executable gold program, a typed compositional operator algebra that
formalises real procurement analytics, refusal on ambiguous, unsupported
and empty questions as scored behaviour, and answer oracles validated by
an independent second implementation at 99.88 percent over 14,770 audited
answers. Because no annotator pool can label compositional procurement
programs at scale, training uses a verified bootstrap. Models propose
candidate programs and deterministic checks, compilation, grounding,
execution and oracle agreement, decide what may be learned from. An
8-billion-parameter local model trained this way reaches 85.65 percent
against 69.76 for the cloud teacher that generated its training data
(paired McNemar p below 1e-15), and 78.31 percent on a sealed challenge
set with an independent surface grammar, a constraint-necessity audit,
and a 28-point margin over the same teacher. Transplanted to
WikiTableQuestions by exchanging only the data-representation layer, the
recipe lifts a 22.5 percent zero-shot floor to 44.4 with answer-only
supervision and 51.8 with gold programs on the official sealed test,
while a four-way attribution audit shows a substantial share of the
apparent expressiveness gap to be representation and compilation debt
rather than missing reasoning capability. We claim no state of the art.
The claim is a task built end to end, and audits strong enough to say
precisely what its system can and cannot yet do.

---
## Reviewer pass
- Length ~230 words, TMLR-appropriate.
- Sentence 2 carries the TBOK claim; TheyBuyForYou positioning lives in
  §1/§2 (abstract stays uncluttered).
- All numbers ledger-backed; "substantial share" avoids over-precision in
  abstract (exact seventh stated in §5.3).
- Zero em-dashes, zero prose colons, no judgement words, final two
  sentences give the scope disclaimer.
