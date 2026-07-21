# §3 System (draft 1)

Figure 1 shows the system. A language model appears in one role only. It
translates a question into a program in a small typed algebra. Everything
after that point is deterministic code, and only that code produces
answers.

**The algebra.** Programs are trees over five value types, record sets,
value sets, group tables, scalars, and booleans. Seventeen node types cover
filtering with negation, disjunction and computed-set membership,
projection, counting, guarded money summation, unique lookup, extremum
records, grouping with count or sum metrics, group reduction by extremum or
top-k, scalar arithmetic and comparison, set union, intersection and
difference, and groupwise combination. A later transfer adds one generic
node, the row-preserving arg-extremum, and nothing else. The grammar is
closed. Every node and enum is fixed, so a recursive type checker decides
membership mechanically and its failures are typed diagnostics. The space
of trees is combinatorially open. This pair of properties is the design
point. Verification stays as mechanical as whitelist membership while
composition becomes free.

**Deterministic execution.** Two evaluators execute every tree. The runtime
evaluator shares data loading with the production system. The independent
evaluator rebuilds the record universe from raw files and shares no code.
Both enforce the same stated conventions, deduplication on the contract
identifier, exclusion of empty group keys, exact decimal money arithmetic
under an additivity guard, and deterministic tie breaks. Agreement between
them is how we measure, rather than assume, that oracles are right.

**Decoding as format authority.** At inference the algebra's grammar is
compiled to a JSON schema and enforced by constrained decoding. Every
system we evaluate, including untrained bases, is format-locked, so
accuracy differences measure planning and not JSON discipline. One
diagnostic seals this point. Decoded freely, the untrained base emits
question-grounded plans in a fluent self-invented schema on 98 of 100
questions but matches the executable contract on 3. The fine-tuned models
match it on 98 and 99. Fine-tuning teaches the contract. Constrained
decoding grants the contract to everyone, which is what makes the ladder a
measure of planning quality.

**Abstention.** Refusal is an output, not a failure. A question may be
ambiguous, may name a concept the schema does not carry, or may match no
records. The system abstains through the same typed channel, and an empty
result whose literals all trace to the question is treated as the data's
answer rather than a defect. A repair loop exists but is gated. It consumes
typed diagnostics from failed checks, proposes one revision, and re-enters
the same checks. Abstentions and semantically meaningful empty results are
final and are never repaired, a rule adopted after an ungated loop
converted six correct refusals into confident wrong answers on a
development slice.

**The verified bootstrap.** Training data is manufactured, not annotated.
A generator authors a program, executes it for the answer under both
evaluators, and only then renders a question surface. A harvest samples
candidate programs from a teacher or from the student itself, executes
them, and keeps a candidate only when every check passes and the answer
matches the oracle. The oracle filters and never authors. No gold program
or answer is copied into training text. What survives trains the next
model, and what fails with a diagnosis becomes labelled negative material.
The recipe has no human annotation and no trusted teacher anywhere in it.

---
## Reviewer pass
- Attack "17+1 feels arbitrary." The one addition is motivated in §5.3 by
  a census and tagged with its own baseline. Sentence already present.
- Attack "two evaluators could share bugs by convention." §4 reports the
  disagreement clusters found and fixed, and the residual 18
  characterised. Forward reference added in §4 draft.
- Attack "constrained decoding is the contribution of vLLM not yours."
  We claim the measurement (3/98/99), not the mechanism. Wording holds.
- Figure 1 spec: three-layer diagram, algebra box with node inventory,
  dual evaluators, bootstrap loop arrows. To draw in LaTeX pass.
- Style contract held. Zero em-dashes, zero colons in prose.
