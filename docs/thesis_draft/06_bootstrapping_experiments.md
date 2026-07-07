# Chapter 6 — Bootstrapping Experiments: Ladder, Yields, and the Five-Pairing Confirmation

Chapter 5 built and hardened the pipeline. This chapter trains models inside it and measures
the result. The claim is that the learning layer of the architecture works, and that it works
three times over, in three progressively stronger forms. First, teacher-filtered distillation:
an 8-billion-parameter student trained only on verifier-passed, oracle-gated teacher outputs
climbs a four-rung training ladder on the development set. Second, verifier-gated
self-improvement: the student replaces its teacher inside the same harvesting harness, and its
rejection-sampled self-harvest yields more verified, oracle-correct training data than the
teacher's harvest did. Third, Step-1 denoising distillation: briefings filtered by
whole-pipeline outcome produce a fully-local two-stage system that runs without any cloud
model. The confirmatory evidence is a single held-out scoreboard on the frozen test set
(n = 2,285): teacher 69.76% < Llama hybrid 75.97% < Qwen hybrid 78.03% < Llama fully-local
83.33% < Qwen fully-local 85.65%. Every adjacent and non-adjacent pairing is significant under
McNemar's test at p < 10⁻¹⁴. Throughout, supervision flowed only from the verifiable region of
the task; the resulting systems still surpass the source of that signal.

## 6.1 Experimental design: two configurations, one metric discipline

All experiments use the benchmark of Chapter 4. Its 12,828 questions are split into a training
pool (9,267), development slices (dev_tune 556, dev_select 671, dev_smoke 49), and a frozen
confirmatory test set, final_test (2,285). Model selection throughout this chapter is done on
compare_set_v4, a 260-question development set stratified as 13 buckets of 20 questions.
final_test is touched exactly once per selected system, after all development decisions were
frozen. Because compare_set_v4 was drawn as a stratified subset of final_test, development
results are reported only in a model-selection role and never as headline numbers; the
headline lives on final_test alone.

Two distinct pipeline configurations appear in this chapter, and they must be kept apart to
read the tables correctly.

The **harvest configuration** is a data engine. It is implemented by `scripts/run_teacher.py`,
which runs the full live two-step pipeline over the training pool. A Step-1 understanding
model produces a dense briefing, and a Step-2 planner produces a typed graph plan. The plan
then passes through deterministic grounding, compilation, exhaustive execution over the
knowledge graph, and the verifier chain. The harvest runs with a repair budget of two
(`--max-repairs 2`). Every verifier-rejected plan is fed back, with structured failure
feedback, for up to two gated replanning attempts, each re-run through the complete
ground–execute–verify path. The goal of this configuration is to maximise the yield of
verified, correct training material. Its output metric is **data-engine yield**: the fraction
of questions for which the engine produced a verified plan whose answer also matches the
held-out oracle.

The **evaluation configuration** is a measurement instrument. It is implemented by
`scripts/run_compare.py` and orchestrated over the ladder by `scripts/eval_ladder.sh`. It
holds a deliberately tighter budget: repair budget one (`max_feedback_replans=1`), two plan
samples per question, and schema-guided JSON decoding on every rung, including the untrained
zero-shot rung. This keeps all cells of the evaluation matrix comparable under an identical
inference budget. Its metric is **system accuracy**: type-aware exact-match against the oracle
on answerable questions, with an unanswerable question scored correct if and only if the
system abstains, exactly as defined in Chapter 4.

Yield and accuracy are different quantities, measured on different question sets under
different budgets, and this chapter never compares one against the other. The teacher appears
twice in what follows, at 65.5% as a data engine and at 69.76% as an evaluated system, and
these are statements about two different things.

Training uses a single recipe across all adapters and both base models, so every rung-to-rung
and base-to-base comparison is a data comparison rather than a hyperparameter comparison. All
adapters are QLoRA fine-tunes (4-bit NF4 quantised base, LoRA rank 64, alpha 128, dropout
0.05, all linear targets), trained with LLaMA-Factory in bf16 with SDPA attention, sequence
cutoff 6,144 tokens, and an effective batch size of 16 in every stage. Stage-specific settings
are as follows. SFT runs at learning rate 1×10⁻⁴ for 3 epochs with cosine decay.
Rejection-sampling SFT (RSFT) continues the SFT adapter at 5×10⁻⁵ for 2 epochs. DPO continues
the RSFT adapter at 5×10⁻⁶ for 1 epoch with sigmoid preference loss and β = 0.1
(`configs/training/*.yaml`). The
two bases are Qwen3-8B (served and trained without thinking blocks, matching its serving
configuration) and Llama-3.1-8B-Instruct.

A final design invariant concerns the training data itself. The exporter
(`scripts/export_llamafactory.py`) renders every training prompt with the *runtime* prompt
builders — the same `typed_plan_messages` and `typed_replan_messages` functions the deployed
pipeline calls — so the distribution the student is trained on is byte-identical to the
distribution it is served on. Targets follow a single serialisation convention: compact JSON
of the compiler-normalised graph plan. To prevent shape memorisation, the exporter caps
emission at 150 samples per template family and 400 per question bucket. The count bucket
spans roughly eight families and would otherwise make up about a third of the pool, teaching a
"when unsure, count" prior. Abstention examples are capped at 10% of the verified plan pool,
because uncapped teacher pools run near 14% abstention against a natural rate of roughly 5%
and would over-teach refusal.

## 6.2 The data engine: teacher harvest over the training pool

The first harvest ran the live teacher — GPT-5.4-nano as Step-1 and Grok-4.1-fast as Step-2,
at temperature 0 — over all 9,267 training questions. Acceptance into the supervised pool is
doubly gated. A plan must pass the full deterministic verifier chain, and its executed answer
must also agree with the held-out oracle (a mechanical answer-shape gate rejects, for example,
a numeric answer to a boolean question). The oracle's role is strictly that of a filter. It
never contributes an answer, a plan, or any content to a training target; it only decides
whether a pipeline-produced artifact may enter the pool. Verifier-passing plans whose answers
fail the oracle gate are the most valuable rejects: they are exactly the plans the runtime
verifier cannot distinguish from correct ones. They are routed to a hard-negative pool and
offered one oracle-*gated* repair attempt, in which the feedback discloses only that the
answer "failed external validation", never the oracle content itself.

Each question therefore lands in exactly one of several sinks. `verified_sft` holds verified
and oracle-correct plans, with briefing and question. `abstain_sft` holds correct abstentions
on unanswerable questions, teaching the abstain plan shape as a first-class success.
`repair_sft` holds feedback-conditioned repair demonstrations whenever success arrived on a
repair attempt. `dpo_pairs` holds the successful attempt as *chosen* against the nearest
preceding failed attempt of the same question as *rejected*; we call this the attempt
protocol. The remaining sinks are `hard_negatives` and `failures`. One point about the attempt
protocol deserves emphasis: preference pairs are not synthesised. They are the repair loop's
own before/after trajectories, so both sides of every pair are plans the pipeline actually
produced for that question.

On the answerable portion of the pool the teacher's data-engine yield was **65.5%**
(5,605 of 8,555 verified and oracle-correct; `data/qa/teacher_full_v1`, recomputed from
traces), with correct abstention on 86.8% (618/712) of unanswerable questions. Of 6,860
verified plans, 4,788 were verified on the first attempt, so the repair loop contributed
roughly two thousand additional verified plans. The repair budget pays for itself at harvest
time even though, as Chapter 5 established, it is roughly accuracy-neutral at inference time.
Routing produced 5,598 verified-SFT rows and 590 abstention rows, alongside 1,262 hard
negatives, 1,725 repair demonstrations, and 390 preference pairs. After family and bucket
caps, the exported round-1 dataset comprised 3,632 plan-SFT training samples (76 validation),
1,679 repair samples, and the 390 pairs.

Two disciplinary notes attach to these numbers. First, 65.5% is a yield, not an accuracy. It
was obtained under the harvest budget on the deliberately hard training pool, and it is not
comparable to any number in Sections 6.7–6.8. Second, the pool is hard by construction. Its
composition is dominated by the difficult buckets (count, bridge_join, factoid), precisely so
that the harvest concentrates supervision where the task is hardest. Per-bucket usable yield
ranged from above 92% on count and sum questions down to 24.9% on bridge_join, which
consequently doubles as the richest source of hard negatives.

## 6.3 Supervised fine-tuning: the first rung on both bases

The SFT rung trains each base on the exported teacher pool (plan-SFT plus repair-SFT). Both
runs were uneventful in the best sense: 840 optimisation steps over 3 epochs, roughly two and
three-quarter hours of wall-clock per base on a single H100. Validation loss on the held-out
plan slice decreased monotonically (Qwen 0.0186 at step 200 to 0.0128 at convergence; Llama
0.0181 to 0.0118), and the two bases' curves did not diverge. Each adapter was smoke-tested on
a small stratified slice behind a vLLM endpoint before entering the evaluation matrix; per the
pre-registered discipline, smoke figures are sanity checks and are not reported as results.

Under the evaluation protocol on the development set (pipeline v2.2), SFT lifts Qwen from a
zero-shot 70.4% to **81.2%**, and Llama from 60.0% to **83.1%**. The jump is larger on the
weaker base. This is consistent with the interpretation, developed in Chapter 7, that
fine-tuning teaches conformance to the executable planning contract rather than planning
ability itself. The full matrix is presented in Section 6.7.

## 6.4 Rejection-sampled self-harvest: the student out-harvests its teacher

The central bootstrapping question is whether the student can replace the teacher *inside the
data engine*. The RSFT round answers it. We re-ran the harvest harness of Section 6.2 with a
single substitution: Step-2 was pointed at the local SFT student (served via vLLM) instead of
the cloud teacher. Sampling ran at temperature 0.7 with four plan samples per question, so
that rejection sampling can explore beyond what greedy decoding already solves. Step-1, the repair
budget of two, the oracle gate, and every sink remained identical.

**Table 6.1 — Data-engine yield on the training pool (harvest configuration, repair budget 2;
oracle-correct on answerable, n = 8,555). Yield figures; not comparable to system accuracy.**

| Data engine (Step-2 policy) | Sampling | Answerable yield | Correct abstention |
|---|---|---|---|
| Teacher (Grok-4.1-fast) | greedy | 65.5% (5,605/8,555) | 86.8% (618/712) |
| Qwen SFT student | temp 0.7 × 4 | **76.6% (6,556/8,555)** | 88.1% (627/712) |
| Llama SFT student | temp 0.7 × 4 | **78.1% (6,685/8,555)** | — |

Both students exceed the teacher's yield by a wide margin: +11.1 points for Qwen and +12.6 for
Llama. Abstention quality does not degrade under temperature sampling. This is the first
exceeds-teacher evidence in the thesis, and its scope must be stated carefully: it is a
statement about the *data engine*, not about model accuracy. Temperature-0.7 rejection
sampling with four candidates, a deterministic verifier gate, and a two-repair loop finds
correct plans beyond both the teacher's greedy decoding and the student's own. The Qwen run's
verified count rose from 4,712 at the first attempt to 7,601 after sampling and repair, a gain
of 2,889 verified plans that comes from the search-and-filter machinery rather than from the
policy alone. The eval-accuracy comparison between student and teacher is a separate claim,
with its own numbers and its own question set (Sections 6.7–6.8).

The honest flags belong in the text rather than a footnote. On the bridge_join bucket the Qwen
self-harvest *verified* 86.7% of questions but was oracle-correct on only 52.3%. This
34.4-point band of verifier-passing-but-wrong plans is the signature of the check-3 blind
region described in Chapter 3: the count-intermediate-set misreading is structurally valid and
executes cleanly. Exactly as designed, this band did not contaminate the supervised pool — the
export remains oracle-gated, and all 6,550 exported verified-SFT rows are both verified and
oracle-correct. Instead the band was converted into fuel: 1,051 hard negatives and 689 new
*on-policy* preference pairs, against the teacher round's 390. Even at 52.3%, the student's
bridge yield roughly doubles the teacher's (~25% usable on the same bucket).

Training on the self-harvest (RSFT: continuing the SFT adapter at 5×10⁻⁵ for 2 epochs on
3,084 plan and 3,169 repair samples) produced a clear dissociation between loss and behaviour.
Validation loss halved relative to the SFT rung (0.0065 terminal), yet development accuracy
moved only from 81.2% to 81.5% on Qwen and from 83.1% to 82.7% on Llama. Under the flat
aggregate, capability was *reshuffled* toward the self-harvest's emphasis. Qwen's RSFT rung
gained four bridge_join questions but collapsed its factoid bucket from 18/20 to 7/20 through
over-abstention. The collapse is a policy-specific data bias, not a property of the method:
Llama's own self-harvest induced no such artifact (its factoid bucket held at 16–17/20), and
the subsequent DPO rung restored Qwen's factoid bucket to 18/20. Self-harvested data inherits
the quirks of the policy that generated it, a caution that recurs in the preference-learning
analysis of Chapter 7.

## 6.5 Preference optimisation on merged pairs

The DPO rung merges the two rounds of attempt-protocol pairs into a single preference pool:
390 off-policy pairs from the teacher harvest plus 689 on-policy pairs from the Qwen
self-harvest, 1,079 pairs in total. The pool includes the oracle-gated wrong-answer-repair
pairs, in which the chosen plan is the repaired one and the rejected plan is its
verifier-passing-but-wrong predecessor. Training continues the RSFT adapter for a single epoch
(68 steps) at 5×10⁻⁶ with β = 0.1.

On the development set, DPO lifts Qwen from 81.5% to **83.5%**, recovering the factoid bucket
and making it the Qwen ladder's champion rung. On Llama the same procedure *regresses* the
ladder from 82.7% to 77.3%. This asymmetry is not noise. Chapter 7 analyses the mechanism:
near-miss on-policy negatives on the bridge bucket are string-adjacent to correct plans, and
suppressing them displaces the correct modes; the pathology arrives at higher dose on Llama.
Two remediation arms (a rebalanced sigmoid pool with a chosen-likelihood anchor, and an IPO
variant) were trained within the pre-committed three-round development budget and did not beat
the original rung (80.0% and 82.3% respectively). Both are reported as mechanism-explained
negative results, and Llama's selected configuration for everything downstream is its
ladder-best SFT rung.

## 6.6 Step-1 distillation: the fully-local stack

Every system so far is a *hybrid*: local Step-2 planning under a cloud Step-1 briefing. The
final form of the learning layer removes this last cloud dependency by distilling Step-1
itself. The filter here must be different, because a briefing is free-form text on the
unverifiable side of the projection boundary, and no deterministic check can accept it
directly. Instead, the training set consists of the 5,091 briefings whose *entire downstream
pipeline outcome* was verified and oracle-correct. A briefing enters the pool only if the plan
built on top of it executed, passed the verifier chain, and matched the oracle. This is the
partial-verifiability filter applied one stage upstream; the verifiable end of the pipeline
vouches for the unverifiable middle. Step-1 adapters were trained on both bases under the
identical QLoRA recipe and paired with each base's champion Step-2 rung (Qwen: DPO; Llama:
SFT). The two adapters were served side by side on a single vLLM instance, with no cloud call
anywhere in the stack.

On the development set the fully-local stacks score **86.2%** (Qwen, +2.7 over its hybrid) and
**84.2%** (Llama, +1.1). The cross-base replication points to a recipe-level effect, and it is
the third independent validation of the partial-verifiability claim, after teacher filtering
and self-harvest. However, the development pairings are individually non-significant
(+13/−6, McNemar p = 0.167 for Qwen; +14/−11, p = 0.690 for Llama), so this chapter draws no
superiority conclusion from them. The fully-local-beats-hybrid claim is licensed only by the
final_test pairings of Section 6.8, where it is tested at nine times the sample size.

## 6.7 The development ladder matrix

Table 6.2 assembles the full evaluation matrix on the development set under pipeline v2.2 and
the uniform evaluation protocol. Its role is model selection and robustness analysis; none of
its cells is a headline claim.

**Table 6.2 — Development ladder (compare_set_v4, n = 260, pipeline v2.2, evaluation protocol:
repair budget 1, two plan samples, guided JSON on every rung). Artifacts under
`outputs/eval/matrix_v2/`.**

| System | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| Zero-shot (full scaffolding, untrained) | 70.4% (183/260) | 60.0% (156/260) |
| SFT | 81.2% (211/260) | 83.1% (216/260) |
| RSFT | 81.5% (212/260) | 82.7% (215/260) |
| DPO | **83.5% (217/260)** | 77.3% (201/260) |
| Fully-local (Step-1 + champion Step-2) | **86.2% (224/260)** | **84.2% (219/260)** |
| Teacher, 3 replicates | mean 72.6% ± 1.1 (71.9, 71.9, 73.9) | — |
| RAG baseline, naive / strong (v1) | 31.5% (82/260) / 28.8% (75/260) | — |

Three readings matter. First, the scaffolding floor. The zero-shot rung — an untrained base
model inside the full deterministic scaffold of grounding, compilation, exhaustive execution,
verification, and guided decoding — already roughly doubles both RAG baselines (under the v1
pipeline, zero-shot Qwen scored 61.5% against 31.5% and 28.8%). Training gains are therefore
added on top of an already-raised floor, and Chapter 7 decomposes the two contributions.
Second, the training gain. SFT adds around 11 points to Qwen and 23 to Llama over their
zero-shot rungs; RSFT is aggregate-neutral but redistributes capability; preference
optimisation adds 2 points on one base and subtracts 5 on the other. Third, robustness to
teacher nondeterminism. Because the cloud teacher is a live, nondeterministic service, we
evaluated it three times, which defines a noise floor of 72.6% ± 1.1. The Qwen DPO rung beats
each of the three replicates individually (discordants +35/−5, +34/−4, +30/−5; McNemar
p = 3×10⁻⁷, 5×10⁻⁷, 2×10⁻⁵), so the development-set student–teacher gap cannot be an artifact
of catching the teacher on a bad run. This three-replicate robustness role is exactly what the
development set contributes to the student-versus-teacher claim; the claim itself is settled
on final_test.

## 6.8 The confirmatory scoreboard on final_test

Each base's hybrid champion and fully-local stack, together with a single teacher replicate,
was evaluated exactly once on the frozen final_test set (n = 2,285, benchmark v4.1, evaluation
protocol). All model selection preceded these runs; nothing was tuned afterwards.

**Table 6.3 — Confirmatory five-system scoreboard (final_test, n = 2,285, v4.1, repair budget
1). All ten pairwise comparisons significant at McNemar p < 10⁻¹⁴. Confidence intervals are
95% intervals on the paired accuracy difference. Artifacts under `outputs/eval/final_test/`.**

| System | Accuracy | Δ vs teacher [95% CI] | Δ vs own hybrid [95% CI] | dev → test decay |
|---|---|---|---|---|
| Teacher (single replicate) | 69.76% (1,594/2,285) | — | — | — |
| Llama hybrid (nano Step-1 + SFT) | 75.97% | +6.21 [+4.68, +7.74] | — | −7.1 pt |
| Qwen hybrid (nano Step-1 + DPO) | 78.03% (1,783/2,285) | +8.27 [+6.70, +9.85] (+263/−74) | — | −5.5 pt |
| Llama fully-local (Step-1 + SFT) | 83.33% | +13.57 [+11.81, +15.32] | +7.35 [+5.73, +8.97] | −0.9 pt |
| Qwen fully-local (Step-1 + DPO) | **85.65% (1,957/2,285)** | **+15.89 [+14.07, +17.70] (+406/−43)** | +7.61 [+6.10, +9.13] (+242/−68) | **−0.5 pt** |

The scoreboard delivers the chapter's three claims at scale. First, every student system beats
the teacher: even the weaker base's hybrid clears it by six points, and the Qwen fully-local
stack answers 406 questions the teacher misses while losing only 43 in the other direction.
Per the pre-committed wording discipline, the student-versus-teacher comparisons in Table 6.3
are pairings against a *single* teacher replicate. The robustness of the student–teacher gap
to provider nondeterminism is established by the three-replicate analysis on the development
set (Section 6.7), and the two statements are kept deliberately separate. Second, on both
bases the fully-local system beats its own cloud-hybrid counterpart by a nearly identical
margin (+7.35 and +7.61 points). The claim that the development pairings could only suggest is
confirmed here with p < 10⁻¹⁴ on both bases. Third, the effect of the distilled briefing stage
is strong enough that the weaker base with local briefings outranks the stronger base with
cloud briefings (Llama fully-local 83.33% vs Qwen hybrid 78.03%). This is a post-hoc,
non-preregistered observation; we note it here and take it up mechanistically in Chapter 7.

The final column contains the most diagnostic pattern in the table. Moving from the
development set to the harder natural distribution of final_test, both hybrids decay steeply
(−5.5 and −7.1 points) while both fully-local systems are essentially flat (−0.5 and −0.9).
The pattern holds in all four cells, across bases and configurations, at n = 2,285. It is the
strongest single piece of evidence that Step-1 distillation acts as *denoising*: the local
briefing stage, trained only on briefings whose downstream outcome was verified correct, stays
in-distribution for the planner exactly where the cloud briefings become irregular. Chapter 7
gives the mechanistic decomposition of this asymmetry.

Finally, the disclosure required by Chapter 4: nineteen of the 2,285 test questions
string-match the `_UNSUPPORTED_CUES` list that was extended during development. The dual
report excluding these rows moves every headline by at most 0.21 points (teacher 69.76 →
69.55; Qwen fully-local 85.65 → 85.57) and changes no ranking and no significance verdict.
The deltas are uniformly *negative* because the cue rows were mostly answered correctly by all
systems. Removing easy rows lowers every accuracy slightly; the deltas do not indicate any
system-specific advantage.

## 6.9 Round-2 self-harvest: bootstrap convergence

Does the loop keep paying? We ran a second self-harvest round (r2) under the fully-local
champion configuration: local Step-1 adapter and the DPO champion as Step-2, temperature
0.7 × 4, with the data engine now containing no cloud model at all. Its outcome was mapped
onto gates pre-committed *before any r2 number existed*. The gates were: a headline swap only
if r2 beat the champion by at least 3 development points with a significant bridge
improvement; a matrix row if it was at least neutral; and an honest diminishing-returns report
otherwise.

At the harvest level the loop is still climbing steeply. Answerable yield reached **86.3%**
(teacher 65.5% → r1 76.6% → r2 86.3%), correct abstention 97.6%, and the bridge_join bucket —
the task's hardest family — jumped from 52% oracle-correct in r1 to **75%** in r2. The DPO
policy samples correct bridge plans far more often than the SFT policy did. At the training
level, however, the gain has stopped. The r2 adapter (iterative RSFT continuing the DPO
champion on its own harvest) scored 86.5% on the development set against the champion's 86.2%:
+0.4 points, discordants +3/−2, McNemar p = 1.0. Its bridge bucket improved by exactly one
question (15/20 vs 14/20, p = 1.0). Per the pre-committed gates this is a supplementary-row
outcome: the headline system remains the fully-local DPO stack, and r2 is reported as a data
point on a convergence curve rather than as a new rung.

The convergence curve is itself a finding. Round 1 converted an 11-point yield gain into a
significant ladder gain; round 2 converted a further 10-point yield gain into statistical
noise. The verifier-gated self-improvement loop has a natural ceiling on this task. Once the
policy already solves, under sampling and repair, nearly everything the verifiable region can
certify, additional verified data re-teaches what the policy already knows. Yield keeps rising
precisely because the policy improved; training gain stalls precisely because the remaining
errors live where the verifier cannot see. The Llama arm of r2 was not trained. The
convergence claim is established on the main base; single-base evidence is enough for a
convergence observation (unlike the DPO asymmetry, no cross-base contrast is claimed), and the
remaining compute budget went to the pre-registered compositional probe.

## 6.10 Cost analysis

The cost asymmetry between the teacher and its replacements is a result in its own right. The
teacher harvest is a metered, latency-bound process. At roughly 13 questions per minute it
took about twelve hours of wall-clock over the 9,267-question pool and consumed on the order
of tens of thousands of billable API calls (each question costs one Step-1 call, one to two
Step-2 calls, and up to two repair round-trips). The self-harvest replaced the metered Step-2
with a local vLLM endpoint and ran at 85 questions per minute — the full pool in under two
hours, at zero marginal cost for every sampled candidate. This is what makes the
temperature-0.7 × 4 rejection-sampling regime economically trivial for a student and
prohibitive for a metered teacher. By r2 the engine contained no cloud model at all.

Training costs are one-off and modest. Each SFT-class rung is roughly three hours on a single
H100 (2h43m for Qwen SFT, 2h22m for Llama SFT, 2h42m for RSFT), the DPO rung thirteen minutes,
and no stage requires more than one GPU. On the inference side, the fully-local champion
serves both stages as two LoRA adapters on one vLLM instance on one GPU; the entire
2,285-question final_test evaluation of the headline system ran without a single external API
call. The deployment case for the fully-local stack is therefore not accuracy traded for cost
but accuracy *and* cost. It is 15.89 points more accurate than the cloud teacher it learned
from, at zero marginal inference cost, and with no procurement data leaving the machine.

## 6.11 Summary

Three instantiations of the same principle produced three compounding results. Teacher-
filtered distillation turned a 65.5%-yield data engine into students that clear the teacher's
own evaluation accuracy. Verifier-gated self-harvest then out-yielded the teacher's engine by
11–13 points and converted its blind-region failures into preference-learning fuel. Step-1
distillation through the whole-pipeline filter removed the final cloud dependency and, on the
held-out confirmatory set, produced the ordering teacher < hybrid < fully-local on both bases,
with every pairing significant at p < 10⁻¹⁴ and near-zero dev-to-test decay. A second
self-harvest round located the ceiling of the loop: harvest yield keeps rising while training
gain stops, which fixes the boundary of what verifier-gated bootstrapping can extract from
this task. Supervision never left the verifiable region; the systems it trained did. Chapter 7
turns from the scoreboard to the mechanisms behind it.
