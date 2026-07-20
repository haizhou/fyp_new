# E — Hand-authored mini-probe v1 (FROZEN before any WTQ self-harvest/SFT)

**Authorship disclosure (material to evidential weight):** these 40 questions
were written by the Claude agent that co-developed the system, at the user's
request (2026-07-20). They were written WITHOUT consulting the training
templates, generator code, or any training example, but the author is not
independent of system development. Treat this as a *hand-authored held-out
probe*, NOT as external human evidence. Independence upgrade path: the user
replaces/edits ≥10 rows before the run; edited rows get author=uceeh01.

Run plan (frozen): checkpoints = composev3 (frozen) + base; guided decoding,
procurement schema; both evaluators; one run each; results frozen beside this
file. Status labels follow PACS v2.2: answerable | ambiguous | empty_result |
unsupported_field | requires_missing_operator.

| # | question | expected operators | status | seen-dist? |
|---|---|---|---|---|
| 1 | surrey county council 2024 contracts — how many | filter,count | answerable | seen |
| 2 | NHS England 一共签了多少 works 类的合同? | filter,count | answerable | unseen-surface |
| 3 | how many contrcts did Lincolnshire County Council award in 2023 (sorry for typo) | filter,count | answerable | unseen-surface |
| 4 | Could you kindly enumerate the distinct suppliers that have been party to at least one services agreement with Surrey County Council? | filter,values | answerable | unseen-surface |
| 5 | which year was busiest for The North Yorkshire Council, by number of awards? | groupby,argext | answerable | seen |
| 6 | top 3 buyers by total spend in 2024, thanks | filter,groupby(sum),top | answerable | seen |
| 7 | is there any supplier that worked for BOTH Surrey County Council and Lincolnshire County Council? name them | values,setop(intersect) | answerable | unseen |
| 8 | suppliers of NHS England in 2023 that did not appear among its 2024 suppliers | values,setop(difference) | answerable | unseen |
| 9 | which buyers have more than 50 goods contracts | groupby,keys_where | answerable | unseen |
| 10 | who spent more in 2024 — Surrey County Council or Lincolnshire County Council? | sum,combine(gt) | answerable | seen |
| 11 | did Biffa Waste Services Ltd win anything in 2025? | filter,exists | answerable | seen |
| 12 | what's the average value of a works contract where NHS England is the buyer | sum,count,combine(ratio) | answerable | unseen |
| 13 | roughly what share of Surrey County Council's contracts are services | count,count,combine(ratio) | answerable | unseen |
| 14 | count Surrey County Council's 2024 contracts that are NOT services | filter(not),count | answerable | seen |
| 15 | how many Lincolnshire County Council contracts are goods or works | filter(any),count | answerable | seen |
| 16 | 帮我看看2024年哪个买家花钱最多 | filter,groupby(sum),argext | answerable | unseen-surface |
| 17 | which year had the LOWEST total spend for NHS England | groupby(sum),argext(argmin) | answerable | seen |
| 18 | number of distinct CPV codes appearing in works contracts | values,size | answerable | seen |
| 19 | is Surrey County Council's total 2025 spend above 100 million pounds? | sum,num,combine(gt) | answerable | unseen |
| 20 | list suppliers of Surrey County Council in 2022 with a contract value over 1 billion | filter,values | empty_result | unseen |
| 21 | how many contracts were released in 2021 | filter,count | empty_result | seen |
| 22 | Ormiston Academies Trust — total value of everything they bought | filter,sum | answerable | seen |
| 23 | who are ESKCARE's buyers | filter,values | answerable | seen |
| 24 | 2023 vs 2024: did the number of services contracts go up? | count,count,combine(lt) | answerable | unseen |
| 25 | which supplier shows up most often for Surrey County Council | filter,groupby,argext | answerable | seen |
| 26 | any goods contract signed by Wales & West Utilities? yes or no is fine | filter,exists | answerable | unseen-surface |
| 27 | buyers common to Biffa Waste Services Ltd and Chiltern Healthcare | values,setop(intersect) | answerable | unseen |
| 28 | how many buyers gave contracts to more than one distinct supplier in 2024 | groupby-ish / values,size per key | answerable (hard) | unseen |
| 29 | difference in total spend between goods and services in 2024 | sum,sum,combine(diff) | answerable | seen |
| 30 | 有没有哪家 supplier 2022到2026每一年都有中标?哪家? | per-year values,setop(intersect) chain | answerable (deep) | unseen |
| 31 | what is the MEDIAN contract value in 2024 | — | requires_missing_operator | n/a |
| 32 | is monthly spending trending upwards through 2024? | — | requires_missing_operator | n/a |
| 33 | what's the average contract DURATION for NHS England deals | — | unsupported_field | n/a |
| 34 | how many of the 2024 winners are small businesses (SMEs)? | — | unsupported_field | n/a |
| 35 | how many contracts last year | — | ambiguous (no reference year) | n/a |
| 36 | how much did the council spend in total? | — | ambiguous (which council) | n/a |
| 37 | why did Surrey County Council pick Biffa over other bidders? | — | unsupported_field (no rationale data) | n/a |
| 38 | who will win the next NHS England tender? | — | unsupported_field (future) | n/a |
| 39 | contracts where the buyer and the supplier are the same organisation — any? | filter(field-vs-field) | requires_missing_operator (no field-to-field predicate) | n/a |
| 40 | show me the biggest contract of 2025 and tell me its title | extreme_rows/extreme+select | answerable | unseen (extension node) |

Notes: #28/#30 are deliberately at or beyond comfortable depth (probe the
composition ceiling); #39 probes a real grammar hole (field-to-field
comparison) and must be abstained, not faked; #40 is answerable via the
extension node on the procurement side (extreme+select over value_amount).
