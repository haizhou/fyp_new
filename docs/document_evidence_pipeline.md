# Document Evidence Pipeline

This project uses documents as passive evidence during answer verification. The
KG locates the contract first; document text is loaded only after the evidence
verdict stage knows the target `ocid`.

## Scope

Default URL scope:

- `data/extracted/documents.parquet`
- `source in ("tender", "award")`
- non-empty `url`
- de-duplicated by cleaned URL hash

This currently yields 55,059 unique external URLs attached to
`tender.documents[]` or `awards[].documents[]`. Submission URLs, party profiles,
buyer profiles, and contact emails are excluded from the default
document-evidence corpus.

Important boundary: these 55,059 URLs are not 55,059 parseable contract
documents. They are external links attached to tender and award document arrays.
The extraction scope is reproducible, but the semantic label is broad:

- `tenderNotice`, `awardNotice`, and `tenderCancellationNotice` are notice
  pages or notice attachments, not necessarily contract files.
- `biddingDocuments` and `technicalSpecifications` are the better first-pass
  document candidates. In the current extract they account for 15,946 rows and
  12,330 unique raw URLs.
- Declared MIME formats within those document candidates identify about 3,191
  unique PDF/Word/spreadsheet URLs before any live availability validation.

A 500-URL validation run over the broad 55,059-URL scope showed that only a
small minority were direct files:

- direct PDF: 3/500
- direct Word: 13/500
- portal landing pages: 157/500
- portal login pages: 56/500
- notice pages: 73/500
- other HTML pages: 164/500
- unreachable: 34/500

The direct-file share in that run was therefore about 3.2%. This should be
treated as a validation finding, not a final population estimate, because the
first run used the manifest order rather than the corrected stratified sample
validation path. However, it is strong enough to set the processing boundary:
non-empty URL, HTTP 200, and `text/html` do not imply that contract text is
available.

Default access classes:

- `direct_pdf`, `direct_word`, `direct_spreadsheet`: parseable document
  candidates.
- `notice_page`: official notice or attachment page; parse only if the page
  exposes useful public text or direct attachments.
- `portal_landing`, `portal_login`: provenance only. Do not spend effort on a
  generic HTML crawler for these, because the limiting factor is portal access
  or account authentication rather than parsing.
- `html_unknown`: probe lightly with title/body checks before deciding whether
  it is a real document page, notice page, or portal page.
- `unreachable`: unavailable for document evidence.

## Design Decision

Do not build an offline document parsing corpus for this project phase. Document
text is used only as opportunistic evidence during the evidence-verdict step.

Reasons:

- The broad tender/award URL scope is mostly notice pages, portal pages, login
  pages, or unavailable links rather than direct contract files.
- Many procurement portals require account access. A crawler or richer HTML
  parser cannot solve an authentication boundary.
- The KG already carries the primary structured evidence. Documents should only
  support, contradict, or add a clause-level citation for a contract that has
  already been located.
- Persisting a document corpus would add a large maintenance surface for a low
  and uneven coverage source.

Therefore there is no full-document crawl, no bulk OCR, no global document
retrieval index, and no persisted chunk parquet by default.

## Trigger Model

Document evidence is not a parallel retrieval system. It is called only after a
KG path has already identified a specific `ocid`.

1. KG retrieval identifies the contract/notice and structured evidence.
2. Evidence verdict decides whether the structured evidence needs support,
   contradiction checks, or clause-level citation.
3. Check whether the contract node for that `ocid` has one or more
   `document_url` values.
4. If no URL exists, skip document evidence and use KG-only evidence.
5. If a URL exists, make a live lightweight request and inspect status,
   redirects, and `Content-Type`.
6. If the response is usable, extract temporary text and select the most
   relevant snippet for the verdict.
7. If the request fails, times out, redirects to login, or returns no useful
   text, skip it and fall back to KG-only evidence.

If no relevant document chunk is found, the verdict should report insufficient
document evidence rather than searching unrelated contracts.

## Live Extraction

Preferred order:

1. For HTML or plain text, strip scripts/styles/navigation and extract visible
   page text with a simple parser.
2. For PDF, use embedded text extraction with a local library such as PyMuPDF.
3. For Word documents, extract text if a local lightweight dependency is
   available.
4. For unsupported files, login pages, 403/404, timeout, or empty text, return
   no document evidence.

No OCR is run in the default verdict path. Scanned PDFs are skipped unless a
later experiment explicitly enables OCR for a small targeted subset.

The extractor returns temporary in-memory text. It does not write document
downloads, extracted text, or chunks to disk.

Within one pipeline run, the verdict layer may keep a small in-memory cache keyed
by `ocid` or normalized URL. This avoids repeated live requests and repeated
temporary chunking when multiple benchmark questions hit the same contract. The
cache is process-local only and is discarded at the end of the run.

## URL Validation Sampling

Do not infer availability from non-empty URLs. Validate a stratified sample
before scaling up.

Recommended strata:

- URL scope: strict `tender+award` versus `tender+award+tender_submission`
- publication year
- source group
- major buyer or buyer family

Validation reports must be read by year as well as overall. Older notices are
more likely to have dead links after portal migrations or buyer website
redesigns.

Record at least:

- HTTP status code
- final URL after redirects
- whether a redirect occurred
- `Content-Type`
- coarse content type group: PDF, HTML, Word, spreadsheet, binary, unknown
- timeout and error type

Treat `200 text/html` carefully. It may be a real notice page, but it may also
be a "document moved" or "not found in new portal" page that returns HTTP 200.

## Temporary Chunk Schema

If extracted text is short, use it as one snippet. If it is long, split it
in-memory by headings and paragraphs, using the same field names as
`procurement_graph.documents.chunking.DocumentChunk` where useful:

- `chunk_id`
- `url_hash`
- `ocid`
- `source`
- `document_type`
- `url`
- `page_number`
- `heading`
- `char_start`
- `char_end`
- `token_count`
- `text`

Chunk size target: about 650 words with about 80 words overlap. The temporary
selector should prefer chunks containing claim terms such as named parties,
amounts, dates, document type labels, award criteria, specifications, payment,
termination, insurance, TUPE, KPIs, social value, and delivery terms.

These chunks are discarded after the verdict call.

## Verdict Integration

`procurement_graph.qa.evidence_verdict` provides the first passive selection
interface. It takes:

- question or claim
- KG evidence
- known `ocid`
- optional required facets

It returns top temporary document snippets only from that `ocid`. This is
intentionally a contract-local selector, not a global document search.

The verdict prompt should treat document snippets as secondary evidence. If the
snippet is missing, stale, inaccessible, or unrelated, the answer should rely on
KG evidence and state that no usable document text was available.
