"""Build and validate contract document URL manifests.

The default scope is intentionally narrow: tender and award document URLs only.
Party profiles, buyer profiles, contact emails, and submission URLs are useful
metadata, but they are not contract-describing document links.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DOCUMENTS_PATH = ROOT / "data" / "extracted" / "documents.parquet"
DEFAULT_RELEASES_PATH = ROOT / "data" / "interim" / "releases.parquet"
DEFAULT_MANIFEST_PATH = ROOT / "data" / "documents" / "contract_document_url_manifest.parquet"
DEFAULT_SAMPLE_PATH = ROOT / "data" / "documents" / "contract_document_url_sample.parquet"
DEFAULT_RESULTS_PATH = ROOT / "data" / "documents" / "contract_document_url_validation.jsonl"
DEFAULT_REPORT_PATH = ROOT / "reports" / "documents" / "contract_document_url_validation_summary.csv"
DEFAULT_GROUP_REPORT_PATH = ROOT / "reports" / "documents" / "contract_document_url_validation_groups.csv"

DEFAULT_SOURCES = ("tender", "award")
USER_AGENT = "procurement-graph-url-validator/0.1"


@dataclass(frozen=True)
class ValidationConfig:
    documents_path: Path = DEFAULT_DOCUMENTS_PATH
    releases_path: Path = DEFAULT_RELEASES_PATH
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    sample_path: Path = DEFAULT_SAMPLE_PATH
    results_path: Path = DEFAULT_RESULTS_PATH
    report_path: Path = DEFAULT_REPORT_PATH
    group_report_path: Path = DEFAULT_GROUP_REPORT_PATH
    sources: tuple[str, ...] = DEFAULT_SOURCES
    sample_size: int | None = None
    sample_only: bool = False
    validate_sample: bool = False
    max_urls: int | None = None
    workers: int = 8
    timeout_seconds: float = 15.0
    resume: bool = True
    manifest_only: bool = False


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]


def normalise_request_url(url: str) -> str:
    cleaned = str(url or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("//"):
        return f"https:{cleaned}"
    if "://" not in cleaned:
        return f"https://{cleaned}"
    return cleaned


def _json_unique(values: pd.Series, limit: int | None = None) -> str:
    unique = sorted({str(value) for value in values if str(value)})
    if limit:
        unique = unique[:limit]
    return json.dumps(unique, ensure_ascii=False)


def _first_non_empty(values: pd.Series) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_done_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = record.get("url_hash")
            if value:
                done.add(str(value))
    return done


def build_manifest(
    documents_path: Path,
    sources: tuple[str, ...] = DEFAULT_SOURCES,
    releases_path: Path = DEFAULT_RELEASES_PATH,
) -> pd.DataFrame:
    docs = pd.read_parquet(documents_path)
    required = {"ocid", "source", "document_type", "url", "title", "description", "format", "date_published"}
    missing = required - set(docs.columns)
    if missing:
        raise ValueError(f"{documents_path} is missing columns: {sorted(missing)}")

    scoped = docs[docs["source"].isin(sources)].copy()
    scoped["url_clean"] = scoped["url"].fillna("").astype(str).str.strip()
    scoped = scoped[scoped["url_clean"].ne("")]
    scoped["request_url"] = scoped["url_clean"].map(normalise_request_url)
    scoped["url_hash"] = scoped["url_clean"].map(url_hash)
    scoped["source_group"] = scoped["source"].astype(str)

    if releases_path.exists():
        releases = pd.read_parquet(releases_path, columns=["ocid", "year", "buyer_name", "buyer_raw_id"])
        scoped = scoped.merge(releases, on="ocid", how="left")
    else:
        scoped["year"] = ""
        scoped["buyer_name"] = ""
        scoped["buyer_raw_id"] = ""

    grouped = (
        scoped.groupby("url_hash", as_index=False)
        .agg(
            url=("url_clean", "first"),
            request_url=("request_url", "first"),
            source_group=("source_group", lambda s: "+".join(sorted(set(map(str, s))))),
            sources=("source", lambda s: json.dumps(sorted(set(map(str, s))), ensure_ascii=False)),
            document_types=("document_type", lambda s: json.dumps(sorted(set(map(str, s))), ensure_ascii=False)),
            formats=("format", lambda s: json.dumps(sorted({str(v) for v in s if str(v)}), ensure_ascii=False)),
            first_year=("year", _first_non_empty),
            years=("year", lambda s: _json_unique(s)),
            first_buyer_name=("buyer_name", _first_non_empty),
            buyer_names=("buyer_name", lambda s: _json_unique(s, limit=10)),
            buyer_raw_ids=("buyer_raw_id", lambda s: _json_unique(s, limit=10)),
            ocid_count=("ocid", "nunique"),
            row_count=("ocid", "size"),
            sample_ocids=("ocid", lambda s: json.dumps(sorted(set(map(str, s)))[:10], ensure_ascii=False)),
            sample_titles=("title", lambda s: json.dumps([str(v) for v in s.dropna().astype(str).head(5)], ensure_ascii=False)),
            sample_descriptions=(
                "description",
                lambda s: json.dumps([str(v) for v in s.dropna().astype(str).head(3)], ensure_ascii=False),
            ),
            first_date_published=("date_published", lambda s: sorted({str(v) for v in s if str(v)})[0] if any(str(v) for v in s) else ""),
        )
        .sort_values(["ocid_count", "row_count", "url_hash"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return grouped


def stratified_sample_manifest(
    manifest: pd.DataFrame,
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    if sample_size <= 0 or len(manifest) <= sample_size:
        return manifest.copy()

    sample = manifest.copy()
    sample["year_stratum"] = sample["first_year"].fillna("").astype(str).replace("", "unknown")
    buyer_counts = sample["first_buyer_name"].fillna("").astype(str).value_counts()
    top_buyers = set(buyer_counts.head(25).index)
    sample["buyer_stratum"] = sample["first_buyer_name"].fillna("").astype(str).where(
        sample["first_buyer_name"].fillna("").astype(str).isin(top_buyers),
        "other",
    )
    strata = ["source_group", "year_stratum", "buyer_stratum"]
    groups = list(sample.groupby(strata, dropna=False))
    per_group = max(1, int(sample_size / max(1, len(groups))))

    parts = []
    remaining = sample_size
    for _, group in groups:
        if remaining <= 0:
            break
        take = min(len(group), per_group, remaining)
        parts.append(group.sample(n=take, random_state=random_state))
        remaining -= take

    if remaining > 0:
        selected = pd.concat(parts) if parts else sample.iloc[0:0]
        rest = sample.drop(index=selected.index)
        if len(rest):
            parts.append(rest.sample(n=min(remaining, len(rest)), random_state=random_state + 1))

    result = pd.concat(parts).drop_duplicates("url_hash").head(sample_size)
    return result.sort_values(["source_group", "first_year", "first_buyer_name", "url_hash"]).reset_index(drop=True)


def _request(url: str, method: str, timeout_seconds: float) -> urllib.response.addinfourl:
    headers = {"User-Agent": USER_AGENT}
    if method == "GET":
        headers["Range"] = "bytes=0-0"
    req = urllib.request.Request(url, method=method, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout_seconds)


def content_type_group(content_type: Any, final_url: Any = "") -> str:
    value = str(content_type or "").lower()
    url = str(final_url or "").lower()
    if "pdf" in value or url.endswith(".pdf"):
        return "pdf"
    if "html" in value or "xhtml" in value:
        return "html"
    if any(token in value for token in ["word", "officedocument", "msword"]) or url.endswith((".doc", ".docx")):
        return "word"
    if any(token in value for token in ["excel", "spreadsheet"]) or url.endswith((".xls", ".xlsx", ".csv")):
        return "spreadsheet"
    if "json" in value:
        return "json"
    if "xml" in value:
        return "xml"
    if "octet-stream" in value:
        return "binary_unknown"
    if value:
        return "other"
    return "unknown"


def redirected(original_url: Any, final_url: Any) -> bool:
    left = normalise_request_url(str(original_url or "")).rstrip("/")
    right = str(final_url or "").strip().rstrip("/")
    return bool(left and right and left != right)


def document_access_class(record: dict[str, Any]) -> str:
    """Classify whether a reachable URL looks directly parseable."""
    type_group = content_type_group(record.get("content_type", ""), record.get("final_url", ""))
    final_url = str(record.get("final_url") or record.get("request_url") or record.get("url") or "").lower()
    doc_types = str(record.get("document_types") or "").lower()
    if type_group in {"pdf", "word", "spreadsheet", "json", "xml"}:
        return f"direct_{type_group}"
    if any(token in final_url for token in ["/login", "account/login", "web/login", "signin", "sign-in"]):
        return "portal_login"
    if any(token in final_url for token in ["/home", "opportunities", "supplier-registration", "welcome"]):
        return "portal_landing"
    if any(token in final_url for token in ["find-tender.service.gov.uk/notice", "contractsfinder.service.gov.uk/notice"]):
        return "notice_page"
    if "notice" in doc_types and type_group == "html":
        return "notice_page"
    if type_group == "html":
        return "html_unknown"
    if not bool(record.get("reachable", False)):
        return "unreachable"
    return "unknown"


def validate_url(record: dict[str, Any], timeout_seconds: float = 15.0) -> dict[str, Any]:
    started_at = time.time()
    url = str(record["url"])
    request_url = str(record.get("request_url") or normalise_request_url(url))
    methods_tried: list[str] = []

    base = {
        "url_hash": record["url_hash"],
        "url": url,
        "request_url": request_url,
        "source_group": str(record.get("source_group", "")),
        "first_year": str(record.get("first_year", "")),
        "first_buyer_name": str(record.get("first_buyer_name", "")),
        "document_types": record.get("document_types", "[]"),
        "formats": record.get("formats", "[]"),
        "ocid_count": int(record.get("ocid_count", 0) or 0),
        "row_count": int(record.get("row_count", 0) or 0),
    }

    for method in ("HEAD", "GET"):
        methods_tried.append(method)
        try:
            with _request(request_url, method, timeout_seconds) as response:
                status = int(getattr(response, "status", response.getcode()))
                headers = response.headers
                finished_at = time.time()
                return base | {
                    "ok": 200 <= status < 400,
                    "reachable": True,
                    "status_code": status,
                    "final_url": response.geturl(),
                    "content_type": headers.get("Content-Type", ""),
                    "content_length": headers.get("Content-Length", ""),
                    "methods_tried": methods_tried,
                    "error_type": "",
                    "error_message": "",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_seconds": round(finished_at - started_at, 4),
                }
        except urllib.error.HTTPError as exc:
            if method == "HEAD" and exc.code in {405, 403, 501}:
                continue
            finished_at = time.time()
            return base | {
                "ok": 200 <= int(exc.code) < 400,
                "reachable": True,
                "status_code": int(exc.code),
                "final_url": exc.url,
                "content_type": exc.headers.get("Content-Type", "") if exc.headers else "",
                "content_length": exc.headers.get("Content-Length", "") if exc.headers else "",
                "methods_tried": methods_tried,
                "error_type": "HTTPError",
                "error_message": str(exc),
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_seconds": round(finished_at - started_at, 4),
            }
        except (urllib.error.URLError, TimeoutError, socket.timeout, ssl.SSLError, OSError) as exc:
            if method == "HEAD":
                continue
            finished_at = time.time()
            return base | {
                "ok": False,
                "reachable": False,
                "status_code": 0,
                "final_url": "",
                "content_type": "",
                "content_length": "",
                "methods_tried": methods_tried,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_seconds": round(finished_at - started_at, 4),
            }

    finished_at = time.time()
    return base | {
        "ok": False,
        "reachable": False,
        "status_code": 0,
        "final_url": "",
        "content_type": "",
        "content_length": "",
        "methods_tried": methods_tried,
        "error_type": "UnknownError",
        "error_message": "No request result",
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(finished_at - started_at, 4),
    }


def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["grouping", "group", "validated_urls", "ok_urls", "reachable_urls"])

    enriched = df.copy()
    enriched["content_type_group"] = enriched.apply(
        lambda row: content_type_group(row.get("content_type", ""), row.get("final_url", "")),
        axis=1,
    )
    enriched["redirected"] = enriched.apply(
        lambda row: redirected(row.get("request_url", row.get("url", "")), row.get("final_url", "")),
        axis=1,
    )
    enriched["document_access_class"] = enriched.apply(lambda row: document_access_class(row.to_dict()), axis=1)
    enriched["first_year"] = enriched.get("first_year", "").fillna("").astype(str).replace("", "unknown")
    enriched["source_group"] = enriched.get("source_group", "").fillna("").astype(str).replace("", "unknown")

    rows: list[dict[str, Any]] = []

    def add_group(grouping: str, cols: list[str]) -> None:
        for key, group in enriched.groupby(cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            label = "|".join(str(part) for part in key)
            rows.append({
                "grouping": grouping,
                "group": label,
                "validated_urls": len(group),
                "ok_urls": int(group["ok"].astype(bool).sum()),
                "reachable_urls": int(group["reachable"].astype(bool).sum()),
                "redirected_urls": int(group["redirected"].astype(bool).sum()),
                "pdf_urls": int(group["content_type_group"].eq("pdf").sum()),
                "html_urls": int(group["content_type_group"].eq("html").sum()),
                "unknown_type_urls": int(group["content_type_group"].eq("unknown").sum()),
                "ok_rate": round(float(group["ok"].astype(bool).mean()), 6),
                "reachable_rate": round(float(group["reachable"].astype(bool).mean()), 6),
            })

    add_group("year", ["first_year"])
    add_group("source", ["source_group"])
    add_group("status", ["status_code"])
    add_group("content_type_group", ["content_type_group"])
    add_group("document_access_class", ["document_access_class"])
    add_group("year_source", ["first_year", "source_group"])
    add_group("year_content_type", ["first_year", "content_type_group"])
    add_group("year_access_class", ["first_year", "document_access_class"])
    return pd.DataFrame(rows).sort_values(["grouping", "group"]).reset_index(drop=True)


def write_summary(
    results_path: Path,
    report_path: Path,
    group_report_path: Path = DEFAULT_GROUP_REPORT_PATH,
) -> pd.DataFrame:
    if not results_path.exists():
        summary = pd.DataFrame([{"metric": "validated_urls", "value": 0}])
        group_summary = pd.DataFrame()
    else:
        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(rows)
        if df.empty:
            summary = pd.DataFrame([{"metric": "validated_urls", "value": 0}])
            group_summary = pd.DataFrame()
        else:
            df["content_type_group"] = df.apply(
                lambda row: content_type_group(row.get("content_type", ""), row.get("final_url", "")),
                axis=1,
            )
            df["redirected"] = df.apply(
                lambda row: redirected(row.get("request_url", row.get("url", "")), row.get("final_url", "")),
                axis=1,
            )
            df["document_access_class"] = df.apply(lambda row: document_access_class(row.to_dict()), axis=1)
            summary_rows: list[dict[str, Any]] = [
                {"metric": "validated_urls", "value": len(df)},
                {"metric": "ok_urls", "value": int(df["ok"].astype(bool).sum())},
                {"metric": "reachable_urls", "value": int(df["reachable"].astype(bool).sum())},
                {"metric": "redirected_urls", "value": int(df["redirected"].astype(bool).sum())},
                {"metric": "pdf_urls", "value": int(df["content_type_group"].eq("pdf").sum())},
                {"metric": "html_urls", "value": int(df["content_type_group"].eq("html").sum())},
                {"metric": "distinct_final_urls", "value": int(df["final_url"].fillna("").replace("", pd.NA).nunique())},
            ]
            for status, count in df["status_code"].value_counts(dropna=False).head(20).items():
                summary_rows.append({"metric": f"status::{status}", "value": int(count)})
            for error_type, count in df["error_type"].replace("", "none").value_counts().head(20).items():
                summary_rows.append({"metric": f"error::{error_type}", "value": int(count)})
            for type_group, count in df["content_type_group"].value_counts().head(20).items():
                summary_rows.append({"metric": f"content_type_group::{type_group}", "value": int(count)})
            for access_class, count in df["document_access_class"].value_counts().head(30).items():
                summary_rows.append({"metric": f"document_access_class::{access_class}", "value": int(count)})
            summary = pd.DataFrame(summary_rows)
            group_summary = build_group_summary(df)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report_path, index=False)
    group_report_path.parent.mkdir(parents=True, exist_ok=True)
    group_summary.to_csv(group_report_path, index=False)
    return summary


def run(config: ValidationConfig) -> dict[str, int]:
    manifest = build_manifest(config.documents_path, sources=config.sources, releases_path=config.releases_path)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(config.manifest_path, index=False)

    stats = {
        "manifest_urls": len(manifest),
        "sample_urls": 0,
        "selected_this_run": 0,
        "validated": 0,
        "skipped_existing": 0,
    }
    if config.sample_size is not None and config.sample_size > 0:
        sample = stratified_sample_manifest(manifest, sample_size=config.sample_size)
        config.sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample.to_parquet(config.sample_path, index=False)
        stats["sample_urls"] = len(sample)
        if config.sample_only:
            write_summary(config.results_path, config.report_path, config.group_report_path)
            return stats
        manifest = sample
    elif config.validate_sample:
        if not config.sample_path.exists():
            raise FileNotFoundError(f"Sample manifest not found: {config.sample_path}")
        manifest = pd.read_parquet(config.sample_path)
        stats["sample_urls"] = len(manifest)

    if config.manifest_only:
        write_summary(config.results_path, config.report_path, config.group_report_path)
        return stats

    done = read_done_hashes(config.results_path) if config.resume else set()
    pending = manifest[~manifest["url_hash"].isin(done)].copy()
    stats["skipped_existing"] = len(manifest) - len(pending)
    if config.max_urls is not None and config.max_urls > 0:
        pending = pending.head(config.max_urls)
    stats["selected_this_run"] = len(pending)

    records = pending.to_dict(orient="records")
    workers = max(1, int(config.workers))
    if workers == 1:
        for record in records:
            append_jsonl(config.results_path, validate_url(record, config.timeout_seconds))
            stats["validated"] += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(validate_url, record, config.timeout_seconds) for record in records]
            for future in as_completed(futures):
                append_jsonl(config.results_path, future.result())
                stats["validated"] += 1

    write_summary(config.results_path, config.report_path, config.group_report_path)
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate contract document URLs")
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS_PATH)
    parser.add_argument("--releases", type=Path, default=DEFAULT_RELEASES_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--group-report", type=Path, default=DEFAULT_GROUP_REPORT_PATH)
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--validate-sample", action="store_true", help="Validate rows from --sample instead of the full manifest")
    parser.add_argument("--max-urls", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args(argv)


def cli_main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stats = run(
        ValidationConfig(
            documents_path=args.documents,
            releases_path=args.releases,
            manifest_path=args.manifest,
            sample_path=args.sample,
            results_path=args.results,
            report_path=args.report,
            group_report_path=args.group_report,
            sources=tuple(args.sources),
            sample_size=args.sample_size,
            sample_only=args.sample_only,
            validate_sample=args.validate_sample,
            max_urls=args.max_urls,
            workers=args.workers,
            timeout_seconds=args.timeout_seconds,
            resume=not args.no_resume,
            manifest_only=args.manifest_only,
        )
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Manifest: {args.manifest}")
    print(f"Sample: {args.sample}")
    print(f"Results: {args.results}")
    print(f"Report: {args.report}")
    print(f"Group report: {args.group_report}")


__all__ = [
    "ValidationConfig",
    "build_manifest",
    "cli_main",
    "run",
    "validate_url",
    "write_summary",
]
