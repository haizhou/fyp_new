"""Run JSONL review queues through an LLM provider.

The queue format is intentionally simple: each row needs a stable `task_id` and
either `messages` or `input`. Decisions are written as JSONL so downstream
safety gates can remain deterministic and provider-independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from procurement_graph.llm.azure_responses import (
    AzureResponsesConfig,
    DEFAULT_SCOPE,
    call_responses_api_audited,
    create_client,
    extract_json_object,
    response_output_text,
    response_to_dict,
)

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_DATA_DIR = ROOT / "data" / "ablation" / "reference"
DEFAULT_QUEUE_PATH = REFERENCE_DATA_DIR / "llm_review_queue.jsonl"
DEFAULT_DECISIONS_PATH = REFERENCE_DATA_DIR / "llm_decisions.jsonl"
DEFAULT_RAW_PATH = REFERENCE_DATA_DIR / "llm_raw_responses.jsonl"
DEFAULT_ERRORS_PATH = REFERENCE_DATA_DIR / "llm_errors.jsonl"


@dataclass(frozen=True)
class QueueRunConfig:
    queue_path: Path = DEFAULT_QUEUE_PATH
    decisions_path: Path = DEFAULT_DECISIONS_PATH
    raw_path: Path = DEFAULT_RAW_PATH
    errors_path: Path = DEFAULT_ERRORS_PATH
    max_tasks: int | None = None
    resume: bool = True
    dry_run: bool = False
    sleep_seconds: float = 0.0
    workers: int = 1
    write_error_decisions: bool = False
    only_task_ids: frozenset[str] | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "task_id" not in record:
                raise ValueError(f"{path} line {line_no} is missing task_id")
            rows.append(record)
    return rows


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def existing_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = record.get("task_id")
            if task_id:
                ids.add(str(task_id))
    return ids


def error_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = record.get("task_id")
            if task_id:
                ids.add(str(task_id))
    return ids


def schema_hash(record: dict[str, Any]) -> str:
    response_format = record.get("response_format")
    if not isinstance(response_format, dict):
        return ""
    raw = json.dumps(response_format, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def queue_record_input(record: dict[str, Any]) -> Any:
    if "messages" in record and record["messages"]:
        return record["messages"]
    if "input" in record:
        return record["input"]
    if "prompt" in record:
        return record["prompt"]
    raise ValueError(f"Queue task {record.get('task_id', '<missing>')} has no messages/input/prompt")


def queue_record_text_format(record: dict[str, Any]) -> dict[str, Any] | None:
    response_format = record.get("response_format")
    return response_format if isinstance(response_format, dict) else None


def conservative_decision(record: dict[str, Any], reason: str, risk_flag: str) -> dict[str, Any]:
    template = record.get("failure_decision_template")
    if isinstance(template, dict):
        decision = dict(template)
        if record.get("prompt_version"):
            decision.setdefault("prompt_version", record["prompt_version"])
        if record.get("schema_version"):
            decision.setdefault("schema_version", record["schema_version"])
        hash_value = schema_hash(record)
        if hash_value:
            decision.setdefault("schema_hash", hash_value)
        decision["reason"] = reason
        flags = decision.get("risk_flags") or []
        if risk_flag not in flags:
            decision["risk_flags"] = [*flags, risk_flag]
        return decision

    reference = record.get("evidence", {}).get("reference", {})
    return {
        "prompt_version": record.get("prompt_version", ""),
        "schema_version": record.get("schema_version", ""),
        "schema_hash": schema_hash(record),
        "task_id": record.get("task_id", ""),
        "decision": "uncertain",
        "confidence": 0.0,
        "approved_entity_ids": [],
        "excluded_entity_ids": [],
        "canonical_reference_id": reference.get("canonical_id", ""),
        "canonical_name": reference.get("matched_name", ""),
        "risk_flags": [risk_flag],
        "reason": reason,
    }


def process_live_task(client: Any, azure_config: AzureResponsesConfig, record: dict[str, Any]) -> dict[str, Any]:
    task_id = str(record["task_id"])
    started_at = time.time()
    response, call_audit = call_responses_api_audited(
        client,
        azure_config,
        queue_record_input(record),
        text_format=queue_record_text_format(record),
    )
    finished_at = time.time()
    text = response_output_text(response)
    decision = extract_json_object(text)
    decision.setdefault("task_id", task_id)
    if record.get("prompt_version"):
        decision.setdefault("prompt_version", record["prompt_version"])
    if record.get("schema_version"):
        decision.setdefault("schema_version", record["schema_version"])
    hash_value = schema_hash(record)
    if hash_value:
        decision.setdefault("schema_hash", hash_value)
    return {
        "task_id": task_id,
        "decision": decision,
        "raw": {
            "task_id": task_id,
            "prompt_version": record.get("prompt_version", ""),
            "schema_version": record.get("schema_version", ""),
            "schema_hash": hash_value,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": round(finished_at - started_at, 4),
            "attempts": call_audit.get("attempts", 1),
            "retry_errors": call_audit.get("retry_errors", []),
            "output_text": text,
            "response": response_to_dict(response),
        },
    }


def run_queue(queue_config: QueueRunConfig, azure_config: AzureResponsesConfig | None = None) -> dict[str, int]:
    queue = read_jsonl(queue_config.queue_path)
    done = existing_task_ids(queue_config.decisions_path) if queue_config.resume else set()
    tasks = [record for record in queue if str(record["task_id"]) not in done]
    if queue_config.only_task_ids is not None:
        tasks = [record for record in tasks if str(record["task_id"]) in queue_config.only_task_ids]
    pending_before_limit = len(tasks)
    if queue_config.max_tasks is not None:
        tasks = tasks[: queue_config.max_tasks]

    stats = {
        "queued": len(queue),
        "skipped_existing": len(done),
        "pending_before_limit": pending_before_limit,
        "selected_this_run": len(tasks),
        "attempted": 0,
        "decisions": 0,
        "errors": 0,
    }

    if queue_config.dry_run:
        for record in tasks:
            decision = conservative_decision(record, "Dry-run placeholder; no live API call made.", "dry_run")
            append_jsonl(queue_config.decisions_path, decision)
            stats["decisions"] += 1
        return stats

    if azure_config is None:
        azure_config = AzureResponsesConfig.from_env()
    client = create_client(azure_config)

    workers = max(1, int(queue_config.workers or 1))

    def handle_success(result: dict[str, Any]) -> None:
        append_jsonl(queue_config.decisions_path, result["decision"])
        append_jsonl(queue_config.raw_path, result["raw"])
        stats["decisions"] += 1

    def handle_error(record: dict[str, Any], exc: Exception) -> None:
        task_id = str(record["task_id"])
        error = {"task_id": task_id, "error": type(exc).__name__, "message": str(exc)}
        append_jsonl(queue_config.errors_path, error)
        if queue_config.write_error_decisions:
            append_jsonl(
                queue_config.decisions_path,
                conservative_decision(record, f"Model call failed or returned invalid JSON: {exc}", "model_call_error"),
            )
            stats["decisions"] += 1
        stats["errors"] += 1

    if workers > 1 and queue_config.sleep_seconds:
        raise ValueError("--sleep-seconds is only supported with --workers 1")

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for record in tasks:
                stats["attempted"] += 1
                futures[executor.submit(process_live_task, client, azure_config, record)] = record
            for future in as_completed(futures):
                record = futures[future]
                try:
                    handle_success(future.result())
                except Exception as exc:  # pragma: no cover - live call/audit path
                    handle_error(record, exc)
        return stats

    for record in tasks:
        task_id = str(record["task_id"])
        stats["attempted"] += 1
        try:
            handle_success(process_live_task(client, azure_config, record))
        except Exception as exc:  # pragma: no cover - live call/audit path
            handle_error(record, exc)
        if queue_config.sleep_seconds:
            time.sleep(queue_config.sleep_seconds)

    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a JSONL LLM review queue through Azure OpenAI Responses")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS_PATH)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS_PATH)
    parser.add_argument("--endpoint", default=None, help="Azure OpenAI endpoint; defaults to AZURE_OPENAI_ENDPOINT")
    parser.add_argument("--deployment", default=None, help="Azure deployment name; defaults to AZURE_OPENAI_DEPLOYMENT")
    parser.add_argument("--scope", default=None, help="Azure auth scope; defaults to AZURE_OPENAI_SCOPE or ai.azure.com")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent API calls; default keeps historical serial behavior")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip task_ids already present in decisions")
    parser.add_argument("--dry-run", action="store_true", help="Write conservative placeholder decisions without API calls")
    parser.add_argument(
        "--only-errors",
        action="store_true",
        help="Retry only task_ids present in the errors JSONL and not already present in decisions",
    )
    parser.add_argument(
        "--write-error-decisions",
        action="store_true",
        help="Also write conservative uncertain decisions for failed API calls; by default failed tasks remain retryable",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-wait-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=None, help="Per-request SDK timeout; defaults to env/config")
    return parser.parse_args(argv)


def cli_main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    only_task_ids = frozenset(error_task_ids(args.errors)) if args.only_errors else None
    queue_config = QueueRunConfig(
        queue_path=args.queue,
        decisions_path=args.decisions,
        raw_path=args.raw,
        errors_path=args.errors,
        max_tasks=args.max_tasks,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        sleep_seconds=args.sleep_seconds,
        workers=args.workers,
        write_error_decisions=args.write_error_decisions,
        only_task_ids=only_task_ids,
    )

    azure_config = None
    if not args.dry_run:
        if args.endpoint and args.deployment:
            azure_config = AzureResponsesConfig(
                endpoint=args.endpoint,
                deployment_name=args.deployment,
                scope=args.scope or DEFAULT_SCOPE,
                max_retries=args.max_retries,
                retry_wait_seconds=args.retry_wait_seconds,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            env_config = AzureResponsesConfig.from_env()
            azure_config = AzureResponsesConfig(
                endpoint=args.endpoint or env_config.endpoint,
                deployment_name=args.deployment or env_config.deployment_name,
                scope=args.scope or env_config.scope,
                max_retries=args.max_retries,
                retry_wait_seconds=args.retry_wait_seconds,
                timeout_seconds=args.timeout_seconds if args.timeout_seconds is not None else env_config.timeout_seconds,
            )

    stats = run_queue(queue_config, azure_config=azure_config)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Decisions: {queue_config.decisions_path}")
    print(f"Raw audit: {queue_config.raw_path}")
    print(f"Errors: {queue_config.errors_path}")


__all__ = [
    "QueueRunConfig",
    "cli_main",
    "conservative_decision",
    "error_task_ids",
    "run_queue",
    "schema_hash",
]
