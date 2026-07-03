"""Reflector memory (adapted from the prior pipeline's step5 append_memory).

Append-only JSONL sink of deterministic reflector diagnoses, so recurring failure modes can be
analysed and — later — fed back into planning (e.g. down-weight a template that repeatedly needs
repair). Reasoning itself stays stateless; this is a side-channel the orchestrator can opt into.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import ReasoningTrace

# Diagnoses that indicate the loop resolved itself and need not be remembered.
_BENIGN = frozenset({"no_repair_needed"})


@dataclass
class ReflectorMemory:
    path: Path

    def record(self, trace: ReasoningTrace, *, context: dict[str, Any] | None = None) -> bool:
        """Append a diagnosis record if the attempt needed (or failed) repair. Returns whether written."""
        reflection = trace.reflection
        if reflection is None or reflection.action in _BENIGN:
            return False
        record = {
            "schema_version": "reasoning_reflector_memory_v1",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "trace_id": trace.trace_id,
            "question": trace.question,
            "diagnosis": reflection.action,
            "reason": reflection.reason,
            "selected_plan_id": trace.selected_plan_id,
            "answered": bool(trace.answer_card and trace.answer_card.answer is not None),
            "attempts": (trace.metadata or {}).get("attempts", []),
            "repairs": (trace.metadata or {}).get("repairs", []),
            "context": context or {},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def diagnosis_counts(self) -> dict[str, int]:
        return dict(Counter(row.get("diagnosis", "") for row in self.load()))


__all__ = ["ReflectorMemory"]
