"""QA benchmark construction framework.

This package contains schema-light benchmark generation primitives. Real KG
loading is intentionally left behind a query backend interface so the framework
can be tested before final entity-resolution outputs are written back.
"""

from .executor import AnswerExecutionError, execute_answer_spec
from .kg_interface import ParquetKGQueryBackend, QueryBackend, TabularQueryBackend
from .models import (
    AnswerSpec,
    BenchmarkExample,
    Constraint,
    GateReport,
    SemanticCheckResult,
)
from .pipeline import BenchmarkPipeline
from .samplers import SamplerConfig, sample_answer_specs
from .stage1 import build_stage1
from .stage2 import build_stage2, load_accepted_specs

__all__ = [
    "AnswerExecutionError",
    "AnswerSpec",
    "BenchmarkExample",
    "BenchmarkPipeline",
    "Constraint",
    "GateReport",
    "ParquetKGQueryBackend",
    "QueryBackend",
    "SemanticCheckResult",
    "SamplerConfig",
    "TabularQueryBackend",
    "build_stage1",
    "build_stage2",
    "execute_answer_spec",
    "load_accepted_specs",
    "sample_answer_specs",
]
