"""Regression tests for safe output handling in partial pipeline runs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
PIPELINES_DIR = PROJECT_ROOT / "pipelines"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from procurement_graph.common.pipeline_paths import resolve_output_target


def load_pipeline(filename: str):
    """Load a numbered pipeline script as an ordinary module."""
    module_name = f"_pipeline_test_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(module_name, PIPELINES_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_run_keeps_canonical_output(tmp_path):
    canonical = tmp_path / "canonical" / "result.parquet"
    assert resolve_output_target(
        canonical,
        None,
        project_root=tmp_path,
        partial=False,
        option_name="--output-path",
    ) == canonical


def test_relative_override_is_rooted_at_project(tmp_path):
    canonical = tmp_path / "canonical" / "result.parquet"
    target = resolve_output_target(
        canonical,
        Path("smoke/result.parquet"),
        project_root=tmp_path,
        partial=True,
        option_name="--output-path",
    )
    assert target == tmp_path / "smoke" / "result.parquet"


def test_partial_run_rejects_explicit_canonical_target(tmp_path):
    canonical = tmp_path / "canonical" / "result.parquet"
    with pytest.raises(ValueError, match="cannot overwrite the canonical output"):
        resolve_output_target(
            canonical,
            canonical,
            project_root=tmp_path,
            partial=True,
            option_name="--output-path",
        )


@pytest.mark.parametrize(
    ("filename", "kwargs", "io_name"),
    [
        ("01_ingest.py", {"years": [2025]}, "flatten_all_years"),
        ("02_er_phase1.py", {"limit": 1}, "load_interim"),
        (
            "03_er_phase2.py",
            {"limit": 1, "run_candidates": False},
            "load_canonical_orgs",
        ),
        ("04_extract.py", {"years": [2025]}, "load_interim"),
    ],
)
def test_partial_pipeline_fails_before_data_io(monkeypatch, filename, kwargs, io_name):
    module = load_pipeline(filename)
    monkeypatch.setattr(
        module,
        io_name,
        lambda *args, **kw: pytest.fail("data I/O ran before the output safety guard"),
    )
    with pytest.raises(ValueError, match="Pass --output-(?:path|dir)"):
        module.main(**kwargs)


def test_ingest_partial_run_uses_explicit_output(monkeypatch, tmp_path):
    module = load_pipeline("01_ingest.py")
    output_path = tmp_path / "ingest" / "releases.parquet"
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-01", tz="UTC")],
            "year": pd.Series([2025], dtype="Int64"),
            "buyer_raw_id": ["GB-COH-1"],
            "contracts_json": ["[]"],
        }
    )
    written = []
    monkeypatch.setattr(module, "flatten_all_years", lambda *args, **kwargs: frame)
    monkeypatch.setattr(module, "write_interim", lambda df, path: written.append(path))

    module.main(years=[2025], output_path=output_path)

    assert written == [output_path]


def test_phase1_partial_run_uses_explicit_output_dir(monkeypatch, tmp_path):
    module = load_pipeline("02_er_phase1.py")
    output_dir = tmp_path / "phase1"
    releases = pd.DataFrame({"ocid": ["ocds-test"]})
    orgs = pd.DataFrame({"canonical_id": ["GB-FTS-1"]})
    aliases = pd.DataFrame({"raw_id": ["GB-FTS-1"]})
    written = []
    monkeypatch.setattr(module, "load_interim", lambda path: releases)
    monkeypatch.setattr(module, "resolve_phase1", lambda df: (orgs, aliases))
    monkeypatch.setattr(
        module, "write_entities", lambda out_orgs, out_aliases, path: written.append(path)
    )
    monkeypatch.setattr(module, "print_report", lambda *args: None)

    module.main(limit=1, output_dir=output_dir)

    assert written == [output_dir]


def test_phase2_partial_run_reads_canonical_and_writes_explicit_dir(monkeypatch, tmp_path):
    module = load_pipeline("03_er_phase2.py")
    output_dir = tmp_path / "phase2"
    orgs = pd.DataFrame(
        {
            "canonical_id": ["GB-FTS-1"],
            "er_status": ["unresolved"],
        }
    )
    aliases = pd.DataFrame(
        {
            "raw_id": ["GB-FTS-1"],
            "canonical_id": ["GB-FTS-1"],
        }
    )
    written = []
    read_dirs = []
    monkeypatch.setattr(
        module,
        "load_canonical_orgs",
        lambda path: read_dirs.append(path) or orgs,
    )
    monkeypatch.setattr(module, "load_alias_map", lambda path: read_dirs.append(path) or aliases)
    monkeypatch.setattr(
        module,
        "resolve_phase2",
        lambda out_orgs, out_aliases, path: (out_orgs, out_aliases, pd.DataFrame()),
    )
    monkeypatch.setattr(
        module,
        "write_phase2_outputs",
        lambda out_orgs, out_aliases, audit, path: written.append(path),
    )
    monkeypatch.setattr(module, "print_report", lambda *args: None)
    monkeypatch.setattr(module, "_count_gov_entries", lambda path: 0)

    module.main(limit=1, run_candidates=False, output_dir=output_dir)

    canonical_dir = module.ROOT / module.load_settings()["data"]["entities_dir"]
    assert read_dirs == [canonical_dir, canonical_dir]
    assert written == [output_dir]


def test_extract_partial_run_uses_explicit_output_dir(monkeypatch, tmp_path):
    module = load_pipeline("04_extract.py")
    output_dir = tmp_path / "extract"
    releases = pd.DataFrame({"ocid": ["ocds-test"], "release_id": ["release-1"]})
    written = []
    monkeypatch.setattr(module, "load_interim", lambda path: releases)
    monkeypatch.setattr(module, "extract_all", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "write_extracted", lambda tables, path: written.append(path))
    monkeypatch.setattr(module, "print_report", lambda *args: None)

    module.main(years=[2025], output_dir=output_dir)

    assert written == [output_dir]
