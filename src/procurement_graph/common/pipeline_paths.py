"""Path guards shared by command-line data pipelines."""

from pathlib import Path


def resolve_output_target(
    canonical: Path,
    requested: Path | str | None,
    *,
    project_root: Path,
    partial: bool,
    option_name: str,
) -> Path:
    """Resolve an output path and protect canonical data from partial runs.

    Relative overrides are interpreted from the project root so a pipeline has
    the same behaviour regardless of the caller's current working directory.
    Full runs retain their canonical default.  A partial run must name a
    distinct destination explicitly.
    """
    canonical = Path(canonical)
    if requested is None:
        target = canonical
    else:
        target = Path(requested).expanduser()
        if not target.is_absolute():
            target = Path(project_root) / target

    if partial and target.resolve() == canonical.resolve():
        raise ValueError(
            "Partial runs cannot overwrite the canonical output "
            f"{canonical}. Pass {option_name} with a separate path."
        )
    return target
