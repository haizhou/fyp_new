"""Name and identifier normalisation utilities."""

from __future__ import annotations

import re
import unicodedata

OFFICIAL_SCHEMES = frozenset(
    ["GB-COH", "GB-NHS", "GB-UKPRN", "GB-CHC", "GB-SC", "GB-NIC", "GB-MPR"]
)

SCHEME_PRIORITY: dict[str, int] = {
    "GB-COH": 1,
    "GB-NHS": 2,
    "GB-UKPRN": 3,
    "GB-CHC": 4,
    "GB-SC": 5,
    "GB-NIC": 6,
    "GB-MPR": 7,
    "GB-FTS": 99,
}

_SUFFIX_LIST = [
    "LIMITED LIABILITY PARTNERSHIP",
    "COMMUNITY INTEREST COMPANY",
    "INCORPORATED",
    "LIMITED",
    "LIMITE",
    "LLP",
    "CIC",
    "PLC",
    "LTD",
    "INC",
    "CO",
]

_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _SUFFIX_LIST) + r")\b\.?$"
)
_AMP_RE = re.compile(r"\s*&\s*")
_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def normalise_name(name: str | None) -> str:
    """Return a canonical normalised form of an organisation name.

    Returns an empty string for missing or blank input.
    """
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.upper().strip()
    text = _AMP_RE.sub(" AND ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _SUFFIX_RE.sub("", text).strip()
    return _SPACE_RE.sub(" ", text).strip()


def normalise_org_id(scheme: str, raw_id: str) -> str:
    """Return a normalised canonical ID for an official-scheme party."""
    scheme = scheme.strip().upper()
    value = (raw_id or "").strip()
    if scheme == "GB-NHS":
        value = value.upper()
    return f"{scheme}-{value}"


def scheme_of(raw_id: str) -> str:
    """Extract the scheme prefix from a raw party ID such as `GB-COH-12345`."""
    if not raw_id:
        return ""
    parts = raw_id.split("-")
    if len(parts) < 3:
        return ""
    return f"{parts[0]}-{parts[1]}"


def value_of(raw_id: str) -> str:
    """Extract the value portion from a raw party ID such as `GB-COH-12345`."""
    if not raw_id:
        return ""
    parts = raw_id.split("-", 2)
    return parts[2] if len(parts) == 3 else ""


def is_official(scheme: str) -> bool:
    """Return True if the scheme is a trusted official registry."""
    return scheme.strip().upper() in OFFICIAL_SCHEMES


def is_fts(scheme: str) -> bool:
    """Return True if the scheme is Contracts Finder's GB-FTS scheme."""
    return scheme.strip().upper() == "GB-FTS"


def canonical_id_from_raw(raw_id: str) -> str | None:
    """Return the canonical ID if `raw_id` uses a trusted official scheme."""
    scheme = scheme_of(raw_id)
    if not is_official(scheme):
        return None
    return normalise_org_id(scheme, value_of(raw_id))


def priority(scheme: str) -> int:
    """Return the scheme priority, where lower is more trusted."""
    return SCHEME_PRIORITY.get(scheme.strip().upper(), 50)

