"""Compatibility wrapper for normalisation utilities.

New code should import from `procurement_graph.common.normalise`. This module
keeps existing flat imports working during the package migration.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OFFICIAL_SCHEMES = frozenset(
    ["GB-COH", "GB-NHS", "GB-UKPRN", "GB-CHC", "GB-SC", "GB-NIC", "GB-MPR"]
)

# Priority: lower number = higher trust
SCHEME_PRIORITY: dict[str, int] = {
    "GB-COH":   1,
    "GB-NHS":   2,
    "GB-UKPRN": 3,
    "GB-CHC":   4,
    "GB-SC":    5,
    "GB-NIC":   6,
    "GB-MPR":   7,
    "GB-FTS":   99,
}

# Legal-form suffixes to strip from company names, longest-first so that
# "LIMITED" is stripped before "LTD" could match a substring.
_SUFFIX_LIST = [
    "LIMITED LIABILITY PARTNERSHIP",
    "COMMUNITY INTEREST COMPANY",
    "INCORPORATED",
    "LIMITED",
    "LIMITE",          # French form occasionally in the data
    "LLP",
    "CIC",
    "PLC",
    "LTD",
    "INC",
    "CO",
]

# Compiled once at import time
_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _SUFFIX_LIST) + r")\b\.?$"
)

# Ampersand / abbreviation normalisations applied in order
_AMP_RE = re.compile(r"\s*&\s*")
_PUNCT_RE = re.compile(r"[^\w\s]")      # remove non-word, non-space chars
_SPACE_RE = re.compile(r"\s+")

# NHS-specific prefixes often added inconsistently
_NHS_PREFIX_RE = re.compile(r"^NHS\s+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalise_name(name: str | None) -> str:
    """Return a canonical normalised form of an organisation name.

    Steps:
    1. Decode unicode to ASCII-safe form (handles curly apostrophes etc.)
    2. Uppercase
    3. Expand & → AND
    4. Remove punctuation except spaces
    5. Strip legal suffixes (LIMITED, LTD, PLC, …)
    6. Collapse whitespace and strip

    Returns "" for None / blank input.
    """
    if not name:
        return ""
    # ASCII transliteration of unicode (e.g. é → e, curly quotes → ')
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", errors="ignore").decode("ascii")
    name = name.upper().strip()
    name = _AMP_RE.sub(" AND ", name)
    name = _PUNCT_RE.sub(" ", name)
    name = _SUFFIX_RE.sub("", name).strip()
    name = _SPACE_RE.sub(" ", name).strip()
    return name


def normalise_org_id(scheme: str, raw_id: str) -> str:
    """Return a normalised canonical_id string for an official-scheme party.

    For GB-NHS codes: uppercase and strip whitespace (codes like "Y56", "QHM").
    For GB-COH numbers: strip whitespace only (preserve original casing of number).
    For all others: strip whitespace.

    Returns "<SCHEME>-<normalised_id>".
    """
    scheme = scheme.strip().upper()
    raw_id = (raw_id or "").strip()
    if scheme == "GB-NHS":
        raw_id = raw_id.upper()
    return f"{scheme}-{raw_id}"


def scheme_of(raw_id: str) -> str:
    """Extract the scheme prefix from a raw party id like 'GB-COH-12345'.

    Returns the scheme string (e.g. 'GB-COH') or '' if unparseable.
    The raw_id format is always <SCHEME>-<VALUE> where SCHEME may itself
    contain hyphens (e.g. GB-COH, GB-NHS, GB-FTS).
    """
    if not raw_id:
        return ""
    # All known schemes have exactly two hyphen-separated parts (e.g. GB-COH)
    # The value follows the second hyphen.
    parts = raw_id.split("-")
    if len(parts) < 3:
        return ""
    return f"{parts[0]}-{parts[1]}"


def value_of(raw_id: str) -> str:
    """Extract the value portion from a raw party id like 'GB-COH-12345'.

    Returns everything after the scheme prefix, e.g. '12345'.
    """
    if not raw_id:
        return ""
    parts = raw_id.split("-", 2)
    return parts[2] if len(parts) == 3 else ""


def is_official(scheme: str) -> bool:
    """Return True if scheme is a trusted official registry (not GB-FTS)."""
    return scheme.strip().upper() in OFFICIAL_SCHEMES


def is_fts(scheme: str) -> bool:
    return scheme.strip().upper() == "GB-FTS"


def canonical_id_from_raw(raw_id: str) -> str | None:
    """If raw_id uses an official scheme, return its canonical_id form.

    Normalises the value portion and returns '<SCHEME>-<value>'.
    Returns None if the scheme is GB-FTS or unrecognised.
    """
    scheme = scheme_of(raw_id)
    if not is_official(scheme):
        return None
    val = value_of(raw_id)
    # Re-normalise through normalise_org_id
    return normalise_org_id(scheme, val)


def priority(scheme: str) -> int:
    """Return sort priority for a scheme (1 = highest trust)."""
    return SCHEME_PRIORITY.get(scheme.strip().upper(), 50)


# Compatibility re-export. New code should import from
# procurement_graph.common.normalise; old flat imports keep working.
from procurement_graph.common.normalise import (  # noqa: E402,F401
    OFFICIAL_SCHEMES,
    SCHEME_PRIORITY,
    canonical_id_from_raw,
    is_fts,
    is_official,
    normalise_name,
    normalise_org_id,
    priority,
    scheme_of,
    value_of,
)
