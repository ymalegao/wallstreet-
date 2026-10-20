"""Parser for Ken French's SIC-to-12-industry classification file."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

_INDUSTRY_HEADER = re.compile(r"^\s*(\d{1,2})\s+([A-Za-z0-9]+)\b")
_SIC_RANGE = re.compile(r"^\s*(\d{4})-(\d{4})\s*$")


def parse_ff12_sic_zip(path: Path) -> dict[int, str]:
    """Expand the official FF12 SIC ranges into an integer-code lookup."""
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if len(members) != 1:
            raise ValueError(f"expected one text member in {path}, found {members}")
        text = archive.read(members[0]).decode("utf-8-sig")

    result: dict[int, str] = {}
    current_industry: str | None = None
    headers: set[int] = set()
    for line in text.splitlines():
        header = _INDUSTRY_HEADER.match(line)
        if header:
            industry_number = int(header.group(1))
            current_industry = header.group(2)
            headers.add(industry_number)
            continue
        sic_range = _SIC_RANGE.match(line)
        if not sic_range:
            continue
        if current_industry is None:
            raise ValueError("FF12 SIC range appears before its industry heading")
        start, end = (int(value) for value in sic_range.groups())
        if end < start:
            raise ValueError(f"invalid SIC range {start}-{end}")
        for sic in range(start, end + 1):
            previous = result.get(sic)
            if previous is not None and previous != current_industry:
                raise ValueError(f"SIC {sic:04d} appears in multiple FF12 industries")
            result[sic] = current_industry
    if headers != set(range(1, 13)):
        raise ValueError(f"FF12 file did not contain all twelve industry headings: {sorted(headers)}")
    return result
