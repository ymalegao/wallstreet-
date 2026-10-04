"""Text cleanup shared by source adapters."""

from __future__ import annotations

import html
import re

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def html_to_text(s: str | None) -> str:
    if not s:
        return ""
    s = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", s)
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", s))).strip()
