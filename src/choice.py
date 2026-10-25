# Multiple-choice answer extraction shared by the MCQ benchmarks.

from __future__ import annotations

import re


def extract_choice(text: str, letters: str) -> str | None:
    """First standalone option letter in `text` (case-insensitive), else the first one anywhere."""
    upper = (text or "").strip().upper()
    m = re.search(rf"\b([{letters}])\b", upper) or re.search(f"[{letters}]", upper)
    return m.group(0) if m else None
