from __future__ import annotations

from typing import List


def parse_objects_field(objects_text: str) -> List[str]:
    """Parse an objects column into a list of object names.

    The SAM3 pipeline uses a dot-separated format like:
        "tree. car. person"

    For robustness we also accept commas and semicolons as separators.
    Empty tokens are filtered out and surrounding whitespace is stripped.
    """
    if not objects_text:
        return []

    normalized = objects_text.replace(";", ".").replace(",", ".")
    parts = [p.strip() for p in normalized.split(".") if p.strip()]
    return parts


def join_objects(objects: List[str]) -> str:
    """Join a list of objects back into the canonical ". " format."""
    if not objects:
        return ""
    return ". ".join(obj.strip() for obj in objects if obj.strip())
