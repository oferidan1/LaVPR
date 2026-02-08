import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union


CSV_REQUIRED_COLUMNS = ["image_path", "description"]


@dataclass
class CaptionRow:
    image_path: str
    description: str


def _normalize_header(name: str) -> str:
    """Normalize CSV headers to a consistent form."""
    return name.strip()


def parse_caption_csv(csv_path: Union[str, Path]) -> Iterator[CaptionRow]:
    """
    Parse a clustered CSV and yield CaptionRow objects.

    The input CSV is expected to contain at least the following columns:
    - image_path
    - description
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")

        # Map normalized header names back to original
        header_map = {_normalize_header(h): h for h in reader.fieldnames}

        missing = [col for col in CSV_REQUIRED_COLUMNS if col not in header_map]
        if missing:
            raise ValueError(
                f"Missing required columns in CSV {csv_path}: {missing}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            image_path = row[header_map["image_path"]]
            description = row[header_map["description"]]
            if image_path is None or description is None:
                # Skip malformed rows instead of failing the whole job
                continue
            yield CaptionRow(
                image_path=str(image_path),
                description=str(description),
            )


def parse_caption_csv_two_columns_no_header(csv_path: Union[str, Path]) -> Iterator[CaptionRow]:
    """
    Parse a simple 2-column CSV with NO header:

        <image_path>,<description>

    This format is commonly produced by captioning pipelines where the description
    may contain commas and is therefore quoted.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader, start=1):
            if not row:
                continue
            # Skip comment-like lines (best-effort)
            if len(row) == 1 and str(row[0]).lstrip().startswith("#"):
                continue
            # If the file actually has a header, skip it (common in the wild).
            if (
                row_idx == 1
                and len(row) >= 2
                and str(row[0]).strip().lower() in {"image_path", "path", "image"}
                and str(row[1]).strip().lower() in {"description", "caption", "text"}
            ):
                continue
            if len(row) < 2:
                # Malformed row; skip rather than failing whole job
                continue
            image_path = str(row[0]).strip()
            # If there are stray extra columns, join them back with commas.
            description = ",".join(str(x) for x in row[1:]).strip()
            if not image_path or not description:
                continue
            yield CaptionRow(image_path=image_path, description=description)
