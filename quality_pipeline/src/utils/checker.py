from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

try:
    # Optional: only used for a nice progress bar on the initial pass.
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):  # type: ignore[no-redef]
        return iterable

from src.utils.object_utils import parse_objects_field, join_objects
from src.utils.llm_clients import VLMClient as LLMClient, build_vlm_client as build_default_client


_LLM_ERRORS_CSV = Path(__file__).resolve().parent / "outs" / "llm_request_errors.csv"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text to a file atomically (best-effort)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        # Best-effort; never break the main loop on logging issues.
        return


def _compute_realtime_stats(
    *,
    input_csv: Path,
    output_csv: Path,
    new_column: str,
) -> Dict[str, Union[float, int, str]]:
    """Compute summary stats from the (growing) SAM3 progress CSV + our output CSV.

    All values are best-effort; if parsing fails (e.g., file mid-write), return zeros.
    """
    total_rows = 0
    sam3_missing_rows = 0

    try:
        with input_csv.open("r", newline="", encoding="utf-8") as f_in:
            reader = csv.DictReader(f_in)
            if reader.fieldnames is None:
                raise ValueError("input CSV has no header")
            for row in reader:
                total_rows += 1
                if (row.get("objects_not_found") or "").strip():
                    sam3_missing_rows += 1
    except Exception:
        total_rows = 0
        sam3_missing_rows = 0

    vllm_no_rows = 0
    vllm_processed_rows = 0
    try:
        if output_csv.is_file():
            with output_csv.open("r", newline="", encoding="utf-8") as f_out:
                reader = csv.DictReader(f_out)
                if reader.fieldnames is None:
                    raise ValueError("output CSV has no header")
                for row in reader:
                    vllm_processed_rows += 1
                    # Count rows where vLLM rejected at least one missing object.
                    if (row.get(new_column) or "").strip():
                        vllm_no_rows += 1
    except Exception:
        vllm_no_rows = 0
        vllm_processed_rows = 0

    missing_pct = (sam3_missing_rows / total_rows * 100.0) if total_rows else 0.0

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "new_column": str(new_column),
        "total_rows": int(total_rows),
        "sam3_missing_rows": int(sam3_missing_rows),
        "sam3_missing_pct": float(missing_pct),
        "vllm_processed_rows": int(vllm_processed_rows),
        "vllm_no_rows": int(vllm_no_rows),
    }


def _format_status_log(stats: Dict[str, Union[float, int, str]]) -> str:
    return (
        f"timestamp: {stats.get('timestamp','')}\n"
        f"input_csv: {stats.get('input_csv','')}\n"
        f"output_csv: {stats.get('output_csv','')}\n"
        f"new_column: {stats.get('new_column','')}\n"
        "\n"
        f"total_rows_in_input_csv: {stats.get('total_rows',0)}\n"
        f"rows_with_sam3_missing_objects: {stats.get('sam3_missing_rows',0)}\n"
        f"sam3_missing_percent: {float(stats.get('sam3_missing_pct',0.0)):.4f}\n"
        "\n"
        f"rows_processed_by_vllm_checker: {stats.get('vllm_processed_rows',0)}\n"
        f"rows_where_vllm_said_no_to_at_least_one_missing_object: {stats.get('vllm_no_rows',0)}\n"
    )


def _log_llm_request_error(
    *,
    image_path: str,
    object_name: str,
    description: str,
    error: Exception,
) -> None:
    """Append an entry to a CSV when an LLM request fails (best-effort)."""
    try:
        _LLM_ERRORS_CSV.parent.mkdir(parents=True, exist_ok=True)
        file_exists = _LLM_ERRORS_CSV.is_file()
        fieldnames = [
            "timestamp",
            "image_path",
            "object_name",
            "description",
            "error_type",
            "error_message",
        ]
        with _LLM_ERRORS_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "image_path": image_path,
                    "object_name": object_name,
                    "description": description,
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:2000],
                }
            )
    except Exception:
        return


@dataclass
class ProgressRow:
    """Represents one row from `sam3_realtime_progress*.csv`.

    Only the fields we actually use are modeled explicitly; all
    remaining columns are preserved in a generic dict when rewriting
    the CSV so the script is robust to schema changes.
    """

    raw: dict  # Full original row as a dict
    image_path: str
    description: str
    objects_not_found: str


def iter_progress_rows(csv_path: Path) -> Iterable[ProgressRow]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")

        for row in reader:
            image_path = (row.get("image_path") or "").strip()
            description = (row.get("description") or "").strip()
            objects_not_found = (row.get("objects_not_found") or "").strip()

            if not image_path:
                # Be forgiving and just skip malformed rows.
                continue

            yield ProgressRow(
                raw=row,
                image_path=image_path,
                description=description,
                objects_not_found=objects_not_found,
            )


def _process_row_with_llm(
    *,
    row: dict,
    writer: csv.DictWriter,
    no_only_writer: Optional[csv.DictWriter],
    no_only_raw_column: str,
    images_root: Optional[Path],
    client: LLMClient,
    new_column: str,
    llm_batch_size: int,
) -> None:
    """Process a single CSV row, writing an augmented row to the writer."""

    pr = ProgressRow(
        raw=row,
        image_path=(row.get("image_path") or "").strip(),
        description=(row.get("description") or "").strip(),
        objects_not_found=(row.get("objects_not_found") or "").strip(),
    )

    if not pr.image_path:
        # Keep malformed rows but leave new column empty.
        row[new_column] = ""
        row["vllm_num_questions"] = 0
        row["vllm_time_s"] = 0.0
        writer.writerow(row)
        return

    # Build full image path:
    # - If image_path is absolute, use it as-is (ignore images_root).
    # - Otherwise, optionally prepend images_root.
    pr_path = Path(pr.image_path)
    if pr_path.is_absolute():
        full_image_path = str(pr_path)
    elif images_root is not None:
        full_image_path = str((images_root / pr_path).resolve())
    else:
        full_image_path = str(pr_path)

    missing_objects = parse_objects_field(pr.objects_not_found)
    if not missing_objects:
        row[new_column] = ""
        row["vllm_num_questions"] = 0
        row["vllm_time_s"] = 0.0
        writer.writerow(row)
        return

    llm_rejected: List[str] = []
    raw_records: list[dict[str, str]] = []
    num_questions = len(missing_objects)
    t0 = time.time()

    # Use batched LLM queries when supported and requested.
    use_batch = llm_batch_size > 1 and hasattr(client, "is_object_in_image_batch")

    if use_batch:
        for i in range(0, len(missing_objects), llm_batch_size):
            chunk = missing_objects[i : i + llm_batch_size]

            queries = [
                {
                    "image_path": full_image_path,
                    "object_name": obj,
                    "description": pr.description,
                }
                for obj in chunk
            ]

            try:
                if hasattr(client, "is_object_in_image_batch_with_raw"):
                    # type: ignore[attr-defined] – runtime guarded by hasattr above
                    present_and_raw = client.is_object_in_image_batch_with_raw(queries)  # type: ignore[assignment]
                    for obj, (present, raw) in zip(chunk, present_and_raw):
                        raw_records.append(
                            {"object_name": str(obj), "raw_output": str(raw or "")}
                        )
                        if not present:
                            llm_rejected.append(obj)
                else:
                    # type: ignore[attr-defined] – guarded by hasattr above
                    present_flags = client.is_object_in_image_batch(queries)  # type: ignore[assignment]
                    for obj, present in zip(chunk, present_flags):
                        raw_records.append({"object_name": str(obj), "raw_output": ""})
                        # We keep objects for which the model said "no".
                        if not present:
                            llm_rejected.append(obj)
            except Exception:
                # If the batch call fails (e.g. transient HTTP timeout), fall back
                # to per-object calls so a single error doesn't abort the whole run.
                for obj in chunk:
                    try:
                        if hasattr(client, "is_object_in_image_with_raw"):
                            # type: ignore[attr-defined]
                            present, raw = client.is_object_in_image_with_raw(  # type: ignore[misc]
                                image_path=full_image_path,
                                object_name=obj,
                                description=pr.description,
                            )
                        else:
                            present = client.is_object_in_image(
                                image_path=full_image_path,
                                object_name=obj,
                                description=pr.description,
                            )
                            raw = ""
                    except Exception as ee:
                        _log_llm_request_error(
                            image_path=full_image_path,
                            object_name=obj,
                            description=pr.description,
                            error=ee,
                        )
                        present = False  # conservative: treat as not present
                        raw = f"[ERROR] {type(ee).__name__}: {ee}"
                    raw_records.append(
                        {"object_name": str(obj), "raw_output": str(raw or "")}
                    )
                    if not present:
                        llm_rejected.append(obj)
    else:
        for obj in missing_objects:
            try:
                if hasattr(client, "is_object_in_image_with_raw"):
                    # type: ignore[attr-defined]
                    present, raw = client.is_object_in_image_with_raw(  # type: ignore[misc]
                        image_path=full_image_path,
                        object_name=obj,
                        description=pr.description,
                    )
                else:
                    present = client.is_object_in_image(
                        image_path=full_image_path,
                        object_name=obj,
                        description=pr.description,
                    )
                    raw = ""
            except Exception as e:
                _log_llm_request_error(
                    image_path=full_image_path,
                    object_name=obj,
                    description=pr.description,
                    error=e,
                )
                present = False  # conservative: treat as not present
                raw = f"[ERROR] {type(e).__name__}: {e}"
            raw_records.append({"object_name": str(obj), "raw_output": str(raw or "")})
            # We keep objects for which the model said "no".
            if not present:
                llm_rejected.append(obj)

    row[new_column] = join_objects(llm_rejected)
    row["vllm_num_questions"] = num_questions
    row["vllm_time_s"] = round(time.time() - t0, 4)
    writer.writerow(row)

    # Optional: write a debug-only CSV with rows where vLLM rejected at least one object.
    if no_only_writer is not None and llm_rejected:
        debug_row = dict(row)
        debug_row[no_only_raw_column] = json.dumps(raw_records, ensure_ascii=False)
        no_only_writer.writerow(debug_row)


def check_csv_with_llm(
    *,
    input_csv: Path,
    output_csv: Optional[Path] = None,
    no_only_csv: Optional[Path] = None,
    no_only_raw_column: str = "vllm_raw_outputs_json",
    images_root: Optional[Path] = None,
    client: Optional[LLMClient] = None,
    new_column: str = "objects_vllm_said_no",
    llm_batch_size: int = 1,
    resume: bool = False,
    follow: bool = False,
    poll_interval: float = 5.0,
    follow_idle_minutes: Optional[int] = None,
) -> Path:
    """Run LLM-based checks over a SAM3 realtime progress CSV.

    For each row where `objects_not_found` is non-empty, we split that
    field into individual objects (dot-separated), ask the LLM a
    yes/no question for each object and image, and write a new CSV
    that copies all original columns plus an extra column containing
    the subset of objects for which the LLM answered "no".

    When `follow=True`, this function keeps watching the input CSV for
    new rows being appended by another process and processes them as
    they appear (similar to `tail -f`). Use Ctrl+C to stop it.
    """

    input_csv = input_csv.resolve()
    if output_csv is None:
        output_csv = input_csv.with_name(input_csv.stem + "_vllm_checked.csv")
    else:
        output_csv = output_csv.resolve()

    if no_only_csv is None:
        # Default debug CSV path lives next to the main output.
        no_only_csv = output_csv.with_name(output_csv.stem + "_no_only_debug.csv")
    else:
        no_only_csv = no_only_csv.resolve()

    if images_root is not None:
        images_root = images_root.resolve()

    client = client or build_default_client()
    if os.environ.get("VLLM_PROGRESS", "0") == "1":
        mode = "follow" if follow else "one-shot"
        print(f"[checker] Stage: starting ({mode})", flush=True)
        print(f"[checker] Stage: input_csv={input_csv}", flush=True)
        print(f"[checker] Stage: output_csv={output_csv}", flush=True)
        if images_root is not None:
            print(f"[checker] Stage: images_root={images_root}", flush=True)

    # Normalize batch size.
    if llm_batch_size <= 0:
        llm_batch_size = 1

    # First, read the header once to determine field ordering.
    with input_csv.open("r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {input_csv}")

        base_fieldnames: List[str] = list(reader.fieldnames)

    fieldnames: List[str] = list(base_fieldnames)
    if new_column not in fieldnames:
        fieldnames.append(new_column)
    for extra in ("vllm_num_questions", "vllm_time_s"):
        if extra not in fieldnames:
            fieldnames.append(extra)

    no_only_fieldnames: List[str] = list(fieldnames)
    if no_only_raw_column not in no_only_fieldnames:
        no_only_fieldnames.append(no_only_raw_column)

    # If we can, count total input rows once so tqdm can show real progress even
    # when resuming (otherwise it can look "stuck" with no output).
    total_input_rows: Optional[int] = None
    if not follow:
        try:
            with input_csv.open("r", newline="", encoding="utf-8") as f_count:
                reader_count = csv.DictReader(f_count)
                if reader_count.fieldnames is None:
                    raise ValueError(f"CSV file has no header: {input_csv}")
                total_input_rows = sum(1 for _ in reader_count)
        except Exception:
            total_input_rows = None

    # Resume mode: if enabled and output exists, count existing rows so we can skip them.
    processed_rows = 0
    output_exists = output_csv.is_file()
    output_has_header = False
    processed_image_paths: set[str] = set()

    if output_exists and resume:
        try:
            with output_csv.open("r", newline="", encoding="utf-8") as f_check:
                reader_check = csv.DictReader(f_check)
                # If the file is non-empty, DictReader should expose fieldnames (header).
                output_has_header = reader_check.fieldnames is not None
                if output_has_header and list(reader_check.fieldnames) != fieldnames:
                    raise ValueError(
                        "Output CSV header does not match expected columns for this run. "
                        "Refusing to --resume to avoid corrupting the file. "
                        f"Expected fieldnames={fieldnames} but found={list(reader_check.fieldnames)}. "
                        "Delete the output CSV, change --new_column to match, or rerun without --resume."
                    )
                # Count data rows (excluding header).
                for out_row in reader_check:
                    processed_rows += 1
                    img = (out_row.get("image_path") or "").strip()
                    if img:
                        processed_image_paths.add(img)
            print(f"Output CSV already has {processed_rows} rows; resuming from there.")
            if processed_image_paths:
                print(
                    f"Resume: loaded {len(processed_image_paths)} unique image_path values from output; "
                    "will skip re-checking those images."
                )
        except Exception:
            # If we can't read it or the header mismatches, do not silently resume.
            raise

    # NO-only debug CSV resume/header check (best-effort; mirrors main output behavior).
    no_only_exists = no_only_csv.is_file()
    no_only_has_header = False
    if no_only_exists and resume:
        try:
            with no_only_csv.open("r", newline="", encoding="utf-8") as f_check_no:
                reader_check_no = csv.DictReader(f_check_no)
                no_only_has_header = reader_check_no.fieldnames is not None
                if no_only_has_header and list(reader_check_no.fieldnames) != no_only_fieldnames:
                    raise ValueError(
                        "NO-only debug CSV header does not match expected columns for this run. "
                        "Refusing to --resume to avoid corrupting the file. "
                        f"Expected fieldnames={no_only_fieldnames} but found={list(reader_check_no.fieldnames)}. "
                        "Delete the NO-only CSV, change --no_only_raw_column, or rerun without --resume."
                    )
        except Exception:
            raise

    # Open the output CSV:
    # - default: write fresh (overwrite)
    # - --resume: append if the output exists and has a header, otherwise write fresh
    if resume and output_exists and output_has_header:
        mode = "a"
    else:
        mode = "w"
    if resume and no_only_exists and no_only_has_header:
        no_mode = "a"
    else:
        no_mode = "w"

    with output_csv.open(mode, newline="", encoding="utf-8") as f_out, \
         no_only_csv.open(no_mode, newline="", encoding="utf-8") as f_no:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        no_writer = csv.DictWriter(f_no, fieldnames=no_only_fieldnames)
        
        # Only write header if we're starting fresh.
        if mode == "w":
            writer.writeheader()
        if no_mode == "w":
            no_writer.writeheader()
        f_out.flush()
        f_no.flush()

        # Follow mode: optionally stop after an idle timeout (no file changes).
        # We detect "changes" by (mtime_ns, size) signature to avoid relying
        # solely on "new_rows == 0", since writers may touch the file.
        idle_timeout_s: Optional[float] = None
        if follow and follow_idle_minutes is not None:
            if follow_idle_minutes <= 0:
                follow_idle_minutes = 20
            idle_timeout_s = float(follow_idle_minutes * 60)

        try:
            st0 = input_csv.stat()
            prev_sig: Optional[Tuple[int, int]] = (st0.st_mtime_ns, st0.st_size)
        except FileNotFoundError:
            prev_sig = None

        idle_elapsed_s = 0.0
        if follow:
            print(
                f"[follow] Watching for new rows in: {input_csv}",
                flush=True,
            )
            print(
                f"[follow] Poll interval: {poll_interval:.1f}s",
                flush=True,
            )
            if idle_timeout_s is not None:
                print(
                    f"[follow] Idle timeout: {int(idle_timeout_s // 60)} minutes "
                    f"(exit when input CSV is unchanged for that long).",
                    flush=True,
                )

        while True:
            new_rows = 0
            # When resuming (no tqdm), print a small heartbeat so it doesn't look hung.
            last_heartbeat_t = time.time()
            heartbeat_every_s = float(os.environ.get("VLLM_CHECKER_HEARTBEAT_S") or 30.0)

            # Each iteration we reopen the input CSV and skip rows we've
            # already processed. This is simple and robust even while
            # another process is appending new rows.
            with input_csv.open("r", newline="", encoding="utf-8") as f_in:
                reader = csv.DictReader(f_in)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV file has no header: {input_csv}")

                # Skip rows we've already processed in previous passes.
                for _ in range(processed_rows):
                    try:
                        next(reader)
                    except StopIteration:
                        break

                # For the initial pass (processed_rows == 0) we keep tqdm
                # for a nice progress bar. For subsequent incremental passes
                # we iterate without tqdm.
                iterable = (
                    tqdm(
                        reader,
                        desc="LLM-checking rows",
                        unit="row",
                        total=total_input_rows,
                        initial=processed_rows,
                    )
                    if not follow
                    else reader
                )

                for row in iterable:
                    # If resuming, never re-check an image_path that already exists in the output CSV.
                    # This protects against accidental duplicates in the input CSV or output truncation.
                    if resume:
                        img = (row.get("image_path") or "").strip()
                        if img and img in processed_image_paths:
                            processed_rows += 1
                            continue
                    _process_row_with_llm(
                        row=row,
                        writer=writer,
                        no_only_writer=no_writer,
                        no_only_raw_column=no_only_raw_column,
                        images_root=images_root,
                        client=client,
                        new_column=new_column,
                        llm_batch_size=llm_batch_size,
                    )
                    new_rows += 1
                    processed_rows += 1
                    if resume:
                        img = (row.get("image_path") or "").strip()
                        if img:
                            processed_image_paths.add(img)
                    f_out.flush()
                    if processed_rows > 0 and processed_rows != new_rows:
                        # Heuristic: if we're resuming, `iterable` is a plain reader.
                        # Print occasionally to show forward progress.
                        if processed_rows > 0 and processed_rows % 50 == 0:
                            print(f"Processed {processed_rows} rows so far...")
                        else:
                            now = time.time()
                            if now - last_heartbeat_t >= heartbeat_every_s:
                                print(f"Processed {processed_rows} rows so far...")
                                last_heartbeat_t = now

            if not follow:
                # One-shot mode: process existing rows and return.
                break

            if new_rows == 0:
                # No new rows since last check. Use file signature to track idle time.
                try:
                    st = input_csv.stat()
                    sig: Optional[Tuple[int, int]] = (st.st_mtime_ns, st.st_size)
                except FileNotFoundError:
                    sig = None

                if sig == prev_sig:
                    idle_elapsed_s += float(poll_interval)
                    if idle_timeout_s is not None:
                        remaining = max(0.0, idle_timeout_s - idle_elapsed_s)
                        if idle_elapsed_s >= idle_timeout_s:
                            print(
                                f"[follow] No input CSV changes for {follow_idle_minutes} minutes. Exiting.",
                                flush=True,
                            )
                            break
                        print(
                            f"[follow] No change detected. Idle "
                            f"{int(idle_elapsed_s // 60)}m{int(idle_elapsed_s % 60):02d}s / "
                            f"{follow_idle_minutes}m (remaining {int(remaining // 60)}m{int(remaining % 60):02d}s).",
                            flush=True,
                        )
                else:
                    prev_sig = sig
                    idle_elapsed_s = 0.0
                    print(
                        "[follow] Input CSV changed, but no new parseable rows detected yet; continuing.",
                        flush=True,
                    )

                # Wait a bit before trying again.
                time.sleep(poll_interval)
            # Otherwise, loop again immediately to see if more rows were
            # appended while we were processing.

    return output_csv


def debug_single_image(
    *,
    input_csv: Path,
    target_image_path: str,
    images_root: Optional[Path] = None,
    client: Optional[LLMClient] = None,
) -> None:
    """Debug mode: process a single image and print results to stdout.
    
    Args:
        input_csv: Path to the input CSV file
        target_image_path: The image_path value to look for (as it appears in CSV)
        images_root: Optional root directory to prepend to image paths
        client: Optional LLM client (will use default if not provided)
    """
    print(f"[1/5] Initializing debug mode...")
    input_csv = input_csv.resolve()
    if images_root is not None:
        images_root = images_root.resolve()
    
    # Enable debug mode for LLM client
    os.environ["VLLM_DEBUG"] = "1"
    
    print(f"[2/5] Building LLM client (this may take a moment to load the model)...")
    client = client or build_default_client()
    print(f"      LLM client ready!")
    
    # Find the row with the matching image_path
    print(f"[3/5] Searching for image '{target_image_path}' in CSV...")
    target_row = None
    row_count = 0
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {input_csv}")
        
        for row in reader:
            row_count += 1
            image_path = (row.get("image_path") or "").strip()
            if image_path == target_image_path:
                target_row = row
                print(f"      Found at row {row_count}!")
                break
            if row_count % 100 == 0:
                print(f"      Searched {row_count} rows so far...")
    
    if target_row is None:
        print(f"ERROR: Image path '{target_image_path}' not found in CSV after searching {row_count} rows.")
        return
    
    # Parse the row
    print(f"[4/5] Parsing row data...")
    pr = ProgressRow(
        raw=target_row,
        image_path=(target_row.get("image_path") or "").strip(),
        description=(target_row.get("description") or "").strip(),
        objects_not_found=(target_row.get("objects_not_found") or "").strip(),
    )
    
    print(f"\n{'='*70}")
    print(f"DEBUG MODE: Processing single image")
    print(f"{'='*70}")
    print(f"Image path: {pr.image_path}")
    print(f"Description: {pr.description}")
    print(f"Objects not found (SAM): {pr.objects_not_found}")
    print(f"{'='*70}\n")
    
    # Build full image path
    if images_root is not None:
        full_image_path = str((images_root / pr.image_path).resolve())
    else:
        full_image_path = pr.image_path
    
    print(f"Full image path: {full_image_path}\n")
    
    missing_objects = parse_objects_field(pr.objects_not_found)
    if not missing_objects:
        print("No objects to check (objects_not_found is empty).")
        return
    
    print(f"[5/5] Checking {len(missing_objects)} objects with LLM...\n")
    
    # Query the LLM for each object
    objects_said_yes: List[str] = []
    objects_said_no: List[str] = []
    
    for idx, obj in enumerate(missing_objects, 1):
        print(f"  [{idx}/{len(missing_objects)}] Querying LLM about '{obj}'...", end=" ", flush=True)
        present = client.is_object_in_image(
            image_path=full_image_path,
            object_name=obj,
            description=pr.description,
        )
        
        if present:
            objects_said_yes.append(obj)
            print(f"✓ YES")
        else:
            objects_said_no.append(obj)
            print(f"✗ NO")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Objects vLLM said YES to ({len(objects_said_yes)}):")
    if objects_said_yes:
        for obj in objects_said_yes:
            print(f"  - {obj}")
    else:
        print("  (none)")
    
    print(f"\nObjects vLLM said NO to ({len(objects_said_no)}):")
    if objects_said_no:
        for obj in objects_said_no:
            print(f"  - {obj}")
    else:
        print("  (none)")
    print(f"{'='*70}\n")

