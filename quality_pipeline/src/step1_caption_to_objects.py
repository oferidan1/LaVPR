from __future__ import annotations

import csv
import sys
import argparse
from pathlib import Path
from functools import lru_cache
from typing import Optional, Set, Tuple

from src.utils.csv_parser import parse_caption_csv, parse_caption_csv_two_columns_no_header
from src.utils.file_lock import FileLock

csv.field_size_limit(sys.maxsize)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    class _DummyTqdm:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            if self.iterable is None:
                return iter([])
            return iter(self.iterable)

        def update(self, n=1):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, total=None, **kwargs):
        return _DummyTqdm(iterable, total, **kwargs)


DEFAULT_PROMPT_TEMPLATE = (
    "I want to use an object detector to check the correctness of an image "
    "caption obtained by an image caption model. Can you help to parse the "
    "caption below and list all objects that could be detected with an object "
    "detection model in the image? Please only list the object name and "
    "ignore the description like colors. Please use singular for all listed objects. "
    "Caption: {caption}. "
    "Please concatenate them together with \". \" as separation. "
    "Do not add any explanation, notes, headings, or markdown; output only the "
    "concatenated object names."
)

DEFAULT_PROMPT_TEMPLATE_STUFF = (
    "I want to use a segmentation model to check the correctness of an image "
    "caption obtained by an image caption model. Can you help to parse the "
    "caption below and list all UNCOUNTABLE REGIONS (stuff) that could be "
    "segmented in the image? Examples: sky, road, grass, pavement, wall, "
    "sidewalk, water, ground. These are materials, surfaces, or regions rather "
    "than individual objects. Please only list the region name and ignore the "
    "description like colors. Please use singular for all listed regions. "
    "Caption: {caption}. "
    "Please concatenate them together with \". \" as separation. "
    "Do not add any explanation, notes, headings, or markdown; output only the "
    "concatenated region names."
)

DEFAULT_PROMPT_TEMPLATE_MERGED = (
    "I want to use both an object detector and a segmentation model to check "
    "the correctness of an image caption obtained by an image caption model. "
    "Can you help to parse the caption below and list: "
    "(1) All objects that could be detected with an object detection model in the image, "
    "and (2) All uncountable regions (stuff) that could be segmented in the image. "
    "Examples of stuff: road, grass, pavement, wall, sidewalk, water, ground. "
    "Stuff are materials, surfaces, or regions rather than individual objects. "
    "Please ignore sky."
    "Please ignore dynamic objects like people and cars"
    "Please only list the object/region names and ignore descriptions like colors. "
    "Please use singular for all listed items. "
    "Please do not list signs or surfaces with written words."

    "Here are some examples of the desired behavior in the format of caption and expected output:\n\n"

    "Example 1\n"
    "Caption: Green utility box/door set in a brick wall, dense green foliage and trees along the left side, paved sidewalk, asphalt road with double yellow lines and white ""DISABLED"" marking, paved sidewalk, black ornate street lamp post, dark metal railings, row of brown brick terraced houses with white window frames and white ground floor sections, distant buildings.\n"
    "utility box. brick wall. green foliage. tree. sidewalk. asphalt road. street lamp post. railings.  terraced houses.  white window frames.  buildings.   ### END_OF_LIST ###\n\n"

    "Example 2\n"
    "Caption: White painted brick wall (upper left), dark brick facade with white-framed multi-pane sash window (upper left), dark brick facade with black-framed door and black iron balcony (upper center), dark brick facade with white-framed multi-pane sash window (upper right), dark grey tiled sloped roof (lower center), black framed door/window unit (lower left), large white-framed multi-pane sash window (lower center), white wall with black wall-mounted lantern and silver intercom/doorbell panel (lower right), black garage door with a narrow window above it (lower right), vertical cream/yellow pipe (far right).\n"
    "wall. window. door. balcony. roof. pipe ### END_OF_LIST ###\n\n"

    "Example 3\n"
    "Caption: River Thames with distant buildings and boats, ornate dark lamppost with criss-cross patterns, bridge parapet with dark top rail and decorative light-on-red criss-cross panels, blue road sign with bus and bicycle pictograms, light-colored lamppost on a column base, grey paved walkway, ""BUS"" road marking with a red line.\n"
    "River. lamppost. bridge parapet. road. sign. walkway  ### END_OF_LIST ###\n\n"

    "Now do the same for this new caption.\n"
    "Caption: {caption}. "
    "Please concatenate all objects and stuff together with \". \" as separation. "
    "Do not add any explanation, notes, headings, or markdown; output only the "
    "concatenated object and region names. "
    "After the last item, output exactly \"### END_OF_LIST ###\" and nothing else."
)

DEFAULT_FILTER_PROMPT_TEMPLATE = (
    "You are helping to filter a list of items for use with a segmentation model (SAM). "
    "The list below was extracted from an image caption using noun-phrase parsing. "
    "Some items may not be suitable for segmentation (e.g., abstract concepts, qualities, "
    "actions, or overly vague terms). "
    "Please review the list and keep ONLY items that are:\n"
    "1. Concrete, physical objects that can be segmented (e.g., car, person, building, tree, bench)\n"
    "2. Visible regions or surfaces that can be segmented (e.g., sky, road, grass, pavement, wall, water)\n"
    "3. Suitable for visual segmentation in an image\n"
    "4. NOT abstract concepts (e.g., beauty, time, happiness, importance)\n"
    "5. NOT actions, verbs, or qualities (e.g., running, brightness, moving)\n"
    "6. NOT overly generic terms that are too vague (e.g., thing, stuff, something, area)\n\n"
    "Input list: {object_list}\n\n"
    "Output ONLY the filtered item names separated by \". \" with NO explanations, "
    "notes, headings, numbering, markdown, or reasoning. Start your response immediately "
    "with the first item name. After the last item, output exactly \"### END_OF_LIST ###\" "
    "and nothing else. If all items should be removed, output only \"### END_OF_LIST ###\".\n\n"
    "Filtered list:"
)


def build_prompt(caption: str, template: str) -> str:
    return template.format(caption=caption)


@lru_cache(maxsize=1)
def _load_spacy_model():
    """
    Lazily load the spaCy English model the first time we need it.

    This keeps spaCy as an optional dependency that is only required when
    using the noun-phrase parsing based method.
    """
    try:
        import spacy  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - dependency error path
        raise SystemExit(
            "spaCy is required when using --use_noun_phrase_parser.\n"
            "Install it with: pip install spacy\n"
            "and download a model, e.g.: python -m spacy download en_core_web_sm"
        ) from exc

    try:
        return spacy.load("en_core_web_sm")
    except Exception as exc:  # pragma: no cover - dependency error path
        raise SystemExit(
            "Failed to load spaCy model 'en_core_web_sm'.\n"
            "Install it with: python -m spacy download en_core_web_sm"
        ) from exc


_NP_BLACKLIST = {
    # Meta / non-visual or rarely useful as a concrete region/object
    "background",
    "foreground",
    "distance",
    "line",
    "area",
    "view",
    "scene",
    "image",
    "photo",
    "picture",
    "something",
    "someone",
    "anything",
}


def _is_valid_np_lemma(lemma: str) -> bool:
    """
    Heuristic filter for noun-phrase head lemmas using a simple blacklist.

    Strategy:
    - drop very short tokens,
    - drop obviously non-visual / meta terms using _NP_BLACKLIST,
    - otherwise keep the lemma (to stay open-set).
    """
    if not lemma or len(lemma) < 2:
        return False

    if lemma in _NP_BLACKLIST:
        return False

    return True


def filter_objects_with_llm(
    object_list: str,
    llm_client,
    filter_template: str,
) -> str:
    """
    Filter a list of object names using an LLM to keep only detectable items.
    
    Args:
        object_list: A dot-separated string of object names (e.g., "car. tree. beauty")
        llm_client: The LLM client to use for filtering
        filter_template: The prompt template for filtering
    
    Returns:
        A filtered dot-separated string of object names
    """
    if not object_list or not object_list.strip():
        return ""
    
    # Calculate dynamic max_tokens based on input list length
    # Strategy: Estimate tokens needed for output (usually <= input length)
    # Add buffer for "### END_OF_LIST ###" marker (5 tokens)
    # Minimum 32 tokens, maximum 256 tokens
    input_length = len(object_list)
    estimated_tokens = max(32, min(256, input_length + 20))  # +20 for marker and safety buffer
    
    prompt = filter_template.format(object_list=object_list)
    filtered_text = llm_client.get_objects_from_caption(prompt, max_new_tokens=estimated_tokens)
    
    # The LLMClient already handles truncation at "### END_OF_LIST ###" marker
    # Just do minimal additional cleanup
    filtered_text = filtered_text.strip()
    
    # Remove trailing punctuation if present (but not internal dots used as separators)
    while filtered_text and filtered_text[-1] in ".!?:":
        filtered_text = filtered_text[:-1].strip()
    
    # Safety check: if output still contains obvious explanation patterns after the marker,
    # truncate at those points as an extra safeguard
    stop_phrases = ["Note:", "Explanation:", "I removed", "I filtered", "\n\n"]
    for phrase in stop_phrases:
        if phrase in filtered_text:
            filtered_text = filtered_text[:filtered_text.find(phrase)].strip()
    
    return filtered_text


def _build_llm_client_or_die(
    *,
    model_name: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    use_4bit: bool = False,
):
    """
    Lazily import and build the local HF LLM client.

    This keeps torch/transformers as optional runtime deps for modes that don't
    need an LLM (e.g. --use_noun_phrase_parser).

    Args:
        model_name: Hugging Face model ID.
        max_new_tokens: Override for max new tokens per generation.
        temperature: Override for generation temperature.
        use_4bit: Enable 4-bit quantization to reduce GPU memory by ~75%.
                  Requires: pip install bitsandbytes accelerate
    """
    try:
        from src.utils.llm_clients import LLMClient, LLMConfig
    except Exception as e:
        raise SystemExit(
            "Failed to import the local LLM backend (torch/transformers).\n\n"
            f"Original error: {type(e).__name__}: {e}\n\n"
            "Fix options:\n"
            "  1) Bypass torch entirely by using:\n"
            "       --use_noun_phrase_parser\n"
            "  2) Fix your PyTorch installation in the current env.\n"
            "     For the specific error `undefined symbol: iJIT_NotifyEvent`, install Intel ITT runtime:\n"
            "       conda install -y -c conda-forge ittapi\n"
            "     Or reinstall PyTorch cleanly into the env (recommended) from conda channels.\n"
        ) from e

    cfg = LLMConfig(model_name=model_name, use_4bit=use_4bit)
    if max_new_tokens is not None:
        cfg.max_new_tokens = int(max_new_tokens)
    if temperature is not None:
        cfg.temperature = float(temperature)
    return LLMClient(cfg)


def get_objects_from_caption_np(caption: str) -> str:
    """
    Extract a deduplicated, lemmatized list of object-like nouns from a caption
    using traditional noun-phrase parsing (spaCy), formatted similarly to the
    LLM-based output: 'obj1. obj2. obj3'.
    """
    nlp = _load_spacy_model()
    doc = nlp(caption)

    seen: set[str] = set()
    objects: list[str] = []

    for chunk in doc.noun_chunks:
        # Collect all noun / proper-noun lemmas inside the chunk. For phrases like
        # "red sign board" this gives ["sign", "board"], which we then join into
        # a single token "sign-board" for use as a detectable object name.
        noun_lemmas: list[str] = []
        for token in chunk:
            if token.pos_ in {"NOUN", "PROPN"}:
                lemma = token.lemma_.strip().lower()
                if not _is_valid_np_lemma(lemma):
                    continue
                noun_lemmas.append(lemma)

        if not noun_lemmas:
            continue

        if len(noun_lemmas) == 1:
            candidate = noun_lemmas[0]
        else:
            candidate = "-".join(noun_lemmas)

        if candidate in seen:
            continue

        seen.add(candidate)
        objects.append(candidate)

    return ". ".join(objects)


def filter_existing_np_csv(
    input_csv: Path,
    filter_llm_client: LLMClient,
    filter_template: str,
) -> None:
    """
    Filter an existing *_np.csv file by adding a 'filtered_by_llm' column.
    
    Args:
        input_csv: Path to an existing CSV file ending with _objects_np.csv
        filter_llm_client: The LLM client to use for filtering
        filter_template: The prompt template for filtering
    """
    import csv
    import tempfile
    
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    
    if not str(input_csv).endswith("_objects_np.csv"):
        raise ValueError(f"Input CSV must end with '_objects_np.csv': {input_csv}")
    
    print(f"Processing: {input_csv}")
    
    # Read existing CSV
    rows = []
    fieldnames = None
    with input_csv.open("r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        
        if not fieldnames or "objects" not in fieldnames:
            raise ValueError(f"CSV must have an 'objects' column: {input_csv}")
        
        rows = list(reader)
    
    # Check if 'filtered_by_llm' column already exists
    if "filtered_by_llm" in fieldnames:
        print(f"  ⚠ Column 'filtered_by_llm' already exists, will overwrite values")
    else:
        # Add new column
        fieldnames = list(fieldnames) + ["filtered_by_llm"]
    
    # Process each row
    for idx, row in enumerate(rows, start=1):
        objects_text = row.get("objects", "").strip()
        
        if not objects_text:
            row["filtered_by_llm"] = ""
            print(f"  Row {idx}: Empty objects list, skipping")
            continue
        
        print(f"  Row {idx}: Filtering '{objects_text[:60]}{'...' if len(objects_text) > 60 else ''}'")
        
        # Apply LLM filtering
        filtered_text = filter_objects_with_llm(
            objects_text,
            filter_llm_client,
            filter_template,
        )
        
        row["filtered_by_llm"] = filtered_text
        print(f"    → '{filtered_text[:60]}{'...' if len(filtered_text) > 60 else ''}'")
    
    # Write back to CSV (atomically using temp file)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=input_csv.parent,
    ) as f_temp:
        writer = csv.DictWriter(f_temp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(f_temp.name)
    
    # Replace original file
    temp_path.replace(input_csv)
    print(f"  ✓ Updated: {input_csv}\n")


def _estimate_max_tokens_for_caption(
    description: str,
    base_extra: int = 32,
    min_tokens: int = 64,
    max_tokens: int = 512,
) -> int:
    """
    Heuristic for deciding max_new_tokens based on the caption length.

    We approximate tokens by the number of characters in the description and add
    a small safety buffer (base_extra) so the model has room for separators,
    spaces, and the end-of-list marker. The value is then clipped to a
    reasonable [min_tokens, max_tokens] range.
    """
    if not description:
        return min_tokens

    approx = len(description) + base_extra
    return max(min_tokens, min(max_tokens, approx))


def _batched_enumerate(iterable, batch_size: int):
    """
    Yield (index, item) pairs in batches of at most batch_size.

    Indices are 1-based to match the previous behavior used for file naming.
    """
    if batch_size <= 0:
        batch_size = 1

    batch = []
    for idx, item in enumerate(iterable, start=1):
        batch.append((idx, item))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run(
    input_csv: Path,
    output_dir: Path,
    output_csv: Optional[Path],
    model_name: str,
    prompt_template: Optional[str],
    prompt_template_stuff: Optional[str],
    prompt_template_merged: Optional[str],
    use_merged_prompt: bool,
    generate_stuff: bool,
    use_noun_phrase_parser: bool,
    filter_with_llm: bool,
    filter_model_name: str,
    filter_prompt_template: Optional[str],
    batch_size: int = 1,
    realtime_progress_csv: Optional[Path] = None,
    skip_existing_llm: bool = False,
    resume: bool = False,
    use_4bit: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    method_suffix = "np" if use_noun_phrase_parser else "llm"
    if use_noun_phrase_parser and filter_with_llm:
        method_suffix = "np_filtered"

    if output_csv is None:
        # Encode the extraction method in the CSV name, e.g.:
        # <stem>_objects_llm.csv or <stem>_objects_np.csv or <stem>_objects_np_filtered.csv
        output_csv = input_csv.with_name(f"{input_csv.stem}_objects_{method_suffix}_v2.csv")

    import csv
    from datetime import datetime

    rows_processed = 0
    start_time = datetime.now()

    # Best-effort extraction of cluster id from the input CSV path.
    # For cluster-based outputs, the parent directory is usually named
    # like 'cluster_123', in which case we store '123' as the cluster id.
    parent_name = input_csv.parent.name
    if parent_name.startswith("cluster_") and len(parent_name) > len("cluster_"):
        cluster_id = parent_name[len("cluster_") :]
    else:
        cluster_id = ""

    # Set up real-time per-image (or per-batch) progress logging, if requested.
    progress_writer = None
    f_progress = None
    progress_fieldnames = [
        "input_csv",
        "output_csv",
        "row_index",
        "image_path",
        "cluster_id",
        "start_time",
        "end_time",
        "duration_seconds",
        "avg_duration_per_image",
        "batch_size",
        "status",
    ]
    if realtime_progress_csv is not None:
        file_exists = realtime_progress_csv.is_file()
        f_progress = realtime_progress_csv.open("a", newline="", encoding="utf-8")
        progress_writer = csv.DictWriter(f_progress, fieldnames=progress_fieldnames)
        if not file_exists:
            progress_writer.writeheader()

    # Optionally skip re-processing if an LLM output CSV already exists.
    # This only applies to pure LLM extraction runs (method_suffix == "llm").
    if skip_existing_llm and method_suffix == "llm" and output_csv.exists():
        print(f"Skipping {input_csv} because LLM output already exists: {output_csv}")
        # Still record this in the realtime progress CSV if requested (summary row).
        if progress_writer is not None and f_progress is not None:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            progress_writer.writerow(
                {
                    "input_csv": str(input_csv),
                    "output_csv": str(output_csv),
                    "row_index": "",
                    "image_path": "",
                    "cluster_id": cluster_id,
                    "start_time": start_time.isoformat(timespec="seconds"),
                    "end_time": end_time.isoformat(timespec="seconds"),
                    "duration_seconds": f"{duration:.2f}",
                    "avg_duration_per_image": "",
                    "batch_size": 0,
                    "status": "skipped_existing_llm",
                }
            )
            f_progress.flush()

        return

    # Resume support for single-CSV mode:
    # - if --resume and output CSV exists (and is non-empty), append new rows
    # - skip already-processed image_path values present in the output CSV
    # Without --resume we keep legacy behavior: delete and regenerate outputs.
    processed_image_paths: set[str] = set()

    llm_client = None
    filter_llm_client = None
    
    if use_noun_phrase_parser:
        # LLM only needed if filtering is enabled
        if filter_with_llm:
            # Use deterministic generation for filtering
            # max_new_tokens will be calculated dynamically based on input list length
            filter_llm_client = _build_llm_client_or_die(
                model_name=filter_model_name,
                max_new_tokens=128,
                temperature=0.0,
                use_4bit=use_4bit,
            )
    else:
        llm_client = _build_llm_client_or_die(model_name=model_name, use_4bit=use_4bit)

    template = prompt_template or DEFAULT_PROMPT_TEMPLATE
    template_stuff = prompt_template_stuff or DEFAULT_PROMPT_TEMPLATE_STUFF
    template_merged = prompt_template_merged or DEFAULT_PROMPT_TEMPLATE_MERGED
    template_filter = filter_prompt_template or DEFAULT_FILTER_PROMPT_TEMPLATE

    # When using merged prompt, create a single column with both objects and stuff
    if use_merged_prompt:
        fieldnames = ["image_path", "description", "objects_and_stuff"]
    else:
        fieldnames = ["image_path", "description", "objects"]
        if generate_stuff:
            fieldnames.append("stuff")

    def _load_processed_image_paths_single_csv(out_csv: Path, expected_fieldnames: list[str]) -> set[str]:
        """
        Load processed image_path values from an existing output CSV.

        This is used for --resume in --input_csv mode to avoid re-processing and,
        critically, to avoid truncating/deleting an existing output CSV at startup.
        """
        import csv as _csv

        processed: set[str] = set()
        if not out_csv.is_file():
            return processed

        with out_csv.open("r", newline="", encoding="utf-8") as f_in:
            r = _csv.DictReader(f_in)
            if r.fieldnames is None:
                # Empty file: treat as fresh run.
                return processed
            if list(r.fieldnames) != list(expected_fieldnames):
                raise ValueError(
                    "Output CSV header does not match expected columns for this run. "
                    "Refusing to --resume to avoid corrupting the file. "
                    f"Expected fieldnames={list(expected_fieldnames)} but found={list(r.fieldnames)}. "
                    "Delete the output CSV or rerun without --resume."
                )
            for row in r:
                p = (row.get("image_path") or "").strip()
                if p:
                    processed.add(p)
        return processed

    try:
        # Load all rows once so we can report a proper tqdm over the total.
        rows = list(parse_caption_csv(input_csv))

        out_mode = "w"
        output_exists = output_csv.exists()
        output_nonempty = output_exists and (output_csv.stat().st_size > 0)

        if resume and output_nonempty:
            processed_image_paths = _load_processed_image_paths_single_csv(output_csv, fieldnames)
            out_mode = "a"
            if processed_image_paths:
                print(
                    f"[resume] Loaded {len(processed_image_paths)} existing image_path values from {output_csv}; "
                    "will skip re-processing those images.",
                    flush=True,
                )
        else:
            # Legacy behavior: regenerate outputs when not resuming (or when file is empty).
            if output_exists and not resume:
                output_csv.unlink()
            # If resume was requested but the existing file is empty, we treat it like a fresh run
            # (write mode + header) without attempting to append.
            out_mode = "w"

        with output_csv.open(out_mode, newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            if out_mode == "w":
                writer.writeheader()
                f_out.flush()

            if use_noun_phrase_parser:
                # Noun-phrase parser path (optionally with LLM filtering).
                for row_idx, row in enumerate(
                    tqdm(
                        rows,
                        total=len(rows),
                        desc=f"{input_csv.name} (NP)",
                        leave=False,
                    ),
                    start=1,
                ):
                    rows_processed = row_idx
                    if resume and processed_image_paths and row.image_path in processed_image_paths:
                        continue

                    row_start_time = datetime.now()

                    objects_text = get_objects_from_caption_np(row.description)

                    # Apply LLM filtering if requested
                    if filter_with_llm and filter_llm_client is not None:
                        objects_text = filter_objects_with_llm(
                            objects_text,
                            filter_llm_client,
                            template_filter,
                        )

                    result_row = {
                        "image_path": row.image_path,
                        "description": row.description,
                        "objects": objects_text,
                    }
                    writer.writerow(result_row)
                    # Ensure outputs are visible on disk after each row.
                    try:
                        f_out.flush()
                    except Exception:
                        pass

                    row_end_time = datetime.now()
                    if progress_writer is not None and f_progress is not None:
                        duration = (row_end_time - row_start_time).total_seconds()
                        progress_writer.writerow(
                            {
                                "input_csv": str(input_csv),
                                "output_csv": str(output_csv),
                                "row_index": row_idx,
                                "image_path": row.image_path,
                                "cluster_id": cluster_id,
                                "start_time": row_start_time.isoformat(
                                    timespec="seconds"
                                ),
                                "end_time": row_end_time.isoformat(
                                    timespec="seconds"
                                ),
                                "duration_seconds": f"{duration:.2f}",
                                "avg_duration_per_image": f"{duration:.2f}",
                                "batch_size": 1,
                                "status": "completed",
                            }
                        )
                        f_progress.flush()
            else:
                # LLM-based extraction: batch captions for faster inference.
                progress_bar = tqdm(
                    total=len(rows),
                    desc=f"{input_csv.name} (LLM)",
                    leave=False,
                )

                for batch in _batched_enumerate(rows, batch_size):
                    # Filter out already-processed rows in resume mode (best-effort).
                    batch_to_process = (
                        [
                            (row_idx, row)
                            for (row_idx, row) in batch
                            if not (resume and processed_image_paths and row.image_path in processed_image_paths)
                        ]
                        if (resume and processed_image_paths)
                        else list(batch)
                    )

                    batch_start_time = datetime.now()

                    # Prepare prompts and max token estimates for this batch
                    if use_merged_prompt:
                        merged_prompts: list[str] = []
                        merged_max_tokens_list: list[int] = []

                        for row_idx, row in batch_to_process:
                            rows_processed = row_idx
                            prompt = build_prompt(row.description, template_merged)
                            merged_prompts.append(prompt)
                            merged_max_tokens_list.append(
                                _estimate_max_tokens_for_caption(row.description)
                            )

                        if llm_client is None:
                            raise RuntimeError(
                                "Internal error: merged prompt requested without an LLM client."
                            )

                        batch_max_tokens = (
                            max(merged_max_tokens_list)
                            if merged_max_tokens_list
                            else None
                        )
                        if batch_to_process:
                            merged_texts = llm_client.get_objects_from_captions_batch(
                                merged_prompts,
                                max_new_tokens=batch_max_tokens,
                            )

                            for (row_idx, row), merged_text in zip(batch_to_process, merged_texts):
                                result_row = {
                                    "image_path": row.image_path,
                                    "description": row.description,
                                    "objects_and_stuff": merged_text,
                                }
                                writer.writerow(result_row)
                                # Ensure outputs are visible on disk after each row.
                                try:
                                    f_out.flush()
                                except Exception:
                                    pass
                    else:
                        # Separate objects (and optional stuff) prompts
                        object_prompts: list[str] = []
                        object_max_tokens_list: list[int] = []
                        stuff_prompts: list[str] = []
                        stuff_max_tokens_list: list[int] = []

                        for row_idx, row in batch_to_process:
                            rows_processed = row_idx

                            prompt = build_prompt(row.description, template)
                            object_prompts.append(prompt)
                            object_max_tokens_list.append(
                                _estimate_max_tokens_for_caption(row.description)
                            )

                            if generate_stuff:
                                prompt_stuff = build_prompt(
                                    row.description, template_stuff
                                )
                                stuff_prompts.append(prompt_stuff)
                                stuff_max_tokens_list.append(
                                    _estimate_max_tokens_for_caption(row.description)
                                )

                        if llm_client is None:
                            raise RuntimeError(
                                "Internal error: LLM client not initialized in LLM mode."
                            )

                        objects_batch_max = (
                            max(object_max_tokens_list)
                            if object_max_tokens_list
                            else None
                        )
                        objects_texts = []
                        if batch_to_process:
                            objects_texts = llm_client.get_objects_from_captions_batch(
                                object_prompts,
                                max_new_tokens=objects_batch_max,
                            )

                        if generate_stuff:
                            stuff_batch_max = (
                                max(stuff_max_tokens_list)
                                if stuff_max_tokens_list
                                else None
                            )
                            stuff_texts = []
                            if batch_to_process:
                                stuff_texts = llm_client.get_objects_from_captions_batch(
                                    stuff_prompts,
                                    max_new_tokens=stuff_batch_max,
                                )
                        else:
                            stuff_texts = []

                        for batch_index, (row_idx, row) in enumerate(batch_to_process):
                            objects_text = (
                                objects_texts[batch_index]
                                if batch_index < len(objects_texts)
                                else ""
                            )

                            result_row = {
                                "image_path": row.image_path,
                                "description": row.description,
                                "objects": objects_text,
                            }

                            if generate_stuff:
                                stuff_text = (
                                    stuff_texts[batch_index]
                                    if batch_index < len(stuff_texts)
                                    else ""
                                )
                                result_row["stuff"] = stuff_text

                            writer.writerow(result_row)
                            # Ensure outputs are visible on disk after each row.
                            try:
                                f_out.flush()
                            except Exception:
                                pass

                    batch_end_time = datetime.now()
                    progress_bar.update(len(batch))

                    if progress_writer is not None and f_progress is not None:
                        batch_duration = (
                            batch_end_time - batch_start_time
                        ).total_seconds()
                        avg_per_image = (
                            batch_duration / len(batch) if len(batch) > 0 else 0.0
                        )
                        for row_idx, row in batch:
                            progress_writer.writerow(
                                {
                                    "input_csv": str(input_csv),
                                    "output_csv": str(output_csv),
                                    "row_index": row_idx,
                                    "image_path": row.image_path,
                                    "cluster_id": cluster_id,
                                    "start_time": batch_start_time.isoformat(
                                        timespec="seconds"
                                    ),
                                    "end_time": batch_end_time.isoformat(
                                        timespec="seconds"
                                    ),
                                    "duration_seconds": f"{batch_duration:.2f}",
                                    "avg_duration_per_image": f"{avg_per_image:.4f}",
                                    "batch_size": len(batch),
                                    "status": "completed",
                                }
                            )
                        f_progress.flush()

                progress_bar.close()
    finally:
        if f_progress is not None:
            f_progress.close()


def _resolve_image_path_from_dir(images_dir: Path, csv_image_path: str) -> Path:
    """
    Resolve an image referenced in a captions CSV to a file inside a flat directory.

    The captions CSV typically contains paths like:
      Images/London/<name>.jpg
    but the user provides a directory containing the actual image files (often flat).
    We therefore match by basename (and try a few common extensions).
    """
    images_dir = Path(images_dir)
    raw = str(csv_image_path).strip()
    if not raw:
        return images_dir / ""

    # 0) If CSV already contains an absolute path and it exists, use it.
    p_raw = Path(raw)
    if p_raw.is_absolute() and p_raw.is_file():
        return p_raw

    # 1) If CSV contains a relative path (possibly with subdirectories),
    # try resolving it under images_dir.
    # Example:
    #   images_dir=/mnt/d/data/gsv_cities
    #   csv_image_path=Images/London/foo.jpg
    #   => /mnt/d/data/gsv_cities/Images/London/foo.jpg
    rel_candidate = images_dir / raw.lstrip("/\\")
    if rel_candidate.is_file():
        return rel_candidate

    # 2) Fallback: match by basename inside a flat images_dir.
    base = p_raw.name
    if not base:
        return rel_candidate

    candidates: list[Path] = []
    # 1) Direct basename (keeps original extension)
    candidates.append(images_dir / base)
    # 2) Try common extensions by stem
    stem = Path(base).stem
    if stem:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidates.append(images_dir / f"{stem}{ext}")

    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return p

    # Final fallback: return the most likely path even if missing.
    #
    # NOTE: Callers should decide what to do when the returned path does not exist.
    # In images-dir mode we default to skipping missing images, so we avoid emitting
    # "made up" paths into the main output CSV.
    return images_dir / base


def run_images_dir_mode(
    *,
    images_dir: Path,
    captions_csv: Path,
    output_dir: Path,
    output_csv: Optional[Path],
    per_image_timing_csv: Optional[Path],
    model_name: str,
    prompt_template: Optional[str],
    prompt_template_stuff: Optional[str],
    prompt_template_merged: Optional[str],
    use_merged_prompt: bool,
    generate_stuff: bool,
    use_noun_phrase_parser: bool,
    filter_with_llm: bool,
    filter_model_name: str,
    filter_prompt_template: Optional[str],
    skip_existing_llm: bool = False,
    resume: bool = False,
    include_missing_images: bool = False,
    use_4bit: bool = False,
) -> None:
    """
    Work mode:
    - user provides a directory containing images (often flat, full of .jpg)
    - user provides a 2-column no-header CSV: <image_path>,<description>
    Output:
    - output CSV: full image path (resolved), description, objects (and optional stuff)
    - timing CSV: per-row duration, status, and resolved image path
    Both are flushed after each processed row.
    """
    import csv
    import time
    from datetime import datetime

    images_dir = Path(images_dir)
    captions_csv = Path(captions_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_csv is None:
        method_suffix = "np" if use_noun_phrase_parser else "llm"
        if use_noun_phrase_parser and filter_with_llm:
            method_suffix = "np_filtered"
        output_csv = captions_csv.with_name(f"{captions_csv.stem}_objects_{method_suffix}_v2.csv")

    if per_image_timing_csv is None:
        per_image_timing_csv = output_csv.with_name(f"{output_csv.stem}_timings.csv")

    # Optionally skip if this is an LLM run and output already exists.
    if skip_existing_llm and (not use_noun_phrase_parser) and output_csv.exists():
        print(f"Skipping because output already exists: {output_csv}")
        return

    # Resume mode: append and skip already-processed rows/images.
    # Default behavior remains: recreate outputs.
    def _load_resume_state(
        *,
        out_csv: Path,
        timing_csv: Path,
        expected_out_fieldnames: list,
        expected_timing_fieldnames: list,
    ) -> Tuple[int, Set[str]]:
        """
        Returns (max_row_index_seen, processed_image_paths).

        - max_row_index_seen is used to fast-skip the first N rows of the captions CSV
          so we truly "continue where we stopped" (best-effort).
        - processed_image_paths ensures we don't re-run the same image twice even if
          the timing CSV is incomplete or was truncated.
        """
        import csv as _csv

        max_row_idx = 0
        processed_paths: Set[str] = set()

        # 1) Load processed image_path values from the output CSV.
        if out_csv.is_file():
            with out_csv.open("r", newline="", encoding="utf-8") as f:
                r = _csv.DictReader(f)
                if r.fieldnames is None:
                    raise ValueError(f"Output CSV has no header: {out_csv}")
                if list(r.fieldnames) != list(expected_out_fieldnames):
                    raise ValueError(
                        "Output CSV header does not match expected columns for this run. "
                        "Refusing to --resume to avoid corrupting the file. "
                        f"Expected fieldnames={list(expected_out_fieldnames)} but found={list(r.fieldnames)}. "
                        "Delete the output CSV or rerun without --resume."
                    )
                for row in r:
                    p = (row.get("image_path") or "").strip()
                    if p:
                        processed_paths.add(p)

        # 2) Load max row_index from timing CSV (best-effort).
        if timing_csv.is_file():
            with timing_csv.open("r", newline="", encoding="utf-8") as f:
                r = _csv.DictReader(f)
                if r.fieldnames is None:
                    raise ValueError(f"Timing CSV has no header: {timing_csv}")
                if list(r.fieldnames) != list(expected_timing_fieldnames):
                    raise ValueError(
                        "Timing CSV header does not match expected columns for this run. "
                        "Refusing to --resume to avoid corrupting the file. "
                        f"Expected fieldnames={list(expected_timing_fieldnames)} but found={list(r.fieldnames)}. "
                        "Delete the timing CSV or rerun without --resume."
                    )
                for row in r:
                    v = (row.get("row_index") or "").strip()
                    if not v:
                        continue
                    try:
                        max_row_idx = max(max_row_idx, int(v))
                    except Exception:
                        continue

        return max_row_idx, processed_paths

    llm_client = None
    filter_llm_client = None

    if use_noun_phrase_parser:
        if filter_with_llm:
            filter_llm_client = _build_llm_client_or_die(
                model_name=filter_model_name,
                max_new_tokens=128,
                temperature=0.0,
                use_4bit=use_4bit,
            )
    else:
        llm_client = _build_llm_client_or_die(model_name=model_name, use_4bit=use_4bit)

    template = prompt_template or DEFAULT_PROMPT_TEMPLATE
    template_stuff = prompt_template_stuff or DEFAULT_PROMPT_TEMPLATE_STUFF
    template_merged = prompt_template_merged or DEFAULT_PROMPT_TEMPLATE_MERGED
    template_filter = filter_prompt_template or DEFAULT_FILTER_PROMPT_TEMPLATE

    if use_merged_prompt:
        out_fieldnames = ["image_path", "description", "objects_and_stuff"]
    else:
        out_fieldnames = ["image_path", "description", "objects"]
        if generate_stuff:
            out_fieldnames.append("stuff")

    timing_fieldnames = [
        "row_index",
        "csv_image_path",
        "resolved_image_path",
        "image_basename",
        "start_time",
        "end_time",
        "duration_seconds",
        "status",
        "error",
    ]

    # Auto-detect whether captions CSV has a header ("image_path,description,...").
    # If it does, use the robust DictReader-based parser; otherwise use the
    # 2-column no-header parser.
    import csv as _csv

    def _has_header_two_cols(path: Path) -> bool:
        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                r = _csv.reader(f)
                for row in r:
                    if not row:
                        continue
                    if len(row) < 2:
                        continue
                    a = str(row[0]).strip().lower()
                    b = str(row[1]).strip().lower()
                    return a == "image_path" and b in {"description", "caption", "text"}
        except Exception:
            return False
        return False

    if _has_header_two_cols(captions_csv):
        rows_iter = parse_caption_csv(captions_csv)
    else:
        rows_iter = parse_caption_csv_two_columns_no_header(captions_csv)

    processed_image_paths: Set[str] = set()
    resume_skip_rows = 0
    out_mode = "w"
    time_mode = "w"

    if resume and (output_csv.exists() or per_image_timing_csv.exists()):
        resume_skip_rows, processed_image_paths = _load_resume_state(
            out_csv=output_csv,
            timing_csv=per_image_timing_csv,
            expected_out_fieldnames=out_fieldnames,
            expected_timing_fieldnames=timing_fieldnames,
        )
        out_mode = "a" if output_csv.exists() else "w"
        time_mode = "a" if per_image_timing_csv.exists() else "w"
        if processed_image_paths:
            print(
                f"[resume] Loaded {len(processed_image_paths)} existing image_path values from {output_csv}; "
                "will skip re-processing those images.",
                flush=True,
            )
        if resume_skip_rows > 0:
            print(
                f"[resume] Will fast-skip the first {resume_skip_rows} rows based on {per_image_timing_csv}.",
                flush=True,
            )
    else:
        # Always recreate output files to avoid mixing old/new runs.
        if output_csv.exists():
            output_csv.unlink()
        if per_image_timing_csv.exists():
            per_image_timing_csv.unlink()

    with output_csv.open(out_mode, newline="", encoding="utf-8") as f_out, per_image_timing_csv.open(
        time_mode, newline="", encoding="utf-8"
    ) as f_time:
        out_writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
        if out_mode == "w":
            out_writer.writeheader()
            f_out.flush()

        time_writer = csv.DictWriter(f_time, fieldnames=timing_fieldnames)
        if time_mode == "w":
            time_writer.writeheader()
            f_time.flush()

        for row_idx, row in enumerate(
            tqdm(rows_iter, desc=f"{captions_csv.name} (images_dir mode)"), start=1
        ):
            if resume_skip_rows and row_idx <= resume_skip_rows:
                continue

            resolved = _resolve_image_path_from_dir(images_dir, row.image_path)
            # If resuming, never re-process an image_path already present in the output CSV.
            if resume and processed_image_paths:
                resolved_key = str(resolved)
                if resolved_key and resolved_key in processed_image_paths:
                    continue
            status = "completed"
            err_msg = ""

            # If the captions CSV contains images from other cities/datasets,
            # don't fabricate "resolved" paths under --images_dir.
            if (not include_missing_images) and (not resolved.is_file()):
                status = "missing_image"
                time_writer.writerow(
                    {
                        "row_index": row_idx,
                        "csv_image_path": row.image_path,
                        "resolved_image_path": str(resolved),
                        "image_basename": resolved.name,
                        "start_time": "",
                        "end_time": "",
                        "duration_seconds": "0.0000",
                        "status": status,
                        "error": "",
                    }
                )
                f_time.flush()
                continue

            start_dt = datetime.now()
            t0 = time.perf_counter()

            try:
                if use_noun_phrase_parser:
                    objects_text = get_objects_from_caption_np(row.description)
                    if filter_with_llm and filter_llm_client is not None:
                        objects_text = filter_objects_with_llm(
                            objects_text,
                            filter_llm_client,
                            template_filter,
                        )

                    out_writer.writerow(
                        {
                            "image_path": str(resolved),
                            "description": row.description,
                            "objects": objects_text,
                        }
                    )
                    f_out.flush()
                else:
                    if llm_client is None:
                        raise RuntimeError("Internal error: LLM client not initialized.")

                    if use_merged_prompt:
                        prompt = build_prompt(row.description, template_merged)
                        max_tokens = _estimate_max_tokens_for_caption(row.description)
                        merged_text = llm_client.get_objects_from_caption(
                            prompt, max_new_tokens=max_tokens
                        )
                        out_writer.writerow(
                            {
                                "image_path": str(resolved),
                                "description": row.description,
                                "objects_and_stuff": merged_text,
                            }
                        )
                        f_out.flush()
                    else:
                        prompt = build_prompt(row.description, template)
                        max_tokens = _estimate_max_tokens_for_caption(row.description)
                        objects_text = llm_client.get_objects_from_caption(
                            prompt, max_new_tokens=max_tokens
                        )

                        result_row: dict[str, str] = {
                            "image_path": str(resolved),
                            "description": row.description,
                            "objects": objects_text,
                        }

                        if generate_stuff:
                            prompt_s = build_prompt(row.description, template_stuff)
                            max_tokens_s = _estimate_max_tokens_for_caption(row.description)
                            stuff_text = llm_client.get_objects_from_caption(
                                prompt_s, max_new_tokens=max_tokens_s
                            )
                            result_row["stuff"] = stuff_text

                        out_writer.writerow(result_row)
                        f_out.flush()

                if not resolved.is_file():
                    status = "missing_image"
            except Exception as e:
                status = "error"
                err_msg = f"{type(e).__name__}: {e}"
                # Still emit a row so downstream tools can keep going.
                if use_merged_prompt:
                    out_writer.writerow(
                        {
                            "image_path": str(resolved),
                            "description": row.description,
                            "objects_and_stuff": "",
                        }
                    )
                else:
                    fallback_row: dict[str, str] = {
                        "image_path": str(resolved),
                        "description": row.description,
                        "objects": "",
                    }
                    if generate_stuff:
                        fallback_row["stuff"] = ""
                    out_writer.writerow(fallback_row)
                f_out.flush()

            end_dt = datetime.now()
            dur = time.perf_counter() - t0
            # Track progress for this run so we don't re-process duplicates even within
            # the same invocation (best-effort).
            if resume:
                processed_image_paths.add(str(resolved))

            time_writer.writerow(
                {
                    "row_index": row_idx,
                    "csv_image_path": row.image_path,
                    "resolved_image_path": str(resolved),
                    "image_basename": resolved.name,
                    "start_time": start_dt.isoformat(timespec="seconds"),
                    "end_time": end_dt.isoformat(timespec="seconds"),
                    "duration_seconds": f"{dur:.4f}",
                    "status": status,
                    "error": (err_msg[:2000] if err_msg else ""),
                }
            )
            f_time.flush()

    print(f"Wrote output CSV to: {output_csv}")
    print(f"Wrote per-image timing CSV to: {per_image_timing_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract detectable objects and segmentable regions from image captions using an LLM."
    )
    parser.add_argument(
        "--input_csv",
        help="Path to a single input CSV with image_path and description columns.",
    )
    parser.add_argument(
        "--input_dir",
        help=(
            "Root directory to recursively search for 'cluster_items.csv' files. "
            "For each match, the script will run on that CSV and save the output CSV "
            "next to it."
        ),
    )
    parser.add_argument(
        "--images_dir",
        default=None,
        help=(
            "Images directory mode: path to a directory full of image files (often flat, e.g. many .jpg). "
            "Used together with --captions_csv."
        ),
    )
    parser.add_argument(
        "--captions_csv",
        default=None,
        help=(
            "Images directory mode: path to a NO-HEADER, 2-column CSV:\n"
            "  <image_path>,<description>\n"
            "The image filename is taken from the first column basename and resolved inside --images_dir."
        ),
    )
    parser.add_argument(
        "--include_missing_images",
        action="store_true",
        help=(
            "Images directory mode: process rows even when the referenced image cannot be found under "
            "--images_dir. When not set (default), missing images are skipped so the output CSV "
            "only contains real files from --images_dir."
        ),
    )
    parser.add_argument(
        "--per_image_timing_csv",
        default=None,
        help=(
            "Images directory mode: optional CSV path for per-image timing logs. "
            "Defaults to <output_csv_stem>_timings.csv next to the output CSV."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume mode: if the output CSV already exists, append new rows and skip "
            "already-processed image_path values. This lets you rerun after interruption "
            "without duplicating work or truncating existing outputs."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory to store generated txt files with object lists. "
            "Required when using --input_csv. "
            "When using --input_dir, if provided, a mirrored subdirectory structure "
            "will be created under this directory; otherwise a local 'objects_debug' "
            "folder will be created next to each found CSV."
        ),
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help=(
            "Path to the output CSV. "
            "Defaults to <input_stem>_objects.csv next to the input CSV."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Hugging Face model identifier to use (local or from the hub). "
            "If not set, --model_preset selects a recommended model."
        ),
    )
    parser.add_argument(
        "--model_preset",
        choices=["quality", "fast"],
        default="quality",
        help=(
            "Quality/speed tradeoff preset used when --model is not provided. "
            "'quality' → microsoft/Phi-3.5-mini-instruct (best quality, default). "
            "'fast' → Qwen/Qwen2.5-1.5B-Instruct (smaller and faster, good quality; not gated)."
        ),
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help=(
            "Use 4-bit quantization (bitsandbytes) to reduce GPU memory usage by ~75%%. "
            "Reduces Phi-3.5-mini memory from ~7.6GB to ~2GB. "
            "Requires: pip install bitsandbytes accelerate"
        ),
    )
    parser.add_argument(
        "--use_noun_phrase_parser",
        action="store_true",
        help=(
            "Use a traditional noun-phrase parsing based method (spaCy) instead of "
            "an LLM for extracting detectable objects."
        ),
    )
    parser.add_argument(
        "--prompt_template",
        default=None,
        help=(
            "Optional custom prompt template for objects (things). "
            "Use {caption} as a placeholder for the caption text."
        ),
    )
    parser.add_argument(
        "--prompt_template_stuff",
        default=None,
        help=(
            "Optional custom prompt template for stuff (uncountable regions). "
            "Use {caption} as a placeholder for the caption text."
        ),
    )
    parser.add_argument(
        "--prompt_template_merged",
        default=None,
        help=(
            "Optional custom prompt template for merged objects and stuff. "
            "Use {caption} as a placeholder for the caption text."
        ),
    )
    parser.add_argument(
        "--use_merged_prompt",
        action="store_true",
        help=(
            "Use a merged prompt that generates both objects and stuff in a single call. "
            "Output will be in one column 'objects_and_stuff' instead of separate columns."
        ),
    )
    parser.add_argument(
        "--generate_stuff",
        action="store_true",
        help="Also generate a list of 'stuff' (uncountable regions) for segmentation.",
    )
    parser.add_argument(
        "--filter_with_llm",
        action="store_true",
        help=(
            "When using --use_noun_phrase_parser, apply an additional LLM filtering step "
            "to remove non-detectable objects from the noun phrase parser output. "
            "This provides a hybrid approach: fast NP extraction + accurate LLM filtering."
        ),
    )
    parser.add_argument(
        "--filter_model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help=(
            "Hugging Face model identifier to use for filtering (when --filter_with_llm is enabled). "
            "Default: Qwen/Qwen2.5-3B-Instruct (recommended for quality and speed)."
        ),
    )
    parser.add_argument(
        "--filter_prompt_template",
        default=None,
        help=(
            "Optional custom prompt template for LLM filtering. "
            "Use {object_list} as a placeholder for the list of objects to filter."
        ),
    )
    parser.add_argument(
        "--filter_existing_np",
        action="store_true",
        help=(
            "Filter existing *_objects_np.csv files by adding a 'filtered_by_llm' column. "
            "Use with --input_csv to process a single CSV, or with --input_dir to find and "
            "process all *_objects_np.csv files recursively. Requires --filter_model."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "Number of captions to send to the LLM together in a single batch "
            "for faster inference (LLM modes only). Must be a positive integer. "
            "Default: 1 (no batching)."
        ),
    )
    parser.add_argument(
        "--realtime_progress_csv",
        default=None,
        help=(
            "Path to a real-time progress tracking CSV file. This file is updated "
            "immediately after each cluster CSV is processed. "
            "If not provided, defaults to 'objects_realtime_progress.csv' "
            "in the input directory (directory scan mode) or next to the input CSV (single mode)."
        ),
    )
    parser.add_argument(
        "--skip_existing_llm",
        action="store_true",
        help=(
            "When using LLM extraction (non-noun-phrase modes), if the target "
            "*_objects_llm.csv already exists, skip processing that input CSV "
            "instead of overwriting it."
        ),
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch_size must be a positive integer.")

    # Resolve effective main LLM model for object extraction (LLM-only modes).
    # Priority:
    #   1) Explicit --model, if provided
    #   2) Preset choice via --model_preset
    if args.model is not None:
        effective_model = args.model
    else:
        if args.model_preset == "quality":
            effective_model = "microsoft/Phi-3.5-mini-instruct"
        elif args.model_preset == "fast":
            effective_model = "Qwen/Qwen2.5-1.5B-Instruct"
        else:  # pragma: no cover - defensive, argparse enforces choices
            parser.error(f"Unknown --model_preset value: {args.model_preset}")

    # Mode validation: noun-phrase parser currently only supports the objects column.
    if args.use_noun_phrase_parser and args.use_merged_prompt:
        parser.error(
            "--use_noun_phrase_parser cannot be combined with --use_merged_prompt; "
            "noun-phrase parsing currently only supports extracting objects."
        )
    if args.use_noun_phrase_parser and args.generate_stuff:
        parser.error(
            "--use_noun_phrase_parser cannot be combined with --generate_stuff; "
            "noun-phrase parsing currently only supports extracting objects."
        )
    if args.filter_with_llm and not args.use_noun_phrase_parser:
        parser.error(
            "--filter_with_llm can only be used with --use_noun_phrase_parser; "
            "it filters the output of the noun phrase parser."
        )
    if args.filter_existing_np:
        # Validate filter_existing_np mode
        if args.use_noun_phrase_parser:
            parser.error(
                "--filter_existing_np cannot be combined with --use_noun_phrase_parser; "
                "use --filter_existing_np to filter already-processed NP CSV files."
            )
        if args.filter_with_llm:
            parser.error(
                "--filter_existing_np cannot be combined with --filter_with_llm; "
                "filtering is automatic in --filter_existing_np mode."
            )
        if args.use_merged_prompt or args.generate_stuff:
            parser.error(
                "--filter_existing_np only works with object lists from NP parsing."
            )
    # Validate input sources (3 mutually exclusive modes):
    #   1) --input_csv (headered CSV)
    #   2) --input_dir (directory scan for cluster_items.csv)
    #   3) --images_dir + --captions_csv (flat images dir + no-header captions CSV)
    images_dir_mode = (args.images_dir is not None) or (args.captions_csv is not None)
    if images_dir_mode:
        if args.images_dir is None or args.captions_csv is None:
            parser.error("Images directory mode requires BOTH --images_dir and --captions_csv.")
        if args.input_csv is not None or args.input_dir is not None:
            parser.error(
                "Images directory mode (--images_dir/--captions_csv) cannot be combined with "
                "--input_csv or --input_dir."
            )
        if args.batch_size != 1 and not args.use_noun_phrase_parser:
            parser.error(
                "Images directory mode requires --batch_size=1 in LLM mode so output/log files "
                "can be updated after every image."
            )
    else:
        if (args.input_csv is None) and (args.input_dir is None):
            parser.error("You must specify either --input_csv, --input_dir, or (--images_dir and --captions_csv).")
        if (args.input_csv is not None) and (args.input_dir is not None):
            parser.error("Please specify only one of --input_csv or --input_dir, not both.")

    # Filter existing NP CSV mode
    if args.filter_existing_np:
        # Initialize LLM for filtering
        filter_llm_client = _build_llm_client_or_die(
            model_name=args.filter_model,
            max_new_tokens=128,
            temperature=0.0,
        )
        template_filter = args.filter_prompt_template or DEFAULT_FILTER_PROMPT_TEMPLATE
        
        if args.input_csv is not None:
            # Single CSV mode
            csv_path = Path(args.input_csv)
            if not csv_path.exists():
                parser.error(f"Input CSV not found: {csv_path}")
            if not str(csv_path).endswith("_objects_np.csv"):
                parser.error(
                    f"With --filter_existing_np, --input_csv must be a *_objects_np.csv file. "
                    f"Got: {csv_path}"
                )
            filter_existing_np_csv(csv_path, filter_llm_client, template_filter)
        else:
            # Directory scan mode
            root_dir = Path(args.input_dir)
            if not root_dir.is_dir():
                parser.error(f"--input_dir '{root_dir}' is not a directory.")
            
            # Find all *_objects_np.csv files
            matched_files = list(root_dir.rglob("*_objects_np.csv"))
            
            if not matched_files:
                raise SystemExit(
                    f"No '*_objects_np.csv' files found under directory '{root_dir}'.\n"
                    f"Run with --use_noun_phrase_parser first to generate these files."
                )
            
            print(f"Found {len(matched_files)} *_objects_np.csv file(s) to process\n")
            
            for csv_path in matched_files:
                filter_existing_np_csv(csv_path, filter_llm_client, template_filter)
            
            print(f"✓ Processed {len(matched_files)} file(s) successfully!")
        
        return  # Exit after processing

    # Images directory mode
    if images_dir_mode:
        images_dir = Path(args.images_dir)
        captions_csv = Path(args.captions_csv)
        if not images_dir.is_dir():
            parser.error(f"--images_dir '{images_dir}' is not a directory.")
        if not captions_csv.is_file():
            parser.error(f"--captions_csv '{captions_csv}' is not a file.")

        # Output dir defaults: next to captions CSV
        out_dir = Path(args.output_dir) if args.output_dir is not None else (captions_csv.parent / "objects_debug")

        run_images_dir_mode(
            images_dir=images_dir,
            captions_csv=captions_csv,
            output_dir=out_dir,
            output_csv=Path(args.output_csv) if args.output_csv else None,
            per_image_timing_csv=Path(args.per_image_timing_csv) if args.per_image_timing_csv else None,
            model_name=effective_model,
            prompt_template=args.prompt_template,
            prompt_template_stuff=args.prompt_template_stuff,
            prompt_template_merged=args.prompt_template_merged,
            use_merged_prompt=args.use_merged_prompt,
            generate_stuff=args.generate_stuff,
            use_noun_phrase_parser=args.use_noun_phrase_parser,
            filter_with_llm=args.filter_with_llm,
            filter_model_name=args.filter_model,
            filter_prompt_template=args.filter_prompt_template,
            skip_existing_llm=args.skip_existing_llm,
            resume=args.resume,
            include_missing_images=args.include_missing_images,
            use_4bit=args.use_4bit,
        )
        return

    # Single CSV mode
    if args.input_csv is not None:
        if args.output_dir is None:
            parser.error("--output_dir is required when using --input_csv.")
        if args.input_dir is not None:
            parser.error("--input_dir cannot be used together with --input_csv.")

        # Determine realtime progress CSV path for single-CSV mode
        input_csv_path = Path(args.input_csv)
        if args.realtime_progress_csv:
            realtime_progress_path = Path(args.realtime_progress_csv)
        else:
            realtime_progress_path = input_csv_path.with_name("objects_realtime_progress.csv")

        # Prevent two concurrent runs from writing the same output CSV (which can
        # otherwise create duplicated/interleaved rows).
        #
        # In the full pipeline we always pass --output_csv explicitly, so we lock on that.
        out_csv_for_lock = Path(args.output_csv).resolve() if args.output_csv else None
        if out_csv_for_lock is not None:
            with FileLock(out_csv_for_lock):
                run(
                    input_csv=input_csv_path,
                    output_dir=Path(args.output_dir),
                    output_csv=Path(args.output_csv) if args.output_csv else None,
                    model_name=effective_model,
                    prompt_template=args.prompt_template,
                    prompt_template_stuff=args.prompt_template_stuff,
                    prompt_template_merged=args.prompt_template_merged,
                    use_merged_prompt=args.use_merged_prompt,
                    generate_stuff=args.generate_stuff,
                    use_noun_phrase_parser=args.use_noun_phrase_parser,
                    filter_with_llm=args.filter_with_llm,
                    filter_model_name=args.filter_model,
                    filter_prompt_template=args.filter_prompt_template,
                    batch_size=args.batch_size,
                    realtime_progress_csv=realtime_progress_path,
                    skip_existing_llm=args.skip_existing_llm,
                    resume=args.resume,
                    use_4bit=args.use_4bit,
                )
        else:
            run(
                input_csv=input_csv_path,
                output_dir=Path(args.output_dir),
                output_csv=Path(args.output_csv) if args.output_csv else None,
                model_name=effective_model,
                prompt_template=args.prompt_template,
                prompt_template_stuff=args.prompt_template_stuff,
                prompt_template_merged=args.prompt_template_merged,
                use_merged_prompt=args.use_merged_prompt,
                generate_stuff=args.generate_stuff,
                use_noun_phrase_parser=args.use_noun_phrase_parser,
                filter_with_llm=args.filter_with_llm,
                filter_model_name=args.filter_model,
                filter_prompt_template=args.filter_prompt_template,
                batch_size=args.batch_size,
                realtime_progress_csv=realtime_progress_path,
                skip_existing_llm=args.skip_existing_llm,
                resume=args.resume,
                use_4bit=args.use_4bit,
            )
        return

    # Directory scan mode
    if args.output_csv is not None:
        parser.error("--output_csv can only be used together with --input_csv.")

    root_dir = Path(args.input_dir)
    if not root_dir.is_dir():
        parser.error(f"--input_dir '{root_dir}' is not a directory.")

    base_output_dir = Path(args.output_dir) if args.output_dir is not None else None

    # Determine realtime progress CSV path for directory scan mode
    if args.realtime_progress_csv:
        realtime_progress_path = Path(args.realtime_progress_csv)
    else:
        realtime_progress_path = root_dir / "objects_realtime_progress.csv"

    # Clear the realtime progress CSV to start fresh (it's opened in append mode by run())
    if realtime_progress_path.exists():
        realtime_progress_path.unlink()
    
    print(f"Real-time progress CSV will be saved to: {realtime_progress_path}")

    # Look for CSV files named 'cluster_items.csv'
    matched_any = False
    for csv_path in root_dir.rglob("cluster_items.csv"):
        matched_any = True
        if base_output_dir is not None:
            # Mirror the directory structure of input_dir under output_dir
            rel_parent = csv_path.parent.relative_to(root_dir)
            per_csv_output_dir = base_output_dir / rel_parent
        else:
            # Default: create a local debug directory next to each CSV
            per_csv_output_dir = csv_path.parent / "objects_debug"

        run(
            input_csv=csv_path,
            output_dir=per_csv_output_dir,
            output_csv=None,  # let run place <input_stem>_objects.csv next to the CSV
            model_name=effective_model,
            prompt_template=args.prompt_template,
            prompt_template_stuff=args.prompt_template_stuff,
            prompt_template_merged=args.prompt_template_merged,
            use_merged_prompt=args.use_merged_prompt,
            generate_stuff=args.generate_stuff,
            use_noun_phrase_parser=args.use_noun_phrase_parser,
            filter_with_llm=args.filter_with_llm,
            filter_model_name=args.filter_model,
            filter_prompt_template=args.filter_prompt_template,
            batch_size=args.batch_size,
            realtime_progress_csv=realtime_progress_path,
            skip_existing_llm=args.skip_existing_llm,
            use_4bit=args.use_4bit,
        )

    if not matched_any:
        raise SystemExit(
            f"No 'cluster_items.csv' files found under directory '{root_dir}'."
        )


if __name__ == "__main__":
    main()


