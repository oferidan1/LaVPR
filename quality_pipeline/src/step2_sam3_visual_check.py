import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Any
import os
import shlex
import shutil
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.utils.file_lock import FileLock

csv.field_size_limit(sys.maxsize)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    # Only for type hints; actual imports are done lazily in load_model().
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401


_CLIP_BPE_DEFAULT_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"


def _resolve_sam3_bpe_path() -> Path:
    """
    Resolve a readable path to the CLIP BPE vocab required by the SAM3 text encoder.

    We prefer an explicit env var if the user has one, otherwise use a cache
    location under ~/.cache so we don't need to write into site-packages.
    """
    # 1) Explicit override via env var(s)
    for k in ("SAM3_BPE_PATH", "CLIP_BPE_PATH", "BPE_PATH"):
        v = (os.environ.get(k) or "").strip()
        if v:
            p = Path(v).expanduser()
            if p.is_file():
                return p

    # 2) Default cache location
    return Path.home() / ".cache" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"


def _ensure_clip_bpe_file(path: Path) -> None:
    """
    Best-effort: download the CLIP BPE vocab file if it does not exist.

    This avoids fragile behavior when the installed `sam3` package does not ship
    the expected `assets/bpe_simple_vocab_16e6.txt.gz` file.
    """
    if path.is_file():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    # Stream download to avoid holding the whole file in memory.
    import urllib.request

    try:
        with urllib.request.urlopen(_CLIP_BPE_DEFAULT_URL, timeout=30) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(path)
    finally:
        try:
            if tmp.exists() and not path.exists():
                tmp.unlink()
        except Exception:
            pass


@dataclass
class DetectionRow:
    image_path: str
    description: str
    objects_raw: str
    stuff_raw: str = ""


def parse_input_csv(
    csv_path: Path,
    use_filtered: bool = False,
) -> Iterator[DetectionRow]:
    """
    Parse an input CSV produced by the caption-to-objects pipeline.

    Supported schemas:
    - Merged schema (LLM merged mode):
      - image_path
      - description
      - objects_and_stuff
      - optional: stuff
    - Split schema (LLM or noun-phrase non-merged mode):
      - image_path
      - description
      - objects
      - optional: stuff
    - Filtered schema (when use_filtered=True):
      - image_path
      - description
      - objects
      - filtered_by_llm  (used as the objects source)

    Args:
        csv_path: Path to the input CSV file.
        use_filtered: If True, use the 'filtered_by_llm' column as the objects source.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")

        fieldnames = set(reader.fieldnames)

        # Basic required columns common to all supported schemas.
        base_required = {"image_path", "description"}
        missing_base = base_required.difference(fieldnames)
        if missing_base:
            raise ValueError(
                f"Missing required columns in CSV {csv_path}: {sorted(missing_base)}. "
                f"Found columns: {reader.fieldnames}"
            )

        # Determine which object schema this CSV uses.
        if use_filtered:
            if "filtered_by_llm" not in fieldnames:
                raise ValueError(
                    f"CSV {csv_path} must contain a 'filtered_by_llm' column when "
                    f"--filtered flag is used. Found columns: {reader.fieldnames}"
                )
            schema = "filtered"
        elif "objects_and_stuff" in fieldnames:
            schema = "merged"
        elif "objects" in fieldnames:
            schema = "split"
        else:
            raise ValueError(
                f"CSV {csv_path} must contain either an 'objects_and_stuff' column "
                f"(merged schema) or an 'objects' column (split schema). "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            image_path = row.get("image_path") or ""
            description = row.get("description") or ""
            if schema == "merged":
                objects_raw = row.get("objects_and_stuff") or ""
            elif schema == "filtered":
                objects_raw = row.get("filtered_by_llm") or ""
            else:
                objects_raw = row.get("objects") or ""
            # Optional 'stuff' column is supported in both schemas.
            stuff_raw = row.get("stuff") or ""
            if not image_path:
                # Skip malformed rows instead of failing the whole job
                continue
            yield DetectionRow(
                image_path=str(image_path),
                description=str(description),
                objects_raw=str(objects_raw),
                stuff_raw=str(stuff_raw),
            )


def parse_objects_field(objects_text: str) -> List[str]:
    """
    Parse the 'objects' column into a list of object names.

    The default LLM-based pipeline uses '. ' as a separator, but we
    also accept commas and semicolons for robustness.
    """
    if not objects_text:
        return []

    normalized = objects_text.replace(";", ".").replace(",", ".")
    parts = [p.strip() for p in normalized.split(".") if p.strip()]
    return parts


def load_image(image_path: str):
    """Load an image for SAM3."""
    image_pil = Image.open(image_path).convert("RGB")
    return image_pil


def load_model(
    device: str,
    confidence_threshold: float = 0.2,
    checkpoint_path: Optional[str] = None,
    verbose: bool = False,
) -> tuple[torch.nn.Module, "Sam3Processor"]:
    """Load a SAM3 model and processor."""
    # Import SAM3 lazily so `--help` (and other non-inference operations) work
    # even when SAM3 dependencies aren't installed.
    try:
        # Prefer the installed sam3 package (with its compiled extensions).
        # This avoids issues where importing directly from the source repo bypasses the
        # compiled extensions and leads to runtime errors.
        from sam3.model_builder import build_sam3_image_model  # type: ignore
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
    except ModuleNotFoundError:
        # Fallback: attempt to use a sibling sam3 repo if the package
        # is not installed in the current environment.
        _THIS_FILE = Path(__file__).resolve()
        _ROOT_DIR = _THIS_FILE.parents[2]  # /mnt/d/dan/git_projects
        _SAM3_DIR = _ROOT_DIR / "sam3"
        if not _SAM3_DIR.is_dir():
            raise

        # Add the repo itself to sys.path
        sys.path.insert(0, str(_SAM3_DIR))

        from sam3.model_builder import build_sam3_image_model  # type: ignore
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore

    # SAM3 needs a CLIP BPE vocab file. Some `sam3` installations do not package
    # it under site-packages/assets, so we resolve (and if needed download) a copy.
    bpe_path = _resolve_sam3_bpe_path()
    try:
        _ensure_clip_bpe_file(bpe_path)
    except Exception as e:
        raise SystemExit(
            "SAM3 failed to locate the required tokenizer vocab file:\n"
            f"  expected/using: {bpe_path}\n\n"
            "Fix options:\n"
            "  1) Set SAM3_BPE_PATH to an existing bpe_simple_vocab_16e6.txt.gz\n"
            f"  2) Ensure this URL is reachable so it can auto-download:\n"
            f"       {_CLIP_BPE_DEFAULT_URL}\n\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    try:
        model = build_sam3_image_model(
            device=device,
            eval_mode=True,
            bpe_path=str(bpe_path),
            checkpoint_path=checkpoint_path,
        )
    except Exception as e:
        raise SystemExit(
            "Failed to build/load the SAM3 model.\n\n"
            "Common causes:\n"
            "  - The Hugging Face repo 'facebook/sam3' is gated and requires authentication.\n"
            "    Fix: request access on Hugging Face, then run `huggingface-cli login` (or set HF_TOKEN).\n"
            "  - No internet access / proxy issues.\n"
            "    Fix: download the checkpoint once on a machine with access and pass --checkpoint_path.\n\n"
            "Fix options:\n"
            "  1) Provide a local checkpoint:\n"
            "       --checkpoint_path /path/to/sam3.pt\n"
            "  2) Authenticate to HF (if you have access):\n"
            "       huggingface-cli login\n\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e
    processor = Sam3Processor(
        model=model, device=device, confidence_threshold=confidence_threshold
    )
    if verbose:
        print(f"Loaded SAM3 model on device: {device}")
    return model, processor


def run_sam3_for_object(
    processor: "Sam3Processor",
    image_pil: Image.Image,
    inference_state: dict,
    object_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run SAM3 for a single object name on an image.

    Args:
        processor: SAM3 processor instance
        image_pil: PIL Image
        inference_state: State dict from set_image (will be reused)
        object_name: Name of the object to detect

    Returns:
        boxes: (N, 4) tensor in xyxy absolute coordinates.
        scores: (N,) tensor of confidence scores in [0, 1].
        masks: (N, H, W) tensor of boolean masks.
    """
    prompt = object_name.lower().strip()

    # Use the existing inference state (image already set)
    # Just update the text prompt
    output_state = processor.set_text_prompt(prompt=prompt, state=inference_state)

    boxes = output_state["boxes"]  # (N, 4) in xyxy format
    scores = output_state["scores"]  # (N,) tensor
    masks = output_state["masks"]  # (N, H, W) boolean tensor

    # If there are multiple detections for the same object above the threshold,
    # keep only the highest-confidence one (top-1) for both output and debug.
    if scores is not None and scores.numel() > 1:
        top_idx = int(torch.argmax(scores).item())
        boxes = boxes[top_idx : top_idx + 1]
        scores = scores[top_idx : top_idx + 1]
        if masks is not None:
            masks = masks[top_idx : top_idx + 1]

    return boxes, scores, masks


def create_color_map(
    labels: List[str],
    box_width: int = 300,
    item_height: int = 30,
) -> Image.Image:
    """
    Create a legend/map image showing color-to-object mappings.
    
    Args:
        labels: List of label strings (e.g., "object_name (0.85)")
        box_width: Width of the legend image
        item_height: Height of each item in the legend
    
    Returns:
        PIL Image with the color map/legend
    """
    if not labels:
        # Return a small empty image
        return Image.new("RGB", (box_width, 50), color="white")
    
    # Get unique labels (in case of duplicates)
    unique_labels = []
    seen = set()
    for label in labels:
        if label not in seen:
            unique_labels.append(label)
            seen.add(label)
    
    # Calculate image height
    padding = 20
    legend_height = len(unique_labels) * item_height + 2 * padding
    
    # Create legend image
    legend_img = Image.new("RGB", (box_width, legend_height), color="white")
    draw = ImageDraw.Draw(legend_img)
    
    # Try to load a font
    try:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        font = None
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, 14)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    
    # Draw title
    title = "Color Map"
    if hasattr(draw, "textbbox"):
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_w = title_bbox[2] - title_bbox[0]
    else:
        title_w, _ = draw.textsize(title, font=font)
    draw.text(((box_width - title_w) // 2, 5), title, fill="black", font=font)
    
    # Draw each item
    color_box_size = 25
    text_x = color_box_size + 15
    y_offset = padding + 25
    
    for idx, label in enumerate(unique_labels):
        y_pos = y_offset + idx * item_height
        
        # Generate the same consistent color as used in the main image
        np.random.seed(hash(label) % (2**32))
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        
        # Draw color box
        color_box = [
            padding,
            y_pos,
            padding + color_box_size,
            y_pos + color_box_size,
        ]
        draw.rectangle(color_box, fill=color, outline="black", width=2)
        
        # Draw label text
        draw.text((text_x, y_pos + 5), label, fill="black", font=font)
    
    return legend_img


def draw_segments_only(
    image_pil: Image.Image,
    boxes: torch.Tensor,
    labels: List[str],
    masks: Optional[torch.Tensor] = None,
) -> Image.Image:
    """
    Draw only the segmentation masks on the image (no labels, no boxes).
    Each segment gets a unique color that matches the color map.

    Args:
        image_pil: Original image.
        boxes: (N, 4) tensor in xyxy format (not used, but kept for compatibility).
        labels: List of N strings (used to generate consistent colors).
        masks: Optional (N, H, W) tensor of boolean masks for segmentation.

    Returns:
        PIL Image with colored segments overlaid.
    """
    if len(boxes) == 0:
        return image_pil

    # Convert to numpy for mask overlay
    img_np = np.array(image_pil).copy()
    H, W = img_np.shape[:2]

    # Draw masks only
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        if masks is not None and idx < masks.shape[0]:
            mask = masks[idx].detach().cpu().numpy()
            # Handle different mask shapes
            if mask.ndim == 3:
                mask = mask[0]
            elif mask.ndim != 2:
                continue
            
            # Ensure mask matches image dimensions
            if mask.shape != (H, W):
                from torch.nn.functional import interpolate
                mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                mask_resized = interpolate(
                    mask_tensor, size=(H, W), mode="bilinear", align_corners=False
                )
                mask = (mask_resized.squeeze().numpy() > 0.5).astype(bool)
            else:
                mask = mask.astype(bool)
            
            # Generate a consistent color for this detection (same as in color map)
            np.random.seed(hash(label) % (2**32))
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            color_np = np.array(color, dtype=np.uint8)
            
            # Overlay mask with transparency
            alpha = 0.5  # Slightly more opaque for better visibility
            img_np[mask] = (alpha * color_np + (1 - alpha) * img_np[mask]).astype(np.uint8)

    # Convert back to PIL
    result_img = Image.fromarray(img_np)
    return result_img


def prepare_clean_directory(dir_path: Path) -> None:
    """
    Ensure dir_path exists and is empty.
    If it already exists, remove all its contents (files and subdirectories).
    """
    if dir_path.exists():
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Expected a directory for debug output: {dir_path}")
        for item in dir_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        dir_path.mkdir(parents=True, exist_ok=True)


def wait_for_file(path: Path, timeout_sec: int = 900, poll_interval: float = 2.0) -> bool:
    """
    Wait for a specific file to appear on disk.

    Returns True if the file was found within the timeout, False otherwise.
    """
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        if path.is_file():
            return True
        time.sleep(poll_interval)
    return False


def maybe_load_cluster_index_map(csv_dir: Path) -> dict[tuple[str, str], str]:
    """
    If a 'cluster_items.csv' file exists in csv_dir, load a mapping from
    (image_path, description) -> idx_in_cluster.

    This lets us propagate idx_in_cluster into the per-object SAM3 CSV.
    """
    cluster_items_path = csv_dir / "cluster_items.csv"
    if not cluster_items_path.is_file():
        return {}

    mapping: dict[tuple[str, str], str] = {}

    with cluster_items_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}

        required = {"image_path", "description", "idx_in_cluster"}
        missing = required.difference(reader.fieldnames)
        if missing:
            # If the expected columns are not present, silently skip mapping.
            return {}

        for row in reader:
            img_path = (row.get("image_path") or "").strip()
            desc = (row.get("description") or "").strip()
            idx = row.get("idx_in_cluster")
            if not img_path or idx is None:
                continue
            mapping[(img_path, desc)] = str(idx)

    return mapping


def _load_processed_image_paths(output_csv: Path) -> set[str]:
    """
    Load a set of image_path values that already exist in a SAM3 output CSV.

    This enables resume/append mode without duplicating work.
    """
    if not output_csv.is_file():
        return set()

    processed: set[str] = set()
    try:
        with output_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "image_path" not in reader.fieldnames:
                return set()
            for row in reader:
                img = (row.get("image_path") or "").strip()
                if img:
                    processed.add(img)
    except Exception:  # noqa: BLE001 - best-effort; resume should not hard-fail
        return set()

    return processed


def _debug_file_prefix(row: DetectionRow, ordinal: int) -> str:
    """
    Create a short, stable-ish prefix for debug image filenames.
    Uses a truncated hash to avoid filename collisions across resume runs.
    """
    stem = Path(row.image_path).stem or "image"
    h = hashlib.sha1(row.image_path.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{ordinal:06d}_{stem}_{h}"


def _stable_image_id(image_path: str) -> str:
    """
    Stable id for an image_path to use in filenames (independent of row ordering).
    """
    stem = Path(image_path).stem or "image"
    h = hashlib.sha1(image_path.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{stem}_{h}"


def _write_sam3_debug_txt(
    *,
    base_dir: Path,
    cluster_dir: str,
    image_path: str,
    description: str,
    objects_raw: str,
    objects_not_found: list[str],
    detected_objects: list[str],
    detected_confidences: list[float],
    extra_lines: Optional[list[str]] = None,
) -> str:
    """
    Write a small per-row debug text file capturing SAM3 results so we can visualize later.

    Returns the path (as a string) to the written .txt file.
    """
    safe_cluster = (cluster_dir or "unknown_cluster").strip() or "unknown_cluster"
    out_dir = (base_dir / "sam3_text_outputs" / safe_cluster)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{_stable_image_id(image_path)}.txt"
    lines: list[str] = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"image_path: {image_path}",
        f"description: {description}",
        f"objects_raw: {objects_raw}",
        "",
        "detected_objects_json: " + json.dumps(detected_objects, ensure_ascii=False),
        "detected_confidences_json: "
        + json.dumps([round(float(x), 6) for x in detected_confidences], ensure_ascii=False),
        "objects_not_found_json: " + json.dumps(objects_not_found, ensure_ascii=False),
    ]
    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out_path)


def run(
    input_csv: Path,
    output_csv: Optional[Path],
    images_root: Optional[Path],
    box_threshold: float,
    cpu_only: bool,
    checkpoint_path: Optional[Path],
    debug: bool,
    debug_dir: Optional[Path],
    max_objects_per_image: int = 0,
    max_stuff_per_image: int = 0,
    no_overwrite: bool = False,
    resume: bool = False,
    use_filtered: bool = False,
    realtime_missing_csv: Optional[Path] = None,
    realtime_progress_csv: Optional[Path] = None,
    follow: bool = False,
    poll_interval: float = 5.0,
    follow_idle_minutes: Optional[int] = None,
    watch: bool = False,
    watch_poll_sec: int = 60,
    watch_idle_minutes: int = 20,
) -> None:
    # Determine suffix based on whether we're using filtered objects.
    suffix = "_sam3_filtered" if use_filtered else "_sam3"

    if output_csv is None:
        output_csv = input_csv.with_name(input_csv.stem + suffix + ".csv")

    # Normalize debug directory if debugging is enabled.
    if debug:
        if debug_dir is None:
            debug_suffix = "_sam3_filtered_debug" if use_filtered else "_sam3_debug"
            debug_dir = input_csv.with_name(input_csv.stem + debug_suffix)

    # If requested, skip processing when outputs already exist.
    if no_overwrite:
        output_exists = output_csv.is_file()
        debug_exists = bool(debug and debug_dir is not None and debug_dir.exists())
        if output_exists or debug_exists:
            print(
                f"[INFO] Skipping input CSV {input_csv} because "
                f"output CSV and/or debug directory already exist and --no_overwrite is set."
            )
            # Optionally log this skip to the realtime progress CSV.
            if realtime_progress_csv is not None:
                realtime_progress_fieldnames = [
                    "timestamp",
                    "input_csv",
                    "cluster_dir",
                    "image_path",
                    "idx_in_cluster",
                    "description",
                    "objects",
                    "objects_not_found",
                    "duration_sec",
                    "status",
                ]
                file_exists = realtime_progress_csv.is_file()
                with realtime_progress_csv.open(
                    "a", newline="", encoding="utf-8"
                ) as prog_f:
                    prog_writer = csv.DictWriter(
                        prog_f, fieldnames=realtime_progress_fieldnames
                    )
                    if not file_exists:
                        prog_writer.writeheader()

                    # Infer cluster directory name (e.g., 'cluster_32') from the CSV path.
                    cluster_dir = ""
                    for parent in input_csv.parents:
                        name = parent.name
                        if name.startswith("cluster_"):
                            cluster_dir = name
                            break
                    if not cluster_dir:
                        cluster_dir = input_csv.parent.name

                    prog_writer.writerow(
                        {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "input_csv": str(input_csv),
                            "cluster_dir": cluster_dir,
                            "image_path": "",
                            "idx_in_cluster": "",
                            "description": "",
                            "objects": "",
                            "objects_not_found": "",
                            "duration_sec": "",
                            "status": "skipped_existing_output",
                        }
                    )
                    prog_f.flush()
            return

    # In non-resume mode we regenerate outputs; clean debug directory first.
    # In resume mode we *append* and should not delete existing debug outputs.
    if debug and debug_dir is not None and not resume:
        prepare_clean_directory(debug_dir)
    if debug and debug_dir is not None and resume:
        debug_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu" if cpu_only or not torch.cuda.is_available() else "cuda"

    model, processor = load_model(
        device=device,
        confidence_threshold=box_threshold,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        verbose=False,
    )

    # Optionally load idx_in_cluster mapping from a sibling cluster_items.csv.
    cluster_index_map = maybe_load_cluster_index_map(input_csv.parent)

    # One row per image with aggregated missing objects.
    fieldnames = [
        "image_path",
        "idx_in_cluster",
        "description",
        "objects",
        "objects_not_found",
        "detected_objects",
        "detected_confidences",
        "sam3_debug_txt_path",
        "debug_image_path",
    ]

    # Open the real-time missing CSV if provided, for writing rows where SAM3
    # did not find all objects.
    realtime_missing_f = None
    realtime_missing_writer = None
    if realtime_missing_csv is not None:
        realtime_missing_fieldnames = [
            "input_csv",
            "image_path",
            "idx_in_cluster",
            "description",
            "objects",
            "objects_not_found",
        ]
        # Open in append mode so multiple CSVs can contribute to the same file.
        file_exists = realtime_missing_csv.is_file()
        realtime_missing_f = realtime_missing_csv.open("a", newline="", encoding="utf-8")
        realtime_missing_writer = csv.DictWriter(
            realtime_missing_f, fieldnames=realtime_missing_fieldnames
        )
        if not file_exists:
            realtime_missing_writer.writeheader()

    # Open the real-time progress CSV if provided, for writing one row per
    # processed image (regardless of whether objects are missing or not).
    realtime_progress_f = None
    realtime_progress_writer = None
    if realtime_progress_csv is not None:
        realtime_progress_fieldnames = [
            "timestamp",
            "input_csv",
            "cluster_dir",
            "image_path",
            "idx_in_cluster",
            "description",
            "objects",
            "objects_not_found",
            "detected_objects",
            "detected_confidences",
            "sam3_debug_txt_path",
            "duration_sec",
            "status",
        ]
        file_exists = realtime_progress_csv.is_file()
        realtime_progress_f = realtime_progress_csv.open(
            "a", newline="", encoding="utf-8"
        )
        realtime_progress_writer = csv.DictWriter(
            realtime_progress_f, fieldnames=realtime_progress_fieldnames
        )
        if not file_exists:
            realtime_progress_writer.writeheader()

    try:
        effective_follow = bool(follow or watch)
        effective_poll_interval = (
            float(watch_poll_sec)
            if watch and float(poll_interval) == 5.0
            else float(poll_interval)
        )
        effective_idle_minutes: Optional[int] = (
            int(watch_idle_minutes) if watch else follow_idle_minutes
        )

        if effective_follow:
            # Follow mode: incrementally process rows as they are appended to the input CSV.
            # Keep the SAM3 model loaded and reuse it across iterations.

            # Resume mode: if enabled and output exists, skip images already written.
            processed_image_paths: set[str] = set()
            if resume:
                processed_image_paths = _load_processed_image_paths(output_csv)
                if processed_image_paths:
                    print(
                        f"[resume] Loaded {len(processed_image_paths)} existing image_path values from "
                        f"{output_csv}; will skip re-processing those images.",
                        flush=True,
                    )

            output_exists = output_csv.is_file()
            output_mode = "a" if (resume and output_exists) else "w"

            # Follow mode: optional idle timeout (no file signature changes).
            idle_timeout_s: Optional[float] = None
            if effective_idle_minutes is not None:
                if effective_idle_minutes <= 0:
                    effective_idle_minutes = 20
                idle_timeout_s = float(effective_idle_minutes * 60)

            try:
                st0 = input_csv.stat()
                prev_sig: Optional[tuple[int, int]] = (st0.st_mtime_ns, st0.st_size)
            except FileNotFoundError:
                prev_sig = None

            idle_elapsed_s = 0.0
            processed_input_rows = 0  # number of data rows read from the input CSV (excluding header)

            print(f"[follow] Watching for new rows in: {input_csv}", flush=True)
            print(f"[follow] Poll interval: {effective_poll_interval:.1f}s", flush=True)
            if idle_timeout_s is not None:
                print(
                    f"[follow] Idle timeout: {int(idle_timeout_s // 60)} minutes "
                    f"(exit when input CSV is unchanged for that long).",
                    flush=True,
                )

            with output_csv.open(output_mode, newline="", encoding="utf-8") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                if output_mode == "w":
                    writer.writeheader()
                    f_out.flush()

                while True:
                    new_rows_seen = 0

                    try:
                        with input_csv.open("r", newline="", encoding="utf-8") as f_in:
                            reader = csv.DictReader(f_in)
                            if reader.fieldnames is None:
                                raise ValueError(f"CSV file has no header: {input_csv}")

                            fieldnames_in = set(reader.fieldnames)
                            if use_filtered:
                                if "filtered_by_llm" not in fieldnames_in:
                                    raise ValueError(
                                        f"CSV {input_csv} must contain a 'filtered_by_llm' column when "
                                        f"--filtered is set. Found columns: {reader.fieldnames}"
                                    )
                                schema = "filtered"
                            elif "objects_and_stuff" in fieldnames_in:
                                schema = "merged"
                            elif "objects" in fieldnames_in:
                                schema = "split"
                            else:
                                raise ValueError(
                                    f"CSV {input_csv} must contain either an 'objects_and_stuff' column "
                                    f"(merged schema) or an 'objects' column (split schema). "
                                    f"Found columns: {reader.fieldnames}"
                                )

                            # Skip already-read rows so we only handle newly appended lines.
                            for _ in range(processed_input_rows):
                                try:
                                    next(reader)
                                except StopIteration:
                                    break

                            for raw in reader:
                                processed_input_rows += 1
                                new_rows_seen += 1
                                global_row_idx = processed_input_rows  # 1-based among data rows

                                image_path = str((raw.get("image_path") or "")).strip()
                                description = str((raw.get("description") or "")).strip()
                                if schema == "merged":
                                    objects_raw = str((raw.get("objects_and_stuff") or "")).strip()
                                elif schema == "filtered":
                                    objects_raw = str((raw.get("filtered_by_llm") or "")).strip()
                                else:
                                    objects_raw = str((raw.get("objects") or "")).strip()
                                stuff_raw = str((raw.get("stuff") or "")).strip()

                                if not image_path:
                                    continue

                                if resume and image_path in processed_image_paths:
                                    continue

                                row = DetectionRow(
                                    image_path=image_path,
                                    description=description,
                                    objects_raw=objects_raw,
                                    stuff_raw=stuff_raw,
                                )

                                objects = parse_objects_field(row.objects_raw)
                                stuff_list = (
                                    parse_objects_field(row.stuff_raw) if row.stuff_raw else []
                                )
                                if max_objects_per_image and max_objects_per_image > 0:
                                    objects = objects[: int(max_objects_per_image)]
                                if max_stuff_per_image and max_stuff_per_image > 0:
                                    stuff_list = stuff_list[: int(max_stuff_per_image)]
                                if not objects and not stuff_list:
                                    continue

                                # Measure per-image processing time (loading image + SAM3 calls).
                                row_start_time = time.perf_counter()

                                try:
                                    row_path = Path(row.image_path)
                                    if row_path.is_absolute():
                                        full_image_path = str(row_path)
                                    elif images_root is not None:
                                        full_image_path = str((images_root / row_path).resolve())
                                    else:
                                        full_image_path = str(row_path)
                                    image_pil = load_image(full_image_path)
                                except FileNotFoundError:
                                    print(
                                        f"[WARN] Image not found, skipping row {global_row_idx}: {row.image_path}",
                                        flush=True,
                                    )
                                    continue
                                except Exception as e:  # noqa: BLE001
                                    print(
                                        f"[WARN] Failed to load image (row {global_row_idx}) {row.image_path}: {e}",
                                        flush=True,
                                    )
                                    continue

                                # Set the image once for all objects (more efficient)
                                inference_state = processor.set_image(image_pil)

                                # Look up idx_in_cluster for this (image_path, description) if available.
                                cluster_idx = cluster_index_map.get(
                                    (row.image_path.strip(), row.description.strip()), ""
                                )

                                missing_objects: List[str] = []
                                detected_objects: List[str] = []
                                detected_confidences: List[float] = []

                                debug_boxes: List[torch.Tensor] = []
                                debug_masks: List[torch.Tensor] = []
                                debug_labels: List[str] = []

                                # Process objects (things)
                                for object_name in objects:
                                    boxes, scores, masks = run_sam3_for_object(
                                        processor=processor,
                                        image_pil=image_pil,
                                        inference_state=inference_state,
                                        object_name=object_name,
                                    )
                                    found = len(scores) > 0
                                    if not found:
                                        missing_objects.append(object_name)
                                    else:
                                        # Record a per-object confidence for downstream analysis.
                                        try:
                                            conf = float(max(scores)) if len(scores) else 0.0
                                        except Exception:
                                            conf = 0.0
                                        detected_objects.append(object_name)
                                        detected_confidences.append(conf)

                                    if found and debug and len(boxes) > 0:
                                        debug_boxes.append(boxes)
                                        debug_masks.append(masks)
                                        debug_labels.extend(
                                            f"{object_name} ({float(s):.2f})" for s in scores
                                        )

                                # Process stuff (regions)
                                for stuff_name in stuff_list:
                                    boxes, scores, masks = run_sam3_for_object(
                                        processor=processor,
                                        image_pil=image_pil,
                                        inference_state=inference_state,
                                        object_name=stuff_name,
                                    )
                                    found = len(scores) > 0
                                    if not found:
                                        missing_objects.append(stuff_name)
                                    else:
                                        try:
                                            conf = float(max(scores)) if len(scores) else 0.0
                                        except Exception:
                                            conf = 0.0
                                        detected_objects.append(stuff_name)
                                        detected_confidences.append(conf)

                                    if found and debug and len(boxes) > 0:
                                        debug_boxes.append(boxes)
                                        debug_masks.append(masks)
                                        debug_labels.extend(
                                            f"{stuff_name} ({float(s):.2f})" for s in scores
                                        )

                                debug_image_path_str = ""
                                if debug and debug_dir is not None and missing_objects:
                                    if debug_boxes:
                                        all_boxes = torch.cat(debug_boxes, dim=0)
                                        all_masks = (
                                            torch.cat(debug_masks, dim=0) if debug_masks else None
                                        )
                                        segments_image = draw_segments_only(
                                            image_pil.copy(),
                                            all_boxes,
                                            debug_labels,
                                            masks=all_masks,
                                        )
                                    else:
                                        segments_image = image_pil.copy()

                                    prefix = (
                                        _debug_file_prefix(row=row, ordinal=global_row_idx)
                                        if resume
                                        else f"{global_row_idx:06d}_{Path(row.image_path).stem}"
                                    )
                                    segments_path = debug_dir / f"{prefix}_segments.jpg"
                                    if not (resume and segments_path.exists()):
                                        segments_image.save(segments_path)

                                    if debug_labels:
                                        color_map_image = create_color_map(debug_labels)
                                        color_map_path = debug_dir / f"{prefix}_colormap.jpg"
                                        if not (resume and color_map_path.exists()):
                                            color_map_image.save(color_map_path)

                                    debug_image_path_str = str(segments_path)

                                objects_not_found_str = (
                                    ". ".join(missing_objects) if missing_objects else ""
                                )

                                # Persist a per-row debug text artifact next to the outputs directory.
                                # This is cheap and makes later visualization/debugging easier.
                                # We always write it (even if nothing is missing) so downstream tools
                                # can rely on the column existing.
                                cluster_dir_for_txt = ""
                                for parent in input_csv.parents:
                                    name = parent.name
                                    if name.startswith("cluster_"):
                                        cluster_dir_for_txt = name
                                        break
                                if not cluster_dir_for_txt:
                                    cluster_dir_for_txt = input_csv.parent.name

                                sam3_debug_txt_path = _write_sam3_debug_txt(
                                    base_dir=realtime_progress_csv.parent
                                    if realtime_progress_csv is not None
                                    else output_csv.parent,
                                    cluster_dir=cluster_dir_for_txt,
                                    image_path=row.image_path,
                                    description=row.description,
                                    objects_raw=row.objects_raw,
                                    objects_not_found=missing_objects,
                                    detected_objects=detected_objects,
                                    detected_confidences=detected_confidences,
                                )

                                detected_objects_json = json.dumps(
                                    detected_objects, ensure_ascii=False
                                )
                                detected_confidences_json = json.dumps(
                                    [round(float(x), 6) for x in detected_confidences],
                                    ensure_ascii=False,
                                )
                                writer.writerow(
                                    {
                                        "image_path": row.image_path,
                                        "idx_in_cluster": cluster_idx,
                                        "description": row.description,
                                        "objects": row.objects_raw,
                                        "objects_not_found": objects_not_found_str,
                                        "detected_objects": detected_objects_json,
                                        "detected_confidences": detected_confidences_json,
                                        "sam3_debug_txt_path": sam3_debug_txt_path,
                                        "debug_image_path": debug_image_path_str,
                                    }
                                )
                                f_out.flush()
                                if resume:
                                    processed_image_paths.add(row.image_path)

                                if missing_objects and realtime_missing_writer is not None:
                                    realtime_missing_writer.writerow(
                                        {
                                            "input_csv": str(input_csv),
                                            "image_path": row.image_path,
                                            "idx_in_cluster": cluster_idx,
                                            "description": row.description,
                                            "objects": row.objects_raw,
                                            "objects_not_found": objects_not_found_str,
                                        }
                                    )
                                    realtime_missing_f.flush()

                                if realtime_progress_writer is not None:
                                    duration_sec = time.perf_counter() - row_start_time
                                    cluster_dir = ""
                                    for parent in input_csv.parents:
                                        name = parent.name
                                        if name.startswith("cluster_"):
                                            cluster_dir = name
                                            break
                                    if not cluster_dir:
                                        cluster_dir = input_csv.parent.name
                                    realtime_progress_writer.writerow(
                                        {
                                            "timestamp": datetime.now().isoformat(
                                                timespec="seconds"
                                            ),
                                            "input_csv": str(input_csv),
                                            "cluster_dir": cluster_dir,
                                            "image_path": row.image_path,
                                            "idx_in_cluster": cluster_idx,
                                            "description": row.description,
                                            "objects": row.objects_raw,
                                            "objects_not_found": objects_not_found_str,
                            "detected_objects": detected_objects_json,
                            "detected_confidences": detected_confidences_json,
                            "sam3_debug_txt_path": sam3_debug_txt_path,
                                            "duration_sec": f"{duration_sec:.4f}",
                                            "status": "processed",
                                        }
                                    )
                                    realtime_progress_f.flush()
                    except FileNotFoundError:
                        # If the upstream process hasn't created the file yet, just wait.
                        pass
                    except Exception as e:  # noqa: BLE001
                        # If file is mid-write / partially written, wait and retry.
                        print(
                            f"[follow] Failed to read/parse input CSV (will retry): "
                            f"{type(e).__name__}: {e}",
                            flush=True,
                        )

                    if new_rows_seen == 0:
                        # No new rows since last check. Use file signature to track idle time.
                        try:
                            st = input_csv.stat()
                            sig: Optional[tuple[int, int]] = (st.st_mtime_ns, st.st_size)
                        except FileNotFoundError:
                            sig = None

                        if sig == prev_sig:
                            idle_elapsed_s += float(effective_poll_interval)
                            if idle_timeout_s is not None and idle_elapsed_s >= idle_timeout_s:
                                print(
                                    f"[follow] No input CSV changes for {effective_idle_minutes} minutes. Exiting.",
                                    flush=True,
                                )
                                break
                        else:
                            prev_sig = sig
                            idle_elapsed_s = 0.0

                        time.sleep(float(effective_poll_interval))
                    # Otherwise, loop again immediately (new rows may have been appended while processing).

            return

        # Preload all rows so we can compute a deterministic progress bar
        # over the total number of samples that SAM3 will process.
        all_rows = list(parse_input_csv(input_csv, use_filtered=use_filtered))
        last_seen_input_rows = len(all_rows)

        # Filter down to only rows that actually have any objects or stuff,
        # which are the ones that will result in SAM3 inference.
        rows_with_ordinals: List[tuple[int, DetectionRow]] = []
        for ordinal, row in enumerate(all_rows, start=1):
            objects = parse_objects_field(row.objects_raw)
            if objects or row.stuff_raw:
                rows_with_ordinals.append((ordinal, row))

        # If resume is requested and the output CSV already exists, skip images
        # that were already processed (based on image_path).
        processed_image_paths: set[str] = set()
        if resume:
            processed_image_paths = _load_processed_image_paths(output_csv)

        rows_to_process: List[tuple[int, DetectionRow]] = []
        if processed_image_paths:
            for ordinal, row in rows_with_ordinals:
                if row.image_path.strip() in processed_image_paths:
                    continue
                rows_to_process.append((ordinal, row))
        else:
            rows_to_process = rows_with_ordinals

        total_samples = len(rows_to_process)

        output_exists = output_csv.is_file()
        output_mode = "a" if (resume and output_exists) else "w"
        with output_csv.open(output_mode, newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            if not (resume and output_exists):
                writer.writeheader()

            if tqdm is not None and total_samples > 0:
                row_iter = tqdm(
                    enumerate(rows_to_process, start=1),
                    total=total_samples,
                    desc=f"SAM3 {input_csv.name}",
                    unit="image",
                )
            else:
                row_iter = enumerate(rows_to_process, start=1)

            for _local_idx, (ordinal, row) in row_iter:
                objects = parse_objects_field(row.objects_raw)
                if max_objects_per_image and max_objects_per_image > 0:
                    objects = objects[: int(max_objects_per_image)]

                # Measure per-image processing time (loading image + SAM3 calls).
                row_start_time = time.perf_counter()

                try:
                    # Resolve image path:
                    # - If CSV provides an absolute path, use it as-is (ignore images_root).
                    # - If CSV provides a relative path, optionally prepend images_root.
                    #
                    # This supports workflows where image_path is already a full path
                    # (e.g. /mnt/d/... in WSL) and avoids accidental double-prefixing.
                    row_path = Path(row.image_path)
                    if row_path.is_absolute():
                        image_path = str(row_path)
                    elif images_root is not None:
                        image_path = str((images_root / row_path).resolve())
                    else:
                        image_path = str(row_path)
                    image_pil = load_image(image_path)
                except FileNotFoundError:
                    print(
                        f"[WARN] Image not found, skipping row {ordinal}: "
                        f"{image_path}"
                    )
                    continue
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] Failed to load image {row.image_path}: {e}")
                    continue

                # Set the image once for all objects (more efficient)
                inference_state = processor.set_image(image_pil)

                # Look up idx_in_cluster for this (image_path, description) if available.
                cluster_idx = cluster_index_map.get(
                    (row.image_path.strip(), row.description.strip()), ""
                )

                # Track objects that SAM3 fails to detect.
                missing_objects: List[str] = []

                debug_boxes: List[torch.Tensor] = []
                debug_masks: List[torch.Tensor] = []
                debug_labels: List[str] = []

                # Process objects (things)
                for object_name in objects:
                    boxes, scores, masks = run_sam3_for_object(
                        processor=processor,
                        image_pil=image_pil,
                        inference_state=inference_state,
                        object_name=object_name,
                    )

                    found = len(scores) > 0
                    if not found:
                        missing_objects.append(object_name)
                    elif debug and len(boxes) > 0:
                        # For visualization, keep the (top-1) box and mask for this object
                        debug_boxes.append(boxes)
                        debug_masks.append(masks)
                        debug_labels.extend(
                            f"{object_name} ({float(s):.2f})" for s in scores
                        )

                # Process stuff (regions) if available
                stuff_list = parse_objects_field(row.stuff_raw) if row.stuff_raw else []
                if max_stuff_per_image and max_stuff_per_image > 0:
                    stuff_list = stuff_list[: int(max_stuff_per_image)]
                for stuff_name in stuff_list:
                    boxes, scores, masks = run_sam3_for_object(
                        processor=processor,
                        image_pil=image_pil,
                        inference_state=inference_state,
                        object_name=stuff_name,
                    )

                    found = len(scores) > 0
                    if not found:
                        missing_objects.append(stuff_name)
                    elif debug and len(boxes) > 0:
                        # For visualization, keep the (top-1) box and mask for this stuff
                        debug_boxes.append(boxes)
                        debug_masks.append(masks)
                        debug_labels.extend(
                            f"{stuff_name} ({float(s):.2f})" for s in scores
                        )

                # By default, no debug image is associated.
                debug_image_path_str = ""

                # Save a debug image only if at least one object/stuff was NOT found.
                if debug and debug_dir is not None and missing_objects:
                    if debug_boxes:
                        all_boxes = torch.cat(debug_boxes, dim=0)
                        all_masks = torch.cat(debug_masks, dim=0) if debug_masks else None

                        # Create segments-only image with found detections overlaid.
                        segments_image = draw_segments_only(
                            image_pil.copy(), all_boxes, debug_labels, masks=all_masks
                        )
                    else:
                        # No detections at all; just save the original image.
                        segments_image = image_pil.copy()

                    if resume:
                        prefix = _debug_file_prefix(row=row, ordinal=ordinal)
                    else:
                        prefix = f"{_local_idx:06d}_{Path(row.image_path).stem}"
                    segments_filename = f"{prefix}_segments.jpg"
                    segments_path = debug_dir / segments_filename
                    # Avoid overwriting existing debug images in resume mode.
                    if not (resume and segments_path.exists()):
                        segments_image.save(segments_path)

                    # Optionally create a color map/legend image if we have any labels.
                    if debug_labels:
                        color_map_image = create_color_map(debug_labels)
                        color_map_filename = f"{prefix}_colormap.jpg"
                        color_map_path = debug_dir / color_map_filename
                        if not (resume and color_map_path.exists()):
                            color_map_image.save(color_map_path)

                    debug_image_path_str = str(segments_path)

                # Aggregate to a single row per image.
                objects_not_found_str = (
                    ". ".join(missing_objects) if missing_objects else ""
                )
                writer.writerow(
                    {
                        "image_path": row.image_path,
                        "idx_in_cluster": cluster_idx,
                        "description": row.description,
                        "objects": row.objects_raw,
                        "objects_not_found": objects_not_found_str,
                        "debug_image_path": debug_image_path_str,
                    }
                )
                try:
                    f_out.flush()
                except Exception:
                    pass

                # Write to real-time missing CSV if there are missing objects.
                if missing_objects and realtime_missing_writer is not None:
                    realtime_missing_writer.writerow(
                        {
                            "input_csv": str(input_csv),
                            "image_path": row.image_path,
                            "idx_in_cluster": cluster_idx,
                            "description": row.description,
                            "objects": row.objects_raw,
                            "objects_not_found": objects_not_found_str,
                        }
                    )
                    realtime_missing_f.flush()  # Flush immediately for real-time writing

                # Write to real-time progress CSV (one row per processed image).
                if realtime_progress_writer is not None:
                    duration_sec = time.perf_counter() - row_start_time

                    # Try to infer the cluster directory name (e.g., 'cluster_32')
                    # from the input CSV path. Fall back to the immediate parent dir.
                    cluster_dir = ""
                    for parent in input_csv.parents:
                        name = parent.name
                        if name.startswith("cluster_"):
                            cluster_dir = name
                            break
                    if not cluster_dir:
                        cluster_dir = input_csv.parent.name

                    realtime_progress_writer.writerow(
                        {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "input_csv": str(input_csv),
                            "cluster_dir": cluster_dir,
                            "image_path": row.image_path,
                            "idx_in_cluster": cluster_idx,
                            "description": row.description,
                            "objects": row.objects_raw,
                            "objects_not_found": objects_not_found_str,
                            "duration_sec": f"{duration_sec:.4f}",
                            "status": "processed",
                        }
                    )
                    realtime_progress_f.flush()
    finally:
        if realtime_missing_f is not None:
            realtime_missing_f.close()
        if realtime_progress_f is not None:
            realtime_progress_f.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM3 on a CSV of images and object lists, "
            "or batch-process a directory of '*_objects.csv' files, "
            "and produce a CSV with per-object detection results."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_csv",
        help="Path to a single input CSV with image_path, description, and objects columns.",
    )
    group.add_argument(
        "--root_dir",
        help=(
            "Root directory to recursively search for CSV files whose names end with "
            "'_objects.csv'. For each such file, SAM3 will be run and aggregate results "
            "will be written to the results directory."
        ),
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help=(
            "Path to the output CSV. "
            "Defaults to <input_stem>_sam3.csv next to the input CSV."
        ),
    )
    parser.add_argument(
        "--images_root",
        default=None,
        help=(
            "Optional root directory to prepend to *relative* image_path values from the CSV "
            "when locating image files on disk. If image_path is absolute, it is used as-is."
        ),
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.2,
        help="Confidence threshold for filtering SAM3 predictions.",
    )
    parser.add_argument(
        "--max_objects_per_image",
        type=int,
        default=0,
        help=(
            "Optional safety cap: when > 0, only process up to this many 'objects' (things) per image. "
            "Extra objects are skipped. Default: 0 (no cap)."
        ),
    )
    parser.add_argument(
        "--max_stuff_per_image",
        type=int,
        default=0,
        help=(
            "Optional safety cap: when > 0, only process up to this many 'stuff' items per image. "
            "Extra stuff entries are skipped. Default: 0 (no cap)."
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help=(
            "Optional local path to the SAM3 checkpoint (.pt). "
            "If not provided, the SAM3 package will attempt to download from Hugging Face "
            "(repo 'facebook/sam3', which may be gated / require login)."
        ),
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Force computation on CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, save visualization images with detections.",
    )
    parser.add_argument(
        "--debug_dir",
        default=None,
        help=(
            "Directory to store debug images. "
            "Defaults to <input_stem>_sam3_debug next to the input CSV."
        ),
    )
    parser.add_argument(
        "--results_dir",
        default=None,
        help=(
            "Directory to store aggregated results and logs when running in batch "
            "mode with --root_dir. Defaults to <root_dir>/sam3_batch_results."
        ),
    )
    parser.add_argument(
        "--debug_root_dir",
        default=None,
        help=(
            "Root directory to store debug images when running in batch mode with "
            "--root_dir. Per-CSV subdirectories will be created inside this directory. "
            "If not provided, debug images are stored next to each input CSV."
        ),
    )
    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help=(
            "If set, do not overwrite existing per-CSV SAM3 output files or debug "
            "directories. Any input CSV whose outputs already exist will be skipped."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If set, resume a previous run by appending to the output CSV (if it exists) "
            "and skipping images already present in that output. Unlike --no_overwrite, "
            "this does not skip the whole CSV; it continues from the remaining images."
        ),
    )
    parser.add_argument(
        "--np",
        action="store_true",
        help=(
            "When running with --root_dir, look for caption CSVs produced by the "
            "noun-phrase pipeline, i.e. files ending with '_objects_np.csv', "
            "instead of the default '*_objects.csv' pattern."
        ),
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help=(
            "When running with --root_dir, look for LLM-caption CSVs whose names end "
            "with '_objects_llm.csv' instead of the default '*_objects.csv' pattern."
        ),
    )
    parser.add_argument(
        "--filtered",
        action="store_true",
        help=(
            "When set, use the 'filtered_by_llm' column from the CSV as the objects "
            "source instead of the 'objects' column. Output files will include "
            "'_filtered' in their names (e.g., '_sam3_filtered.csv')."
        ),
    )
    parser.add_argument(
        "--realtime_missing_csv",
        default=None,
        help=(
            "Path to the real-time missing summary CSV file. This file is updated "
            "immediately whenever SAM3 fails to find all objects in a row. "
            "If not provided, defaults to 'sam3_realtime_missing[_filtered].csv' "
            "in the results directory (batch mode) or next to the input CSV (single mode)."
        ),
    )
    parser.add_argument(
        "--realtime_progress_csv",
        default=None,
        help=(
            "Path to the real-time per-image progress CSV file. This file is updated "
            "immediately after each image is processed (one row per image). "
            "If not provided, defaults to 'sam3_realtime_progress[_filtered].csv' "
            "in the results directory (batch mode) or next to the input CSV (single mode)."
        ),
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help=(
            "Keep watching the input CSV for new rows being appended and process them as they appear "
            "(similar to `tail -f`). Use Ctrl+C to stop."
        ),
    )
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=5.0,
        help="Seconds to wait between checks for new rows when --follow is set.",
    )
    parser.add_argument(
        "--follow_idle_minutes",
        type=int,
        default=None,
        help=(
            "Optional: when --follow is set, exit automatically after this many minutes "
            "with no detected changes to the input CSV (mtime/size unchanged). "
            "If omitted, follow mode runs indefinitely until Ctrl+C."
        ),
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help=(
            "Convenience alias for --follow with a 1-minute sampling cadence and an "
            "idle-exit timeout. This exits when the input CSV has not changed for "
            "a configurable amount of time."
        ),
    )
    parser.add_argument(
        "--watch_poll_sec",
        type=int,
        default=60,
        help="Polling cadence in seconds for --watch (default: 60).",
    )
    parser.add_argument(
        "--watch_idle_minutes",
        type=int,
        default=20,
        help="Exit after this many minutes with no input CSV changes in --watch (default: 20).",
    )
    parser.add_argument(
        "--input_csv_wait_seconds",
        type=int,
        default=900,
        help=(
            "If the input CSV does not exist yet, wait up to this many seconds for it to appear "
            "before exiting (default: 900 = 15 minutes)."
        ),
    )

    args = parser.parse_args()

    # Batch mode: recursively process all '*_objects.csv' files under root_dir.
    if args.root_dir:
        root_dir = Path(args.root_dir).resolve()
        results_dir = (
            Path(args.results_dir).resolve()
            if args.results_dir
            else root_dir / "sam3_batch_results"
        )
        debug_root_dir = (
            Path(args.debug_root_dir).resolve()
            if args.debug_root_dir
            else None
        )
        images_root = (
            Path(args.images_root).resolve() if args.images_root else None
        )

        run_batch(
            root_dir=root_dir,
            results_dir=results_dir,
            images_root=images_root,
            box_threshold=args.box_threshold,
            cpu_only=args.cpu_only,
            checkpoint_path=Path(args.checkpoint_path).resolve()
            if args.checkpoint_path
            else None,
            debug=args.debug,
            debug_root_dir=debug_root_dir,
            no_overwrite=args.no_overwrite,
            resume=args.resume,
            use_np=args.np,
            use_llm=args.llm,
            use_filtered=args.filtered,
            max_objects_per_image=int(args.max_objects_per_image),
            max_stuff_per_image=int(args.max_stuff_per_image),
            input_csv_wait_seconds=int(args.input_csv_wait_seconds),
            realtime_missing_csv=Path(args.realtime_missing_csv).resolve()
            if args.realtime_missing_csv
            else None,
        )
    else:
        # Single-CSV mode: optionally wait for the input CSV to be created.
        input_csv_path = Path(args.input_csv).resolve()
        if not input_csv_path.is_file():
            timeout_sec = int(args.input_csv_wait_seconds)
            found = wait_for_file(input_csv_path, timeout_sec=timeout_sec)
            if not found:
                print(
                    f"Input CSV not found: {input_csv_path}. "
                    f"Waited {timeout_sec} seconds for it to appear; exiting."
                )
                sys.exit(1)

        # Determine realtime missing CSV path for single-CSV mode.
        if args.realtime_missing_csv:
            realtime_missing_path = Path(args.realtime_missing_csv).resolve()
        else:
            suffix = "_filtered" if args.filtered else ""
            realtime_missing_path = input_csv_path.with_name(
                f"sam3_realtime_missing{suffix}.csv"
            )

        # Determine realtime progress CSV path for single-CSV mode.
        if args.realtime_progress_csv:
            realtime_progress_path = Path(args.realtime_progress_csv).resolve()
        else:
            suffix = "_filtered" if args.filtered else ""
            realtime_progress_path = input_csv_path.with_name(
                f"sam3_realtime_progress{suffix}.csv"
            )

        # Prevent concurrent runs from writing the same realtime progress CSV.
        # This is the most common cause of "exactly doubled" line counts.
        with FileLock(realtime_progress_path):
            run(
                input_csv=input_csv_path,
                output_csv=Path(args.output_csv).resolve() if args.output_csv else None,
                images_root=Path(args.images_root).resolve()
                if args.images_root
                else None,
                box_threshold=args.box_threshold,
                cpu_only=args.cpu_only,
                checkpoint_path=Path(args.checkpoint_path).resolve()
                if args.checkpoint_path
                else None,
                debug=args.debug,
                debug_dir=Path(args.debug_dir).resolve() if args.debug_dir else None,
                max_objects_per_image=int(args.max_objects_per_image),
                max_stuff_per_image=int(args.max_stuff_per_image),
                no_overwrite=args.no_overwrite,
                resume=args.resume,
                use_filtered=args.filtered,
                realtime_missing_csv=realtime_missing_path,
                realtime_progress_csv=realtime_progress_path,
                follow=bool(args.follow or args.watch),
                poll_interval=(
                    float(args.watch_poll_sec)
                    if args.watch and float(args.poll_interval) == 5.0
                    else float(args.poll_interval)
                ),
                follow_idle_minutes=(
                    int(args.watch_idle_minutes) if args.watch else args.follow_idle_minutes
                ),
                watch=bool(args.watch),
                watch_poll_sec=int(args.watch_poll_sec),
                watch_idle_minutes=int(args.watch_idle_minutes),
            )


def run_batch(
    root_dir: Path,
    results_dir: Path,
    images_root: Optional[Path],
    box_threshold: float,
    cpu_only: bool,
    checkpoint_path: Optional[Path],
    debug: bool,
    debug_root_dir: Optional[Path],
    max_objects_per_image: int = 0,
    max_stuff_per_image: int = 0,
    no_overwrite: bool = False,
    resume: bool = False,
    use_np: bool = False,
    use_llm: bool = False,
    use_filtered: bool = False,
    input_csv_wait_seconds: int = 900,
    realtime_missing_csv: Optional[Path] = None,
    realtime_progress_csv: Optional[Path] = None,
) -> None:
    """
    Recursively search root_dir for CSV files that end with '_objects.csv',
    run SAM3 on each, and aggregate cases where SAM3 could not find the
    requested object.

    The function writes:
    - Per-CSV SAM3 outputs next to each input CSV (same behavior as single mode),
      one row per image with aggregated missing objects.
    - A detailed CSV of all failed detections in results_dir.
    - A summary CSV with counts per CSV in results_dir.
    - A text log file in results_dir that includes the command line used.
    """
    root_dir = root_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if debug and debug_root_dir is not None:
        debug_root_dir.mkdir(parents=True, exist_ok=True)

    # Discover all relevant CSV files, waiting up to a configurable amount of time for them to appear.
    # Patterns:
    # - '*_objects.csv' (default, standard objects CSV)
    # - '*_objects_np.csv' (noun-phrase pipeline) when --np is set
    # - '*_objects_llm_v2.csv' (LLM-expanded objects, v2 naming) when --llm is set
    timeout_sec = int(input_csv_wait_seconds)
    poll_interval = 5.0
    start_time = time.time()
    csv_files: List[Path] = []
    # Resolve pattern priority: --llm > --np > default.
    if use_llm:
        # Newer LLM extraction script writes '*_objects_llm_v2.csv'.
        pattern = "*_objects_llm_v2.csv"
    elif use_np:
        pattern = "*_objects_np.csv"
    else:
        pattern = "*_objects.csv"
    while True:
        csv_files = sorted(root_dir.rglob(pattern))
        if csv_files:
            break
        if time.time() - start_time >= timeout_sec:
            print(
                f"No '{pattern}' files found under {root_dir} after "
                f"{timeout_sec} seconds; exiting."
            )
            return
        time.sleep(poll_interval)

    # Use appropriate file names based on filtered mode.
    suffix = "_filtered" if use_filtered else ""
    detailed_path = results_dir / f"sam3_missing_detections{suffix}.csv"
    summary_path = results_dir / f"sam3_missing_summary{suffix}.csv"
    log_path = results_dir / f"sam3_batch_log{suffix}.txt"

    # Determine realtime missing CSV path.
    if realtime_missing_csv is not None:
        realtime_missing_path = realtime_missing_csv
    else:
        realtime_missing_path = results_dir / f"sam3_realtime_missing{suffix}.csv"

    # Determine realtime progress CSV path.
    if realtime_progress_csv is not None:
        realtime_progress_path = realtime_progress_csv
    else:
        realtime_progress_path = results_dir / f"sam3_realtime_progress{suffix}.csv"

    # Clear the realtime CSVs to start fresh (they are opened in append mode by run()).
    if realtime_missing_path.exists():
        realtime_missing_path.unlink()
    if realtime_progress_path.exists():
        realtime_progress_path.unlink()

    print(f"Realtime missing CSV will be saved to: {realtime_missing_path}")
    print(f"Realtime progress CSV will be saved to: {realtime_progress_path}")

    # Prepare logging.
    command_line = " ".join(shlex.quote(arg) for arg in sys.argv)
    start_time = datetime.now().isoformat(timespec="seconds")

    with log_path.open("a", encoding="utf-8") as log_f, \
         detailed_path.open("w", newline="", encoding="utf-8") as det_f:
        log_f.write("\n")
        log_f.write(f"=== SAM3 batch run @ {start_time} ===\n")
        log_f.write(f"Command: {command_line}\n")
        log_f.write(f"Root dir: {root_dir}\n")
        log_f.write(f"Results dir: {results_dir}\n")
        if debug and debug_root_dir is not None:
            log_f.write(f"Debug root dir: {debug_root_dir}\n")
        if use_filtered:
            log_f.write("Mode: Using 'filtered_by_llm' column as objects source\n")
        log_f.write(f"Realtime missing CSV: {realtime_missing_path}\n")
        log_f.write(f"Realtime progress CSV: {realtime_progress_path}\n")
        log_f.write(f"Found {len(csv_files)} CSV files matching '{pattern}'.\n")

        detailed_fieldnames = [
            "input_csv",
            "output_csv",
            "image_path",
            "idx_in_cluster",
            "description",
            "objects",
            "object_name",
            "debug_image_path",
        ]
        detailed_writer = csv.DictWriter(det_f, fieldnames=detailed_fieldnames)
        detailed_writer.writeheader()

        # Track per-CSV counts for the summary.
        summary_counts: dict[str, dict[str, float]] = {}

        # Track which cluster directories we've already checked for their
        # companion cluster_items CSV to avoid redundant waits.
        checked_cluster_dirs: set[Path] = set()

        for idx, csv_path in enumerate(csv_files, start=1):
            csv_path = csv_path.resolve()
            log_f.write(f"[{idx}/{len(csv_files)}] Processing CSV: {csv_path}\n")
            log_f.flush()

            # When using LLM-based object CSVs, ensure that for each cluster_* directory
            # we also have the corresponding 'cluster_items_objects_llm_v2.csv'.
            # If it does not exist yet, wait up to 5 minutes for it to appear; if it
            # still does not show up, exit with a meaningful error.
            if use_llm:
                cluster_dir: Optional[Path] = None
                for parent in csv_path.parents:
                    if parent.name.startswith("cluster_"):
                        cluster_dir = parent
                        break
                if cluster_dir is None:
                    cluster_dir = csv_path.parent

                if cluster_dir not in checked_cluster_dirs:
                    cluster_items_name = "cluster_items_objects_llm_v2.csv"
                    cluster_items_path = cluster_dir / cluster_items_name
                    if not cluster_items_path.is_file():
                        timeout_sec = int(input_csv_wait_seconds)
                        poll_interval = 5.0
                        print(
                            f"[INFO] Waiting up to {timeout_sec} seconds for "
                            f"'{cluster_items_name}' in cluster directory "
                            f"{cluster_dir}..."
                        )
                        found = wait_for_file(
                            cluster_items_path,
                            timeout_sec=timeout_sec,
                            poll_interval=poll_interval,
                        )
                        if not found:
                            print(
                                f"[ERROR] Expected '{cluster_items_name}' not found in "
                                f"{cluster_dir} after waiting {timeout_sec} seconds. "
                                "Exiting."
                            )
                            sys.exit(1)

                    checked_cluster_dirs.add(cluster_dir)

            # Determine per-CSV debug directory in batch mode.
            per_csv_debug_dir: Optional[Path] = None
            debug_suffix_dir = "_sam3_filtered_debug" if use_filtered else "_sam3_debug"
            if debug:
                if debug_root_dir is not None:
                    try:
                        rel_parent = csv_path.parent.relative_to(root_dir)
                    except ValueError:
                        # If for some reason the CSV is not under root_dir, just use its parent name.
                        rel_parent = Path(csv_path.parent.name)
                    per_csv_debug_dir = (
                        debug_root_dir / rel_parent / f"{csv_path.stem}{debug_suffix_dir}"
                    )
                else:
                    # Default: store debug images next to each CSV, similar to single-CSV mode.
                    per_csv_debug_dir = csv_path.with_name(
                        f"{csv_path.stem}{debug_suffix_dir}"
                    )

            # Run SAM3 on this CSV (per-image results written next to the CSV).
            run(
                input_csv=csv_path,
                output_csv=None,
                images_root=images_root,
                box_threshold=box_threshold,
                cpu_only=cpu_only,
                checkpoint_path=checkpoint_path,
                debug=debug,
                debug_dir=per_csv_debug_dir,
                max_objects_per_image=int(max_objects_per_image),
                max_stuff_per_image=int(max_stuff_per_image),
                no_overwrite=no_overwrite,
                resume=resume,
                use_filtered=use_filtered,
                realtime_missing_csv=realtime_missing_path,
                realtime_progress_csv=realtime_progress_path,
            )

            output_suffix = "_sam3_filtered.csv" if use_filtered else "_sam3.csv"
            output_csv = csv_path.with_name(csv_path.stem + output_suffix)
            if not output_csv.is_file():
                log_f.write(
                    f"  [WARN] Expected output CSV not found, skipping aggregation: "
                    f"{output_csv}\n"
                )
                log_f.flush()
                continue

            # Aggregate failures from this CSV.
            total_objects = 0
            missing_objects = 0

            with output_csv.open("r", newline="", encoding="utf-8") as f_out:
                reader = csv.DictReader(f_out)
                for out_row in reader:
                    # Each row represents one image; count total and missing objects.
                    objects_list = parse_objects_field(out_row.get("objects", ""))
                    objects_not_found_list = parse_objects_field(
                        out_row.get("objects_not_found", "")
                    )
                    total_objects += len(objects_list)
                    missing_objects += len(objects_not_found_list)

                    # Emit one detailed row per missing object, if any.
                    for missing_name in objects_not_found_list:
                        detailed_writer.writerow(
                            {
                                "input_csv": str(csv_path),
                                "output_csv": str(output_csv),
                                "image_path": out_row.get("image_path", ""),
                                "idx_in_cluster": out_row.get("idx_in_cluster", ""),
                                "description": out_row.get("description", ""),
                                "objects": out_row.get("objects", ""),
                                "object_name": missing_name,
                                "debug_image_path": out_row.get(
                                    "debug_image_path", ""
                                ),
                            }
                        )

            if total_objects > 0:
                missing_rate = missing_objects / float(total_objects)
            else:
                missing_rate = 0.0

            summary_counts[str(csv_path)] = {
                "total_objects": float(total_objects),
                "missing_objects": float(missing_objects),
                "missing_rate": missing_rate,
            }

            log_f.write(
                f"  Total objects: {total_objects}, "
                f"missing: {missing_objects} "
                f"({missing_rate:.2%})\n"
            )
            log_f.flush()

    # Write per-CSV summary CSV.
    with summary_path.open("w", newline="", encoding="utf-8") as sum_f:
        summary_fieldnames = [
            "input_csv",
            "total_objects",
            "missing_objects",
            "missing_rate",
        ]
        summary_writer = csv.DictWriter(sum_f, fieldnames=summary_fieldnames)
        summary_writer.writeheader()
        for input_csv_str, stats in sorted(summary_counts.items()):
            summary_writer.writerow(
                {
                    "input_csv": input_csv_str,
                    "total_objects": int(stats["total_objects"]),
                    "missing_objects": int(stats["missing_objects"]),
                    "missing_rate": f"{stats['missing_rate']:.6f}",
                }
            )



if __name__ == "__main__":
    main()

