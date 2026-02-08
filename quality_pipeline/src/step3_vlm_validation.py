from __future__ import annotations

import argparse
import sys
import csv
import time
from datetime import datetime
from pathlib import Path
csv.field_size_limit(sys.maxsize)
from src.utils.checker import check_csv_with_llm, debug_single_image
from src.utils.file_lock import FileLock
from src.utils.llm_clients import (
    build_default_client,
    GeminiVLMClient,
    OpenAICompatVLMClient,
    TextOnlyLLMClient,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process a sam3_realtime_progress CSV with an LLM that "
            "answers per-object yes/no questions and adds a new column "
            "listing objects the LLM still considers missing."
        )
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help=(
            "Global API key for remote backends. This is a convenience flag so you don't need "
            "provider-specific flags.\n"
            "  - When using --vlm_provider gemini: used as GEMINI_API_KEY (if --gemini_api_key is not set)\n"
            "  - When using --vlm_provider openai_compat or when OPENAI_BASE_URL is set: used as OPENAI_API_KEY "
            "(if --openai_api_key is not set)\n"
            "Also exported as VLLM_API_KEY for shared resolution in vllm_checker.llm_client."
        ),
    )
    parser.add_argument(
        "--prompt_style",
        default="strict_yn",
        choices=["strict_yn", "describe_then_yesno"],
        help=(
            "Prompt style for the VLM.\n"
            "  - strict_yn: legacy behavior: ask only a strict yes/no question.\n"
            "  - describe_then_yesno: ask the model to briefly describe the image first, "
            "then decide if the object is present, with a final yes/no answer."
        ),
    )
    parser.add_argument(
        "--max_tokens_describe_then_yesno",
        type=int,
        default=None,
        help=(
            "Token budget used when --prompt_style describe_then_yesno. "
            "If omitted, defaults to 64 (or VLLM_MAX_TOKENS_DESCRIBE_THEN_YESNO env var)."
        ),
    )
    parser.add_argument(
        "--max_tokens_strict_yn",
        type=int,
        default=None,
        help=(
            "Token budget used when --prompt_style strict_yn. "
            "If omitted, defaults to backend defaults (or VLLM_MAX_TOKENS_STRICT_YN env var)."
        ),
    )
    parser.add_argument(
        "--prompt_examples_path",
        default=None,
        help=(
            "Optional path to a text file containing few-shot examples to include in the prompt. "
            "If provided, this text is inserted verbatim before the actual question/instructions. "
            "Most useful with --prompt_style describe_then_yesno."
        ),
    )
    parser.add_argument(
        "--prompt_disable_default_examples",
        action="store_true",
        help=(
            "Disable built-in prompt examples. If you provide --prompt_examples_path, "
            "you usually don't need the defaults."
        ),
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help=(
            "Path to the input sam3_realtime_progress CSV (e.g. "
            "sam3_realtime_progress.csv)."
        ),
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help=(
            "Optional path for the augmented CSV. Defaults to "
            "<input_stem>_vllm_checked.csv next to the input file."
        ),
    )
    parser.add_argument(
        "--no_only_csv",
        default=None,
        help=(
            "Optional path for an additional debug CSV that contains ONLY rows where the model "
            "said 'no' to at least one missing object. If omitted, defaults to "
            "<output_stem>_no_only_debug.csv next to the output CSV."
        ),
    )
    parser.add_argument(
        "--no_only_raw_column",
        default="vllm_raw_outputs_json",
        help=(
            "Column name to store per-object raw model outputs in the NO-only debug CSV "
            "(default: vllm_raw_outputs_json)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If output_csv already exists, append to it and skip rows already written "
            "(continue from where the output CSV ended). Also skips any input rows whose "
            "image_path already appears in the existing output (so images are never re-checked). "
            "If not set, output_csv is overwritten from scratch."
        ),
    )
    parser.add_argument(
        "--images_root",
        default=None,
        help=(
            "Optional root directory to prepend to each image_path from "
            "the CSV when constructing the full image path."
        ),
    )
    parser.add_argument(
        "--new_column",
        default="objects_vllm_said_no",
        help=(
            "Name of the additional column to write with objects that "
            "the LLM answered 'no' for."
        ),
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=1,
        help=(
            "Maximum number of objects to send to the LLM in a single batch. "
            "Use a value > 1 to enable batched LLM calls."
        ),
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help=(
            "Keep watching the input CSV for new rows being appended and "
            "process them as they appear (similar to `tail -f`). Use Ctrl+C "
            "to stop."
        ),
    )
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=5.0,
        help=(
            "Seconds to wait between checks for new rows when --follow is set."
        ),
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
        "--debug-image",
        default=None,
        help=(
            "Debug mode: process only the row with this image_path value "
            "(as it appears in the CSV). Prints results to stdout without "
            "creating an output CSV."
        ),
    )
    parser.add_argument(
        "--vlm_provider",
        default="auto",
        choices=["auto", "gemini", "openai_compat", "local_hf"],
        help=(
            "Which VLM backend to use. "
            "auto: preserve legacy behavior (OPENAI_BASE_URL -> OpenAI-compatible HTTP, else local HF). "
            "gemini: Google AI Studio / Gemini Developer API. "
            "openai_compat: force OpenAI-compatible HTTP. "
            "local_hf: force local HuggingFace backend."
        ),
    )
    parser.add_argument(
        "--local_model",
        default=None,
        help=(
            "Optional: override the local HuggingFace model to load (only applies when NOT using "
            "--openai_base_url).\n"
            "Examples:\n"
            "  - Qwen/Qwen2.5-VL-72B-Instruct\n"
            "  - qwen2.5-vl-72b   (alias)\n"
            "  - Qwen/Qwen2-VL-2B-Instruct"
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print lightweight progress stages during model init/inference (sets VLLM_PROGRESS=1).",
    )
    parser.add_argument(
        "--openai_base_url",
        default=None,
        help=(
            "Optional: use an OpenAI-compatible HTTP VLM backend (e.g. vLLM in OpenAI mode). "
            "Example: http://localhost:8000 . If provided, this is used instead of the local "
            "HuggingFace backend (which requires torch/transformers)."
        ),
    )
    parser.add_argument(
        "--openai_model",
        default=None,
        help=(
            "Optional model name for the OpenAI-compatible backend. "
            "If omitted, the backend default is used (or OPENAI_MODEL env var)."
        ),
    )
    parser.add_argument(
        "--openai_api_key",
        default=None,
        help=(
            "Optional API key for OpenAI-compatible backend. If omitted, OPENAI_API_KEY is used, "
            "otherwise 'EMPTY'."
        ),
    )
    parser.add_argument(
        "--openai_timeout_s",
        type=float,
        default=None,
        help=(
            "HTTP timeout in seconds for the OpenAI-compatible backend (connect + read). "
            "If omitted, uses the default (120s) or OPENAI_TIMEOUT_S env var."
        ),
    )
    parser.add_argument(
        "--openai_max_retries",
        type=int,
        default=None,
        help=(
            "Max retries for transient OpenAI-compatible HTTP failures. "
            "If omitted, uses the default (2) or OPENAI_MAX_RETRIES env var."
        ),
    )
    parser.add_argument(
        "--openai_retry_backoff_s",
        type=float,
        default=None,
        help=(
            "Retry backoff base (seconds). Actual sleeps are backoff * 2^attempt. "
            "If omitted, uses the default (1.0) or OPENAI_RETRY_BACKOFF_S env var."
        ),
    )
    parser.add_argument(
        "--gemini_model",
        default=None,
        help=(
            "Optional: Gemini model name when using --vlm_provider gemini. "
            "Defaults to GEMINI_MODEL env var or 'gemini-2.5-flash'."
        ),
    )
    parser.add_argument(
        "--gemini_api_key",
        default=None,
        help=(
            "Optional: Gemini API key (Google AI Studio). If omitted, GEMINI_API_KEY (or GOOGLE_API_KEY) is used."
        ),
    )
    parser.add_argument(
        "--log_llm_io",
        action="store_true",
        help=(
            "Debug: log each VLM prompt + raw output to vllm_checker/outs/llm_io_log.csv "
            "(enable with VLLM_LOG_IO=1 under the hood)."
        ),
    )
    return parser


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _client_model_label(client: object) -> str:
    """Best-effort label of the active backend + model for logging."""
    try:
        if isinstance(client, OpenAICompatVLMClient):
            return f"openai_compat_http model={client.config.model} base_url={client.config.base_url}"
        if isinstance(client, GeminiVLMClient):
            return f"gemini model={client.config.model}"
        if isinstance(client, TextOnlyLLMClient):
            dev = getattr(client, "_device", None)
            dev_str = str(dev) if dev is not None else "unknown"
            accel = "GPU" if "cuda" in dev_str.lower() else "CPU"
            return f"local_hf model={client.config.model_name} device={dev_str} ({accel})"
        # Fallback for custom client types.
        return type(client).__name__
    except Exception:
        return type(client).__name__

def _wait_for_file(path: Path, timeout_sec: int, poll_interval: float = 2.0) -> bool:
    """
    Wait for `path` to become an existing file, up to `timeout_sec`.
    Returns True if the file appeared, else False.
    """
    start = time.time()
    while time.time() - start < timeout_sec:
        if path.is_file():
            return True
        time.sleep(poll_interval)
    return path.is_file()


def main() -> None:
    parser = build_arg_parser()
    parser.add_argument(
        "--input_csv_wait_seconds",
        type=int,
        default=900,
        help=(
            "If the input CSV does not exist yet and --follow/--watch is set, wait up to this "
            "many seconds for it to appear before exiting (default: 900 = 15 minutes)."
        ),
    )
    args = parser.parse_args()

    import os

    # Global API key (read by vllm_checker.llm_client via env vars).
    # Provider-specific flags still take precedence, but this enables a single-flag UX.
    if args.api_key:
        os.environ["VLLM_API_KEY"] = str(args.api_key)

    # Prompt configuration (read by vllm_checker.llm_client via env vars).
    os.environ["VLLM_PROMPT_STYLE"] = str(args.prompt_style)
    if args.max_tokens_describe_then_yesno is not None:
        os.environ["VLLM_MAX_TOKENS_DESCRIBE_THEN_YESNO"] = str(int(args.max_tokens_describe_then_yesno))
    if args.max_tokens_strict_yn is not None:
        os.environ["VLLM_MAX_TOKENS_STRICT_YN"] = str(int(args.max_tokens_strict_yn))
    if args.prompt_examples_path:
        os.environ["VLLM_PROMPT_EXAMPLES_PATH"] = str(args.prompt_examples_path)
    if args.prompt_disable_default_examples:
        os.environ["VLLM_PROMPT_DISABLE_DEFAULT_EXAMPLES"] = "1"

    if args.progress:
        os.environ["VLLM_PROGRESS"] = "1"

    if args.log_llm_io:
        os.environ["VLLM_LOG_IO"] = "1"

    if args.local_model:
        # Used by build_default_client() for the local HF backend.
        os.environ["VLLM_HF_MODEL"] = str(args.local_model)

    # Gemini backend config via env vars (respected by build_default_client()).
    if args.gemini_model:
        os.environ["GEMINI_MODEL"] = str(args.gemini_model)
    if args.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = str(args.gemini_api_key)

    # If user specified OpenAI-compatible backend via CLI args, set env vars that
    # `build_default_client()` already respects.
    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = str(args.openai_base_url)
        if args.openai_model:
            os.environ["OPENAI_MODEL"] = str(args.openai_model)
        if args.openai_api_key:
            os.environ["OPENAI_API_KEY"] = str(args.openai_api_key)
        if args.openai_timeout_s is not None:
            os.environ["OPENAI_TIMEOUT_S"] = str(args.openai_timeout_s)
        if args.openai_max_retries is not None:
            os.environ["OPENAI_MAX_RETRIES"] = str(args.openai_max_retries)
        if args.openai_retry_backoff_s is not None:
            os.environ["OPENAI_RETRY_BACKOFF_S"] = str(args.openai_retry_backoff_s)

    input_csv = Path(args.input_csv).resolve()
    images_root = Path(args.images_root).resolve() if args.images_root else None

    # In watch/follow mode, the input CSV may not exist yet (upstream step still writing it).
    if not input_csv.is_file():
        if args.follow or args.watch:
            timeout_sec = int(args.input_csv_wait_seconds)
            print(
                f"[{_ts()}] [main] Input CSV not found yet: {input_csv}. "
                f"Waiting up to {timeout_sec} seconds for it to appear...",
                flush=True,
            )
            found = _wait_for_file(input_csv, timeout_sec=timeout_sec, poll_interval=2.0)
            if not found:
                print(
                    f"[{_ts()}] [main] Input CSV still not found: {input_csv}. "
                    f"Waited {timeout_sec} seconds; exiting.",
                    flush=True,
                )
                sys.exit(1)
        else:
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Debug mode: process a single image and print results
    if args.debug_image:
        # Build client after enabling debug mode so any backend logs show up.
        os.environ["VLLM_DEBUG"] = "1"
        print(f"[{_ts()}] [main] Building LLM client (may load model)...", flush=True)
        t0 = time.time()
        client = build_default_client(provider=str(args.vlm_provider))
        print(f"[{_ts()}] [main] LLM client ready (took {time.time() - t0:.1f}s)", flush=True)
        print(f"[{_ts()}] [main] Using LLM: {_client_model_label(client)}", flush=True)
        debug_single_image(
            input_csv=input_csv,
            target_image_path=args.debug_image,
            images_root=images_root,
            client=client,
        )
        return

    # Normal mode: process entire CSV
    output_csv = Path(args.output_csv).resolve() if args.output_csv else None
    no_only_csv = Path(args.no_only_csv).resolve() if args.no_only_csv else None

    # Build the client once (so we fail fast if the backend is unreachable) and
    # reuse it for the full run.
    print(f"[{_ts()}] [main] Building LLM client (may load model)...", flush=True)
    t0 = time.time()
    client = build_default_client(provider=str(args.vlm_provider))
    print(f"[{_ts()}] [main] LLM client ready (took {time.time() - t0:.1f}s)", flush=True)
    print(f"[{_ts()}] [main] Using LLM: {_client_model_label(client)}", flush=True)
    # Print what backend is being used + CPU/GPU if local.
    if isinstance(client, OpenAICompatVLMClient):
        print(f"LLM backend: openai_compat_http (base_url={client.config.base_url}, model={client.config.model})")
    elif isinstance(client, GeminiVLMClient):
        print(f"LLM backend: gemini (model={client.config.model})")
    elif isinstance(client, TextOnlyLLMClient):
        dev = getattr(client, "_device", None)
        dev_str = str(dev) if dev is not None else "unknown"
        accel = "GPU" if "cuda" in dev_str.lower() else "CPU"
        print(f"LLM backend: local_hf (model={client.config.model_name}, device={dev_str} => {accel})")
    else:
        print(f"LLM backend: {type(client).__name__}")
    # Prevent two concurrent vllm_checker runs from writing the same output CSV.
    # In the full pipeline we always pass --output_csv explicitly.
    if output_csv is not None:
        with FileLock(output_csv):
            final_path = check_csv_with_llm(
                input_csv=input_csv,
                output_csv=output_csv,
                no_only_csv=no_only_csv,
                no_only_raw_column=str(args.no_only_raw_column),
                images_root=images_root,
                new_column=args.new_column,
                llm_batch_size=args.llm_batch_size,
                resume=bool(args.resume),
                follow=bool(args.follow or args.watch),
                poll_interval=(
                    float(args.watch_poll_sec)
                    if args.watch and float(args.poll_interval) == 5.0
                    else float(args.poll_interval)
                ),
                follow_idle_minutes=(
                    int(args.watch_idle_minutes) if args.watch else args.follow_idle_minutes
                ),
                client=client,
            )
    else:
        final_path = check_csv_with_llm(
            input_csv=input_csv,
            output_csv=output_csv,
            no_only_csv=no_only_csv,
            no_only_raw_column=str(args.no_only_raw_column),
            images_root=images_root,
            new_column=args.new_column,
            llm_batch_size=args.llm_batch_size,
            resume=bool(args.resume),
            follow=bool(args.follow or args.watch),
            poll_interval=(
                float(args.watch_poll_sec)
                if args.watch and float(args.poll_interval) == 5.0
                else float(args.poll_interval)
            ),
            follow_idle_minutes=(
                int(args.watch_idle_minutes) if args.watch else args.follow_idle_minutes
            ),
            client=client,
        )

    print(f"Wrote LLM-checked CSV to: {final_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

