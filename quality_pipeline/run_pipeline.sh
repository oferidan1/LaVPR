#!/usr/bin/env bash
set -euo pipefail

# VPR Quality Pipeline - Main orchestration script
# Runs all 4 stages of the quality validation pipeline

# Ensure the repo root is on the Python path so "from src.utils..." imports work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
VPR Quality Pipeline

USAGE:
  ./run_pipeline.sh [OPTIONS]

REQUIRED:
  --input_csv PATH        Input CSV with columns: image_path, description
  
OPTIONAL:
  --output_dir PATH       Output directory (default: ./output)
  --images_root PATH      Root directory for resolving relative image paths
                          (default: directory containing input CSV)
  --config FILE           YAML config file (default: configs/default_config.yaml)
  --resume                Resume from existing outputs if present
  --help                  Show this help message

STEP-SPECIFIC OPTIONS:
  # Step 1: Caption → Objects
  --step1_model MODEL     HuggingFace model ID (default: microsoft/Phi-3.5-mini-instruct)
  --step1_batch_size N    Batch size for Step 1 (default: 1)
  --step1_use_4bit        Use 4-bit quantization (requires bitsandbytes)
  --step1_force_cpu       Force CPU for Step 1
  
  # Step 2: SAM3 Visual Check
  --sam3_box_threshold F  SAM3 confidence threshold (default: 0.2)
  --sam3_checkpoint PATH  Path to local SAM3 checkpoint
  
  # Step 3: VLM Validation
  --step3_llm_batch_size N        Batch size for Step 3 (default: 1)
  --vllm_prompt_style STYLE       strict_yn | describe_then_yesno (default: strict_yn)
  
  # Step 4: LLM Filter
  --filter_backend BACKEND        hf | openai_compat (default: hf)
  --filter_model MODEL            Model ID for Step 4
  --filter_device DEVICE          cpu | cuda | cuda:0 (default: cpu)
  --filter_batch_size N           Batch size for Step 4 (default: 64)

EXAMPLES:
  # Run on example dataset
  ./run_pipeline.sh \
    --input_csv examples/sample_input.csv \
    --images_root examples/images \
    --output_dir examples/output

  # Run with GPU optimizations
  ./run_pipeline.sh \
    --input_csv data.csv \
    --config configs/gpu_config.yaml \
    --step1_use_4bit \
    --filter_device cuda

  # Resume interrupted run
  ./run_pipeline.sh \
    --input_csv data.csv \
    --output_dir output \
    --resume

For more information, see README.md and docs/
EOF
}

# Default values
INPUT_CSV=""
OUTPUT_DIR="./output"
IMAGES_ROOT=""
CONFIG_FILE="configs/default_config.yaml"
RESUME="0"

# Step defaults
STEP1_MODEL="microsoft/Phi-3.5-mini-instruct"
STEP1_BATCH_SIZE="1"
STEP1_USE_4BIT="0"
STEP1_FORCE_CPU="0"

SAM3_BOX_THRESHOLD="0.2"
SAM3_CHECKPOINT=""

STEP3_LLM_BATCH_SIZE="1"
VLLM_PROMPT_STYLE="strict_yn"

FILTER_BACKEND="hf"
FILTER_MODEL="microsoft/Phi-3.5-mini-instruct"
FILTER_DEVICE="cpu"
FILTER_BATCH_SIZE="64"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --input_csv)
      INPUT_CSV="${2:-}"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --images_root)
      IMAGES_ROOT="${2:-}"
      shift 2
      ;;
    --config)
      CONFIG_FILE="${2:-}"
      shift 2
      ;;
    --resume)
      RESUME="1"
      shift 1
      ;;
    --step1_model)
      STEP1_MODEL="${2:-}"
      shift 2
      ;;
    --step1_batch_size)
      STEP1_BATCH_SIZE="${2:-}"
      shift 2
      ;;
    --step1_use_4bit)
      STEP1_USE_4BIT="1"
      shift 1
      ;;
    --step1_force_cpu)
      STEP1_FORCE_CPU="1"
      shift 1
      ;;
    --sam3_box_threshold)
      SAM3_BOX_THRESHOLD="${2:-}"
      shift 2
      ;;
    --sam3_checkpoint)
      SAM3_CHECKPOINT="${2:-}"
      shift 2
      ;;
    --step3_llm_batch_size)
      STEP3_LLM_BATCH_SIZE="${2:-}"
      shift 2
      ;;
    --vllm_prompt_style)
      VLLM_PROMPT_STYLE="${2:-}"
      shift 2
      ;;
    --filter_backend)
      FILTER_BACKEND="${2:-}"
      shift 2
      ;;
    --filter_model)
      FILTER_MODEL="${2:-}"
      shift 2
      ;;
    --filter_device)
      FILTER_DEVICE="${2:-}"
      shift 2
      ;;
    --filter_batch_size)
      FILTER_BATCH_SIZE="${2:-}"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Validate required arguments
if [[ -z "${INPUT_CSV}" ]]; then
  echo "[ERROR] --input_csv is required" >&2
  usage >&2
  exit 1
fi

if [[ ! -f "${INPUT_CSV}" ]]; then
  echo "[ERROR] Input CSV not found: ${INPUT_CSV}" >&2
  exit 1
fi

# Resolve paths
INPUT_CSV_ABS="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "${INPUT_CSV}")"
OUTPUT_DIR_ABS="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "${OUTPUT_DIR}")"

# Determine images_root
if [[ -z "${IMAGES_ROOT}" ]]; then
  IMAGES_ROOT="$(dirname "${INPUT_CSV_ABS}")"
else
  IMAGES_ROOT="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "${IMAGES_ROOT}")"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR_ABS}"

# Define output files
OBJECTS_CSV="${OUTPUT_DIR_ABS}/objects.csv"
SAM3_PROGRESS_CSV="${OUTPUT_DIR_ABS}/sam3_progress.csv"
VLLM_CHECKED_CSV="${OUTPUT_DIR_ABS}/vllm_checked.csv"
FILTERED_CSV="${OUTPUT_DIR_ABS}/filtered.csv"
FILTERED_SUMMARY_CSV="${OUTPUT_DIR_ABS}/filtered_summary.csv"

echo "========================================"
echo "VPR Quality Pipeline"
echo "========================================"
echo "Input:        ${INPUT_CSV_ABS}"
echo "Images root:  ${IMAGES_ROOT}"
echo "Output:       ${OUTPUT_DIR_ABS}"
echo "Resume:       ${RESUME}"
echo ""

# Build common args
COMMON_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  COMMON_ARGS+=(--resume)
fi

# ============================================================================
# Step 1: Caption → Objects
# ============================================================================
echo "[Step 1/4] Extracting objects from captions..."
STEP1_ARGS=(
  --input_csv "${INPUT_CSV_ABS}"
  --output_dir "${OUTPUT_DIR_ABS}"
  --output_csv "${OBJECTS_CSV}"
  --use_merged_prompt
  --batch_size "${STEP1_BATCH_SIZE}"
  --model "${STEP1_MODEL}"
)
if [[ "${RESUME}" == "1" ]]; then
  STEP1_ARGS+=(--resume)
fi
if [[ "${STEP1_USE_4BIT}" == "1" ]]; then
  STEP1_ARGS+=(--use_4bit)
fi
if [[ "${STEP1_FORCE_CPU}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

python3 src/step1_caption_to_objects.py "${STEP1_ARGS[@]}"
echo "[Step 1/4] Complete: ${OBJECTS_CSV}"
echo ""

# ============================================================================
# Step 2: SAM3 Visual Verification
# ============================================================================
echo "[Step 2/4] Running SAM3 visual verification..."
STEP2_ARGS=(
  --input_csv "${OBJECTS_CSV}"
  --realtime_progress_csv "${SAM3_PROGRESS_CSV}"
  --box_threshold "${SAM3_BOX_THRESHOLD}"
  --images_root "${IMAGES_ROOT}"
  --resume
)
if [[ -n "${SAM3_CHECKPOINT}" ]]; then
  STEP2_ARGS+=(--checkpoint_path "${SAM3_CHECKPOINT}")
fi

python3 src/step2_sam3_visual_check.py "${STEP2_ARGS[@]}"
echo "[Step 2/4] Complete: ${SAM3_PROGRESS_CSV}"
echo ""

# ============================================================================
# Step 3: VLM Validation
# ============================================================================
echo "[Step 3/4] Running VLM validation..."
STEP3_ARGS=(
  --input_csv "${SAM3_PROGRESS_CSV}"
  --output_csv "${VLLM_CHECKED_CSV}"
  --images_root "${IMAGES_ROOT}"
  --llm_batch_size "${STEP3_LLM_BATCH_SIZE}"
  --prompt_style "${VLLM_PROMPT_STYLE}"
  --resume
)

python3 src/step3_vlm_validation.py "${STEP3_ARGS[@]}"
echo "[Step 3/4] Complete: ${VLLM_CHECKED_CSV}"
echo ""

# ============================================================================
# Step 4: LLM Filter (Detectability + Description)
# ============================================================================
echo "[Step 4/4] Running final LLM filtering..."
STEP4_ARGS=(
  --input_csv "${VLLM_CHECKED_CSV}"
  --output_csv "${FILTERED_CSV}"
  --summary_csv "${FILTERED_SUMMARY_CSV}"
  --backend "${FILTER_BACKEND}"
  --batch_size "${FILTER_BATCH_SIZE}"
  --objects_col "objects_vllm_said_no"
  --description_col "description"
  --image_col "image_path"
  --resume
)

if [[ "${FILTER_BACKEND}" == "hf" ]]; then
  STEP4_ARGS+=(--hf_model "${FILTER_MODEL}")
  STEP4_ARGS+=(--hf_device "${FILTER_DEVICE}")
else
  STEP4_ARGS+=(--openai_model "${FILTER_MODEL}")
fi

python3 src/step4_llm_filter.py "${STEP4_ARGS[@]}"
echo "[Step 4/4] Complete: ${FILTERED_CSV}"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo "Output files:"
echo "  1. ${OBJECTS_CSV}"
echo "  2. ${SAM3_PROGRESS_CSV}"
echo "  3. ${VLLM_CHECKED_CSV}"
echo "  4. ${FILTERED_CSV}"
echo "  5. ${FILTERED_SUMMARY_CSV}"
echo ""
echo "To view results:"
echo "  head -n 5 ${FILTERED_CSV}"
echo "  cat ${FILTERED_SUMMARY_CSV}"
echo ""
