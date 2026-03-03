#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  topreward/scripts/run_predict.sh --config-name CONFIG [HYDRA_OVERRIDES...]

Examples:
  topreward/scripts/run_predict.sh --config-name predict_gvl dataset=nyudoor model=gemini
  topreward/scripts/run_predict.sh --config-name predict_topreward dataset=austin_sirius_dataset model=qwen prediction.add_chat_template=true

Common Hydra override types:
  Group selection:
    dataset=nyudoor model=qwen data_loader=huggingface mapper=gemini prompts=default
  Scalar field overrides:
    prediction.num_examples=20 prediction.output_dir=./results model.model_name=Qwen/Qwen3-VL-8B-Instruct
  Boolean/toggles:
    prediction.add_chat_template=true prediction.use_video_description=false
  Add missing keys explicitly:
    ++dataset.num_context_episodes=0

Notes:
  - --config-name should match a config in configs/experiments (without .yaml).
  - All non-wrapper arguments are passed directly to Hydra.
  - Use native Hydra overrides instead of shell-specific flags.
USAGE
}

CONFIG_NAME=""
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
    --config-name)
        if [[ $# -lt 2 || -z "${2:-}" || "${2}" == --* ]]; then
            echo "Missing value for --config-name." >&2
            usage
            exit 1
        fi
        CONFIG_NAME="${2:-}"
        shift 2
        ;;
    --config-name=*)
        CONFIG_NAME="${1#*=}"
        if [[ -z "$CONFIG_NAME" ]]; then
            echo "Missing value for --config-name." >&2
            usage
            exit 1
        fi
        shift
        ;;
    --help|-h)
        usage
        exit 0
        ;;
    --)
        shift
        ARGS+=("$@")
        break
        ;;
    *)
        ARGS+=("$1")
        shift
        ;;
    esac
done

if [[ -z "$CONFIG_NAME" ]]; then
    echo "--config-name is required." >&2
    usage
    exit 1
fi

HYDRA_FULL_ERROR=1 PYTHONPATH=. uv run python3 -m topreward.scripts.predict \
    --config-dir configs/experiments \
    --config-name "${CONFIG_NAME}" \
    "${ARGS[@]}"
