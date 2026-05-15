#!/bin/bash
# Dynamically generate and submit evaluation jobs.
# Modes:
#  - all: evaluate on all supported datasets (trirex-bite, trirex, simple-questions, web-qsp)
#  - models: iterate training configs and eval on specified datasets
#  - <CONFIG_DIR>: evaluate all .yaml configs in the given directory (excluding debug.yaml)
# For each config, this creates an sbatch script mirroring launchers/evaluate.sh
# with the config baked in, then submits it.

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 models|all|<CONFIG_DIR> [OUTPUT_PREFIX=eval/eval-] [BATCH_SIZE=16] [MAX_SAMPLES=None] [SPLIT=test] [MODELS_DIR=configs/1-trirex/TESTS] [DATASETS=all]"
    echo "       models -> iterate training configs (default: configs/1-trirex/TESTS) and eval on specified datasets"
    echo "       all    -> evaluate the four dataset templates once (no model sweep)"
    echo "       <CONFIG_DIR> -> evaluate all .yaml configs in the given directory (excluding debug.yaml)"
    echo ""
    echo "DATASETS can be:"
    echo "  all                     -> all datasets (trirex-bite, trirex, simple-questions, web-qsp)"
    echo "  trirex-bite             -> only trirex-bite"
    echo "  trirex                  -> only trirex"
    echo "  simple-questions        -> only simple-questions"
    echo "  web-qsp                 -> only web-qsp"
    echo "  comma-separated list    -> e.g., 'trirex-bite,simple-questions'"
    exit 1
fi

MODE_OR_DIR="$1"
OUTPUT_FILE="${2:-eval/eval-}"
BATCH_SIZE="${3:-16}"
MAX_SAMPLES="${4:-None}"
SPLIT="${5:-test}"
MODELS_DIR_ARG="${6:-}"
DATASETS_ARG="${7:-all}"

# Parse datasets argument
parse_datasets() {
    local datasets_input="$1"
    local -a available_datasets=("trirex-bite" "trirex" "simple-questions" "web-qsp")
    local -a selected_datasets=()
    
    if [ "$datasets_input" = "all" ]; then
        selected_datasets=("${available_datasets[@]}")
    else
        IFS=',' read -ra dataset_list <<< "$datasets_input"
        for dataset in "${dataset_list[@]}"; do
            dataset=$(echo "$dataset" | xargs)  # trim whitespace
            if [[ " ${available_datasets[*]} " =~ " ${dataset} " ]]; then
                selected_datasets+=("$dataset")
            else
                echo "Error: Unknown dataset '$dataset'. Available: ${available_datasets[*]}"
                exit 1
            fi
        done
    fi
    
    printf '%s\n' "${selected_datasets[@]}"
}

# Get selected datasets
mapfile -t SELECTED_DATASETS < <(parse_datasets "$DATASETS_ARG")

CONFIG_FILES=()
MODE="$MODE_OR_DIR"
if [ "$MODE" = "all" ]; then
    # Build config files list based on selected datasets
    declare -A DATASET_TO_CONFIG=(
        [trirex-bite]="configs/3-eval/eval_bite.yaml"
        [trirex]="configs/3-eval/eval_trirex.yaml"
        [simple-questions]="configs/3-eval/eval_simpleQA.yaml"
        [web-qsp]="configs/3-eval/eval_web_qsp.yaml"
    )
    
    for dataset in "${SELECTED_DATASETS[@]}"; do
        CONFIG_FILES+=("${DATASET_TO_CONFIG[$dataset]}")
    done
elif [ "$MODE" = "models" ]; then
    : # handled in a dedicated block below
else
    CONFIG_DIR="$MODE_OR_DIR"
    # Find YAML configs excluding debug.yaml
    mapfile -t CONFIG_FILES < <(find "$CONFIG_DIR" -type f -name "*.yaml" ! -name "debug.yaml" | sort)
    if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
        echo "No .yaml configs found for the provided input: $MODE_OR_DIR"
        exit 0
    fi
fi

# Ensure output and generated job directories exist
OUT_DIR="out"
GEN_DIR="launchers/generated_eval_jobs"
mkdir -p "$OUT_DIR" "$GEN_DIR"

# Helper: generate sbatch script for a single config
generate_and_submit_job() {
    local CONFIG="$1"
    local NAME_NO_EXT="$2"
    local OUTPUT_JSON_SUFFIX="$3"  # e.g., dataset-runname

    local JOB_NAME="eval-${NAME_NO_EXT}"
    local JOB_SCRIPT="${GEN_DIR}/eval_${NAME_NO_EXT}.sh"

    # if output json file already exists do not execute
    if [ -f "${OUTPUT_FILE}${OUTPUT_JSON_SUFFIX}.json" ]; then
        echo "Output JSON file already exists, skipping job submission: ${OUTPUT_FILE}${OUTPUT_JSON_SUFFIX}.json"
        return
    fi

    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --output=${OUT_DIR}/${JOB_NAME}_%j.out
#SBATCH --account=iscrc_kg-lfm
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --chdir=.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=davide.cavicchini@studenti.unitn.it

source ./prepare_env.sh

CONFIG_FILE="${CONFIG}"
OUTPUT_FILE="${OUTPUT_FILE}${OUTPUT_JSON_SUFFIX}.json"
BATCH_SIZE="${BATCH_SIZE}"
MAX_SAMPLES="${MAX_SAMPLES}"
SPLIT="${SPLIT}"

accelerate launch --config-file configs/accelerate_evalconfig.yaml \
  evaluate.py --config "\${CONFIG_FILE}" --output_file "\${OUTPUT_FILE}" --batch_size "\${BATCH_SIZE}" --max_samples "\${MAX_SAMPLES}" --no_baseline --split "\${SPLIT}"
EOF
    chmod +x "$JOB_SCRIPT"
    sbatch "$JOB_SCRIPT"
}

if [ "$MODE" = "models" ]; then
    # Sweep over training configs and evaluate on selected datasets
    if [ -n "$MODELS_DIR_ARG" ]; then
        MODELS_DIR="$MODELS_DIR_ARG"
    else
        MODELS_DIR="${MODELS_DIR:-configs/1-trirex/TESTS}"
    fi
    echo "Using MODELS_DIR=$MODELS_DIR"
    echo "Selected datasets: ${SELECTED_DATASETS[*]}"
    
    mapfile -t TRAIN_CONFIGS < <(find "$MODELS_DIR" -type f -name "*.yaml" ! -name "debug.yaml" | sort)
    if [ ${#TRAIN_CONFIGS[@]} -eq 0 ]; then
        echo "No training configs found in $MODELS_DIR"
        exit 0
    fi

    # Dataset -> template mapping
    declare -A DATASET_TEMPLATES=(
        [trirex-bite]="configs/3-eval/eval_bite.yaml"
        [trirex]="configs/3-eval/eval_trirex.yaml"
        [simple-questions]="configs/3-eval/eval_simpleQA.yaml"
        [web-qsp]="configs/3-eval/eval_web_qsp.yaml"
    )

    GEN_CFG_DIR="launchers/generated_eval_configs"
    mkdir -p "$GEN_CFG_DIR"

    for TRAIN_CFG in "${TRAIN_CONFIGS[@]}"; do
        TRAIN_BASE=$(basename "$TRAIN_CFG")
        TRAIN_STEM="${TRAIN_BASE%.yaml}"
        # Extract run_name from the training config; fallback to tri-KG-LFM-<file-stem>
        RUN_NAME=$(awk -F': ' '/^[[:space:]]*run_name:[[:space:]]*/ {gsub(/"/,"",$2); print $2; exit}' "$TRAIN_CFG")
        if [ -z "$RUN_NAME" ]; then
            RUN_NAME="tri-KG-LFM-${TRAIN_STEM}"
        fi
        CKPT_PATH="/leonardo_work/IscrC_KG-LFM/checkpoints/${RUN_NAME}/best_model"

        for DATASET in "${SELECTED_DATASETS[@]}"; do
            TEMPLATE="${DATASET_TEMPLATES[$DATASET]}"
            GEN_CFG_SUBDIR="${GEN_CFG_DIR}/${RUN_NAME}"
            mkdir -p "$GEN_CFG_SUBDIR"
            GEN_CFG_PATH="${GEN_CFG_SUBDIR}/eval_${DATASET}_${RUN_NAME}.yaml"

            # Generate config overriding start_from_checkpoint
            awk -v path="$CKPT_PATH" '{
                if ($0 ~ /^[[:space:]]*start_from_checkpoint:/) {
                    print "  start_from_checkpoint: \"" path "\""
                } else {
                    print $0
                }
            }' "$TEMPLATE" > "$GEN_CFG_PATH"

            NAME_NO_EXT="${DATASET}-${RUN_NAME}"
            generate_and_submit_job "$GEN_CFG_PATH" "$NAME_NO_EXT" "$NAME_NO_EXT"
            echo "Submitted: $DATASET with checkpoint $CKPT_PATH from $TRAIN_CFG"
            echo ""
            echo ""
        done
    done

    echo "All evaluation jobs submitted for models (${#TRAIN_CONFIGS[@]} models x ${#SELECTED_DATASETS[@]} datasets)."
else
    # Default behavior: submit each provided config as-is
    for CONFIG in "${CONFIG_FILES[@]}"; do
        BASENAME=$(basename "$CONFIG")
        NAME_NO_EXT="${BASENAME%.yaml}"
        generate_and_submit_job "$CONFIG" "$NAME_NO_EXT" "$NAME_NO_EXT"
        echo "Submitted evaluation job for configuration: $CONFIG -> eval-${NAME_NO_EXT}"
    done
    echo "All evaluation jobs submitted (${#CONFIG_FILES[@]} total)."
fi
