#!/usr/bin/env bash
# Run the remaining parameter-randomization checks after all 1B/1.5B cells.

set -uo pipefail

ROOT="/home/jl3499/JacobianScopes/JacobianScopes"
REBUTTAL="$ROOT/Neurips_rebuttal"
RUNNER="$REBUTTAL/run_saliency_randomization.py"
RESULTS="$REBUTTAL/saliency_randomization_results"
LOG_DIR="$REBUTTAL/logs"
QUEUE_LOG="$LOG_DIR/saliency_large_queue.log"
RETRY_SLEEP_SECONDS=300
SEED=20260723
NUM_PROMPTS=200

mkdir -p "$LOG_DIR"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec 9>"$LOG_DIR/saliency_large_queue.lock"
if ! flock -n 9; then
    echo "Another large-model saliency queue already holds the lock." >&2
    exit 1
fi

log() {
    echo "$*" | tee -a "$QUEUE_LOG"
}

small_cells_complete() {
    python - "$RESULTS" <<'PY'
import json
import sys
from pathlib import Path

results = Path(sys.argv[1])
models = ("Llama-3.2-1B", "Qwen2.5-1.5B", "gemma-3-1b-pt")
datasets = ("lmbd1000", "IWSLT2017DE_EN")
expected_methods = ["Semantic", "Temperature", "Fisher"]

missing = []
for model in models:
    for dataset in datasets:
        metadata_path = (
            results / f"{model}_{dataset}" / "saliency_randomization_metadata.json"
        )
        if not metadata_path.exists():
            missing.append(f"{model}/{dataset}: no metadata")
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("methods") != expected_methods or metadata.get("num_prompts") != 200:
            missing.append(f"{model}/{dataset}: incomplete or stale")

if missing:
    print("\n".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
}

run_cell() {
    local model="$1"
    local dataset="$2"
    local visible_gpus="$3"
    local attempt=1
    local cell_log="$LOG_DIR/saliency_${model}_${dataset}_scope_rerun.log"

    while true; do
        log "START $model $dataset GPUs=$visible_gpus attempt=$attempt $(date --iso-8601=seconds)"
        if env CUDA_VISIBLE_DEVICES="$visible_gpus" \
            python "$RUNNER" \
                --model "$model" \
                --dataset "$dataset" \
                --device auto \
                --max-memory-per-gpu-gib 14 \
                --num-prompts "$NUM_PROMPTS" \
                --seed "$SEED" \
                --stage all >>"$cell_log" 2>&1; then
            log "DONE $model $dataset $(date --iso-8601=seconds)"
            return 0
        fi
        log "FAILED $model $dataset attempt=$attempt; retrying in ${RETRY_SLEEP_SECONDS}s"
        sleep "$RETRY_SLEEP_SECONDS"
        attempt=$((attempt + 1))
    done
}

run_model_pair() {
    local model="$1"
    local visible_gpus="$2"
    run_cell "$model" "lmbd1000" "$visible_gpus"
    run_cell "$model" "IWSLT2017DE_EN" "$visible_gpus"
}

log "QUEUE START $(date --iso-8601=seconds)"
if ! small_cells_complete >>"$QUEUE_LOG" 2>&1; then
    log "ABORT: all 1B/1.5B cells must finish before large models"
    exit 1
fi
log "VERIFIED all 1B/1.5B cells complete"

# Keep all four GPUs occupied while the two 3B model families run.
run_model_pair "llama-3b" "0,1" &
llama_pid=$!
run_model_pair "qwen-3b" "2,3" &
qwen_pid=$!
wait "$llama_pid"
wait "$qwen_pid"
log "ALL 3B CELLS COMPLETE $(date --iso-8601=seconds)"

# Gemma-4B in FP32 requires all four 16 GiB GPUs.
run_cell "gemma-4b" "lmbd1000" "0,1,2,3"
run_cell "gemma-4b" "IWSLT2017DE_EN" "0,1,2,3"
log "ALL 3B/4B CELLS COMPLETE $(date --iso-8601=seconds)"

python "$REBUTTAL/generate_saliency_check_all.py" >>"$QUEUE_LOG" 2>&1
