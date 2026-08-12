#!/usr/bin/env bash
# Durable rebuttal queue: filler PAD sweeps first, then saliency, then LOO.
# On failure (including CUDA OOM), wait 30 minutes and retry. Sweep scripts
# resume from checkpointed JSON, so retries continue rather than restart.

set -uo pipefail

ROOT="/home/jl3499/JacobianScopes/JacobianScopes"
LOG_DIR="$ROOT/Neurips_rebuttal/logs"
RESULT_DIR="$ROOT/Neurips_rebuttal/filler_aopc_results"
SWEEP="$ROOT/Neurips_rebuttal/run_filler_aopc_sweep.py"
RETRY_SLEEP_SECONDS=1800
MAX_EXAMPLES=1000

mkdir -p "$LOG_DIR"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

log() {
    # stdout is already redirected into rebuttal_queue.log by the nohup launcher.
    echo "$@"
}

model_meta() {
    # Sets MODEL_SHORT and FILLER for a model key.
    case "$1" in
        llama-1b) MODEL_SHORT="Llama-3.2-1B"; FILLER="right_pad" ;;
        llama-3b) MODEL_SHORT="Llama-3.2-3B"; FILLER="right_pad" ;;
        qwen-1.5b) MODEL_SHORT="Qwen2.5-1.5B"; FILLER="pad_eos" ;;
        qwen-3b) MODEL_SHORT="Qwen2.5-3B"; FILLER="pad_eos" ;;
        gemma-1b) MODEL_SHORT="gemma-3-1b-pt"; FILLER="pad" ;;
        gemma-4b) MODEL_SHORT="gemma-3-4b-pt"; FILLER="pad" ;;
        *) log "Unknown model key: $1"; return 1 ;;
    esac
}

filler_count() {
    local model_short="$1"
    local method="$2"
    local dataset="$3"
    local filler="$4"
    local path="$RESULT_DIR/${model_short}__${method}_${dataset}_top0.05_filler_${filler}_results.json"
    python - "$path" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
print(0 if not path.exists() else len(json.loads(path.read_text())["results"]))
PY
}

job_complete() {
    local model_short="$1"
    local dataset="$2"
    local filler="$3"
    local method n
    for method in random_ablation IG gradient_x_input Semantic Temperature Fisher_k_1; do
        n="$(filler_count "$model_short" "$method" "$dataset" "$filler")"
        if [[ "$n" -lt "$MAX_EXAMPLES" ]]; then
            return 1
        fi
    done
    return 0
}

run_with_retry() {
    local desc="$1"
    local log_file="$2"
    shift 2
    local attempt=1
    while true; do
        log "START $desc attempt=$attempt $(date --iso-8601=seconds)"
        {
            echo "===== attempt=$attempt $(date --iso-8601=seconds) ====="
            "$@"
        } >>"$log_file" 2>&1
        local rc=$?
        if [[ "$rc" -eq 0 ]]; then
            log "DONE $desc $(date --iso-8601=seconds)"
            return 0
        fi
        log "FAILED $desc rc=$rc attempt=$attempt; sleeping ${RETRY_SLEEP_SECONDS}s before retry $(date --iso-8601=seconds)"
        sleep "$RETRY_SLEEP_SECONDS"
        attempt=$((attempt + 1))
    done
}

run_filler_job() {
    local model="$1"
    local dataset="$2"
    local visible_gpus="$3"
    local device="$4"
    local max_memory="${5:-}"
    local MODEL_SHORT FILLER
    model_meta "$model" || return 1

    if job_complete "$MODEL_SHORT" "$dataset" "$FILLER"; then
        log "SKIP filler: $model $dataset already complete ($MODEL_SHORT)"
        return 0
    fi

    local log_file="$LOG_DIR/${model}_${dataset}_filler.log"
    if [[ -n "$max_memory" ]]; then
        run_with_retry "filler: $model $dataset (gpus=$visible_gpus device=$device mem=${max_memory}GiB)" \
            "$log_file" \
            env CUDA_VISIBLE_DEVICES="$visible_gpus" \
            python "$SWEEP" \
            --model "$model" \
            --dataset "$dataset" \
            --device "$device" \
            --max-memory-per-gpu-gib "$max_memory" \
            --max-examples "$MAX_EXAMPLES" \
            --save-every 10
    else
        run_with_retry "filler: $model $dataset (gpus=$visible_gpus device=$device)" \
            "$log_file" \
            env CUDA_VISIBLE_DEVICES="$visible_gpus" \
            python "$SWEEP" \
            --model "$model" \
            --dataset "$dataset" \
            --device "$device" \
            --max-examples "$MAX_EXAMPLES" \
            --save-every 10
    fi
}

run_filler_pair() {
    local model="$1"
    local gpu_a="$2"
    local gpu_b="$3"
    local device_mode="${4:-single}"
    local max_memory="${5:-}"
    local lambada_pid iwslt_pid status=0

    if [[ "$device_mode" == "auto" ]]; then
        run_filler_job "$model" "lmbd1000" "$gpu_a" "auto" "$max_memory" &
        lambada_pid=$!
        run_filler_job "$model" "IWSLT2017DE_EN" "$gpu_b" "auto" "$max_memory" &
        iwslt_pid=$!
    else
        # Each child sees only one GPU, remapped to cuda:0.
        run_filler_job "$model" "lmbd1000" "$gpu_a" "cuda:0" &
        lambada_pid=$!
        run_filler_job "$model" "IWSLT2017DE_EN" "$gpu_b" "cuda:0" &
        iwslt_pid=$!
    fi
    wait "$lambada_pid" || status=1
    wait "$iwslt_pid" || status=1
    return "$status"
}

log "=== Rebuttal queue restart $(date --iso-8601=seconds) ==="
log "GPUs available: 0,1,2,3 | OOM/failure retry sleep: ${RETRY_SLEEP_SECONDS}s"
log "Saliency and LOO continue automatically after filler sweeps succeed."

# Highest priority: finish remaining PAD/filler sweeps.
# Qwen-3B LAMBADA is done; resume IWSLT Fisher on GPUs 0,1 (prior single-GPU OOM).
# Gemma-1B runs both datasets in parallel on GPUs 2 and 3 at the same time.
status=0
run_filler_job "qwen-3b" "IWSLT2017DE_EN" "0,1" "auto" "14" &
qwen_pid=$!
run_filler_pair "gemma-1b" "2" "3" "single" &
gemma1_pid=$!
wait "$qwen_pid" || status=1
wait "$gemma1_pid" || status=1
if [[ "$status" -ne 0 ]]; then
    log "WARN: qwen-3b/gemma-1b stage reported failure; continuing with retries already applied inside jobs"
fi

# Gemma-4B FP32 needs sharding; run datasets sequentially across all four GPUs.
run_filler_job "gemma-4b" "lmbd1000" "0,1,2,3" "auto" "14"
run_filler_job "gemma-4b" "IWSLT2017DE_EN" "0,1,2,3" "auto" "14"

log "ALL FILLER SWEEPS COMPLETE $(date --iso-8601=seconds)"

# Second priority: saliency randomization on two models and both datasets.
# The two datasets run in parallel; each cell uses the same fixed seed and 200
# correctly predicted passages.
status=0
run_with_retry "saliency: llama-1b lmbd1000" \
    "$LOG_DIR/saliency_llama-1b_lmbd1000.log" \
    env CUDA_VISIBLE_DEVICES=0 \
    python "$ROOT/Neurips_rebuttal/run_saliency_randomization.py" \
    --model llama-1b \
    --dataset lmbd1000 \
    --device cuda:0 \
    --num-prompts 200 \
    --seed 20260723 &
saliency_1b_lambada_pid=$!
run_with_retry "saliency: llama-1b IWSLT2017DE_EN" \
    "$LOG_DIR/saliency_llama-1b_IWSLT2017DE_EN.log" \
    env CUDA_VISIBLE_DEVICES=1 \
    python "$ROOT/Neurips_rebuttal/run_saliency_randomization.py" \
    --model llama-1b \
    --dataset IWSLT2017DE_EN \
    --device cuda:0 \
    --num-prompts 200 \
    --seed 20260723 &
saliency_1b_iwslt_pid=$!
wait "$saliency_1b_lambada_pid" || status=1
wait "$saliency_1b_iwslt_pid" || status=1
if [[ "$status" -ne 0 ]]; then
    log "WARN: LLaMA-1B saliency stage reported failure"
fi

status=0
run_with_retry "saliency: llama-3b lmbd1000" \
    "$LOG_DIR/saliency_llama-3b_lmbd1000.log" \
    env CUDA_VISIBLE_DEVICES=0,1 \
    python "$ROOT/Neurips_rebuttal/run_saliency_randomization.py" \
    --model llama-3b \
    --dataset lmbd1000 \
    --device auto \
    --max-memory-per-gpu-gib 14 \
    --num-prompts 200 \
    --seed 20260723 &
saliency_3b_lambada_pid=$!
run_with_retry "saliency: llama-3b IWSLT2017DE_EN" \
    "$LOG_DIR/saliency_llama-3b_IWSLT2017DE_EN.log" \
    env CUDA_VISIBLE_DEVICES=2,3 \
    python "$ROOT/Neurips_rebuttal/run_saliency_randomization.py" \
    --model llama-3b \
    --dataset IWSLT2017DE_EN \
    --device auto \
    --max-memory-per-gpu-gib 14 \
    --num-prompts 200 \
    --seed 20260723 &
saliency_3b_iwslt_pid=$!
wait "$saliency_3b_lambada_pid" || status=1
wait "$saliency_3b_iwslt_pid" || status=1
if [[ "$status" -ne 0 ]]; then
    log "WARN: LLaMA-3B saliency stage reported failure"
fi

run_with_retry "prefix concentration" \
    "$LOG_DIR/prefix_concentration.log" \
    python "$ROOT/Neurips_rebuttal/analyze_prefix_concentration.py" \
    --device cuda:0

log "DONE saliency checks $(date --iso-8601=seconds)"

# Lowest priority: IWSLT LOO for LLaMA models.
run_with_retry "IWSLT LOO LLaMA-1B" \
    "$LOG_DIR/llama-1b_IWSLT_loo.log" \
    python "$ROOT/paper/benchmarks/LAMBADA_LOO_benchmarks.py" \
    --model meta-llama/Llama-3.2-1B \
    --dataset IWSLT2017DE_EN \
    --devices 0 \
    --cutoff 1000 \
    --fisher-k 4

run_with_retry "IWSLT LOO LLaMA-3B" \
    "$LOG_DIR/llama-3b_IWSLT_loo.log" \
    python "$ROOT/paper/benchmarks/LAMBADA_LOO_benchmarks.py" \
    --model meta-llama/Llama-3.2-3B \
    --dataset IWSLT2017DE_EN \
    --devices 0,1 \
    --cutoff 1000 \
    --fisher-k 4

log "ALL REBUTTAL JOBS COMPLETE $(date --iso-8601=seconds)"
