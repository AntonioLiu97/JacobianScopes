#!/usr/bin/env bash
set -u

repo_root=/home/jl3499/JacobianScopes/JacobianScopes
output_root=/data/jl3499/jacobian_scope_ig_alpha_sweep
python_bin=/home/jl3499/conda/LLM1/bin/python
runner="$repo_root/paper/benchmarks/ig_alpha_sampling_sweep.py"
plotter="$repo_root/paper/figures/plot_ig_alpha_sampling_sweep.py"
log_dir="$output_root/logs"
mkdir -p "$log_dir"

run_model() {
    local gpu_ids=$1
    local model=$2
    local alpha_batch_size=$3
    local log_name=${model//\//_}
    CUDA_VISIBLE_DEVICES="$gpu_ids" HF_HOME=/data/huggingface_cache \
        "$python_bin" -u "$runner" \
        --model "$model" \
        --datasets lmbd1000 IWSLT2017DE_EN \
        --repo-root "$repo_root" \
        --output-root "$output_root" \
        --num-prompts 1000 \
        --cache-dtype float16 \
        --alpha-batch-size "$alpha_batch_size" \
        --local-files-only \
        >"$log_dir/$log_name.log" 2>&1
}

status=0
run_model 2 Llama-3.2-1B 8 || status=1
run_model 2 Qwen2.5-1.5B 6 || status=1
run_model 2 Gemma-3-1B 6 || status=1

"$python_bin" "$plotter" --output-root "$output_root" \
    >"$log_dir/plot.log" 2>&1 || status=1

printf '%s\n' "$status" >"$output_root/exit_status"
exit "$status"
