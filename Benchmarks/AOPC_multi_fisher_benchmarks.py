
from pathlib import Path

# fisher_k = [16, 64, 256, 1024, 2048]
# method = "low_rank"
fisher_k = [2048]
method = "full"

top_k_fractions = [0.05, 0.1, 0.2]
model_name = "meta-llama/Llama-3.2-1B"
# model_name = "meta-llama/Llama-3.2-3B"
# model_name = "meta-llama/Llama-3.1-8B"
# model_name = "Qwen/Qwen2.5-3B"
# model_name = "Qwen/Qwen2.5-1.5B"

prompts_path = Path("../data/lambada_prompts_1000.json")
dataset_name_short = "lmbd1000"
num_prompts = 1000

import numpy as np
np.random.seed(42)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch import nn
import torch
import math

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc

import sys
sys.path.append('..')
import JCBScope_utils
import JacobianScopes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name_short = model_name.split("/")[-1]
if device.type == "cpu":
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

embedding_layer = model.get_input_embeddings()
embed_device = embedding_layer.weight.device
lm_head = JCBScope_utils.get_lm_head(model)

front_pad = 0
back_pad = 0
front_strip = 0

bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id


def change_in_max_log_p_with_ablation(string, fisher_k_val, top_k_fraction=0.1, verbose=True):
    """Compute Fisher scope with given k, then ablation impact. top_k_fraction can be float or list."""
    top_k_list = [top_k_fraction] if not isinstance(top_k_fraction, (list, tuple)) else list(top_k_fraction)
    input_ids_list = tokenizer(string, add_special_tokens=False)["input_ids"]
    if eos_token_id is not None:
        input_ids_list += [eos_token_id] * back_pad

    input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(embed_device)
    attention_mask = torch.ones_like(input_ids, device=embed_device)
    seq_len = input_ids.size(1)

    decoded_tokens = tokenizer.batch_decode([[tid] for tid in input_ids[0].tolist()], skip_special_tokens=True)
    grad_idx = [idx for idx in range(front_pad, len(decoded_tokens), 1)][front_strip:]
    tick_label_text = [decoded_tokens[idx] for idx in grad_idx]

    d_model = embedding_layer.embedding_dim
    residual = nn.Parameter(torch.zeros(len(grad_idx), d_model, device=embed_device))
    presence = torch.ones(len(decoded_tokens), 1, device=embed_device)
    model.eval()
    forward_pass = JCBScope_utils.customize_forward_pass(
        model, residual, presence, input_ids, grad_idx, attention_mask
    )
    loss_position = seq_len - 2
    n_tokens = len(grad_idx)

    if verbose:
        print(f"Computing Fisher Scope (k={fisher_k_val})...")
    grad_vals, logits_orig = JacobianScopes.fisher_scope_scores(
        forward_pass, residual, loss_position, lm_head, method=method, k=fisher_k_val
    )
    if grad_vals.ndim > 1:
        grad_vals = grad_vals.squeeze()

    max_idx_orig = logits_orig[loss_position].argmax().item()
    max_log_prob_orig = torch.log_softmax(logits_orig[loss_position], dim=-1)[max_idx_orig].item()
    logits_orig_cpu = logits_orig.detach().cpu()
    del logits_orig
    torch.cuda.empty_cache()
    del forward_pass

    delta_log_prob_dict = {}
    ablated_indices_dict = {}
    for k in top_k_list:
        n_top = max(1, int(n_tokens * k))
        ablated_indices = grad_vals.argsort()[::-1][:n_top]
        presence_ablated = presence.clone()
        presence_ablated[[grad_idx[i] for i in ablated_indices], 0] = 0.0

        forward_pass_ablated = JCBScope_utils.customize_forward_pass(
            model, residual, presence_ablated, input_ids, grad_idx, attention_mask
        )
        with torch.no_grad():
            _, logits_ablated = forward_pass_ablated(
                loss_position=loss_position,
                hidden_norm_as_loss=False,
                unnormalized_logits=True,
                tie_input_output_embed=False,
            )
        max_log_prob_ablated = torch.log_softmax(logits_ablated[loss_position], dim=-1)[max_idx_orig].item()
        delta_log_prob_dict[k] = max_log_prob_ablated - max_log_prob_orig
        ablated_indices_dict[k] = ablated_indices
        del logits_ablated, forward_pass_ablated
        torch.cuda.empty_cache()

    true_token_id = input_ids[0, loss_position + 1].item()
    predicted_token_id = max_idx_orig
    true_token_str = tokenizer.decode([true_token_id])
    predicted_token_str = tokenizer.decode([predicted_token_id])

    return delta_log_prob_dict, grad_vals, ablated_indices_dict, tick_label_text, logits_orig_cpu, true_token_str, predicted_token_str


import json

with open(prompts_path, "r", encoding="utf-8") as f:
    all_prompts_data = json.load(f)

if isinstance(all_prompts_data, dict) and "prompts" in all_prompts_data:
    prompts_list = all_prompts_data["prompts"]
else:
    prompts_list = all_prompts_data

prompts_to_process = prompts_list[:num_prompts]

fisher_k_list = [fisher_k] if not isinstance(fisher_k, (list, tuple)) else list(fisher_k)
print("Fisher k values:", fisher_k_list)

for k_val in fisher_k_list:
    print(f"Processing Fisher k={k_val}")
    mode_name = f"Fisher_k_{k_val}"
    labels_by_k = {
        k: f"{model_name_short}__{mode_name}_{dataset_name_short}_top{k}"
        for k in top_k_fractions
    }
    results_by_k = {k: [] for k in top_k_fractions}

    for i, item in enumerate(tqdm(prompts_to_process, desc="Processing prompts")):
        prompt = item["text"] if isinstance(item, dict) else item
        delta_log_prob_dict, grad_vals, ablated_indices_dict, tick_label_text, logits_orig, true_token, predicted_token = change_in_max_log_p_with_ablation(
            string=prompt, fisher_k_val=k_val, top_k_fraction=top_k_fractions, verbose=False
        )
        for k in top_k_fractions:
            entry = {
                "delta_log_prob": delta_log_prob_dict[k],
                "prompt": prompt,
                "true_token": true_token,
                "predicted_token": predicted_token,
                "index": i,
            }
            if isinstance(item, dict):
                entry.update({key: val for key, val in item.items() if key != "prompt"})
            results_by_k[k].append(entry)

    result_dir = Path("../results")
    result_dir.mkdir(parents=True, exist_ok=True)
    master_path = Path("../results/master_results.json")
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master = {}
    if master_path.exists():
        with open(master_path, "r", encoding="utf-8") as f:
            master = json.load(f)

    for k in top_k_fractions:
        label = labels_by_k[k]
        results = results_by_k[k]
        # Save per–top_k result file
        result_path = result_dir / f"{label}_results.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"label": label, "top_k_fraction": k, "results": results}, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(results)} results to {result_path}")
        # Summary stats and master entry
        num_correct = sum(1 for r in results if r.get('true_token', '') == r.get('predicted_token', ''))
        total = len(results)
        accuracy = num_correct / total if total > 0 else 0.0
        deltas = [r["delta_log_prob"] for r in results if r.get("delta_log_prob") is not None]
        avg_delta = sum(deltas) / len(deltas) if deltas else float("nan")
        variance_delta = (
            sum((x - avg_delta) ** 2 for x in deltas) / len(deltas) if deltas else float("nan")
        )
        if deltas and len(deltas) > 1:
            std_delta = math.sqrt(variance_delta)
            sem_delta = std_delta / math.sqrt(len(deltas))
        else:
            sem_delta = float("nan")
        print(f"\n--- top_k_fraction = {k} (label={label}) ---")
        print(f"Correct prediction rate: {num_correct}/{total} = {accuracy:.2%}")
        print(f"Average ablation impact (mean delta_log_prob): {avg_delta:.4f}")
        print(f"Variance of ablation impact (delta_log_prob): {variance_delta:.6f}")
        print(f"Standard error of the mean (SEM) for |delta_log_prob|: {sem_delta:.6f}")
        master[label] = {
            "top_k_fraction": k,
            "avg_delta": avg_delta if deltas and not np.isnan(avg_delta) else None,
            "variance_delta": variance_delta if deltas and not np.isnan(variance_delta) else None,
            "sem_delta": sem_delta,
            "accuracy": accuracy,
            "num_correct": num_correct,
            "total": total,
        }

    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)
    print(f"\nSaved all entries to {master_path}")
