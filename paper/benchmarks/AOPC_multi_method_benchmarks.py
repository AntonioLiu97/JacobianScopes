
from pathlib import Path
modes = ['Temperature', 'Semantic', 'gradient_x_input','Fisher', "random_ablation", 'IG']
# modes = ['IG']
modes = ['Fisher']
# fisher_k = 4
# fisher_k = 16
fisher_k = 1

# path intervals for integrated gradients
presence_list = [0.2, 0.4, 0.6, 0.8, 1]

top_k_fractions = [0.05, 0.1, 0.2]
model_name = "meta-llama/Llama-3.2-1B"
# model_name = "meta-llama/Llama-3.2-3B"
# model_name = "meta-llama/Llama-3.1-8B"
# model_name = "Qwen/Qwen2.5-3B"
# model_name = "Qwen/Qwen2.5-1.5B"

prompts_path = Path("../data/lambada_prompts_1000.json")
dataset_name_short = "lmbd1000"
num_prompts = 1000
# num_prompts = 200

# prompts_path = Path("../data/IWSLT2017DE_EN.json")
# dataset_name_short = "IWSLT2017DE_EN"
# num_prompts = 1000
# num_prompts = 200

import numpy as np  
np.random.seed(42)    


import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  
from torch import nn                                                            
import torch                                          
import math
                              
from transformers import AutoModelForCausalLM, AutoTokenizer  
from tqdm import tqdm


from JacobianScopes import JacobianScopes_utils as JCBScope_utils
from JacobianScopes import JacobianScopes

# Move to GPU with optimal dtype
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"

# Load the tokenizer and model


model_name_short = model_name.split("/")[-1]
if device == "cpu":
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
embedding_layer = model.get_input_embeddings()
embed_device = embedding_layer.weight.device    

if 'Fisher' in modes:
    lm_head = JCBScope_utils.get_lm_head(model)

front_pad = 0
back_pad = 0

front_strip = 0
# num_prompts = 1


# Get special tokens if available
bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id



def change_in_max_log_p_with_ablation(string, mode, top_k_fraction=0.1, verbose=True):
    """Optimized: single model load, retain_graph=False, free logits before ablation.
    top_k_fraction can be a float or a list of floats. Returns delta_log_prob dict keyed by k.
    fisher_k_override: if set, used as k for Fisher scope instead of global fisher_k."""
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
    hidden_norm_as_loss = (mode == 'Temperature')
    loss_position = seq_len - 2

    if verbose:
        print(f"Computing {mode} Scope...")
    n_tokens = len(grad_idx)

    if mode == "random_ablation":
        with torch.no_grad():
            _, logits_orig = forward_pass(
                loss_position=loss_position,
                hidden_norm_as_loss=hidden_norm_as_loss,
                unnormalized_logits=True,
                tie_input_output_embed=False,
            )
        grad_vals = None
    else:
        if mode == "IG":
            grad_list = []
            for alpha in presence_list:
                loss, logits_orig = forward_pass(
                    loss_position=loss_position,
                    hidden_norm_as_loss=hidden_norm_as_loss,
                    unnormalized_logits=False,
                    tie_input_output_embed=False,
                    alpha=alpha,
                )
                g = torch.autograd.grad(loss, residual, retain_graph=False)[0]
                grad_list.append(g.detach().clone())
                del loss
            grads = torch.stack(grad_list).mean(dim=0)
            del grad_list
            with torch.no_grad():
                token_embeds = JCBScope_utils.embedding_lookup(input_ids[0, grad_idx], embedding_layer)
            grad_vals = (grads * token_embeds.to(grads.device)).norm(dim=-1).squeeze().cpu().numpy()
            del grads
        elif mode == "Temperature":
            grad_vals, logits_orig = JacobianScopes.temperature_scope_scores(forward_pass, residual, loss_position)
        elif mode == "Semantic":
            grad_vals, logits_orig = JacobianScopes.semantic_scope_scores(forward_pass, residual, loss_position)
        elif mode == "gradient_x_input":
            grad_vals, logits_orig = JacobianScopes.gradient_x_input_scores(forward_pass, residual, loss_position, embedding_layer, input_ids, grad_idx)
        elif mode == "Fisher":
            grad_vals, logits_orig = JacobianScopes.fisher_scope_scores(forward_pass,residual,loss_position,lm_head,method="low_rank",
            k=fisher_k)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
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
        if mode == "random_ablation":
            ablated_indices = np.random.choice(n_tokens, size=min(n_top, n_tokens), replace=False)
        else:
            ablated_indices = grad_vals.argsort()[::-1][:n_top]
        presence_ablated = presence.clone()
        presence_ablated[[grad_idx[i] for i in ablated_indices], 0] = 0.0

        forward_pass_ablated = JCBScope_utils.customize_forward_pass(
            model, residual, presence_ablated, input_ids, grad_idx, attention_mask
        )
        with torch.no_grad():
            _, logits_ablated = forward_pass_ablated(
                loss_position=loss_position,
                hidden_norm_as_loss=hidden_norm_as_loss,
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

    if verbose:
        last_k = top_k_list[-1]
        dropped_words = []
        for i in ablated_indices_dict[last_k]:
            tok = input_ids[0, grad_idx[int(i)]].item()
            word = tokenizer.decode([tok]).strip()
            dropped_words.append(word)

    return delta_log_prob_dict, grad_vals, ablated_indices_dict, tick_label_text, logits_orig_cpu, true_token_str, predicted_token_str

import json



with open(prompts_path, "r", encoding="utf-8") as f:
    all_prompts_data = json.load(f)

# Handle both list-of-dicts and {prompts: [...]} formats
if isinstance(all_prompts_data, dict) and "prompts" in all_prompts_data:
    prompts_list = all_prompts_data["prompts"]
else:
    prompts_list = all_prompts_data

prompts_to_process = prompts_list[:num_prompts]

print("All modes:", modes)

for mode in modes:
    print(f"Processing {mode}")
    # Multiple labels, one per k
    mode_name = mode
    if mode == "Fisher":
        mode_name = f"{mode}_k_{fisher_k}"
    labels_by_k = {
        k: f"{model_name_short}__{mode_name}_{dataset_name_short}_top{k}"
        for k in top_k_fractions
    }
    results_by_k = {k: [] for k in top_k_fractions}

    for i, item in enumerate(tqdm(prompts_to_process, desc="Processing prompts")):
        prompt = item["text"] if isinstance(item, dict) else item

        delta_log_prob_dict, grad_vals, ablated_indices_dict, tick_label_text, logits_orig, true_token, predicted_token = change_in_max_log_p_with_ablation(
            string=prompt,mode=mode, top_k_fraction=top_k_fractions, verbose=False
        )
        _last_grad_vals, _last_ablated_dict, _last_tick = grad_vals, ablated_indices_dict, tick_label_text
        _last_logits = logits_orig.detach().cpu() if hasattr(logits_orig, "cpu") else logits_orig
        del logits_orig
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

    # Save results: one json per k
    result_dir = Path("../results")
    result_dir.mkdir(parents=True, exist_ok=True)
    for k in top_k_fractions:
        label = labels_by_k[k]
        result_path = result_dir / f"{label}_results.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"label": label, "top_k_fraction": k, "results": results_by_k[k]}, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(results_by_k[k])} results to {result_path}")

    # Restore last successful result for visualization cells (use last k)
    if "_last_grad_vals" in dir():
        _last_k = top_k_fractions[-1]
        grad_vals, ablated_indices, tick_label_text = _last_grad_vals, _last_ablated_dict[_last_k], _last_tick
        logits_orig = _last_logits
        
    # ## Summary of delta_log_prob for each prompt (per k)
    # for k in top_k_fractions:
    #     print(f"\n--- top_k_fraction = {k} ---")
    #     for r in results_by_k[k]:
    #         d = r.get("delta_log_prob")
    #         d_str = f"{d:.4f}" if d is not None else "ERROR"
    #         true_token = r.get('true_token', '')
    #         pred_token = r.get('predicted_token', '')
    #         is_correct = (true_token == pred_token)
    #         correctness = "✔" if is_correct else "✘"
    #         print(f"[{r['index']}] delta_log_prob={d_str}  true={true_token!r} pred={pred_token!r}  -> {correctness}")
        

    master_path = Path("../results/master_results.json")
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master = {}
    if master_path.exists():
        with open(master_path, "r", encoding="utf-8") as f:
            master = json.load(f)

    for k in top_k_fractions:
        results = results_by_k[k]
        label = labels_by_k[k]
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

        entry = {
            "top_k_fraction": k,
            "avg_delta": avg_delta if deltas and not np.isnan(avg_delta) else None,
            "variance_delta": variance_delta if deltas and not np.isnan(variance_delta) else None,
            "sem_delta": sem_delta,
            "accuracy": accuracy,
            "num_correct": num_correct,
            "total": total,
        }
        master[label] = entry

    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)
    print(f"\nSaved all entries to {master_path}")
            