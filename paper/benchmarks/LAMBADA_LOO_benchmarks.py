import argparse
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS = {
    "lambada": SCRIPT_DIR.parent / "data" / "lambada_prompts_1000.json",
    "IWSLT2017DE_EN": SCRIPT_DIR.parent / "data" / "IWSLT2017DE_EN.json",
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=("meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"),
    default="meta-llama/Llama-3.2-3B",
)
parser.add_argument("--dataset", choices=DATASETS, default="lambada")
parser.add_argument("--devices", default="0,1")
parser.add_argument("--cutoff", type=int, default=1000)
parser.add_argument("--fisher-k", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
model_name = args.model
prompts_path = DATASETS[args.dataset]
model_name_short = model_name.split("/")[-1]
save_path = (
    SCRIPT_DIR.parent
    / "results"
    / f"{model_name_short}__LOO_KL_{args.dataset}_loo_results.json"
)
mode_list = ['temperature', 'semantic', 'gradient_x_input', 'ig', 'random', 'fisher']
fisher_k = args.fisher_k
cutoff = args.cutoff
print(f"Results will be saved to {save_path}")

from torch import nn


from math import sqrt
import numpy as np                                       
                           
import torch                                          

                              
from transformers import AutoModelForCausalLM, AutoTokenizer  
from tqdm import tqdm
import matplotlib.pyplot as plt  
import torch.nn.functional as F
import gc
import re
import copy

np.random.seed(args.seed)
torch.manual_seed(args.seed)

from JacobianScopes import JacobianScopes_utils as JCBScope_utils
from JacobianScopes import JacobianScopes

# Move to GPU with optimal dtype
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"

import json


with open(prompts_path, "r", encoding="utf-8") as f:
    all_prompts_data = json.load(f)
    
prompts_list = [entry['text'] for entry in all_prompts_data]
    
len(prompts_list)

# Load the tokenizer and model



if device == "cpu":
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
embedding_layer = model.get_input_embeddings()
embed_device = embedding_layer.weight.device    

front_pad = 0
back_pad = 0

front_strip = 0

# Get special tokens if available
bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

presence_ratios = np.linspace(0, 1.0, 11, endpoint=True)[1:]

def compute_kl_divergence(log_probs_p, log_probs_q, eps=1e-10):
    """KL(P || Q) = sum P * (log P - log Q). Inputs are log-probabilities."""
    p = torch.exp(log_probs_p).clamp(min=eps)
    q = torch.exp(log_probs_q).clamp(min=eps)
    return (p * (log_probs_p - torch.log(q))).sum().item()

def loo_kl_ranking(string, verbose=False):
    """
    Leave-One-Out (LOO) ranking: for each token, zero it out and record the change in
    KL divergence of the predicted token distribution. Returns ranked token indices,
    decoded tokens, and corresponding KL divergences.
    """
    input_ids_list = tokenizer(string, add_special_tokens=False)["input_ids"]
    if eos_token_id is not None:
        input_ids_list += [eos_token_id] * back_pad

    input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(embed_device)
    attention_mask = torch.ones_like(input_ids, device=embed_device)
    seq_len = input_ids.size(1)

    decoded_tokens = tokenizer.batch_decode([[tid] for tid in input_ids[0].tolist()], skip_special_tokens=True)

    
    token_idx = [idx for idx in range(front_pad, len(decoded_tokens), 1)][front_strip:]
    decoded_tokens_list = [decoded_tokens[idx] for idx in token_idx]

    d_model = embedding_layer.embedding_dim
    # Use plain tensor (no gradient needed for LOO)
    residual = torch.zeros(len(token_idx), d_model, device=embed_device)
    presence = torch.ones(len(decoded_tokens), 1, device=embed_device)

    loss_position = seq_len - 2

    model.eval()
    forward_pass = JCBScope_utils.customize_forward_pass(
        model, residual, presence, input_ids, token_idx, attention_mask
    )

    # Full-context forward pass (no ablation) - no gradient needed
    with torch.no_grad():
        _, logits_orig = forward_pass(
            loss_position=loss_position,
            hidden_norm_as_loss=False,
            unnormalized_logits=False,
            tie_input_output_embed=False,
        )
    log_probs_orig = F.log_softmax(logits_orig[loss_position].float(), dim=-1)
    del logits_orig
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    if verbose:
        print(f"Computing LOO KL ranking ({len(token_idx)} tokens)...")

    # Inner loop: zero out each token one at a time, compute KL divergence
    kl_values = []
    # for i, idx in enumerate(tqdm(token_idx, desc="LOO tokens", leave=False, disable=not verbose)):
    for i, idx in enumerate(token_idx):
        presence_loo = presence.clone()
        presence_loo[idx, 0] = 0.0

        forward_pass_loo = JCBScope_utils.customize_forward_pass(
            model, residual, presence_loo, input_ids, token_idx, attention_mask
        )
        with torch.no_grad():
            _, logits_loo = forward_pass_loo(
                loss_position=loss_position,
                hidden_norm_as_loss=False,
                unnormalized_logits=False,
                tie_input_output_embed=False,
            )
        log_probs_loo = F.log_softmax(logits_loo[loss_position].float(), dim=-1)
        kl = compute_kl_divergence(log_probs_orig.cpu(), log_probs_loo.cpu())
        kl_values.append(kl)
        del logits_loo, forward_pass_loo
        if device != "cpu":
            torch.cuda.empty_cache()

    del forward_pass

    # Rank by KL divergence (descending: higher KL = more important token)
    kl_values = np.array(kl_values)
    rank_order = np.argsort(kl_values)[::-1]

    ranked_decoded_tokens = [decoded_tokens_list[j] for j in rank_order]

    true_token_id = input_ids[0, loss_position + 1].item()
    predicted_token_id = log_probs_orig.argmax().item()
    true_token_str = tokenizer.decode([true_token_id])
    predicted_token_str = tokenizer.decode([predicted_token_id])
    ranked_token_indices = [int(token_idx[j]) for j in rank_order]
    
    if verbose:
        print(string)
        print(f"True: {true_token_str!r} | Predicted: {predicted_token_str!r}")
        print(f"Top-5 LOO tokens (by KL): {ranked_decoded_tokens[:5]}")

    return {
        "ranked_token_indices": ranked_token_indices,
        "decoded_tokens": decoded_tokens_list,
        "kl_divergences": kl_values.tolist(),
        "true_token": true_token_str,
        "predicted_token": predicted_token_str,
    }    

def get_influence_ranking(string, scope, fisher_method='low_rank', fisher_k = None):
    """Compute influence scores via JacobianScopes. scope from global mode if None; fisher_method when scope='fisher'; semantic_path_integral when scope='semantic'."""   
    input_ids_list = tokenizer(string, add_special_tokens=False)["input_ids"]
    if eos_token_id is not None:
        input_ids_list += [eos_token_id] * back_pad

    decoded_tokens = tokenizer.batch_decode([[tid] for tid in input_ids_list], skip_special_tokens=True)
    grad_idx = [idx for idx in range(front_pad, len(decoded_tokens), 1)][front_strip:]
    tick_label_text = [decoded_tokens[idx] for idx in grad_idx]

    if scope == 'random':
        most_influential_local_idx = int(np.random.randint(0, len(grad_idx)))
        most_influential_idx = grad_idx[most_influential_local_idx]
        grad_vals = np.random.random(len(grad_idx)).astype(np.float32)
        ablated_indices = np.array([most_influential_local_idx])
        return most_influential_idx, grad_vals, ablated_indices, tick_label_text, grad_idx

    input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(embed_device)
    attention_mask = torch.ones_like(input_ids, device=embed_device)
    seq_len = input_ids.size(1)

    d_model = embedding_layer.embedding_dim
    residual = nn.Parameter(torch.zeros(len(grad_idx), d_model, device=embed_device))
    presence = torch.ones(len(decoded_tokens), 1, device=embed_device)
    model.eval()
    forward_pass = JCBScope_utils.customize_forward_pass(
        model, residual, presence, input_ids, grad_idx, attention_mask
    )
    loss_position = seq_len - 2

    if scope == 'fisher':
        lm_head = JCBScope_utils.get_lm_head(model)
        grad_vals, _ = JacobianScopes.fisher_scope_scores(
            forward_pass, residual, loss_position, lm_head,method=fisher_method,
            k = fisher_k
        )
    elif scope == 'temperature':
        grad_vals, _ = JacobianScopes.temperature_scope_scores(forward_pass, residual, loss_position)
        del _
    elif scope == 'semantic':
        grad_vals, _ = JacobianScopes.semantic_scope_scores(
            forward_pass, residual, loss_position,
            path_integral=False, grad_idx=grad_idx
        )
        del _
    elif scope == 'gradient_x_input':
        grad_vals, _ = JacobianScopes.gradient_x_input_scores(
            forward_pass, residual, loss_position, embedding_layer, input_ids, grad_idx
        )
        del _
    elif scope == 'ig':
        grad_vals, _ = JacobianScopes.semantic_scope_scores(
            forward_pass, residual, loss_position,
            grad_idx=grad_idx,
            path_integral=True,
            presence_ratios=presence_ratios,
        )
        del _

    else:
        raise ValueError(f"Unknown scope: {scope!r}")
    if grad_vals.ndim > 1:
        grad_vals = grad_vals.squeeze()
    if not isinstance(grad_vals, np.ndarray):
        grad_vals = np.asarray(grad_vals, dtype=np.float32)

    most_influential_local_idx = int(np.argmax(grad_vals))
    most_influential_idx = grad_idx[most_influential_local_idx]
    ablated_indices = np.array([most_influential_local_idx])

    # del model
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    return most_influential_idx, grad_vals, ablated_indices, tick_label_text, grad_idx

for mode in mode_list:
    # Store mode-specific info (overwrites if rerunning same mode)
    mode_name = mode
    
    if mode == "fisher":
        mode_name += f"_k{fisher_k}"
    
    print(f"Processing mode: {mode_name}")        

    # Fields we want to cache from loo_kl_ranking
    loo_out_fields = [
        "ranked_token_indices",
        "decoded_tokens",
        "kl_divergences",
        "true_token",
        "predicted_token",
    ]

    # Load existing results if present, otherwise start fresh
    results = []
    if save_path.exists():
        with open(save_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
            results = existing.get("results", [])
    else:
        results = []

    for i, prompt in enumerate(tqdm(prompts_list[:cutoff], desc="Processing prompts")):
        # print(prompt)

        # Get or create the per-prompt result dict
        if i < len(results):
            item = results[i]
            # Optional safety check:
            # if item.get("prompt") != prompt:
            #     print(f"Warning: prompt mismatch at index {i}")
        else:
            item = {"prompt": prompt}
            results.append(item)

        # 1) Use cached LOO outputs if ranked_token_indices already exists
        if "ranked_token_indices" in item:
            # Reconstruct a LOO_out-like dict from cached fields
            LOO_out = {k: item[k] for k in loo_out_fields if k in item}
        else:
            # 2) Otherwise compute LOO and store all its outputs
            LOO_out = loo_kl_ranking(prompt, verbose=False)
            for k in loo_out_fields:
                item[k] = LOO_out[k]
        
        ranked_token_indices = item["ranked_token_indices"]

        # 3) Continue with mode-specific influence scoring
        most_influential_idx, grad_vals, ablated_indices, tick_label_text, grad_idx = get_influence_ranking(prompt, mode, fisher_k=fisher_k)

        if most_influential_idx in ranked_token_indices:
            rank = ranked_token_indices.index(most_influential_idx)
            ranking_pct = (rank / len(ranked_token_indices)) * 100
        else:
            rank = None
            ranking_pct = None

        item[f"{mode_name}_rank"] = rank
        item[f"{mode_name}_ranking_pct"] = ranking_pct
        item[f"{mode_name}_influence_scores"] = grad_vals.tolist()

        # print(
        #     f"[{i}] {mode_name}_rank = {rank}, {mode_name}_ranking_pct = {ranking_pct:.2g}"
        #     if ranking_pct is not None
        #     else f"[{i}] {mode_name}_rank = {rank}, {mode_name}_ranking_pct = None"
        # )
        # most_influential_token_scope = tick_label_text[grad_idx[most_influential_idx]]
        # print(f"Most influential token index (by {mode_name} scope): {most_influential_token_scope}")
        # most_influential_token_loo = tick_label_text[ranked_token_indices[0]]
        # print(f"Most influential token index (by LOO): {most_influential_token_loo}")

        del grad_vals, ablated_indices, tick_label_text, grad_idx, most_influential_idx, rank, ranking_pct
        if device != "cpu":
            torch.cuda.empty_cache()
        gc.collect()

        # Persist after each iteration, preserving all modes collected so far
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)
            

    ################################################################################################
    
    # Report ranking stats: where does the most influential token (by {mode} scope) rank in LOO?
    rank_key, pct_key = f"{mode_name}_rank", f"{mode_name}_ranking_pct"
    ranks = [r[rank_key] for r in results if r.get(rank_key) is not None]
    pcts = [r[pct_key] for r in results if r.get(pct_key) is not None]
    total = len(results)

    if ranks:
        avg_rank = sum(ranks) / len(ranks)
        avg_pct = sum(pcts) / len(pcts) if pcts else float("nan") 

        # Compute SEM (standard error of the mean) for mean_ranking_pct if possible
        sem_ranking_pct = None
        if pcts and len(pcts) > 1:
            mean_pct = sum(pcts) / len(pcts)
            variance_pct = sum((x - mean_pct) ** 2 for x in pcts) / (len(pcts) - 1)
            sem_ranking_pct = sqrt(variance_pct / len(pcts))

        print(f"\n{mode_name} scope vs LOO ranking ({len(ranks)}/{total} prompts):")
        print(f"  Mean rank of most influential token in LOO: {avg_rank:.2f}")
        if sem_ranking_pct is not None:
            print(f"  Mean ranking percentile: {avg_pct:.2f} ± {sem_ranking_pct:.2f}%")
        else:
            print(f"  Mean ranking percentile: {avg_pct:.2f}%")
    else:
        print("No rank data. Run the processing cell first.")

    # Calculate how often most influential token's ranking percentile is within 5% of the top
    within_5_pct_count = sum(1 for r in results if r.get(pct_key) is not None and r[pct_key] <= 5)
    fraction_within_5_pct = within_5_pct_count / total if total else 0
    print(f"\n{mode_name} scope most influential token is within top 5% of LOO ranking for {within_5_pct_count}/{total} prompts ({fraction_within_5_pct:.1%})")

    # Save to master_results.json
    label = f"{model_name_short}__{mode_name}_{args.dataset}_loo_rank"
    master_path = SCRIPT_DIR.parent / "results" / "master_results.json"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master = {}
    if master_path.exists():
        with open(master_path, "r", encoding="utf-8") as f:
            master = json.load(f)

    entry = {
        "mean_rank": sum(ranks) / len(ranks) if ranks else None,
        "mean_ranking_pct": sum(pcts) / len(pcts) if pcts else None,
        "sem_mean_ranking_pct": sem_ranking_pct,
        "n_with_rank": len(ranks),
        "total": total,
        "within_top5_pct_count": within_5_pct_count,
        "fraction_within_top5_pct": fraction_within_5_pct,
    }
    master[label] = entry
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)
    print(f"\nSaved to {master_path} (label={label})")

            
