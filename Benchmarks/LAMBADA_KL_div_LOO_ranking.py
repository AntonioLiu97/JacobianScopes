import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  


model_name = "meta-llama/Llama-3.2-1B"
# model_name = "meta-llama/Llama-3.2-3B"
# model_name = "meta-llama/Llama-3.1-8B"

import numpy as np                                                                  
import torch                                          

                              
from transformers import AutoModelForCausalLM, AutoTokenizer  
from tqdm import tqdm
import torch.nn.functional as F
import gc

import sys
sys.path.append('..')
import JCBScope_utils

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
front_pad = 0
back_pad = 0
front_strip = 0

# num_prompts = 100
# num_prompts = 300
num_prompts = 1000

# Get special tokens if available
bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
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

    ranked_token_indices = [int(token_idx[j]) for j in rank_order]
    ranked_decoded_tokens = [decoded_tokens_list[j] for j in rank_order]
    ranked_kl_divergences = [float(kl_values[j]) for j in rank_order]

    true_token_id = input_ids[0, loss_position + 1].item()
    predicted_token_id = log_probs_orig.argmax().item()
    true_token_str = tokenizer.decode([true_token_id])
    predicted_token_str = tokenizer.decode([predicted_token_id])

    if verbose:
        print(string)
        print(f"True: {true_token_str!r} | Predicted: {predicted_token_str!r}")
        print(f"Top-5 LOO tokens (by KL): {ranked_decoded_tokens[:5]}")

    return {
        "ranked_token_indices": ranked_token_indices,
        "decoded_tokens": ranked_decoded_tokens,
        "kl_divergences": ranked_kl_divergences,
        "true_token": true_token_str,
        "predicted_token": predicted_token_str,
    }
    
import json
from pathlib import Path

# Load prompts from JSON
prompts_path = Path("../data/lambada_prompts_1000.json")
with open(prompts_path, "r", encoding="utf-8") as f:
    all_prompts_data = json.load(f)

# Handle both list-of-dicts and {prompts: [...]} formats
if isinstance(all_prompts_data, dict) and "prompts" in all_prompts_data:
    prompts_list = all_prompts_data["prompts"]
else:
    prompts_list = all_prompts_data

prompts_to_process = prompts_list[:num_prompts]

# Label for LOO results
label = f"{model_name_short}__LOO_KL_lambada"
results = []
for i, item in enumerate(tqdm(prompts_to_process, desc="Processing prompts")):
    print(f"processing {i+1} of {len(prompts_to_process)} prompts")
    
    prompt = item["text"] if isinstance(item, dict) else item
    # print(prompt)
    loo_result = loo_kl_ranking(string=prompt, verbose=False)
    entry = {
        "prompt": prompt,
        "index": i,
        "ranked_token_indices": loo_result["ranked_token_indices"],
        "decoded_tokens": loo_result["decoded_tokens"],
        "kl_divergences": loo_result["kl_divergences"],
        "true_token": loo_result["true_token"],
        "predicted_token": loo_result["predicted_token"],
    }
    if isinstance(item, dict):
        entry.update({k: v for k, v in item.items() if k != "text" and k != "prompt"})
    results.append(entry)


# Save LOO results to JSON
result_path = Path(f"../results/{label}_loo_results.json")
result_path.parent.mkdir(parents=True, exist_ok=True)
with open(result_path, "w", encoding="utf-8") as f:
    json.dump({"label": label, "results": results}, f, indent=2, ensure_ascii=False)

print(f"Saved {len(results)} LOO results to {result_path}")

