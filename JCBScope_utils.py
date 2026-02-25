import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn

def get_lm_head(model):
    if hasattr(model, 'lm_head'):  # LLaMA models
        return model.lm_head
    elif hasattr(model, 'embed_out'):  # GPTNeoX (Pythia) models
        return model.embed_out
    else:
        raise ValueError(f"Unsupported model architecture: {type(model)}")

def embedding_lookup(indices, embedding_layer):
    """
    Look up embeddings at given indices. Uses F.embedding directly to avoid
    Accelerate hook device mismatches when using device_map="auto".
    """
    w_device = embedding_layer.weight.device
    indices = indices.to(w_device) if indices.device != w_device else indices
    return F.embedding(indices, embedding_layer.weight)


def get_input_embeddings(model):
    if hasattr(model, 'get_input_embeddings'):
        return model.get_input_embeddings()
    elif hasattr(model, 'gpt_neox'):  # GPTNeoX (Pythia) models
        return model.gpt_neox.embed_in
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):  # LLaMA models
        return model.model.embed_tokens
    else:
        raise ValueError(f"Unsupported model architecture: {type(model)}")

def customize_forward_pass(model, residual, presence, input_ids, grad_idx, attention_mask, ):
    lm_head = get_lm_head(model)
    embedding_layer = model.get_input_embeddings()
    vocab_embed = embedding_layer.weight
    embed_device = embedding_layer.weight.device
    residual = residual.to(embed_device)
    presence = presence.to(embed_device) 

    with torch.no_grad():
        input_ids_to_dev = input_ids.to(embed_device)
        # Use F.embedding directly to bypass Accelerate hooks that can move inputs to the wrong device
        base_embeds = F.embedding(input_ids_to_dev, embedding_layer.weight) 
        # base_embeds = embedding_layer(input_ids.to(embedding_layer.weight.device))
        
    def build_inputs(residual_override=None, residual_batch=None):
        """residual_override: [n_tokens, d] for single. residual_batch: [B, n_tokens, d] for batched (e.g. finite-diff JVP)."""
        if residual_batch is not None:
            B = residual_batch.shape[0]
            embeds = base_embeds.expand(B, -1, -1).clone()
            embeds[:, grad_idx, :] += residual_batch
            return embeds
        r = residual if residual_override is None else residual_override
        embeds = base_embeds.clone()
        embeds[0, grad_idx, :] += r
        return embeds

    def compute_logits(hidden, built_input_embeds):
        """
            Modify logit of target token to use updated embedding for prediction
        """
        L = built_input_embeds.size(1)
        logits = hidden[0].to(lm_head.device) @ vocab_embed.T
        hidden_prev = hidden[0, :-1]  # [L-1, d]
        embeds_next = built_input_embeds[0, 1:].to(hidden.device)  # [L-1, d]
        target_logits = (hidden_prev * embeds_next).sum(dim=-1)  # [L-1]
        targets = input_ids[0, 1:].to(logits.device)
        logits[torch.arange(L - 1, device=logits.device), targets] = target_logits.to(logits.device)
        return logits


    def forward_pass(loss_position='all', hidden_norm_as_loss=False, unnormalized_logits=False, projection_probe=None,tie_input_output_embed = False, return_input_embeds = False, alpha=1, residual_override=None, residual_batch=None, return_hidden=False):
        embeds = build_inputs(residual_override=residual_override, residual_batch=residual_batch)
        input_embeds = embeds
        batched = residual_batch is not None
        if batched:
            input_embeds = input_embeds * presence.unsqueeze(0)
            input_embeds = input_embeds * alpha
            attn = attention_mask.expand(input_embeds.shape[0], -1)
        else:
            input_embeds[0, :, :] *= presence
            input_embeds[0, :, :] *= alpha
            attn = attention_mask
        
        out = model.model(inputs_embeds=input_embeds,
                        attention_mask=attn,
                        use_cache=False)

        hidden = out.last_hidden_state       # [1, L, d] or [B, L, d]
        # print("hidden", hidden.shape)

        if return_hidden and loss_position != 'all':
            if not torch.is_tensor(loss_position):
                lp = torch.tensor(loss_position, device=hidden.device)
            else:
                lp = loss_position.to(hidden.device)
            return hidden[0, lp, :] if not batched else hidden[:, lp, :]

        if tie_input_output_embed:
            readout_embeds = embeds            
            logits = compute_logits(hidden, readout_embeds)        
        else:
            # lm_head.weight = lm_head.weight.to(hidden.device)
            # lm_head = lm_head.to(hidden.device)
            # logits = hidden[0,] @ lm_head.weight.T
            lm_head_on_device = lm_head.to(hidden.device)
            logits = hidden[0] @ lm_head_on_device.weight.T
        
        targets = input_ids[0, 1:].to(logits.device)
        # print("lm_head ", lm_head.weight.shape)
        # print("logits ",logits.shape)    
        # print("targets ",targets.shape)  
        
        
        ### Total energy for anomaly detection      
        if loss_position == 'all':
            if unnormalized_logits:
                # Extract logits at target positions and sum
                target_logits = logits[torch.arange(len(targets)), targets]  # [L-1]
                loss_full = -target_logits
                loss = loss_full.mean()  # or .sum()
            else:
                loss = nn.CrossEntropyLoss()(logits[:-1], targets)
                loss_full = nn.CrossEntropyLoss(reduction='none')(logits[:-1], targets).detach()  # shape: (seq_len,)
                # loss = nn.CrossEntropyLoss()(logits[5:-1], targets[5:])
                
            return loss, logits, loss_full
        
        
        if not torch.is_tensor(loss_position):
            loss_position = torch.tensor(loss_position)
        
        
        ### random projection for Hutchinson  
        if projection_probe is not None:          
            projection_probe = projection_probe / projection_probe.norm(dim=-1, keepdim=True)
            loss_position = loss_position.to(hidden.device)
            # loss = (hidden[0, loss_position, :] * projection_probe.to(hidden.device)).sum()
            loss = (hidden[0, loss_position, :] * projection_probe.to(hidden.device)).sum(dim=-1)
            if return_input_embeds:
                return loss, logits, input_embeds.detach()
            else:
                return loss, logits
            
        ### Loss at chosen location 
        if hidden_norm_as_loss == True:            
            ### Temperature scope
            hidden_act = hidden[:, loss_position, :].detach()
            hidden_act = hidden_act / hidden_act.norm(dim=-1, keepdim=True)
            # print("hidden_act", hidden_act.shape)
            # print("hidden", hidden.shape)
            loss_position = loss_position.to(hidden.device)
            loss = (hidden[0, loss_position, :] * hidden_act).sum(dim=-1)
            if return_input_embeds:
                return loss, logits, input_embeds.detach()
            else:
                return loss, logits
        
        else:
            target_chosen = targets[loss_position].unsqueeze(0)
            loss_position = loss_position.to(logits.device)
            target_chosen = target_chosen.to(logits.device)
            if unnormalized_logits:
                
                loss = -logits[loss_position,target_chosen]
            else:
                assert isinstance(loss_position, int) or (
                    torch.is_tensor(loss_position) and loss_position.dim() == 0
                ), "loss_position must be either an integer, a 0D tensor, or str(all)"
                
                logits_chosen = logits[loss_position, :].unsqueeze(0)  # [1, V]
                loss = nn.CrossEntropyLoss()(logits_chosen, target_chosen)
            if return_input_embeds:
                return loss, logits, input_embeds.detach()
            else:
                return loss, logits

    return forward_pass 