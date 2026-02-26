"""
JacobianScopes: Fisher, Temperature, and Semantic scope implementations.
Uses JCBScope_utils.customize_forward_pass for the forward interface.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import JCBScope_utils


def fisher_hidden_from_logits_and_W(logits_t, W, chunk_size=8192):
    """
    Fisher information matrix at hidden state for a single position.
    Fx [d,d] = W^T (Diag(p) - p p^T) W.

    Args:
        logits_t: [V] logits at the target position
        W: [V, d] lm_head weight
        chunk_size: chunk size for vocab loop to control memory

    Returns:
        Fx: [d, d] symmetric positive semi-definite matrix
    """
    if logits_t.ndim != 1:
        raise ValueError(f"logits_t must be 1D [V], got {tuple(logits_t.shape)}")
    if logits_t.numel() != W.shape[0]:
        raise ValueError(f"logits_t has {logits_t.numel()} entries but W has vocab {W.shape[0]}")

    p = F.softmax(logits_t, dim=-1).to(dtype=W.dtype)
    V, d = W.shape
    S = torch.zeros(d, d, device=W.device, dtype=W.dtype)
    mu = torch.zeros(d, device=W.device, dtype=W.dtype)

    for i in range(0, V, chunk_size):
        Wi = W[i : i + chunk_size]
        pi = p[i : i + chunk_size].unsqueeze(1)
        Ww = Wi * pi
        S += Wi.T @ Ww
        mu += Ww.sum(dim=0)

    return S - torch.outer(mu, mu)


def fisher_scope_scores(
    forward_pass,
    residual,
    loss_position,
    lm_head,
    method="full",
    batch_size=16,
    k=8,
    n_hutchinson_samples=8,
    eps_finite_diff=1e-5,
    progress=False,
):
    """
    Compute Fisher scope influence scores per token: tr(J^T Fx J) per token.

    Args:
        forward_pass: from JCBScope_utils.customize_forward_pass
        residual: nn.Parameter [n_tokens, d_model]
        loss_position: int or tensor, position for loss
        lm_head: model lm_head (for W and logits)
        n_tokens: len(grad_idx)
        d_model: residual.shape[-1]
        method: 'full' | 'low_rank' | 'finite_diff'
        batch_size: for full Jacobian backward batches
        k: top-k for low_rank
        n_hutchinson_samples: for finite_diff
        eps_finite_diff: finite-diff step for finite_diff
        progress: if True, use tqdm for finite_diff loop

    Returns:
        (scores, logits) for method 'full' or 'low_rank'; (scores,) for 'finite_diff'.
        scores: np.ndarray [n_tokens], float32
    """
    W = lm_head.weight.to(residual.device, dtype=residual.dtype)
    n_tokens = residual.shape[0]
    d_model = residual.shape[1]
    if method == "full":
        projection_probes = torch.eye(d_model, d_model, device=residual.device, dtype=residual.dtype)
        losses, logits = forward_pass(loss_position=loss_position, projection_probe=projection_probes)
        num_losses = losses.numel()
        grads_list = []
        for i in range(0, num_losses, batch_size):
            end = min(i + batch_size, num_losses)
            eye_batch = torch.eye(num_losses, device=losses.device, dtype=losses.dtype)[i:end]
            g = torch.autograd.grad(
                outputs=losses,
                inputs=residual,
                grad_outputs=eye_batch,
                is_grads_batched=True,
                retain_graph=(end < num_losses),
            )[0]
            grads_list.append(g)
        grads = torch.cat(grads_list, dim=0)
        del grads_list, losses
        J_all = grads.detach().swapaxes(0, 1)
        del grads
        logits_t = logits[loss_position].to(residual.device)
        Fx = fisher_hidden_from_logits_and_W(logits_t, W)
        tmp = Fx.unsqueeze(0) @ J_all
        F_all = J_all.transpose(1, 2) @ tmp
        scores = np.array([torch.trace(F_all[i]).item() for i in range(n_tokens)], dtype=np.float32)
        return scores, logits

    if method == "low_rank":
        _, logits = forward_pass(loss_position=loss_position)
        logits_t = logits[loss_position].to(residual.device)
        with torch.no_grad():
            Fx = fisher_hidden_from_logits_and_W(logits_t, W)
            orig_device = Fx.device
            Fx_cpu = Fx.float().contiguous().cpu()
            U_full, eigvals, _ = torch.linalg.svd(Fx_cpu)
            eigvals = eigvals.numpy()
            U_full = U_full.to(orig_device)
        k_actual = min(k, len(eigvals))
        idx_top = np.argsort(eigvals)[::-1][:k_actual].copy()
        U_k = U_full[:, idx_top].clone().to(residual.device)
        S_k = torch.tensor(eigvals[idx_top].copy(), device=residual.device, dtype=W.dtype)
        projection_probes_lowrank = U_k.T
        losses_lr, _ = forward_pass(loss_position=loss_position, projection_probe=projection_probes_lowrank)
        eye_k = torch.eye(k_actual, device=losses_lr.device, dtype=losses_lr.dtype)
        grads_lr = torch.autograd.grad(
            outputs=losses_lr, inputs=residual, grad_outputs=eye_k, is_grads_batched=True
        )[0]
        grads_lr = grads_lr.contiguous()
        S_k_diag = torch.diag(S_k).contiguous().to(grads_lr.device)
        scores = np.zeros(n_tokens, dtype=np.float32)
        for tau in range(grads_lr.shape[1]):
            UkT_J = grads_lr[:, tau, :]
            M = UkT_J @ UkT_J.T  # (U_k^T J)(J^T U_k)
            scores[tau] = (S_k_diag @ M).trace().item()
        return scores, logits

    if method == "finite_diff":
        _, logits = forward_pass(loss_position=loss_position)
        logits_t = logits[loss_position].to(residual.device)
        with torch.no_grad():
            Fx = fisher_hidden_from_logits_and_W(logits_t, W)
        torch.manual_seed(42)
        hutch_probes = (
            torch.randint(0, 2, (n_hutchinson_samples, d_model), device=residual.device) * 2 - 1
        ).to(residual.dtype)
        hidden_base = forward_pass(loss_position=loss_position, return_hidden=True)
        hutchinson_Jz_all = torch.zeros(
            residual.shape[0], n_hutchinson_samples, d_model, device=residual.device, dtype=residual.dtype
        )
        iterator = range(residual.shape[0])
        if progress and tqdm is not None:
            iterator = tqdm(iterator, desc="Finite-diff JVP")
        for tau in iterator:
            perturb = torch.zeros(
                n_hutchinson_samples, residual.shape[0], d_model, device=residual.device, dtype=residual.dtype
            )
            perturb[:, tau, :] = hutch_probes
            r_batch = residual.unsqueeze(0) + eps_finite_diff * perturb
            hidden_pert = forward_pass(loss_position=loss_position, residual_batch=r_batch, return_hidden=True)
            Jz = (hidden_pert - hidden_base.unsqueeze(0)) / eps_finite_diff
            hutchinson_Jz_all[tau] = Jz.detach()
        g = hutchinson_Jz_all.to(Fx.device, dtype=Fx.dtype)
        FxJz = g @ Fx
        quad = (FxJz * g).sum(dim=-1)
        scores = quad.mean(dim=1).cpu().numpy().astype(np.float32)
        return scores,logits

    raise ValueError(f"method must be 'full', 'low_rank', or 'finite_diff', got {method!r}")


def temperature_scope_scores(forward_pass, residual, loss_position):
    """
    Temperature scope: gradient of hidden-norm loss w.r.t. residual; score = grad norm per token.

    Returns:
        scores: np.ndarray [n_tokens], float32
    """
    loss, logits = forward_pass(loss_position=loss_position, hidden_norm_as_loss=True)
    grads = torch.autograd.grad(loss, residual, retain_graph=False)[0]
    scores = grads.norm(dim=-1).squeeze().cpu().numpy().astype(np.float32)
    if scores.ndim > 1:
        scores = scores.squeeze()
    return scores, logits


def semantic_scope_scores(
    forward_pass,
    residual,
    loss_position,
    path_integral=False,
    presence_ratios=None,
    grad_idx=None,
    return_grads_per_step=False,
):
    """
    Semantic scope: single-pass (hidden_norm_as_loss=False, unnormalized_logits) or path-integrated.
    When path_integral=False: gradient norm per token.
    When path_integral=True: Path_integrated_grad = mean(grads) * (x_final - x_initial) at grad_idx, scores = norm.

    Args:
        forward_pass: from JCBScope_utils.customize_forward_pass
        residual: nn.Parameter [n_tokens, d_model]
        loss_position: int or tensor
        path_integral: if True, use path integration as in Path_Integrated_Semantic_Scope.ipynb
        presence_ratios: for path_integral; default np.linspace(0.01, 1, 100)
        grad_idx: indices of token positions for path_integral slice; if None use range(n_tokens)
        return_grads_per_step: if True and path_integral, also return (grads_per_step, input_embeds_per_step)

    Returns:
        scores: np.ndarray [n_tokens], float32
        If return_grads_per_step and path_integral: (scores, grads_per_step, input_embeds_per_step)
    """
    if not path_integral:
        loss, logits = forward_pass(
            loss_position=loss_position,
            hidden_norm_as_loss=False,
            unnormalized_logits=True,
        )
        grads = torch.autograd.grad(loss, residual, retain_graph=False)[0]
        scores = grads.norm(dim=-1).squeeze().cpu().numpy().astype(np.float32)
        if scores.ndim > 1:
            scores = scores.squeeze()
        return scores, logits

    if grad_idx is None:
        grad_idx = list(range(residual.shape[0]))

    presence_ratios = presence_ratios if presence_ratios is not None else np.linspace(0.01, 1.0, 10)
    grads_per_step = []
    input_embeds_per_step = []

    for presence_ratio in presence_ratios:
        loss, logits, input_embeds = forward_pass(
            loss_position=loss_position,
            hidden_norm_as_loss=False,
            unnormalized_logits=True,
            return_input_embeds=True,
            alpha=float(presence_ratio),
        )
        input_embeds_per_step.append(input_embeds.detach().cpu().clone())
        residual_grad = torch.autograd.grad(loss, residual, retain_graph=False)[0] / presence_ratio
        grads_per_step.append(residual_grad.detach().cpu().clone())

    grad_stack = torch.stack(grads_per_step)
    input_final = input_embeds_per_step[-1][0, grad_idx, :]
    input_initial = input_embeds_per_step[0][0, grad_idx, :]
    path_integrated_grad = grad_stack.mean(dim=0) * (input_final - input_initial)
    scores = path_integrated_grad.norm(dim=-1).numpy().astype(np.float32)

    if return_grads_per_step:
        return scores, grads_per_step, input_embeds_per_step
    return scores


def gradient_x_input_scores(forward_pass, residual, loss_position, embedding_layer, input_ids, grad_idx):
    """
    Gradient times input (grad * input_embeds) norm per token.

    Returns:
        scores: np.ndarray [n_tokens], float32
    """
    loss, logits = forward_pass(
        loss_position=loss_position,
        hidden_norm_as_loss=False,
        unnormalized_logits=False,
    )
    grads = torch.autograd.grad(loss, residual, retain_graph=False)[0]
    with torch.no_grad():
        token_embeds = JCBScope_utils.embedding_lookup(input_ids[0, grad_idx], embedding_layer)
    scores = (grads * token_embeds.to(grads.device)).norm(dim=-1).squeeze().cpu().numpy().astype(np.float32)
    if scores.ndim > 1:
        scores = scores.squeeze()
    return scores, logits


def setup_scope_context(model, tokenizer, string, front_pad=0, back_pad=0, front_strip=0, eos_token_id=None):
    """
    Build input_ids, attention_mask, grad_idx, decoded_tokens, residual, presence, forward_pass, d_model, embed_device
    for use with scope functions. Caller must still set model.eval() and pass loss_position.

    Returns:
        dict with keys: input_ids, attention_mask, grad_idx, decoded_tokens, residual, presence,
        forward_pass, d_model, embed_device
    """
    input_ids_list = tokenizer(string, add_special_tokens=False)["input_ids"]
    if eos_token_id is not None:
        input_ids_list += [eos_token_id] * back_pad
    decoded_tokens = tokenizer.batch_decode([[tid] for tid in input_ids_list], skip_special_tokens=True)
    grad_idx = list(range(front_pad, len(decoded_tokens)))[front_strip:]

    embedding_layer = model.get_input_embeddings()
    embed_device = embedding_layer.weight.device
    d_model = embedding_layer.embedding_dim

    input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(embed_device)
    attention_mask = torch.ones_like(input_ids, device=embed_device)
    residual = torch.nn.Parameter(torch.zeros(len(grad_idx), d_model, device=embed_device))
    presence = torch.ones(len(decoded_tokens), 1, device=embed_device)
    forward_pass = JCBScope_utils.customize_forward_pass(
        model, residual, presence, input_ids, grad_idx, attention_mask
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "grad_idx": grad_idx,
        "decoded_tokens": decoded_tokens,
        "residual": residual,
        "presence": presence,
        "forward_pass": forward_pass,
        "d_model": d_model,
        "embed_device": embed_device,
    }
