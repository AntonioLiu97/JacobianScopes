#!/usr/bin/env python3
"""Quantify prefix/BOS concentration along the zero-to-input IG path.

This is a one-prompt diagnostic matching the executed path-integrand notebook,
not a benchmark-level claim. It explicitly includes the two BOS tokens in the
attribution index so their mass can be measured rather than inferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from JacobianScopes import JacobianScopes
from JacobianScopes import JacobianScopes_utils as scope_utils


MODEL_NAME = "meta-llama/Llama-3.2-1B"
PROMPT = (
    "Tom loves Korean food. He grew up in the South and goes to Columbia. "
    "Tom is a liberal"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--num-alpha", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "prefix_concentration_results",
    )
    return parser.parse_args()


def normalized_prefix_mass(scores: np.ndarray, count: int) -> float:
    denominator = scores.sum()
    return float(scores[:count].sum() / denominator) if denominator > 0 else float("nan")


def endpoint_scope_scores(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    method: str,
) -> np.ndarray:
    embedding_layer = model.get_input_embeddings()
    grad_idx = list(range(input_ids.shape[1]))
    residual = torch.nn.Parameter(
        torch.zeros(
            len(grad_idx),
            embedding_layer.embedding_dim,
            device=embedding_layer.weight.device,
            dtype=embedding_layer.weight.dtype,
        )
    )
    presence = torch.ones(
        len(grad_idx),
        1,
        device=embedding_layer.weight.device,
        dtype=embedding_layer.weight.dtype,
    )
    forward_pass = scope_utils.customize_forward_pass(
        model, residual, presence, input_ids, grad_idx, attention_mask
    )
    loss_position = input_ids.shape[1] - 2
    if method == "Semantic":
        scores, _ = JacobianScopes.semantic_scope_scores(
            forward_pass, residual, loss_position
        )
    elif method == "Temperature":
        scores, _ = JacobianScopes.temperature_scope_scores(
            forward_pass, residual, loss_position
        )
    elif method == "Fisher":
        scores, _ = JacobianScopes.fisher_scope_scores(
            forward_pass,
            residual,
            loss_position,
            scope_utils.get_lm_head(model),
            method="low_rank",
            k=1,
        )
    else:
        raise ValueError(method)
    return np.asarray(scores, dtype=float).reshape(-1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.to(device).eval().requires_grad_(False)

    content_ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor(
        [[tokenizer.bos_token_id, tokenizer.bos_token_id, *content_ids]],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    loss_position = input_ids.shape[1] - 2
    target_id = int(input_ids[0, -1])
    with torch.no_grad():
        input_embeds = model.get_input_embeddings()(input_ids).detach()

    alphas = np.linspace(0.01, 1.0, args.num_alpha)
    integrand_vectors = []
    rows = []
    for alpha in alphas:
        path_embeds = (alpha * input_embeds).detach().requires_grad_(True)
        output = model(
            inputs_embeds=path_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        loss = torch.nn.functional.cross_entropy(
            output.logits[:, loss_position, :].float(),
            torch.tensor([target_id], device=device),
        )
        gradient = torch.autograd.grad(loss, path_embeds)[0]
        integrand = (gradient * input_embeds).detach()
        integrand_vectors.append(integrand)
        scores = integrand.norm(dim=-1)[0].float().cpu().numpy()
        rows.append(
            {
                "alpha": float(alpha),
                "bos_1_mass": normalized_prefix_mass(scores, 1),
                "bos_2_mass": normalized_prefix_mass(scores, 2),
                "bos_plus_2_content_mass": normalized_prefix_mass(scores, 4),
            }
        )
        del path_embeds, output, loss, gradient

    curve = pd.DataFrame(rows)
    curve.to_csv(args.output_dir / "ig_integrand_prefix_mass.csv", index=False)

    integrated_vector = torch.stack(integrand_vectors).mean(dim=0)
    integrated_scores = integrated_vector.norm(dim=-1)[0].float().cpu().numpy()
    endpoint_rows = [
        {
            "method": "Integrated Gradients (path aggregate)",
            "bos_1_mass": normalized_prefix_mass(integrated_scores, 1),
            "bos_2_mass": normalized_prefix_mass(integrated_scores, 2),
            "bos_plus_2_content_mass": normalized_prefix_mass(integrated_scores, 4),
        }
    ]
    for method in ("Semantic", "Temperature", "Fisher"):
        scores = endpoint_scope_scores(model, input_ids, attention_mask, method)
        endpoint_rows.append(
            {
                "method": f"{method} Scope (α=1)",
                "bos_1_mass": normalized_prefix_mass(scores, 1),
                "bos_2_mass": normalized_prefix_mass(scores, 2),
                "bos_plus_2_content_mass": normalized_prefix_mass(scores, 4),
            }
        )
    endpoint = pd.DataFrame(endpoint_rows)
    endpoint.to_csv(args.output_dir / "aggregate_prefix_mass.csv", index=False)

    fig, axis = plt.subplots(figsize=(6.2, 3.8), dpi=200)
    axis.plot(curve["alpha"], curve["bos_2_mass"], label="IG integrand: 2 BOS")
    axis.plot(
        curve["alpha"],
        curve["bos_plus_2_content_mass"],
        label="IG integrand: 2 BOS + 2 content",
    )
    for _, row in endpoint.iterrows():
        axis.axhline(
            row["bos_2_mass"],
            linestyle="--",
            linewidth=1,
            label=f"{row['method']}: 2 BOS",
        )
    axis.set_xlabel(r"Interpolation coefficient $\alpha$")
    axis.set_ylabel("Fraction of attribution mass on prefix")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "prefix_mass_vs_alpha.png")
    plt.close(fig)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    metadata = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "tokens": tokens,
        "bos_count": 2,
        "target_token": tokenizer.decode([target_id]),
        "num_alpha": args.num_alpha,
        "interpretation": (
            "One-prompt diagnostic matching the notebook setup; not a "
            "benchmark-level estimate of attention-sink prevalence."
        ),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote prefix-concentration analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
