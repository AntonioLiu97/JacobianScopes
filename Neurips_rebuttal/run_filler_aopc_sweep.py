#!/usr/bin/env python3
"""Run all-method AOPC with length-preserving filler-token replacement.

Unlike the IG baseline sweep, this changes the AOPC intervention itself:
top-ranked token embeddings are replaced by a model-family filler embedding
while their attention-mask entries remain one.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from JacobianScopes import JacobianScopes
from JacobianScopes import JacobianScopes_utils as scope_utils


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "llama-1b": ("meta-llama/Llama-3.2-1B", "Llama-3.2-1B", 0),
    "llama-3b": ("meta-llama/Llama-3.2-3B", "Llama-3.2-3B", 0),
    "qwen-1.5b": ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B", 0),
    "qwen-3b": ("Qwen/Qwen2.5-3B", "Qwen2.5-3B", 0),
    "gemma-1b": ("google/gemma-3-1b-pt", "gemma-3-1b-pt", 1),
    "gemma-4b": ("google/gemma-3-4b-pt", "gemma-3-4b-pt", 1),
}
DATASETS = {
    "lmbd1000": ROOT / "paper" / "data" / "lambada_prompts_1000.json",
    "IWSLT2017DE_EN": ROOT / "paper" / "data" / "IWSLT2017DE_EN.json",
}
METHODS = (
    "random_ablation",
    "IG",
    "gradient_x_input",
    "Semantic",
    "Temperature",
    "Fisher_k_1",
)
FRACTIONS = (0.05, 0.1, 0.2)
IG_ALPHAS = (0.2, 0.4, 0.6, 0.8, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-memory-per-gpu-gib",
        type=int,
        default=None,
        help=(
            "With --device auto, cap each visible GPU so Accelerate shards "
            "the model instead of filling one GPU."
        ),
    )
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "filler_aopc_results",
    )
    return parser.parse_args()


def filler_token(tokenizer, model_key: str) -> tuple[str, int]:
    if model_key.startswith("llama"):
        token = "<|finetune_right_pad_id|>"
        token_id = tokenizer.convert_tokens_to_ids(token)
    else:
        token = tokenizer.pad_token
        token_id = tokenizer.pad_token_id
    if token is None or token_id is None:
        raise ValueError(f"No configured filler token for {model_key}")
    return token, int(token_id)


def load_prompts(dataset: str, limit: int) -> list[str]:
    payload = json.loads(DATASETS[dataset].read_text())
    if isinstance(payload, dict) and "prompts" in payload:
        payload = payload["prompts"]
    return [
        row["text"] if isinstance(row, dict) else row
        for row in payload[:limit]
    ]


def output_path(
    output_dir: Path,
    model_short: str,
    method: str,
    dataset: str,
    fraction: float,
    filler_name: str,
) -> Path:
    label = (
        f"{model_short}__{method}_{dataset}_top{fraction}"
        f"_filler_{filler_name}"
    )
    return output_dir / f"{label}_results.json"


def load_existing(
    output_dir: Path,
    model_short: str,
    method: str,
    dataset: str,
    filler_name: str,
) -> dict[float, list[dict]]:
    results = {fraction: [] for fraction in FRACTIONS}
    for fraction in FRACTIONS:
        path = output_path(
            output_dir, model_short, method, dataset, fraction, filler_name
        )
        if path.exists():
            results[fraction] = json.loads(path.read_text())["results"]
    index_sets = [
        frozenset(int(row["index"]) for row in rows) for rows in results.values()
    ]
    if len(set(index_sets)) != 1:
        raise ValueError("Existing files contain inconsistent prompt indices")
    return results


def save_results(
    output_dir: Path,
    model_short: str,
    method: str,
    dataset: str,
    filler_name: str,
    filler_token_string: str,
    filler_token_id: int,
    results: dict[float, list[dict]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fraction, rows in results.items():
        path = output_path(
            output_dir, model_short, method, dataset, fraction, filler_name
        )
        label = path.name.removesuffix("_results.json")
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "label": label,
                    "top_k_fraction": fraction,
                    "intervention": "filler_embedding_replacement",
                    "filler_name": filler_name,
                    "filler_token": filler_token_string,
                    "filler_token_id": filler_token_id,
                    "results": rows,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        temporary.replace(path)


def build_context(model, tokenizer, prompt: str, front_pad: int) -> dict:
    embedding_layer = model.get_input_embeddings()
    embed_device = embedding_layer.weight.device
    content_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prefix = [tokenizer.bos_token_id] * front_pad if front_pad else []
    input_ids = torch.tensor(
        [[*prefix, *content_ids]], dtype=torch.long, device=embed_device
    )
    grad_idx = list(range(front_pad, input_ids.shape[1]))
    attention_mask = torch.ones_like(input_ids)
    residual = nn.Parameter(
        torch.zeros(
            len(grad_idx),
            embedding_layer.embedding_dim,
            device=embed_device,
            dtype=embedding_layer.weight.dtype,
        )
    )
    presence = torch.ones(
        input_ids.shape[1],
        1,
        device=embed_device,
        dtype=embedding_layer.weight.dtype,
    )
    forward_pass = scope_utils.customize_forward_pass(
        model, residual, presence, input_ids, grad_idx, attention_mask
    )
    return {
        "embedding_layer": embedding_layer,
        "input_ids": input_ids,
        "grad_idx": grad_idx,
        "residual": residual,
        "forward_pass": forward_pass,
        "loss_position": input_ids.shape[1] - 2,
    }


def attribution_scores(model, context: dict, method: str) -> tuple[np.ndarray, torch.Tensor]:
    forward_pass = context["forward_pass"]
    residual = context["residual"]
    loss_position = context["loss_position"]
    if method == "random_ablation":
        with torch.no_grad():
            _, logits = forward_pass(
                loss_position=loss_position, unnormalized_logits=True
            )
        scores = np.random.random(len(context["grad_idx"]))
    elif method == "IG":
        gradients = []
        for alpha in IG_ALPHAS:
            loss, logits = forward_pass(
                loss_position=loss_position,
                hidden_norm_as_loss=False,
                unnormalized_logits=False,
                alpha=alpha,
            )
            gradients.append(
                torch.autograd.grad(loss, residual, retain_graph=False)[0].detach()
            )
        average_gradient = torch.stack(gradients).mean(dim=0)
        with torch.no_grad():
            token_embeds = scope_utils.embedding_lookup(
                context["input_ids"][0, context["grad_idx"]],
                context["embedding_layer"],
            )
        scores = (
            average_gradient * token_embeds.to(average_gradient.device)
        ).norm(dim=-1).float().cpu().numpy()
    elif method == "gradient_x_input":
        scores, logits = JacobianScopes.gradient_x_input_scores(
            forward_pass,
            residual,
            loss_position,
            context["embedding_layer"],
            context["input_ids"],
            context["grad_idx"],
        )
    elif method == "Semantic":
        scores, logits = JacobianScopes.semantic_scope_scores(
            forward_pass, residual, loss_position
        )
    elif method == "Temperature":
        scores, logits = JacobianScopes.temperature_scope_scores(
            forward_pass, residual, loss_position
        )
    elif method == "Fisher_k_1":
        scores, logits = JacobianScopes.fisher_scope_scores(
            forward_pass,
            residual,
            loss_position,
            scope_utils.get_lm_head(model),
            method="low_rank",
            k=1,
        )
    else:
        raise ValueError(method)
    return np.asarray(scores).reshape(-1), logits


@torch.no_grad()
def intervention_metrics(
    context: dict,
    logits_original: torch.Tensor,
    scores: np.ndarray,
    filler_embedding: torch.Tensor,
) -> dict[float, dict]:
    loss_position = context["loss_position"]
    original_distribution = torch.log_softmax(
        logits_original[loss_position].float(), dim=-1
    )
    predicted_id = int(logits_original[loss_position].argmax())
    original_log_prob = original_distribution[predicted_id]
    ranked = np.argsort(scores)[::-1]
    input_embeds = scope_utils.embedding_lookup(
        context["input_ids"][0, context["grad_idx"]],
        context["embedding_layer"],
    )
    rows = {}
    for fraction in FRACTIONS:
        count = max(1, int(len(ranked) * fraction))
        selected = ranked[:count].copy()
        replacement_residual = torch.zeros_like(context["residual"])
        replacement_residual[selected] = (
            filler_embedding.to(input_embeds.device) - input_embeds[selected]
        )
        _, logits_ablated = context["forward_pass"](
            loss_position=loss_position,
            unnormalized_logits=True,
            residual_override=replacement_residual,
        )
        ablated_distribution = torch.log_softmax(
            logits_ablated[loss_position].float(), dim=-1
        )
        ablated_predicted_id = int(logits_ablated[loss_position].argmax())
        kl = torch.sum(
            original_distribution.exp()
            * (original_distribution - ablated_distribution)
        )
        rows[fraction] = {
            "delta_log_prob": float(
                (ablated_distribution[predicted_id] - original_log_prob).item()
            ),
            "kl_divergence": float(kl.item()),
            "prediction_flipped": ablated_predicted_id != predicted_id,
            "ablated_predicted_token_id": ablated_predicted_id,
        }
    return rows


def main() -> None:
    args = parse_args()
    model_name, model_short, front_pad = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.device == "auto":
        max_memory = None
        if args.max_memory_per_gpu_gib is not None:
            max_memory = {
                index: f"{args.max_memory_per_gpu_gib}GiB"
                for index in range(torch.cuda.device_count())
            }
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map="auto",
            max_memory=max_memory,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        )
        model.to(torch.device(args.device))
    model.eval().requires_grad_(False)
    filler_string, filler_id = filler_token(tokenizer, args.model)
    if args.model.startswith("llama"):
        filler_name = "right_pad"
    elif args.model.startswith("qwen"):
        filler_name = "pad_eos"
    else:
        filler_name = "pad"
    filler_embedding = scope_utils.embedding_lookup(
        torch.tensor([filler_id], device=model.get_input_embeddings().weight.device),
        model.get_input_embeddings(),
    )[0]
    prompts = load_prompts(args.dataset, args.max_examples)
    np.random.seed(42)

    for method in METHODS:
        results = load_existing(
            args.output_dir, model_short, method, args.dataset, filler_name
        )
        completed = {int(row["index"]) for row in results[FRACTIONS[0]]}
        pending = [
            (index, prompt)
            for index, prompt in enumerate(prompts)
            if index not in completed
        ]
        print(
            f"{model_short} {args.dataset} {method}: "
            f"{len(completed)} complete, {len(pending)} pending"
        )
        for processed, (index, prompt) in enumerate(
            tqdm(pending, desc=method), start=1
        ):
            context = build_context(model, tokenizer, prompt, front_pad)
            scores, logits_original = attribution_scores(model, context, method)
            metrics = intervention_metrics(
                context, logits_original, scores, filler_embedding
            )
            predicted_id = int(
                logits_original[context["loss_position"]].argmax()
            )
            target_id = int(context["input_ids"][0, -1])
            for fraction in FRACTIONS:
                ablated_predicted_id = metrics[fraction].pop(
                    "ablated_predicted_token_id"
                )
                results[fraction].append(
                    {
                        **metrics[fraction],
                        "prompt": prompt,
                        "true_token": tokenizer.decode([target_id]),
                        "predicted_token": tokenizer.decode([predicted_id]),
                        "ablated_predicted_token": tokenizer.decode(
                            [ablated_predicted_id]
                        ),
                        "index": index,
                    }
                )
                results[fraction].sort(key=lambda row: int(row["index"]))
            if processed % args.save_every == 0:
                save_results(
                    args.output_dir,
                    model_short,
                    method,
                    args.dataset,
                    filler_name,
                    filler_string,
                    filler_id,
                    results,
                )
            del context, scores, logits_original
            gc.collect()
            torch.cuda.empty_cache()
        save_results(
            args.output_dir,
            model_short,
            method,
            args.dataset,
            filler_name,
            filler_string,
            filler_id,
            results,
        )


if __name__ == "__main__":
    main()
