#!/usr/bin/env python3
"""Adebayo-style cascading parameter-randomization sanity check.

The model, prompts, tokenization, and gold target are fixed. Transformer blocks
are reinitialized from the output side toward the input side, and complete
attribution maps are recomputed at each checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from JacobianScopes import JacobianScopes
from JacobianScopes import JacobianScopes_utils as scope_utils


MODEL_NAME = "meta-llama/Llama-3.2-1B"
MODEL_SHORT = "Llama-3.2-1B"
METHODS = ("Semantic", "Temperature", "Fisher", "InputXGradient", "IG")
IG_ALPHAS = (0.2, 0.4, 0.6, 0.8, 1.0)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--fisher-k", type=int, default=1)
    parser.add_argument(
        "--cached-correct-path",
        type=Path,
        default=root
        / "paper"
        / "results"
        / f"{MODEL_SHORT}__Semantic_lmbd1000_top0.05_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "saliency_randomization_results",
    )
    return parser.parse_args()


def load_fixed_prompts(
    cached_correct_path: Path,
    num_prompts: int,
    seed: int,
) -> list[tuple[int, str]]:
    cached = json.loads(cached_correct_path.read_text())["results"]
    correct = {
        int(row["index"]): row["prompt"]
        for row in cached
        if row["true_token"] == row["predicted_token"]
    }
    rng = random.Random(seed)
    selected = sorted(rng.sample(list(correct), min(num_prompts, len(correct))))
    return [(index, correct[index]) for index in selected]


def build_context(model, tokenizer, prompt: str) -> dict:
    embedding_layer = model.get_input_embeddings()
    device = embedding_layer.weight.device
    input_ids = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    if input_ids.shape[1] < 2:
        raise ValueError("Prompt has fewer than two tokens")
    grad_idx = list(range(input_ids.shape[1]))
    residual = torch.nn.Parameter(
        torch.zeros(
            len(grad_idx),
            embedding_layer.embedding_dim,
            dtype=embedding_layer.weight.dtype,
            device=device,
        )
    )
    presence = torch.ones(
        input_ids.shape[1], 1, dtype=embedding_layer.weight.dtype, device=device
    )
    attention_mask = torch.ones_like(input_ids)
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
        "target_id": int(input_ids[0, -1]),
    }


def attribution_scores(model, tokenizer, prompt: str, method: str, fisher_k: int) -> np.ndarray:
    context = build_context(model, tokenizer, prompt)
    forward_pass = context["forward_pass"]
    residual = context["residual"]
    loss_position = context["loss_position"]

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
            k=fisher_k,
        )
    elif method == "InputXGradient":
        scores, _ = JacobianScopes.gradient_x_input_scores(
            forward_pass,
            residual,
            loss_position,
            context["embedding_layer"],
            context["input_ids"],
            context["grad_idx"],
        )
    elif method == "IG":
        gradients = []
        for alpha in IG_ALPHAS:
            loss, _ = forward_pass(
                loss_position=loss_position,
                hidden_norm_as_loss=False,
                unnormalized_logits=False,
                alpha=alpha,
            )
            gradient = torch.autograd.grad(loss, residual, retain_graph=False)[0]
            gradients.append(gradient.detach())
        average_gradient = torch.stack(gradients).mean(dim=0)
        with torch.no_grad():
            token_embeds = scope_utils.embedding_lookup(
                context["input_ids"][0, context["grad_idx"]],
                context["embedding_layer"],
            )
        scores = (
            average_gradient * token_embeds.to(average_gradient.device)
        ).norm(dim=-1).float().cpu().numpy()
    else:
        raise ValueError(f"Unknown method: {method}")

    del context
    return np.asarray(scores, dtype=np.float64).reshape(-1)


def top_fraction_set(scores: np.ndarray, fraction: float = 0.1) -> set[int]:
    count = max(1, int(math.ceil(len(scores) * fraction)))
    return set(np.argsort(scores)[-count:].tolist())


def map_metrics(
    original: np.ndarray,
    randomized: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    correlation = float(spearmanr(original, randomized).statistic)
    if not np.isfinite(correlation):
        correlation = 0.0
    original_top = top_fraction_set(original)
    randomized_top = top_fraction_set(randomized)
    union = original_top | randomized_top
    jaccard = len(original_top & randomized_top) / len(union)

    random_correlations = []
    random_jaccards = []
    for _ in range(20):
        permutation = rng.permutation(len(original)).astype(float)
        random_correlations.append(float(spearmanr(original, permutation).statistic))
        permutation_top = top_fraction_set(permutation)
        permutation_union = original_top | permutation_top
        random_jaccards.append(
            len(original_top & permutation_top) / len(permutation_union)
        )
    return {
        "spearman": correlation,
        "top10_jaccard": jaccard,
        "random_floor_spearman": float(np.mean(random_correlations)),
        "random_floor_top10_jaccard": float(np.mean(random_jaccards)),
    }


def randomize_blocks(model, block_indices: range, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    layers = model.model.layers
    for index in block_indices:
        layers[index].apply(model._init_weights)


def summarize(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    metrics = [
        "spearman",
        "top10_jaccard",
        "random_floor_spearman",
        "random_floor_top10_jaccard",
    ]
    summary_rows = []
    for (checkpoint, method), group in frame.groupby(["checkpoint", "method"], sort=False):
        row = {"checkpoint": checkpoint, "method": method, "n": len(group)}
        for metric in metrics:
            row[f"mean_{metric}"] = group[metric].mean()
            row[f"sem_{metric}"] = group[metric].std(ddof=1) / math.sqrt(len(group))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def render_markdown(summary: pd.DataFrame, num_layers: int) -> str:
    lines = [
        "# Parameter-randomization saliency sanity check",
        "",
        f"LLaMA-3.2 1B has {num_layers} transformer blocks. Blocks were reinitialized "
        "cumulatively from the output side. Values are mean ± SEM across fixed, correctly "
        "predicted LAMBADA prompts; lower similarity indicates greater dependence on learned parameters.",
        "",
        "| Checkpoint | Method | Spearman ρ | Top-10% Jaccard | Random ρ floor | Random Jaccard floor |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['checkpoint']} | {row['method']} | "
            f"{row['mean_spearman']:.3f} ± {row['sem_spearman']:.3f} | "
            f"{row['mean_top10_jaccard']:.3f} ± {row['sem_top10_jaccard']:.3f} | "
            f"{row['mean_random_floor_spearman']:.3f} | "
            f"{row['mean_random_floor_top10_jaccard']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Protocol: Adebayo et al. (2018), with text-ranking metrics following "
            "Kokhlikyan et al. (2021). The target token is held fixed even if the "
            "randomized model's prediction changes.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.to(device).eval().requires_grad_(False)

    fixed_prompts = load_fixed_prompts(
        args.cached_correct_path,
        args.num_prompts,
        args.seed,
    )
    original_maps: dict[int, dict[str, np.ndarray]] = {}
    for index, prompt in tqdm(fixed_prompts, desc="Original maps"):
        original_maps[index] = {
            method: attribution_scores(model, tokenizer, prompt, method, args.fisher_k)
            for method in METHODS
        }

    num_layers = len(model.model.layers)
    quarter_start = math.floor(3 * num_layers / 4)
    half_start = math.floor(num_layers / 2)
    checkpoints = [
        ("final_quarter", range(quarter_start, num_layers)),
        ("final_half", range(half_start, quarter_start)),
        ("all_blocks", range(0, half_start)),
    ]

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    for checkpoint_number, (checkpoint, newly_randomized) in enumerate(checkpoints, start=1):
        randomize_blocks(
            model,
            newly_randomized,
            seed=args.seed + checkpoint_number,
        )
        model.eval().requires_grad_(False)
        for index, prompt in tqdm(fixed_prompts, desc=checkpoint):
            for method in METHODS:
                randomized_map = attribution_scores(
                    model, tokenizer, prompt, method, args.fisher_k
                )
                metrics = map_metrics(original_maps[index][method], randomized_map, rng)
                rows.append(
                    {
                        "index": index,
                        "checkpoint": checkpoint,
                        "method": method,
                        "n_tokens": len(randomized_map),
                        **metrics,
                    }
                )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    raw = pd.DataFrame(rows)
    summary = summarize(rows)
    raw.to_csv(args.output_dir / "saliency_randomization_per_prompt.csv", index=False)
    summary.to_csv(args.output_dir / "saliency_randomization_summary.csv", index=False)
    (args.output_dir / "saliency_randomization_summary.md").write_text(
        render_markdown(summary, num_layers)
    )
    (args.output_dir / "saliency_randomization_metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "num_layers": num_layers,
                "num_prompts": len(fixed_prompts),
                "prompt_indices": [index for index, _ in fixed_prompts],
                "methods": METHODS,
                "ig_alphas": IG_ALPHAS,
                "fisher_k": args.fisher_k,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    print(f"Wrote randomization results to {args.output_dir}")


if __name__ == "__main__":
    main()
