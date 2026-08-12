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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from JacobianScopes import JacobianScopes
from JacobianScopes import JacobianScopes_utils as scope_utils


MODELS = {
    "llama-1b": (
        "meta-llama/Llama-3.2-1B",
        "Llama-3.2-1B",
        "LLaMA-3.2 1B",
    ),
    "llama-3b": (
        "meta-llama/Llama-3.2-3B",
        "Llama-3.2-3B",
        "LLaMA-3.2 3B",
    ),
    "qwen-1.5b": (
        "Qwen/Qwen2.5-1.5B",
        "Qwen2.5-1.5B",
        "Qwen2.5 1.5B",
    ),
    "qwen-3b": (
        "Qwen/Qwen2.5-3B",
        "Qwen2.5-3B",
        "Qwen2.5 3B",
    ),
    "gemma-1b": (
        "google/gemma-3-1b-pt",
        "gemma-3-1b-pt",
        "Gemma-3 1B",
    ),
    "gemma-4b": (
        "google/gemma-3-4b-pt",
        "gemma-3-4b-pt",
        "Gemma-3 4B",
    ),
}
DATASETS = {
    "lmbd1000": "LAMBADA",
    "IWSLT2017DE_EN": "IWSLT2017 DE→EN",
}
METHODS = ("Semantic", "Temperature", "Fisher")
REPORT_METHODS = (
    "Semantic",
    "Temperature",
    "Fisher",
    "InputXGradient",
    "IG",
)
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
        help="With --device auto, cap each visible GPU for model sharding.",
    )
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--fisher-k", type=int, default=1)
    parser.add_argument(
        "--stage",
        choices=("all", "original", "final_quarter", "final_half", "all_blocks", "finalize"),
        default="all",
        help="Run one independently resumable stage, all stages, or final aggregation.",
    )
    parser.add_argument("--num-prompt-shards", type=int, default=1)
    parser.add_argument("--prompt-shard-index", type=int, default=0)
    parser.add_argument(
        "--cached-correct-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
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
    selected = sorted(rng.sample(sorted(correct), min(num_prompts, len(correct))))
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


@torch.no_grad()
def randomize_blocks(model, block_indices: range, seed: int) -> None:
    """Reinitialize transformer blocks and fail if any target block is unchanged.

    Do not use ``model._init_weights`` here. In some Transformers versions its
    initialization helpers are disabled after ``from_pretrained``, making that
    call silently leave pretrained parameters untouched.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    indices = tuple(block_indices)
    layers = model.model.layers
    initializer_std = float(getattr(model.config, "initializer_range", 0.02) or 0.02)
    changed_blocks = 0

    def reinitialize_module(module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=initializer_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif (
            isinstance(module, torch.nn.LayerNorm)
            or "RMSNorm" in module.__class__.__name__
        ):
            if getattr(module, "weight", None) is not None:
                torch.nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)

    for index in indices:
        layer = layers[index]
        probe = next((parameter for parameter in layer.parameters() if parameter.numel()), None)
        if probe is None:
            raise RuntimeError(f"Transformer block {index} has no parameters")
        before = probe.detach().reshape(-1)[:1024].clone()
        layer.apply(reinitialize_module)
        if torch.equal(before, probe.detach().reshape(-1)[: len(before)]):
            raise RuntimeError(
                f"Transformer block {index} did not change during reinitialization"
            )
        changed_blocks += 1

    if changed_blocks != len(indices):
        raise RuntimeError(
            f"Changed {changed_blocks} of {len(indices)} requested transformer blocks"
        )
    print(
        f"Reinitialized and verified {changed_blocks} blocks "
        f"(seed={seed}, std={initializer_std:g})"
    )


def save_original_maps(path: Path, maps: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **maps)
    temporary.replace(path)


def load_original_maps(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        with np.load(path) as saved:
            if not set(METHODS).issubset(saved.files):
                return None
            return {
                method: np.asarray(saved[method], dtype=np.float64)
                for method in METHODS
            }
    except (OSError, ValueError):
        return None


def save_prompt_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(rows))
    temporary.replace(path)


def load_prompt_rows(
    path: Path,
    index: int,
    checkpoint: str,
) -> list[dict] | None:
    if not path.exists():
        return None
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(rows, list)
        or len(rows) != len(METHODS)
        or {row.get("method") for row in rows} != set(METHODS)
        or any(
            row.get("index") != index or row.get("checkpoint") != checkpoint
            for row in rows
        )
    ):
        return None
    return rows


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


def render_markdown(
    summary: pd.DataFrame,
    num_layers: int,
    model_label: str,
    dataset_label: str,
) -> str:
    lines = [
        "# Parameter-randomization saliency sanity check",
        "",
        f"{model_label} has {num_layers} transformer blocks. Blocks were reinitialized "
        "cumulatively from the output side. Values are mean ± SEM across fixed, correctly "
        f"predicted {dataset_label} prompts; lower similarity indicates greater dependence "
        "on learned parameters.",
        "",
        "| Checkpoint | Method | Spearman ρ | Top-10% Jaccard | Random ρ floor | Random Jaccard floor |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    indexed = summary.set_index(["checkpoint", "method"])
    for checkpoint, _ in checkpoint_specs(num_layers):
        for method in REPORT_METHODS:
            if (checkpoint, method) not in indexed.index:
                lines.append(f"| {checkpoint} | {method} | NA | NA | NA | NA |")
                continue
            row = indexed.loc[(checkpoint, method)]
            lines.append(
                f"| {checkpoint} | {method} | "
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


def checkpoint_specs(num_layers: int) -> list[tuple[str, range]]:
    quarter_start = math.floor(3 * num_layers / 4)
    half_start = math.floor(num_layers / 2)
    return [
        ("final_quarter", range(quarter_start, num_layers)),
        ("final_half", range(half_start, quarter_start)),
        ("all_blocks", range(0, half_start)),
    ]


def write_final_results(
    output_dir: Path,
    checkpoint_dir: Path,
    fixed_prompts: list[tuple[int, str]],
    num_layers: int,
    model_name: str,
    model_short: str,
    model_label: str,
    dataset: str,
    dataset_label: str,
    fisher_k: int,
    seed: int,
) -> None:
    rows = []
    missing = []
    for checkpoint, _ in checkpoint_specs(num_layers):
        for index, _ in fixed_prompts:
            path = checkpoint_dir / "metrics" / checkpoint / f"{index}.json"
            prompt_rows = load_prompt_rows(path, index, checkpoint)
            if prompt_rows is None:
                missing.append((checkpoint, index))
            else:
                rows.extend(prompt_rows)
    if missing:
        raise RuntimeError(
            f"Cannot finalize: {len(missing)} prompt-stage checkpoints are missing"
        )

    raw = pd.DataFrame(rows)
    summary = summarize(rows)
    raw.to_csv(output_dir / "saliency_randomization_per_prompt.csv", index=False)
    summary.to_csv(output_dir / "saliency_randomization_summary.csv", index=False)
    (output_dir / "saliency_randomization_summary.md").write_text(
        render_markdown(summary, num_layers, model_label, dataset_label)
    )
    (output_dir / "saliency_randomization_metadata.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "model_short": model_short,
                "dataset": dataset,
                "dataset_label": dataset_label,
                "num_layers": num_layers,
                "num_prompts": len(fixed_prompts),
                "prompt_indices": [index for index, _ in fixed_prompts],
                "methods": METHODS,
                "ig_alphas": IG_ALPHAS,
                "fisher_k": fisher_k,
                "seed": seed,
            },
            indent=2,
        )
    )
    print(f"Wrote randomization results to {output_dir}")


def main() -> None:
    args = parse_args()
    if args.num_prompt_shards < 1:
        raise ValueError("--num-prompt-shards must be positive")
    if not 0 <= args.prompt_shard_index < args.num_prompt_shards:
        raise ValueError("--prompt-shard-index must be within the shard count")

    model_name, model_short, model_label = MODELS[args.model]
    dataset_label = DATASETS[args.dataset]
    root = Path(__file__).resolve().parents[1]
    cached_correct_path = args.cached_correct_path or (
        root
        / "paper"
        / "results"
        / f"{model_short}__Semantic_{args.dataset}_top0.05_results.json"
    )
    output_dir = args.output_dir or (
        Path(__file__).resolve().parent
        / "saliency_randomization_results"
        / f"{model_short}_{args.dataset}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_prompts = load_fixed_prompts(
        cached_correct_path,
        args.num_prompts,
        args.seed,
    )
    checkpoint_dir = (
        output_dir
        / "checkpoints_v1"
        / f"seed_{args.seed}_n_{len(fixed_prompts)}_fisher_{args.fisher_k}"
    )
    config = AutoConfig.from_pretrained(model_name)
    num_layers = int(config.num_hidden_layers)
    if args.stage == "finalize":
        write_final_results(
            output_dir,
            checkpoint_dir,
            fixed_prompts,
            num_layers,
            model_name,
            model_short,
            model_label,
            args.dataset,
            dataset_label,
            args.fisher_k,
            args.seed,
        )
        return

    shard_prompts = [
        prompt
        for position, prompt in enumerate(fixed_prompts)
        if position % args.num_prompt_shards == args.prompt_shard_index
    ]
    print(
        f"Stage={args.stage}; prompt shard {args.prompt_shard_index + 1}/"
        f"{args.num_prompt_shards} ({len(shard_prompts)} prompts)"
    )

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

    original_maps: dict[int, dict[str, np.ndarray]] = {}
    resumed_original = 0
    for index, prompt in tqdm(shard_prompts, desc="Original maps"):
        checkpoint_path = checkpoint_dir / "original_maps" / f"{index}.npz"
        maps = load_original_maps(checkpoint_path)
        if maps is None:
            if args.stage not in ("all", "original"):
                raise RuntimeError(
                    f"Missing original-map checkpoint for prompt {index}; "
                    "finish the original stage first"
                )
            maps = {
                method: attribution_scores(
                    model, tokenizer, prompt, method, args.fisher_k
                )
                for method in METHODS
            }
            save_original_maps(checkpoint_path, maps)
        else:
            resumed_original += 1
        original_maps[index] = maps
    print(
        f"Original-map checkpoints: resumed {resumed_original}, "
        f"computed {len(shard_prompts) - resumed_original}"
    )
    if args.stage == "original":
        return

    checkpoints = checkpoint_specs(num_layers)
    if len(model.model.layers) != num_layers:
        raise RuntimeError(
            f"Config reports {num_layers} layers but model has {len(model.model.layers)}"
        )
    if args.stage == "all":
        targets = checkpoints
    else:
        target_number = next(
            number
            for number, (checkpoint, _) in enumerate(checkpoints)
            if checkpoint == args.stage
        )
        for checkpoint_number, (_, newly_randomized) in enumerate(
            checkpoints[: target_number + 1], start=1
        ):
            randomize_blocks(
                model,
                newly_randomized,
                seed=args.seed + checkpoint_number,
            )
        model.eval().requires_grad_(False)
        targets = [checkpoints[target_number]]

    for checkpoint, newly_randomized in targets:
        checkpoint_number = next(
            number
            for number, (name, _) in enumerate(checkpoints, start=1)
            if name == checkpoint
        )
        if args.stage == "all":
            randomize_blocks(
                model,
                newly_randomized,
                seed=args.seed + checkpoint_number,
            )
            model.eval().requires_grad_(False)
        resumed_prompts = 0
        for index, prompt in tqdm(shard_prompts, desc=checkpoint):
            checkpoint_path = (
                checkpoint_dir / "metrics" / checkpoint / f"{index}.json"
            )
            prompt_rows = load_prompt_rows(checkpoint_path, index, checkpoint)
            if prompt_rows is not None:
                resumed_prompts += 1
                continue

            prompt_rows = []
            for method_number, method in enumerate(METHODS):
                randomized_map = attribution_scores(
                    model, tokenizer, prompt, method, args.fisher_k
                )
                metric_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [args.seed, checkpoint_number, index, method_number]
                    )
                )
                metrics = map_metrics(
                    original_maps[index][method], randomized_map, metric_rng
                )
                prompt_rows.append(
                    {
                        "index": index,
                        "checkpoint": checkpoint,
                        "method": method,
                        "n_tokens": len(randomized_map),
                        **metrics,
                    }
                )
            save_prompt_rows(checkpoint_path, prompt_rows)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(
            f"{checkpoint} checkpoints: resumed {resumed_prompts}, "
            f"computed {len(shard_prompts) - resumed_prompts}"
        )

    if args.stage == "all" and args.num_prompt_shards == 1:
        write_final_results(
            output_dir,
            checkpoint_dir,
            fixed_prompts,
            num_layers,
            model_name,
            model_short,
            model_label,
            args.dataset,
            dataset_label,
            args.fisher_k,
            args.seed,
        )


if __name__ == "__main__":
    main()
