#!/usr/bin/env python3
"""Run a resumable Integrated Gradients alpha-resolution sweep.

The implementation intentionally mirrors paper/benchmarks/
AOPC_multi_method_benchmarks.py. It evaluates a 20-point right-endpoint
alpha grid once, caches every per-token residual-gradient vector, and derives
5/10/20-point token rankings from evenly spaced subsets.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from JacobianScopes import JacobianScopes_utils as scope_utils

MODELS = {
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B",
    "Gemma-3-1B": "google/gemma-3-1b-pt",
    "Gemma-3-4B": "google/gemma-3-4b-pt",
}

DATASETS = {
    "lmbd1000": "paper/data/lambada_prompts_1000.json",
    "IWSLT2017DE_EN": "paper/data/IWSLT2017DE_EN.json",
}

ALPHAS = np.linspace(0.05, 1.0, 20, endpoint=True, dtype=np.float64)
SAMPLE_COUNTS = (5, 10, 20)
ABLATION_FRACTIONS = (0.05, 0.1, 0.2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=DATASETS,
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--alpha-batch-size", type=int, default=1)
    parser.add_argument(
        "--cache-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_prompts(path: Path, limit: int) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    prompts = payload["prompts"] if isinstance(payload, dict) and "prompts" in payload else payload
    return prompts[:limit]


def prompt_text(item: Any) -> str:
    return item["text"] if isinstance(item, dict) else item


def prompt_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def result_key(sample_count: int, fraction: float) -> str:
    return f"samples_{sample_count}/top_{fraction}"


def selected_alpha_indices(sample_count: int) -> list[int]:
    stride = len(ALPHAS) // sample_count
    return list(range(stride - 1, len(ALPHAS), stride))


def build_metadata(args: argparse.Namespace, model_name: str) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_short": args.model,
        "model_name": model_name,
        "alpha_grid": ALPHAS.tolist(),
        "sample_counts": list(SAMPLE_COUNTS),
        "selected_alpha_indices": {
            str(count): selected_alpha_indices(count) for count in SAMPLE_COUNTS
        },
        "selected_alphas": {
            str(count): [float(ALPHAS[i]) for i in selected_alpha_indices(count)]
            for count in SAMPLE_COUNTS
        },
        "ablation_fractions": list(ABLATION_FRACTIONS),
        "cache_dtype": args.cache_dtype,
        "alpha_batch_size": args.alpha_batch_size,
        "gradient_definition": (
            "d(cross_entropy_at_last_prediction)/d(residual), with the residual "
            "inserted before alpha scaling; matches AOPC_multi_method_benchmarks.py"
        ),
        "score_definition": "norm(mean(selected gradients) * token_embedding, dim=-1)",
        "created_at_unix": time.time(),
    }


def summarize(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, rows in results.items():
        deltas = np.asarray(
            [row["delta_log_prob"] for row in rows if row.get("delta_log_prob") is not None],
            dtype=np.float64,
        )
        correct = sum(row.get("true_token", "") == row.get("predicted_token", "") for row in rows)
        total = len(rows)
        if deltas.size:
            avg = float(deltas.mean())
            variance = float(np.mean((deltas - avg) ** 2))
            sem = float(math.sqrt(variance) / math.sqrt(deltas.size)) if deltas.size > 1 else None
        else:
            avg = variance = sem = None
        summary[key] = {
            "avg_delta": avg,
            "variance_delta": variance,
            "sem_delta": sem,
            "accuracy": correct / total if total else None,
            "num_correct": correct,
            "total": total,
        }

    for sample_count in SAMPLE_COUNTS:
        points = [summary.get(result_key(sample_count, fraction)) for fraction in ABLATION_FRACTIONS]
        if not all(points) or any(point["avg_delta"] is None for point in points):
            continue
        vals = [point["avg_delta"] for point in points]
        sems = [point["sem_delta"] for point in points]
        aupc = 0.0
        variance_aupc = 0.0
        for index in range(len(ABLATION_FRACTIONS) - 1):
            width = ABLATION_FRACTIONS[index + 1] - ABLATION_FRACTIONS[index]
            weight = 0.5 * width
            aupc += weight * (vals[index] + vals[index + 1])
            if sems[index] is not None and sems[index + 1] is not None:
                variance_aupc += weight**2 * (sems[index] ** 2 + sems[index + 1] ** 2)
        summary[f"samples_{sample_count}/aupc"] = {
            "aupc": aupc,
            "sem_aupc": math.sqrt(variance_aupc),
        }
    return summary


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "completed_prompt_indices": [],
            "results": {
                result_key(count, fraction): []
                for count in SAMPLE_COUNTS
                for fraction in ABLATION_FRACTIONS
            },
            "summary": {},
        }
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cache_matches(
    cache: dict[str, Any],
    text: str,
    token_ids: torch.Tensor,
    model_compute_dtype: torch.dtype,
) -> bool:
    return (
        cache.get("format_version") == 3
        and cache.get("prompt_sha256") == prompt_digest(text)
        and cache.get("input_ids") == token_ids.detach().cpu().tolist()
        and cache.get("alpha_grid") == ALPHAS.tolist()
        and cache.get("model_compute_dtype") == str(model_compute_dtype)
    )


def last_token_logits(
    *,
    model: Any,
    lm_head: Any,
    base_embeddings: torch.Tensor,
    residual: torch.Tensor,
    presence: torch.Tensor,
    attention_mask: torch.Tensor,
    grad_idx: list[int],
    loss_position: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    embeddings = base_embeddings.clone()
    embeddings[0, grad_idx, :] += residual
    embeddings *= presence
    embeddings *= alpha
    hidden = model.model(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        use_cache=False,
    ).last_hidden_state[0, loss_position, :]
    return F.linear(hidden.to(lm_head.weight.device), lm_head.weight)


def alpha_gradient_batch(
    *,
    model: Any,
    lm_head: Any,
    base_embeddings: torch.Tensor,
    presence: torch.Tensor,
    attention_mask: torch.Tensor,
    grad_idx: list[int],
    loss_position: int,
    target_token_id: int,
    alphas: np.ndarray,
) -> torch.Tensor:
    batch_size = len(alphas)
    residual_batch = nn.Parameter(
        torch.zeros(
            batch_size,
            len(grad_idx),
            base_embeddings.shape[-1],
            device=base_embeddings.device,
            dtype=torch.float32,
        )
    )
    embeddings = base_embeddings.expand(batch_size, -1, -1).clone()
    embeddings[:, grad_idx, :] += residual_batch
    embeddings *= presence.unsqueeze(0)
    alpha_tensor = torch.as_tensor(
        alphas, device=embeddings.device, dtype=embeddings.dtype
    ).reshape(batch_size, 1, 1)
    embeddings *= alpha_tensor
    hidden = model.model(
        inputs_embeds=embeddings,
        attention_mask=attention_mask.expand(batch_size, -1),
        use_cache=False,
    ).last_hidden_state[:, loss_position, :]
    logits = F.linear(hidden.to(lm_head.weight.device), lm_head.weight)
    targets = torch.full(
        (batch_size,), target_token_id, device=logits.device, dtype=torch.long
    )
    loss = F.cross_entropy(logits, targets, reduction="sum")
    gradients = torch.autograd.grad(loss, residual_batch, retain_graph=False)[0]
    return gradients.detach()


def compute_prompt(
    *,
    item: Any,
    prompt_index: int,
    cache_path: Path,
    cache_dtype: torch.dtype,
    tokenizer: Any,
    model: Any,
    embedding_layer: Any,
    lm_head: Any,
    alpha_batch_size: int,
) -> dict[str, dict[float, dict[str, Any]]]:
    text = prompt_text(item)
    input_ids_list = tokenizer(text, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=embedding_layer.weight.device)
    attention_mask = torch.ones_like(input_ids)
    decoded_tokens = tokenizer.batch_decode(
        [[token_id] for token_id in input_ids[0].tolist()], skip_special_tokens=True
    )
    grad_idx = list(range(len(decoded_tokens)))
    loss_position = input_ids.size(1) - 2
    residual = nn.Parameter(
        torch.zeros(
            len(grad_idx),
            embedding_layer.embedding_dim,
            device=embedding_layer.weight.device,
        )
    )
    presence = torch.ones(len(decoded_tokens), 1, device=embedding_layer.weight.device)
    with torch.no_grad():
        base_embeddings = F.embedding(input_ids, embedding_layer.weight)
    target_token_id = input_ids[0, loss_position + 1].item()
    forward_pass = scope_utils.customize_forward_pass(
        model, residual, presence, input_ids, grad_idx, attention_mask
    )

    cached_scores: dict[int, np.ndarray] | None = None
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache_matches(cache, text, input_ids, embedding_layer.weight.dtype):
            cached_scores = {
                int(count): scores.float().numpy()
                for count, scores in cache["scores_by_sample_count"].items()
            }

    logits_orig = None
    gradients_float_cpu: list[torch.Tensor] | None = None

    if cached_scores is None:
        gradients_float_cpu = [torch.empty(0)] * len(ALPHAS)
        batch_start = 0
        effective_batch_size = alpha_batch_size
        while batch_start < len(ALPHAS):
            batch_end = min(batch_start + effective_batch_size, len(ALPHAS))
            try:
                gradient_batch = alpha_gradient_batch(
                    model=model,
                    lm_head=lm_head,
                    base_embeddings=base_embeddings,
                    presence=presence,
                    attention_mask=attention_mask,
                    grad_idx=grad_idx,
                    loss_position=loss_position,
                    target_token_id=target_token_id,
                    alphas=ALPHAS[batch_start:batch_end],
                )
            except torch.OutOfMemoryError:
                if effective_batch_size == 1:
                    raise
                effective_batch_size = max(1, effective_batch_size // 2)
                print(
                    f"alpha batch OOM for prompt {prompt_index}; "
                    f"retrying with batch size {effective_batch_size}",
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()
                continue
            for offset, gradient in enumerate(gradient_batch):
                gradients_float_cpu[batch_start + offset] = gradient.to(
                    device="cpu", dtype=torch.float32
                )
            del gradient_batch
            batch_start = batch_end

        if alpha_batch_size > 1:
            for alpha_index in selected_alpha_indices(5):
                exact_gradient = alpha_gradient_batch(
                    model=model,
                    lm_head=lm_head,
                    base_embeddings=base_embeddings,
                    presence=presence,
                    attention_mask=attention_mask,
                    grad_idx=grad_idx,
                    loss_position=loss_position,
                    target_token_id=target_token_id,
                    alphas=ALPHAS[alpha_index : alpha_index + 1],
                )[0]
                gradients_float_cpu[alpha_index] = exact_gradient.to(
                    device="cpu", dtype=torch.float32
                )
                del exact_gradient

    with torch.no_grad():
        _, logits_orig_full = forward_pass(
            loss_position=loss_position,
            hidden_norm_as_loss=False,
            unnormalized_logits=True,
            tie_input_output_embed=False,
        )
    logits_orig = logits_orig_full[loss_position].detach()
    del logits_orig_full

    max_index = logits_orig.argmax().item()
    max_log_probability = torch.log_softmax(logits_orig, dim=-1)[max_index].item()
    true_token = tokenizer.decode([target_token_id])
    predicted_token = tokenizer.decode([max_index])

    with torch.no_grad():
        token_embeddings = F.embedding(
            input_ids[0, grad_idx], embedding_layer.weight
        ).float()

    scores_by_sample_count: dict[int, np.ndarray] = {}
    if cached_scores is not None:
        scores_by_sample_count = cached_scores
    else:
        assert gradients_float_cpu is not None
        for count in SAMPLE_COUNTS:
            selected_gradients = torch.stack(
                [
                    gradients_float_cpu[index].to(residual.device)
                    for index in selected_alpha_indices(count)
                ]
            )
            mean_gradient = selected_gradients.mean(dim=0)
            scores_by_sample_count[count] = (
                (mean_gradient * token_embeddings).norm(dim=-1).squeeze().cpu().numpy()
            )
            del selected_gradients, mean_gradient

        gradient_tensor = torch.stack(
            [gradient.to(dtype=cache_dtype) for gradient in gradients_float_cpu]
        )
        atomic_torch_save(
            {
                "format_version": 3,
                "prompt_index": prompt_index,
                "prompt_sha256": prompt_digest(text),
                "input_ids": input_ids.detach().cpu().tolist(),
                "grad_idx": grad_idx,
                "alpha_grid": ALPHAS.tolist(),
                "model_compute_dtype": str(embedding_layer.weight.dtype),
                "gradient_dtype": str(gradient_tensor.dtype),
                "gradient_shape": list(gradient_tensor.shape),
                "gradients": gradient_tensor,
                "scores_by_sample_count": {
                    str(count): torch.from_numpy(scores_by_sample_count[count]).float()
                    for count in SAMPLE_COUNTS
                },
            },
            cache_path,
        )
        del gradients_float_cpu, gradient_tensor

    output: dict[str, dict[float, dict[str, Any]]] = {}
    for count in SAMPLE_COUNTS:
        scores = scores_by_sample_count[count]
        ranking = scores.argsort()[::-1]
        output[str(count)] = {}
        for fraction in ABLATION_FRACTIONS:
            num_top = max(1, int(len(grad_idx) * fraction))
            ablated_indices = ranking[:num_top]
            ablated_presence = presence.clone()
            ablated_presence[[grad_idx[int(i)] for i in ablated_indices], 0] = 0.0
            ablated_forward = scope_utils.customize_forward_pass(
                model, residual, ablated_presence, input_ids, grad_idx, attention_mask
            )
            with torch.no_grad():
                _, logits_ablated_full = ablated_forward(
                    loss_position=loss_position,
                    hidden_norm_as_loss=False,
                    unnormalized_logits=True,
                    tie_input_output_embed=False,
                )
            logits_ablated = logits_ablated_full[loss_position]
            ablated_log_probability = torch.log_softmax(
                logits_ablated, dim=-1
            )[max_index].item()
            row: dict[str, Any] = {
                "delta_log_prob": ablated_log_probability - max_log_probability,
                "prompt": text,
                "true_token": true_token,
                "predicted_token": predicted_token,
                "index": prompt_index,
                "ablated_indices": [int(i) for i in ablated_indices],
            }
            if isinstance(item, dict):
                row.update({key: value for key, value in item.items() if key != "prompt"})
            output[str(count)][fraction] = row
            del logits_ablated, logits_ablated_full, ablated_forward, ablated_presence

    del logits_orig, residual, presence, forward_pass, base_embeddings, token_embeddings
    torch.cuda.empty_cache()
    gc.collect()
    return output


def run_dataset(
    *,
    args: argparse.Namespace,
    dataset: str,
    tokenizer: Any,
    model: Any,
    embedding_layer: Any,
    lm_head: Any,
) -> None:
    dataset_path = args.repo_root / DATASETS[dataset]
    prompts = load_prompts(dataset_path, args.num_prompts)
    destination = args.output_root / args.model / dataset
    gradient_dir = destination / "gradients"
    checkpoint_path = destination / "results.json"
    metadata_path = destination / "metadata.json"
    gradient_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_path)
    completed = set(checkpoint["completed_prompt_indices"])
    cache_dtype = getattr(torch, args.cache_dtype)

    for prompt_index in tqdm(
        range(args.start_index, len(prompts)),
        desc=f"{args.model} / {dataset}",
        dynamic_ncols=True,
    ):
        if prompt_index in completed:
            continue
        cache_path = gradient_dir / f"prompt_{prompt_index:04d}.pt"
        prompt_results = compute_prompt(
            item=prompts[prompt_index],
            prompt_index=prompt_index,
            cache_path=cache_path,
            cache_dtype=cache_dtype,
            tokenizer=tokenizer,
            model=model,
            embedding_layer=embedding_layer,
            lm_head=lm_head,
            alpha_batch_size=args.alpha_batch_size,
        )
        for sample_count in SAMPLE_COUNTS:
            for fraction in ABLATION_FRACTIONS:
                checkpoint["results"][result_key(sample_count, fraction)].append(
                    prompt_results[str(sample_count)][fraction]
                )
        completed.add(prompt_index)
        checkpoint["completed_prompt_indices"] = sorted(completed)
        checkpoint["summary"] = summarize(checkpoint["results"])
        checkpoint["updated_at_unix"] = time.time()
        atomic_json_dump(checkpoint, checkpoint_path)

    metadata = build_metadata(args, MODELS[args.model])
    metadata.update(
        {
            "dataset": dataset,
            "dataset_path": str(dataset_path),
            "num_prompts_requested": args.num_prompts,
            "num_prompts_completed": len(completed),
            "completed_at_unix": time.time(),
        }
    )
    atomic_json_dump(metadata, metadata_path)


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    os.chdir(args.repo_root)
    np.random.seed(42)
    torch.manual_seed(42)

    model_name = MODELS[args.model]
    print(f"Loading {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float32,
        local_files_only=args.local_files_only,
    )
    model.eval()
    embedding_layer = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    print(
        f"Loaded {model_name}; embedding={tuple(embedding_layer.weight.shape)} "
        f"dtype={embedding_layer.weight.dtype} device={embedding_layer.weight.device}",
        flush=True,
    )

    root_metadata = build_metadata(args, model_name)
    root_metadata.update(
        {
            "datasets": args.datasets,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "transformers_version": sys.modules["transformers"].__version__,
        }
    )
    atomic_json_dump(root_metadata, args.output_root / args.model / "run_metadata.json")

    for dataset in args.datasets:
        run_dataset(
            args=args,
            dataset=dataset,
            tokenizer=tokenizer,
            model=model,
            embedding_layer=embedding_layer,
            lm_head=lm_head,
        )


if __name__ == "__main__":
    main()
