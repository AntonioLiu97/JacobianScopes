#!/usr/bin/env python3
"""Rerun LLaMA-3.2-1B IG with a selected token-embedding baseline.

Only cached correctly predicted passages are processed. The five integration
points and zero-embedding AOPC intervention match the original benchmark, so
the only intended change is the IG path baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "meta-llama/Llama-3.2-1B"
MODEL_SHORT = "Llama-3.2-1B"
PAD_TOKEN = "<|finetune_right_pad_id|>"
FRACTIONS = (0.05, 0.1, 0.2)
ALPHAS = (0.2, 0.4, 0.6, 0.8, 1.0)
DATASETS = ("lmbd1000", "IWSLT2017DE_EN")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline", choices=("pad", "bos", "eos"), default="pad")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--only-index", type=int)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--cached-results-dir", type=Path, default=root / "paper" / "results")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = (
            Path(__file__).resolve().parent / f"ig_{args.baseline}_results"
        )
    return args


def load_correct_entries(cached_results_dir: Path, dataset: str) -> list[tuple[int, str]]:
    path = cached_results_dir / f"{MODEL_SHORT}__Semantic_{dataset}_top0.05_results.json"
    payload = json.loads(path.read_text())
    return sorted(
        (
            int(row["index"]),
            row["prompt"],
        )
        for row in payload["results"]
        if row["true_token"] == row["predicted_token"]
    )


def save_results(
    output_dir: Path,
    dataset: str,
    baseline_name: str,
    results: dict[float, list[dict]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fraction, rows in results.items():
        label = f"{MODEL_SHORT}__IG_{baseline_name}_{dataset}_top{fraction}"
        path = output_dir / f"{label}_results.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {"label": label, "top_k_fraction": fraction, "results": rows},
                indent=2,
                ensure_ascii=False,
            )
        )
        temporary.replace(path)


def load_existing(
    output_dir: Path, dataset: str, baseline_name: str
) -> dict[float, list[dict]]:
    results: dict[float, list[dict]] = {fraction: [] for fraction in FRACTIONS}
    for fraction in FRACTIONS:
        label = f"{MODEL_SHORT}__IG_{baseline_name}_{dataset}_top{fraction}"
        path = output_dir / f"{label}_results.json"
        if path.exists():
            results[fraction] = json.loads(path.read_text())["results"]
    completed_sets = [{int(row["index"]) for row in rows} for rows in results.values()]
    if len({frozenset(indices) for indices in completed_sets}) != 1:
        raise ValueError("Existing output files do not contain the same indices")
    return results


def model_logits(
    model: AutoModelForCausalLM,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    return model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits


def integrated_gradients_pad_scores(
    model: AutoModelForCausalLM,
    input_embeds: torch.Tensor,
    baseline_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_position: int,
    target_id: int,
) -> np.ndarray:
    difference = input_embeds - baseline_embeds
    gradients = []
    for alpha in ALPHAS:
        path_embeds = (baseline_embeds + alpha * difference).detach().requires_grad_(True)
        logits = model_logits(model, path_embeds, attention_mask)
        loss = torch.nn.functional.cross_entropy(
            logits[:, loss_position, :].float(),
            torch.tensor([target_id], device=logits.device),
        )
        gradient = torch.autograd.grad(loss, path_embeds, retain_graph=False)[0]
        gradients.append(gradient.detach())
        del logits, loss, path_embeds
    average_gradient = torch.stack(gradients).mean(dim=0)
    scores = (average_gradient * difference).norm(dim=-1)[0]
    return scores.float().cpu().numpy()


@torch.no_grad()
def ablation_deltas(
    model: AutoModelForCausalLM,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_position: int,
    predicted_id: int,
    ranked_indices: np.ndarray,
) -> dict[float, float]:
    original_logits = model_logits(model, input_embeds, attention_mask)
    original_log_prob = torch.log_softmax(
        original_logits[0, loss_position].float(), dim=-1
    )[predicted_id]
    deltas = {}
    for fraction in FRACTIONS:
        n_top = max(1, int(input_embeds.shape[1] * fraction))
        ablated = input_embeds.clone()
        selected = torch.as_tensor(
            ranked_indices[:n_top].copy(), dtype=torch.long, device=ablated.device
        )
        ablated[0, selected] = 0
        ablated_logits = model_logits(model, ablated, attention_mask)
        ablated_log_prob = torch.log_softmax(
            ablated_logits[0, loss_position].float(), dim=-1
        )[predicted_id]
        deltas[fraction] = float((ablated_log_prob - original_log_prob).item())
        del ablated, ablated_logits
    return deltas


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    baseline_tokens = {
        "pad": (PAD_TOKEN, tokenizer.convert_tokens_to_ids(PAD_TOKEN)),
        "bos": (tokenizer.bos_token, tokenizer.bos_token_id),
        "eos": (tokenizer.eos_token, tokenizer.eos_token_id),
    }
    baseline_token, baseline_token_id = baseline_tokens[args.baseline]
    if baseline_token_id == tokenizer.unk_token_id or baseline_token_id is None:
        raise ValueError(
            f"{baseline_token!r} is not a recognized LLaMA tokenizer token"
        )

    # The original AOPC benchmark did not request a reduced dtype, so it ran in
    # FP32. Preserve that precision to keep cached-correct membership exact.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.to(device).eval().requires_grad_(False)
    embedding_layer = model.get_input_embeddings()

    correct = load_correct_entries(args.cached_results_dir, args.dataset)
    if args.only_index is not None:
        correct = [entry for entry in correct if entry[0] == args.only_index]
        if not correct:
            raise ValueError(
                f"Index {args.only_index} is not cached as correct for {args.dataset}"
            )
    if args.max_examples is not None:
        correct = correct[: args.max_examples]

    results = load_existing(args.output_dir, args.dataset, args.baseline)
    completed = {int(row["index"]) for row in results[FRACTIONS[0]]}
    pending = [
        (index, prompt)
        for index, prompt in correct
        if index not in completed
    ]
    print(
        f"{args.dataset}: {len(correct)} correctly predicted entries; "
        f"{len(completed)} complete, {len(pending)} pending; "
        f"{args.baseline.upper()} id={baseline_token_id}"
    )

    for count, (index, prompt) in enumerate(tqdm(pending, desc=args.dataset), start=1):
        input_ids = tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)
        if input_ids.shape[1] < 2:
            raise ValueError(f"Prompt {index} has fewer than two tokens")
        attention_mask = torch.ones_like(input_ids)
        loss_position = input_ids.shape[1] - 2
        target_id = int(input_ids[0, loss_position + 1])

        with torch.no_grad():
            input_embeds = embedding_layer(input_ids).detach()
            baseline_embeds = input_embeds.clone()
            replace_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=device)
            if tokenizer.bos_token_id is not None and int(input_ids[0, 0]) == tokenizer.bos_token_id:
                replace_mask[0] = False
            baseline_ids = torch.full_like(input_ids, baseline_token_id)
            token_baseline_embeds = embedding_layer(baseline_ids).detach()
            baseline_embeds[0, replace_mask] = token_baseline_embeds[0, replace_mask]

            original_logits = model_logits(model, input_embeds, attention_mask)
            predicted_id = int(original_logits[0, loss_position].argmax())
            del original_logits
        if predicted_id != target_id:
            raise ValueError(
                f"Cached-correct entry {index} is not correct in this run: "
                f"target={target_id}, predicted={predicted_id}"
            )

        scores = integrated_gradients_pad_scores(
            model=model,
            input_embeds=input_embeds,
            baseline_embeds=baseline_embeds,
            attention_mask=attention_mask,
            loss_position=loss_position,
            target_id=target_id,
        )
        ranked = np.argsort(scores)[::-1]
        deltas = ablation_deltas(
            model=model,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            loss_position=loss_position,
            predicted_id=predicted_id,
            ranked_indices=ranked,
        )

        common = {
            "prompt": prompt,
            "true_token": tokenizer.decode([target_id]),
            "predicted_token": tokenizer.decode([predicted_id]),
            "index": index,
            "ig_baseline_name": args.baseline,
            "ig_baseline_token": baseline_token,
            "ig_baseline_token_id": baseline_token_id,
            "integration_alphas": ALPHAS,
        }
        for fraction in FRACTIONS:
            results[fraction].append(
                {"delta_log_prob": deltas[fraction], **common}
            )
            results[fraction].sort(key=lambda row: int(row["index"]))

        if count % args.save_every == 0:
            save_results(args.output_dir, args.dataset, args.baseline, results)
        del input_ids, input_embeds, baseline_embeds, token_baseline_embeds, scores

    save_results(args.output_dir, args.dataset, args.baseline, results)
    print(f"Saved {len(results[FRACTIONS[0]])} entries for {args.dataset}")


if __name__ == "__main__":
    main()
