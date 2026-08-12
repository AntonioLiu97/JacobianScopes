#!/usr/bin/env python3
"""Combine parameter-randomization results into one Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCOPES = ("Semantic", "Temperature", "Fisher")
CHECKPOINTS = ("final_quarter", "final_half", "all_blocks")
MODELS = (
    ("Llama-3.2-1B", "LLaMA-3.2 1B"),
    ("Llama-3.2-3B", "LLaMA-3.2 3B"),
    ("Qwen2.5-1.5B", "Qwen2.5 1.5B"),
    ("Qwen2.5-3B", "Qwen2.5 3B"),
    ("gemma-3-1b-pt", "Gemma-3 1B"),
    ("gemma-3-4b-pt", "Gemma-3 4B"),
)
DATASETS = (
    ("lmbd1000", "LAMBADA"),
    ("IWSLT2017DE_EN", "IWSLT2017 DE→EN"),
)
EXPERIMENTS = tuple(
    (model_short, model_label, dataset, dataset_label)
    for model_short, model_label in MODELS
    for dataset, dataset_label in DATASETS
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir / "saliency_randomization_results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "saliency_check_all.md",
    )
    return parser.parse_args()


def load_results(
    experiment_dir: Path,
) -> tuple[dict[tuple[str, str], dict[str, str]], int | None]:
    metadata_path = experiment_dir / "saliency_randomization_metadata.json"
    summary_path = experiment_dir / "saliency_randomization_summary.csv"
    if not metadata_path.exists() or not summary_path.exists():
        return {}, None

    metadata = json.loads(metadata_path.read_text())
    # Older outputs were produced before block reinitialization was fixed and
    # contain the two baselines. Do not silently report those stale results.
    if tuple(metadata.get("methods", ())) != SCOPES:
        return {}, None

    with summary_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {
        (row["checkpoint"], row["method"]): row
        for row in rows
        if row["checkpoint"] in CHECKPOINTS and row["method"] in SCOPES
    }
    if any((checkpoint, method) not in indexed for checkpoint in CHECKPOINTS for method in SCOPES):
        return {}, None
    return indexed, int(metadata["num_prompts"])


def format_estimate(row: dict[str, str], metric: str) -> str:
    return f"{float(row[f'mean_{metric}']):.3f} ± {float(row[f'sem_{metric}']):.3f}"


def render_experiment(
    results: dict[tuple[str, str], dict[str, str]],
) -> list[str]:
    lines = [
        "| Scope | Similarity | Original | Final quarter | Final half | All blocks |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for method in SCOPES:
        for label, metric in (
            ("Spearman ρ", "spearman"),
            ("Top-10% Jaccard", "top10_jaccard"),
        ):
            estimates = [
                format_estimate(results[(checkpoint, method)], metric)
                if results
                else "NA"
                for checkpoint in CHECKPOINTS
            ]
            lines.append(
                f"| {method} | {label} | 1.000 | "
                f"{estimates[0]} | {estimates[1]} | {estimates[2]} |"
            )
    return lines


def render(results_dir: Path) -> str:
    lines = [
        "# Parameter-randomization saliency sanity checks",
        "",
        "Each table compares an attribution map with its original-model map as "
        "transformer blocks are cumulatively reinitialized from the output side. "
        "Values are mean ± SEM across fixed, correctly predicted prompts; lower "
        "similarity indicates greater dependence on learned parameters. `NA` marks "
        "an unfinished rerun.",
        "",
    ]
    for model_short, model_label, dataset, dataset_label in EXPERIMENTS:
        experiment_dir = results_dir / f"{model_short}_{dataset}"
        results, num_prompts = load_results(experiment_dir)
        sample_text = f" (*n* = {num_prompts})" if num_prompts is not None else ""
        lines.extend(
            [
                f"## {model_label} — {dataset_label}{sample_text}",
                "",
                *render_experiment(results),
                "",
            ]
        )
    lines.extend(
        [
            "Protocol: Adebayo et al. (2018), with text-ranking metrics following "
            "Kokhlikyan et al. (2021). The target token is held fixed even if the "
            "randomized model's prediction changes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output.write_text(render(args.results_dir))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
