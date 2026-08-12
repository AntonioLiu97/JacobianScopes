#!/usr/bin/env python3
"""Regenerate paper Table 1 using PAD/filler-token AOPC ablation results.

AOPC is reconstructed per passage via trapezoidal integration over
k in {5%, 10%, 20%}, then reported as mean ± SEM. More-negative is better.
Incomplete or missing method–dataset cells are reported as NA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FRACTIONS = (0.05, 0.1, 0.2)
MODELS = (
    ("Llama-3.2-1B", "LLaMA-3.2 1B", "right_pad"),
    ("Llama-3.2-3B", "LLaMA-3.2 3B", "right_pad"),
    ("Qwen2.5-1.5B", "Qwen2.5 1.5B", "pad_eos"),
    ("Qwen2.5-3B", "Qwen2.5 3B", "pad_eos"),
    ("gemma-3-1b-pt", "Gemma-3 1B", "pad"),
    ("gemma-3-4b-pt", "Gemma-3 4B", "pad"),
)
DATASETS = (
    ("lmbd1000", "LAMBADA"),
    ("IWSLT2017DE_EN", "IWSLT2017 DE→EN"),
)
METHODS = (
    ("random_ablation", "Random"),
    ("IG", "Integrated Gradients"),
    ("gradient_x_input", "Input × Gradient"),
    ("Semantic", "Semantic Scope"),
    ("Temperature", "Temperature Scope"),
    ("Fisher_k_1", "Fisher Scope"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filler-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "filler_aopc_results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "pad_table1.md",
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=1000,
        help="Require this many matched indices across all three fractions; else NA.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Display precision for mean ± SEM (paper-style rebuttal table uses 2).",
    )
    return parser.parse_args()


def result_path(
    filler_dir: Path,
    model: str,
    method: str,
    dataset: str,
    fraction: float,
    filler: str,
) -> Path:
    return (
        filler_dir
        / f"{model}__{method}_{dataset}_top{fraction}_filler_{filler}_results.json"
    )


def load_matched_records(
    filler_dir: Path,
    model: str,
    method: str,
    dataset: str,
    filler: str,
) -> dict[float, dict[int, dict]] | None:
    by_fraction: dict[float, dict[int, dict]] = {}
    for fraction in FRACTIONS:
        path = result_path(filler_dir, model, method, dataset, fraction, filler)
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        by_fraction[fraction] = {int(row["index"]): row for row in payload["results"]}
    return by_fraction


def mean_aopc_sem(
    records: dict[float, dict[int, dict]], min_examples: int
) -> tuple[float, float, int] | None:
    indices = sorted(
        set.intersection(*(set(records[fraction]) for fraction in FRACTIONS))
    )
    if len(indices) < min_examples:
        return None
    values = np.asarray(
        [
            np.trapezoid(
                [records[fraction][index]["delta_log_prob"] for fraction in FRACTIONS],
                FRACTIONS,
            )
            for index in indices
        ],
        dtype=float,
    )
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, sem, len(values)


def format_cell(
    stats: tuple[float, float, int] | None, decimals: int, bold: bool
) -> str:
    if stats is None:
        return "NA"
    mean, sem, _n = stats
    text = f"{mean:.{decimals}f} ± {sem:.{decimals}f}"
    return f"**{text}**" if bold else text


def build_table(
    filler_dir: Path, min_examples: int, decimals: int
) -> tuple[str, pd.DataFrame]:
    rows: list[dict] = []
    stats_grid: dict[tuple[str, str, str], tuple[float, float, int] | None] = {}

    for model_key, model_label, filler in MODELS:
        for method_key, method_label in METHODS:
            row: dict = {"Model": model_label, "Method": method_label}
            for dataset_key, dataset_label in DATASETS:
                records = load_matched_records(
                    filler_dir, model_key, method_key, dataset_key, filler
                )
                stats = (
                    None
                    if records is None
                    else mean_aopc_sem(records, min_examples)
                )
                stats_grid[(model_label, method_label, dataset_label)] = stats
                row[dataset_label] = stats
                rows.append(
                    {
                        "model": model_label,
                        "method": method_label,
                        "dataset": dataset_label,
                        "n": None if stats is None else stats[2],
                        "mean_aopc": None if stats is None else stats[0],
                        "sem_aopc": None if stats is None else stats[1],
                    }
                )
            # keep one logical table row per model/method; CSV gets long form above
            _ = row

    # Best (most negative mean) per model × dataset among non-NA cells.
    # Bold all entries tied at the displayed precision, since readers cannot
    # distinguish differences hidden by paper-style rounding.
    best: set[tuple[str, str, str]] = set()
    for _model_key, model_label, _filler in MODELS:
        for _dataset_key, dataset_label in DATASETS:
            candidates = []
            for _method_key, method_label in METHODS:
                key = (model_label, method_label, dataset_label)
                stats = stats_grid[key]
                if stats is not None:
                    candidates.append((stats[0], key))
            if candidates:
                displayed_best = min(round(mean, decimals) for mean, _key in candidates)
                best.update(
                    key
                    for mean, key in candidates
                    if round(mean, decimals) == displayed_best
                )

    lines = [
        "# Table 1 (PAD / filler-token ablation)",
        "",
        "Regenerated from length-preserving filler-token AOPC sweeps. "
        "AOPC is the per-passage trapezoidal integral over "
        r"$k\in\{5\%,10\%,20\%\}$"
        f", reported as mean ± SEM. More-negative values are better. "
        f"Cells with fewer than {min_examples} matched examples across all "
        "three ablation fractions are shown as NA. "
        "The best or tied-best score at the displayed precision in each "
        "model×dataset column is bolded.",
        "",
        "Filler tokens: LLaMA `<|finetune_right_pad_id|>`; "
        "Qwen shared PAD/EOS `<|endoftext|>`; Gemma `<pad>`.",
        "",
        "| Model | Method | LAMBADA | IWSLT2017 DE→EN |",
        "| --- | --- | --- | --- |",
    ]

    for model_key, model_label, filler in MODELS:
        first = True
        for method_key, method_label in METHODS:
            lambada = format_cell(
                stats_grid[(model_label, method_label, "LAMBADA")],
                decimals,
                (model_label, method_label, "LAMBADA") in best,
            )
            iwslt = format_cell(
                stats_grid[(model_label, method_label, "IWSLT2017 DE→EN")],
                decimals,
                (model_label, method_label, "IWSLT2017 DE→EN") in best,
            )
            model_cell = model_label if first else ""
            lines.append(
                f"| {model_cell} | {method_label} | {lambada} | {iwslt} |"
            )
            first = False

    lines.append("")
    return "\n".join(lines), pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    markdown, frame = build_table(
        args.filler_dir, args.min_examples, args.decimals
    )
    args.output.write_text(markdown)
    frame.to_csv(args.output.with_suffix(".csv"), index=False)
    print(markdown)
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
