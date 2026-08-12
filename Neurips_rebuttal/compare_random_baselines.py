#!/usr/bin/env python3
"""Compare random-token AOPC under zero and PAD replacement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ZERO_DIR = ROOT / "paper" / "results"
PAD_DIR = ROOT / "Neurips_rebuttal" / "filler_aopc_results"
FRACTIONS = (0.05, 0.1, 0.2)
MODELS = (
    ("Llama-3.2-1B", "LLaMA-3.2 1B", "right_pad"),
    ("Llama-3.2-3B", "LLaMA-3.2 3B", "right_pad"),
    ("Qwen2.5-1.5B", "Qwen2.5 1.5B", "pad_eos"),
    ("Qwen2.5-3B", "Qwen2.5 3B", "pad_eos"),
    ("gemma-3-1b-pt", "Gemma-3 1B", "pad"),
    ("gemma-3-4b-pt", "Gemma-3 4B", "pad"),
)
DATASETS = (("lmbd1000", "LAMBADA"), ("IWSLT2017DE_EN", "IWSLT2017 DE→EN"))


def load_records(paths: list[Path]) -> list[dict[int, dict]]:
    records = []
    for path in paths:
        payload = json.loads(path.read_text())
        records.append({int(row["index"]): row for row in payload["results"]})
    return records


def per_passage_aopc(records: list[dict[int, dict]], indices: list[int]) -> np.ndarray:
    return np.asarray(
        [
            np.trapezoid(
                [records[j][index]["delta_log_prob"] for j in range(len(FRACTIONS))],
                FRACTIONS,
            )
            for index in indices
        ]
    )


def main() -> None:
    rows = []
    for model, model_label, filler in MODELS:
        for dataset, dataset_label in DATASETS:
            zero_paths = [
                ZERO_DIR
                / f"{model}__random_ablation_{dataset}_top{fraction}_results.json"
                for fraction in FRACTIONS
            ]
            pad_paths = [
                PAD_DIR
                / (
                    f"{model}__random_ablation_{dataset}_top{fraction}"
                    f"_filler_{filler}_results.json"
                )
                for fraction in FRACTIONS
            ]
            zero_records = load_records(zero_paths)
            pad_records = load_records(pad_paths)
            common = sorted(
                set.intersection(
                    *(set(records) for records in [*zero_records, *pad_records])
                )
            )
            zero = per_passage_aopc(zero_records, common)
            pad = per_passage_aopc(pad_records, common)
            rows.append(
                {
                    "model": model_label,
                    "dataset": dataset_label,
                    "n": len(common),
                    "zero_mean": zero.mean(),
                    "zero_sem": zero.std(ddof=1) / np.sqrt(len(zero)),
                    "pad_mean": pad.mean(),
                    "pad_sem": pad.std(ddof=1) / np.sqrt(len(pad)),
                    "zero_magnitude": abs(zero.mean()),
                    "pad_magnitude": abs(pad.mean()),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(Path(__file__).with_name("random_baseline_comparison.csv"), index=False)

    zero = frame["zero_magnitude"]
    pad = frame["pad_magnitude"]
    zero_mean, pad_mean = zero.mean(), pad.mean()
    zero_sem = zero.std(ddof=1) / np.sqrt(len(zero))
    pad_sem = pad.std(ddof=1) / np.sqrt(len(pad))
    zero_wins = int((zero < pad).sum())

    summary = [
        "# Random-ablation baseline comparison",
        "",
        "Aggregate values are the mean magnitude of the cell-level mean AOPC. "
        "Error bars are SEM across the 12 model–dataset cells.",
        "",
        f"- Zero: **{zero_mean:.3f} ± {zero_sem:.3f}**",
        f"- PAD: **{pad_mean:.3f} ± {pad_sem:.3f}**",
        f"- Zero has smaller magnitude in **{zero_wins} of {len(frame)}** cells.",
        "",
    ]
    Path(__file__).with_name("random_baseline_comparison.md").write_text(
        "\n".join(summary)
    )
    print("\n".join(summary))


if __name__ == "__main__":
    main()
