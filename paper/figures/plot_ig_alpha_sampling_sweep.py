#!/usr/bin/env python3
"""Plot AUPC and perturbation scores across IG alpha sampling rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = (
    "Llama-3.2-1B",
    "Qwen2.5-1.5B",
    "Gemma-3-1B",
    "Llama-3.2-3B",
    "Qwen2.5-3B",
    "Gemma-3-4B",
)
DATASETS = ("lmbd1000", "IWSLT2017DE_EN")
SAMPLE_COUNTS = (5, 10, 20)
FRACTIONS = (0.05, 0.1, 0.2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def load_summary(root: Path, model: str, dataset: str) -> dict:
    path = root / model / dataset / "results.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle).get("summary", {})


def plot_aupc(root: Path, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    for axis, dataset in zip(axes, DATASETS):
        for model in MODELS:
            summary = load_summary(root, model, dataset)
            points = [summary.get(f"samples_{count}/aupc") for count in SAMPLE_COUNTS]
            if not all(points):
                continue
            values = [point["aupc"] for point in points]
            errors = [point["sem_aupc"] for point in points]
            axis.errorbar(SAMPLE_COUNTS, values, yerr=errors, marker="o", capsize=3, label=model)
        axis.set_title("LAMBADA-1000" if dataset == "lmbd1000" else "IWSLT2017 DE-EN")
        axis.set_xlabel("Number of alpha samples")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("AUPC")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "ig_aupc_by_alpha_samples.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "ig_aupc_by_alpha_samples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_perturbation_scores(root: Path, figure_dir: Path) -> None:
    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharex=True)
        for axis, fraction in zip(axes, FRACTIONS):
            for model in MODELS:
                summary = load_summary(root, model, dataset)
                points = [summary.get(f"samples_{count}/top_{fraction}") for count in SAMPLE_COUNTS]
                if not all(points):
                    continue
                values = [point["avg_delta"] for point in points]
                errors = [point["sem_delta"] for point in points]
                axis.errorbar(SAMPLE_COUNTS, values, yerr=errors, marker="o", capsize=3, label=model)
            axis.set_title(f"Top {int(100 * fraction)}% ablated")
            axis.set_xlabel("Number of alpha samples")
            axis.grid(True, alpha=0.25)
        axes[0].set_ylabel("Mean change in log probability")
        axes[-1].legend(frameon=False, fontsize=8)
        fig.suptitle("LAMBADA-1000" if dataset == "lmbd1000" else "IWSLT2017 DE-EN")
        fig.tight_layout()
        stem = f"ig_perturbation_by_alpha_samples_{dataset}"
        fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(figure_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def write_csv(root: Path, figure_dir: Path) -> None:
    lines = ["model,dataset,alpha_samples,ablation_fraction,mean_delta,sem_delta,aupc,sem_aupc"]
    for model in MODELS:
        for dataset in DATASETS:
            summary = load_summary(root, model, dataset)
            for count in SAMPLE_COUNTS:
                aupc = summary.get(f"samples_{count}/aupc", {})
                for fraction in FRACTIONS:
                    point = summary.get(f"samples_{count}/top_{fraction}", {})
                    lines.append(
                        ",".join(
                            str(value)
                            for value in (
                                model,
                                dataset,
                                count,
                                fraction,
                                point.get("avg_delta", ""),
                                point.get("sem_delta", ""),
                                aupc.get("aupc", ""),
                                aupc.get("sem_aupc", ""),
                            )
                        )
                    )
    (figure_dir / "ig_alpha_sampling_sweep.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    figure_dir = args.output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_aupc(args.output_root, figure_dir)
    plot_perturbation_scores(args.output_root, figure_dir)
    write_csv(args.output_root, figure_dir)


if __name__ == "__main__":
    main()
