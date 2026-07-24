#!/usr/bin/env python3
"""Measure full-ranking agreement between attribution scores and LOO-KL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


MODELS = {
    "Llama-3.2-1B": "LLaMA-3.2 1B",
    "Llama-3.2-3B": "LLaMA-3.2 3B",
}
METHODS = {
    "temperature": "Temperature Scope",
    "semantic": "Semantic Scope",
    "gradient_x_input": "Input × Gradient",
    "fisher_k4": "Fisher Scope (k=4)",
    "ig": "Integrated Gradients",
    "random": "Random",
}
METRICS = ("spearman", "kendall_tau_b", "top10_jaccard")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=root / "paper" / "results")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def top_fraction(scores: np.ndarray, fraction: float = 0.1) -> set[int]:
    count = max(1, int(math.ceil(len(scores) * fraction)))
    return set(np.argsort(scores, kind="stable")[-count:].tolist())


def ranking_metrics(loo_scores: np.ndarray, method_scores: np.ndarray) -> dict[str, float]:
    if len(loo_scores) != len(method_scores):
        raise ValueError(
            f"Score-length mismatch: LOO={len(loo_scores)}, method={len(method_scores)}"
        )
    spearman = float(spearmanr(loo_scores, method_scores).statistic)
    kendall = float(kendalltau(loo_scores, method_scores, variant="b").statistic)
    loo_top = top_fraction(loo_scores)
    method_top = top_fraction(method_scores)
    jaccard = len(loo_top & method_top) / len(loo_top | method_top)
    return {
        "spearman": spearman,
        "kendall_tau_b": kendall,
        "top10_jaccard": jaccard,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int,
    chunk_size: int = 500,
) -> tuple[float, float]:
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def load_rows(results_dir: Path) -> list[dict]:
    rows = []
    for model_short, model_display in MODELS.items():
        path = results_dir / f"{model_short}__LOO_KL_lambada_loo_results.json"
        records = json.loads(path.read_text())["results"]
        for prompt_index, record in enumerate(records):
            loo_scores = np.asarray(record["kl_divergences"], dtype=float)
            is_correct = record.get("true_token") == record.get("predicted_token")
            for method_key, method_display in METHODS.items():
                field = f"{method_key}_influence_scores"
                if field not in record:
                    continue
                method_scores = np.asarray(record[field], dtype=float)
                metrics = ranking_metrics(loo_scores, method_scores)
                rows.append(
                    {
                        "model": model_display,
                        "prompt_index": prompt_index,
                        "method": method_display,
                        "n_tokens": len(loo_scores),
                        "is_correct": is_correct,
                        **metrics,
                    }
                )
    return rows


def summarize(
    per_prompt: pd.DataFrame,
    rng: np.random.Generator,
    n_resamples: int,
) -> pd.DataFrame:
    rows = []
    subset_frames = {
        "all": per_prompt,
        "correct_only": per_prompt[per_prompt["is_correct"]],
    }
    for subset, frame in subset_frames.items():
        for (model, method), group in frame.groupby(["model", "method"], sort=False):
            row = {
                "subset": subset,
                "model": model,
                "method": method,
                "n": len(group),
                "mean_tokens": group["n_tokens"].mean(),
            }
            for metric in METRICS:
                values = group[metric].dropna().to_numpy()
                low, high = bootstrap_mean_ci(values, rng, n_resamples)
                row[f"mean_{metric}"] = values.mean()
                row[f"sem_{metric}"] = values.std(ddof=1) / math.sqrt(len(values))
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            rows.append(row)
    return pd.DataFrame(rows)


def render_markdown(summary: pd.DataFrame) -> str:
    lines = [
        "# Full-ranking agreement with LOO-KL interventions",
        "",
        "Agreement is computed independently for each passage between the complete "
        "token ranking from an attribution method and the ranking induced by "
        "single-token LOO KL divergence. Values are prompt-level means with paired "
        "bootstrap 95% confidence intervals. Higher is better.",
        "",
    ]
    method_order = list(METHODS.values())
    for subset in ("all", "correct_only"):
        lines.extend([f"## {subset}", ""])
        for model in MODELS.values():
            lines.extend(
                [
                    f"### {model}",
                    "",
                    "| Method | n | Spearman ρ [95% CI] | Kendall τ-b [95% CI] | Top-10% Jaccard [95% CI] |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            block = summary[
                (summary["subset"] == subset) & (summary["model"] == model)
            ].set_index("method")
            for method in method_order:
                if method not in block.index:
                    continue
                row = block.loc[method]
                lines.append(
                    f"| {method} | {int(row['n'])} | "
                    f"{row['mean_spearman']:.3f} "
                    f"[{row['spearman_ci_low']:.3f}, {row['spearman_ci_high']:.3f}] | "
                    f"{row['mean_kendall_tau_b']:.3f} "
                    f"[{row['kendall_tau_b_ci_low']:.3f}, {row['kendall_tau_b_ci_high']:.3f}] | "
                    f"{row['mean_top10_jaccard']:.3f} "
                    f"[{row['top10_jaccard_ci_low']:.3f}, {row['top10_jaccard_ci_high']:.3f}] |"
                )
            lines.append("")
    lines.extend(
        [
            "This is an interventional ranking-agreement analysis. It complements, "
            "but does not replace, parameter-randomization sanity checks.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    per_prompt = pd.DataFrame(load_rows(args.results_dir))
    summary = summarize(per_prompt, rng, args.bootstrap_resamples)
    per_prompt.to_csv(args.output_dir / "loo_ranking_agreement_per_prompt.csv", index=False)
    summary.to_csv(args.output_dir / "loo_ranking_agreement_summary.csv", index=False)
    (args.output_dir / "loo_ranking_agreement.md").write_text(
        render_markdown(summary)
    )
    print(
        f"Wrote {len(per_prompt)} prompt-method comparisons to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
