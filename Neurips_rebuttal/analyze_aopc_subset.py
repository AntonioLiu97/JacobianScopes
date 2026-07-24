#!/usr/bin/env python3
"""Reanalyze cached AOPC results with paired, per-passage statistics.

The primary artifact is the correctly predicted subset, where the gold target
used by target-specific attribution methods equals the model's pre-ablation
argmax token measured by AOPC.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


FRACTIONS = (0.05, 0.1, 0.2)
MODELS = {
    "Llama-3.2-1B": "LLaMA-3.2 1B",
    "Llama-3.2-3B": "LLaMA-3.2 3B",
    "Qwen2.5-1.5B": "Qwen2.5 1.5B",
    "Qwen2.5-3B": "Qwen2.5 3B",
    "gemma-3-1b-pt": "Gemma-3 1B",
    "gemma-3-4b-pt": "Gemma-3 4B",
}
DATASETS = {
    "lmbd1000": "LAMBADA",
    "IWSLT2017DE_EN": "IWSLT2017 DE→EN",
}
METHODS = {
    "random_ablation": "Random",
    "IG": "Integrated Gradients",
    "gradient_x_input": "Input × Gradient",
    "Semantic": "Semantic Scope",
    "Temperature": "Temperature Scope",
    "Fisher_k_1": "Fisher Scope",
}
SCOPES = ("Semantic", "Temperature", "Fisher_k_1")
NON_SCOPE_BASELINES = ("IG", "gradient_x_input")


@dataclass(frozen=True)
class CachedMethod:
    per_fraction: dict[float, dict[int, dict]]
    indices: tuple[int, ...]

    def per_example_aopc(self, indices: set[int] | None = None) -> pd.Series:
        selected = self.indices if indices is None else tuple(i for i in self.indices if i in indices)
        values = []
        for idx in selected:
            deltas = [self.per_fraction[f][idx]["delta_log_prob"] for f in FRACTIONS]
            values.append(float(np.trapezoid(deltas, FRACTIONS)))
        return pd.Series(values, index=selected, dtype=float)


def result_path(results_dir: Path, model: str, method: str, dataset: str, fraction: float) -> Path:
    return results_dir / f"{model}__{method}_{dataset}_top{fraction}_results.json"


def load_method(results_dir: Path, model: str, method: str, dataset: str) -> CachedMethod:
    per_fraction: dict[float, dict[int, dict]] = {}
    expected_indices: tuple[int, ...] | None = None
    for fraction in FRACTIONS:
        path = result_path(results_dir, model, method, dataset, fraction)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached result: {path}")
        payload = json.loads(path.read_text())
        records = {int(row["index"]): row for row in payload["results"]}
        indices = tuple(sorted(records))
        if expected_indices is None:
            expected_indices = indices
        elif indices != expected_indices:
            raise ValueError(f"Indices differ across fractions for {model}, {method}, {dataset}")
        per_fraction[fraction] = records
    assert expected_indices is not None
    return CachedMethod(per_fraction=per_fraction, indices=expected_indices)


def subset_indices(
    methods: dict[str, CachedMethod], model: str, dataset: str
) -> tuple[set[int], set[int], set[int]]:
    reference = methods["Semantic"].per_fraction[FRACTIONS[0]]
    correct_reference = {
        idx
        for idx, row in reference.items()
        if row.get("true_token") == row.get("predicted_token")
    }
    common = set.intersection(*(set(cached.indices) for cached in methods.values()))
    correct_common = correct_reference & common

    for method, cached in methods.items():
        for idx in common:
            ref_row = reference[idx]
            row = cached.per_fraction[FRACTIONS[0]][idx]
            metadata = (row.get("true_token"), row.get("predicted_token"))
            reference_metadata = (ref_row.get("true_token"), ref_row.get("predicted_token"))
            if metadata != reference_metadata:
                raise ValueError(
                    f"Prediction metadata differs for {model}, {dataset}, {method}, index {idx}"
                )
    if len(common) != len(reference):
        print(
            f"Note: {model}, {dataset} uses {len(common)}/{len(reference)} indices "
            "available for every method."
        )
    return common, correct_common, correct_reference


def sem(values: pd.Series) -> float:
    return float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("nan")


def bootstrap_mean_ci(
    differences: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int,
    chunk_size: int = 500,
) -> tuple[float, float]:
    means = np.empty(n_resamples, dtype=float)
    n = len(differences)
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        sampled_indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = differences[sampled_indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm step-down family-wise correction."""
    order = np.argsort(p_values.to_numpy())
    sorted_p = p_values.to_numpy()[order]
    adjusted_sorted = np.maximum.accumulate((len(sorted_p) - np.arange(len(sorted_p))) * sorted_p)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return pd.Series(adjusted, index=p_values.index)


def paired_row(
    model: str,
    dataset: str,
    subset: str,
    scope: str,
    baseline: str,
    scope_values: pd.Series,
    baseline_values: pd.Series,
    rng: np.random.Generator,
    n_resamples: int,
) -> dict:
    common = scope_values.index.intersection(baseline_values.index)
    differences = (scope_values.loc[common] - baseline_values.loc[common]).to_numpy()
    ci_low, ci_high = bootstrap_mean_ci(differences, rng, n_resamples)
    if np.allclose(differences, 0):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(differences, alternative="two-sided", method="auto").pvalue)
    return {
        "model": MODELS[model],
        "dataset": DATASETS[dataset],
        "subset": subset,
        "scope": METHODS[scope],
        "baseline": METHODS[baseline],
        "n": len(common),
        "mean_scope_minus_baseline": float(differences.mean()),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "wilcoxon_p": p_value,
        "scope_win_fraction": float(np.mean(differences < 0)),
        "tie_fraction": float(np.mean(differences == 0)),
    }


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def render_subset_markdown(summary: pd.DataFrame, accuracies: pd.DataFrame) -> str:
    lines = [
        "# AOPC on correctly predicted passages",
        "",
        "AOPC is reconstructed per passage using trapezoidal integration over "
        r"$k\\in\\{5\\%,10\\%,20\\%\\}$, then aggregated. More-negative values are better.",
        "",
        "## Subset sizes",
        "",
    ]
    accuracy_display = accuracies.copy()
    accuracy_display["accuracy"] = accuracy_display["accuracy"].map(lambda value: f"{100 * value:.1f}%")
    lines.append(
        markdown_table(
            accuracy_display,
            [
                "model",
                "dataset",
                "n_correct",
                "n_total",
                "accuracy",
                "n_common_correct",
            ],
            ["Model", "Dataset", "Correct", "Total", "Accuracy", "Analyzed"],
        )
    )

    for model in MODELS.values():
        lines.extend(["", f"## {model}", ""])
        model_rows = summary[summary["model"] == model].copy()
        pivot_value = model_rows.pivot(index="method", columns="dataset", values="formatted")
        ordered_methods = [name for name in METHODS.values() if name in pivot_value.index]
        display = pd.DataFrame({"method": ordered_methods})
        for dataset in DATASETS.values():
            display[dataset] = [pivot_value.loc[method, dataset] for method in ordered_methods]
        lines.append(
            markdown_table(
                display,
                ["method", *DATASETS.values()],
                ["Method", *DATASETS.values()],
            )
        )
    return "\n".join(lines) + "\n"


def render_paired_markdown(paired: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = [
        "# Paired AOPC analysis",
        "",
        "Each comparison uses within-passage differences "
        "`AOPC(scope) - AOPC(baseline)`; negative differences favor the Scope. "
        "Holm correction is applied across all Scope-versus-baseline comparisons within each subset.",
        "",
    ]
    for subset in ("all", "correct_only"):
        block = paired[paired["subset"] == subset]
        significant = block[block["holm_p"] < 0.05]
        favor_scope = significant[significant["mean_scope_minus_baseline"] < 0]
        favor_baseline = significant[significant["mean_scope_minus_baseline"] > 0]
        lines.extend(
            [
                f"## {subset}",
                "",
                f"- Comparisons: {len(block)}",
                f"- Significant in favor of a Scope: {len(favor_scope)}",
                f"- Significant in favor of a baseline: {len(favor_baseline)}",
                f"- Not significant after correction: {len(block) - len(significant)}",
                "",
            ]
        )
        if len(significant):
            display = significant.copy()
            display["difference"] = display["mean_scope_minus_baseline"].map(lambda value: f"{value:.4f}")
            display["ci"] = display.apply(
                lambda row: f"[{row['bootstrap_ci_low']:.4f}, {row['bootstrap_ci_high']:.4f}]",
                axis=1,
            )
            display["holm"] = display["holm_p"].map(lambda value: f"{value:.3g}")
            lines.append(
                markdown_table(
                    display,
                    ["model", "dataset", "scope", "baseline", "difference", "ci", "holm"],
                    ["Model", "Dataset", "Scope", "Baseline", "Paired difference", "95% CI", "Holm p"],
                )
            )
            lines.append("")

    lines.extend(
        [
            "## Best-performing Scope versus best-performing baseline",
            "",
            "For each model–dataset combination, this section compares the Scope and "
            "non-Scope baseline with the lowest mean AOPC in the corresponding subset. "
            "Holm p-values retain the correction across all Scope-versus-baseline "
            "comparisons within that subset.",
            "",
        ]
    )
    best_counts = {}
    for subset in ("all", "correct_only"):
        block = paired[paired["subset"] == subset]
        subset_summary = summary[summary["subset"] == subset]
        selected_rows = []
        for model in MODELS.values():
            for dataset in DATASETS.values():
                cell = subset_summary[
                    (subset_summary["model"] == model) & (subset_summary["dataset"] == dataset)
                ]
                best_scope = cell[cell["method"].isin([METHODS[name] for name in SCOPES])].nsmallest(
                    1, "mean_aopc"
                ).iloc[0]
                best_baseline = cell[
                    cell["method"].isin([METHODS[name] for name in NON_SCOPE_BASELINES])
                ].nsmallest(1, "mean_aopc").iloc[0]
                selected = block[
                    (block["model"] == model)
                    & (block["dataset"] == dataset)
                    & (block["scope"] == best_scope["method"])
                    & (block["baseline"] == best_baseline["method"])
                ].iloc[0].copy()
                selected["scope_mean"] = best_scope["mean_aopc"]
                selected["baseline_mean"] = best_baseline["mean_aopc"]
                selected_rows.append(selected)
        selected_block = pd.DataFrame(selected_rows)
        best_counts[subset] = {
            "scope_wins": int(
                (
                    (selected_block["holm_p"] < 0.05)
                    & (selected_block["mean_scope_minus_baseline"] < 0)
                ).sum()
            ),
            "baseline_wins": int(
                (
                    (selected_block["holm_p"] < 0.05)
                    & (selected_block["mean_scope_minus_baseline"] > 0)
                ).sum()
            ),
            "total": len(selected_block),
        }
        lines.extend([f"### {subset}", ""])
        display = selected_block.copy()
        display["scope_mean"] = display["scope_mean"].map(lambda value: f"{value:.3f}")
        display["baseline_mean"] = display["baseline_mean"].map(lambda value: f"{value:.3f}")
        display["difference"] = display["mean_scope_minus_baseline"].map(lambda value: f"{value:.4f}")
        display["ci"] = display.apply(
            lambda row: f"[{row['bootstrap_ci_low']:.4f}, {row['bootstrap_ci_high']:.4f}]",
            axis=1,
        )
        display["holm"] = display["holm_p"].map(lambda value: f"{value:.3g}")
        lines.append(
            markdown_table(
                display,
                [
                    "model",
                    "dataset",
                    "scope",
                    "scope_mean",
                    "baseline",
                    "baseline_mean",
                    "difference",
                    "ci",
                    "holm",
                ],
                [
                    "Model",
                    "Dataset",
                    "Best Scope",
                    "Scope AOPC",
                    "Best baseline",
                    "Baseline AOPC",
                    "Paired difference",
                    "95% CI",
                    "Holm p",
                ],
            )
        )
        lines.append("")
    lines.extend(
        [
            "### Summary and statistical interpretation",
            "",
            f"On all passages, the best Scope significantly beats the best non-Scope "
            f"baseline in **{best_counts['all']['scope_wins']}/{best_counts['all']['total']}** "
            f"model–dataset combinations. On correctly predicted passages, it does so in "
            f"**{best_counts['correct_only']['scope_wins']}/"
            f"{best_counts['correct_only']['total']}** combinations. There are "
            f"**{best_counts['all']['baseline_wins']}** significant best-baseline wins on "
            f"all passages and **{best_counts['correct_only']['baseline_wins']}** on the "
            "correct-only subset.",
            "",
            "The 95% CI is a bootstrap confidence interval for the mean paired difference. "
            "We resampled passages with replacement 10,000 times, preserving each passage's "
            "Scope–baseline pairing, computed the mean difference for every resample, and "
            "reported the 2.5th and 97.5th percentiles. Thus, we did perform the "
            "reviewer-suggested bootstrapping. An interval below zero supports the Scope; "
            "an interval crossing zero indicates that zero remains plausible.",
            "",
            "The Holm p-value is the two-sided paired Wilcoxon signed-rank p-value after "
            "Holm correction across all 72 Scope-versus-baseline comparisons in the "
            "corresponding subset. A Holm p-value below 0.05 is treated as significant. "
            "The confidence interval is bootstrap-based, whereas the p-value comes from "
            "the Wilcoxon test; they are complementary rather than the same calculation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "paper" / "results",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    summary_rows: list[dict] = []
    accuracy_rows: list[dict] = []
    paired_rows: list[dict] = []

    for model in MODELS:
        for dataset in DATASETS:
            cached = {
                method: load_method(args.results_dir, model, method, dataset)
                for method in METHODS
            }
            common, correct, correct_reference = subset_indices(cached, model, dataset)
            total = len(cached["Semantic"].indices)
            accuracy_rows.append(
                {
                    "model": MODELS[model],
                    "dataset": DATASETS[dataset],
                    "n_correct": len(correct_reference),
                    "n_total": total,
                    "accuracy": len(correct_reference) / total,
                    "n_common_all": len(common),
                    "n_common_correct": len(correct),
                }
            )

            values_by_subset: dict[str, dict[str, pd.Series]] = {}
            for subset, indices in (("all", common), ("correct_only", correct)):
                values_by_subset[subset] = {}
                for method, method_cache in cached.items():
                    values = method_cache.per_example_aopc(indices)
                    values_by_subset[subset][method] = values
                    summary_rows.append(
                        {
                            "model": MODELS[model],
                            "dataset": DATASETS[dataset],
                            "subset": subset,
                            "method": METHODS[method],
                            "n": len(values),
                            "mean_aopc": float(values.mean()),
                            "sem_aopc": sem(values),
                        }
                    )

                for scope in SCOPES:
                    for baseline in NON_SCOPE_BASELINES:
                        paired_rows.append(
                            paired_row(
                                model=model,
                                dataset=dataset,
                                subset=subset,
                                scope=scope,
                                baseline=baseline,
                                scope_values=values_by_subset[subset][scope],
                                baseline_values=values_by_subset[subset][baseline],
                                rng=rng,
                                n_resamples=args.bootstrap_resamples,
                            )
                        )

    summary = pd.DataFrame(summary_rows)
    summary["formatted"] = summary.apply(
        lambda row: f"{row['mean_aopc']:.3f} ± {row['sem_aopc']:.3f}", axis=1
    )
    accuracies = pd.DataFrame(accuracy_rows)
    paired = pd.DataFrame(paired_rows)
    paired["holm_p"] = np.nan
    for subset in paired["subset"].unique():
        mask = paired["subset"] == subset
        paired.loc[mask, "holm_p"] = holm_adjust(paired.loc[mask, "wilcoxon_p"]).to_numpy()

    summary.to_csv(args.output_dir / "aopc_subset_summary.csv", index=False)
    accuracies.to_csv(args.output_dir / "prediction_accuracies.csv", index=False)
    paired.to_csv(args.output_dir / "paired_aopc_tests.csv", index=False)

    correct_summary = summary[summary["subset"] == "correct_only"]
    (args.output_dir / "correct_subset_aopc_table.md").write_text(
        render_subset_markdown(correct_summary, accuracies)
    )
    (args.output_dir / "paired_aopc_summary.md").write_text(render_paired_markdown(paired, summary))

    machine_readable = {
        "fractions": FRACTIONS,
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
        "accuracies": accuracy_rows,
        "summary": summary_rows,
        "paired_tests": paired.to_dict(orient="records"),
    }
    (args.output_dir / "aopc_reanalysis.json").write_text(
        json.dumps(machine_readable, indent=2, allow_nan=False)
    )

    print(f"Wrote AOPC reanalysis to {args.output_dir}")
    print(
        "Correctly predicted passages: "
        f"{accuracies['n_correct'].sum()}/{accuracies['n_total'].sum()} "
        f"({accuracies['n_correct'].sum() / accuracies['n_total'].sum():.1%})"
    )


if __name__ == "__main__":
    main()
