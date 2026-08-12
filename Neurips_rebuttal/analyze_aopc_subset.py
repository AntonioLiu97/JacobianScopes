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
from scipy.stats import t, wilcoxon


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


def paired_row(
    model: str,
    dataset: str,
    subset: str,
    scope: str,
    baseline: str,
    scope_values: pd.Series,
    baseline_values: pd.Series,
) -> dict:
    common = scope_values.index.intersection(baseline_values.index)
    differences = (scope_values.loc[common] - baseline_values.loc[common]).to_numpy()
    mean_difference = float(differences.mean())
    standard_error = float(differences.std(ddof=1) / math.sqrt(len(differences)))
    margin = float(t.ppf(0.975, df=len(differences) - 1) * standard_error)
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
        "mean_scope_minus_baseline": mean_difference,
        "ci_low": mean_difference - margin,
        "ci_high": mean_difference + margin,
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
        "The Scope and baseline for each model–dataset cell are fixed to the methods "
        "with the lowest mean AOPC in the original all-passage table. Each comparison "
        "uses within-passage differences `AOPC(scope) - AOPC(baseline)`; negative "
        "differences favor the Scope. The 95% confidence interval is the standard "
        "Student-t interval for the mean paired difference. Reported p-values are "
        "from two-sided paired Wilcoxon signed-rank tests.",
        "",
    ]
    original = summary[summary["subset"] == "all"]
    selected_methods: dict[tuple[str, str], tuple[str, str]] = {}
    for model in MODELS.values():
        for dataset in DATASETS.values():
            cell = original[
                (original["model"] == model) & (original["dataset"] == dataset)
            ]
            scope = cell[
                cell["method"].isin([METHODS[name] for name in SCOPES])
            ].nsmallest(1, "mean_aopc").iloc[0]["method"]
            baseline = cell[
                cell["method"].isin([METHODS[name] for name in NON_SCOPE_BASELINES])
            ].nsmallest(1, "mean_aopc").iloc[0]["method"]
            selected_methods[(model, dataset)] = (scope, baseline)

    counts: dict[str, dict[str, int]] = {}
    for subset in ("all", "correct_only"):
        block = paired[paired["subset"] == subset]
        selected_rows = []
        for (model, dataset), (scope, baseline) in selected_methods.items():
            selected_rows.append(
                block[
                    (block["model"] == model)
                    & (block["dataset"] == dataset)
                    & (block["scope"] == scope)
                    & (block["baseline"] == baseline)
                ].iloc[0]
            )
        selected = pd.DataFrame(selected_rows)
        significant = selected["wilcoxon_p"] < 0.05
        counts[subset] = {
            "scope_wins": int(
                (significant & (selected["mean_scope_minus_baseline"] < 0)).sum()
            ),
            "baseline_wins": int(
                (significant & (selected["mean_scope_minus_baseline"] > 0)).sum()
            ),
            "total": len(selected),
        }

        display = selected.copy()
        display["difference"] = display["mean_scope_minus_baseline"].map(
            lambda value: f"{value:.4f}"
        )
        display["ci"] = display.apply(
            lambda row: f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]", axis=1
        )
        display["p_value"] = display["wilcoxon_p"].map(lambda value: f"{value:.3g}")
        lines.extend(
            [
                f"## {subset}",
                "",
                markdown_table(
                    display,
                    ["model", "dataset", "scope", "baseline", "difference", "ci", "p_value"],
                    [
                        "Model",
                        "Dataset",
                        "Top-performing Scope",
                        "Top-performing baseline",
                        "Mean paired difference",
                        "95% CI",
                        "Wilcoxon p",
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Summary",
            "",
            f"On all passages, the top-performing Scope significantly beats the "
            f"top-performing baseline in **{counts['all']['scope_wins']}/"
            f"{counts['all']['total']}** model–dataset combinations. On correctly "
            f"predicted passages, it does so in **{counts['correct_only']['scope_wins']}/"
            f"{counts['correct_only']['total']}** combinations. There are "
            f"**{counts['all']['baseline_wins']}** significant baseline wins on all "
            f"passages and **{counts['correct_only']['baseline_wins']}** on the "
            "correct-only subset.",
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
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
                            )
                        )

    summary = pd.DataFrame(summary_rows)
    summary["formatted"] = summary.apply(
        lambda row: f"{row['mean_aopc']:.3f} ± {row['sem_aopc']:.3f}", axis=1
    )
    accuracies = pd.DataFrame(accuracy_rows)
    paired = pd.DataFrame(paired_rows)

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
