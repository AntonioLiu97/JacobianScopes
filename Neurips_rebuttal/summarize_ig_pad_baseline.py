#!/usr/bin/env python3
"""Compare cached zero-baseline IG with a token-baseline rerun."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


MODEL = "Llama-3.2-1B"
FRACTIONS = (0.05, 0.1, 0.2)
DATASETS = {
    "lmbd1000": "LAMBADA",
    "IWSLT2017DE_EN": "IWSLT2017 DE→EN",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=("pad", "bos", "eos"), default="pad")
    parser.add_argument("--original-dir", type=Path, default=root / "paper" / "results")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    if args.baseline_dir is None:
        args.baseline_dir = (
            Path(__file__).resolve().parent / f"ig_{args.baseline}_results"
        )
    return args


def load_aopc(directory: Path, method: str, dataset: str) -> pd.Series:
    records_by_fraction = {}
    expected = None
    for fraction in FRACTIONS:
        path = directory / f"{MODEL}__{method}_{dataset}_top{fraction}_results.json"
        payload = json.loads(path.read_text())["results"]
        records = {int(row["index"]): float(row["delta_log_prob"]) for row in payload}
        if expected is None:
            expected = set(records)
        elif set(records) != expected:
            raise ValueError(f"Indices differ across fractions in {path}")
        records_by_fraction[fraction] = records
    assert expected is not None
    return pd.Series(
        {
            index: float(
                np.trapezoid(
                    [records_by_fraction[fraction][index] for fraction in FRACTIONS],
                    FRACTIONS,
                )
            )
            for index in sorted(expected)
        }
    )


def bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, n_resamples: int
) -> tuple[float, float]:
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, 500):
        stop = min(start + 500, n_resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for dataset, display_name in DATASETS.items():
        original = load_aopc(args.original_dir, "IG", dataset)
        token_baseline = load_aopc(
            args.baseline_dir, f"IG_{args.baseline}", dataset
        )
        common = original.index.intersection(token_baseline.index)
        if len(common) != len(token_baseline):
            raise ValueError(
                f"{args.baseline.upper()} rerun for {dataset} contains unexpected indices"
            )
        original = original.loc[common]
        token_baseline = token_baseline.loc[common]
        difference = (token_baseline - original).to_numpy()
        low, high = bootstrap_ci(difference, rng, args.bootstrap_resamples)
        rows.append(
            {
                "dataset": display_name,
                "n": len(common),
                "zero_mean_aopc": original.mean(),
                "zero_sem": original.std(ddof=1) / math.sqrt(len(original)),
                "token_baseline_mean_aopc": token_baseline.mean(),
                "token_baseline_sem": token_baseline.std(ddof=1)
                / math.sqrt(len(token_baseline)),
                "token_baseline_minus_zero": difference.mean(),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "wilcoxon_p": wilcoxon(difference, method="auto").pvalue,
            }
        )
    frame = pd.DataFrame(rows)
    order = np.argsort(frame["wilcoxon_p"].to_numpy())
    sorted_p = frame["wilcoxon_p"].to_numpy()[order]
    adjusted_sorted = np.minimum(
        1.0,
        np.maximum.accumulate((len(sorted_p) - np.arange(len(sorted_p))) * sorted_p),
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    frame["holm_p"] = adjusted
    frame.to_csv(
        args.baseline_dir / f"ig_{args.baseline}_baseline_comparison.csv",
        index=False,
    )

    lines = [
        f"# LLaMA-3.2 1B IG {args.baseline.upper()}-baseline sensitivity",
        "",
        "AOPC on the same correctly predicted passages; more-negative is better. "
        f"The paired difference is {args.baseline.upper()}-baseline IG minus zero-baseline IG.",
        "",
        f"| Dataset | n | Zero baseline | {args.baseline.upper()}-token baseline | "
        "Paired difference [95% CI] | Holm p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['n'])} | "
            f"{row['zero_mean_aopc']:.3f} ± {row['zero_sem']:.3f} | "
            f"{row['token_baseline_mean_aopc']:.3f} ± {row['token_baseline_sem']:.3f} | "
            f"{row['token_baseline_minus_zero']:.3f} "
            f"[{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}] | "
            f"{row['holm_p']:.3g} |"
        )
    (args.baseline_dir / f"ig_{args.baseline}_baseline_comparison.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(f"Wrote IG baseline comparison to {args.baseline_dir}")


if __name__ == "__main__":
    main()
