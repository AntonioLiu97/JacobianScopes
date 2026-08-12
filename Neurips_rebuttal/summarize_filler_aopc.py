#!/usr/bin/env python3
"""Summarize zeroing versus filler-token AOPC on matched prompt indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FRACTIONS = (0.05, 0.1, 0.2)
METHODS = (
    "random_ablation",
    "IG",
    "gradient_x_input",
    "Semantic",
    "Temperature",
    "Fisher_k_1",
)
METHOD_LABELS = {
    "random_ablation": "Random",
    "IG": "Integrated Gradients",
    "gradient_x_input": "Input × Gradient",
    "Semantic": "Semantic Scope",
    "Temperature": "Temperature Scope",
    "Fisher_k_1": "Fisher Scope",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-short", required=True)
    parser.add_argument("--filler-name", required=True)
    parser.add_argument(
        "--filler-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "filler_aopc_results",
    )
    parser.add_argument(
        "--original-dir", type=Path, default=ROOT / "paper" / "results"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "filler_aopc_summary.md",
    )
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def path_for(
    directory: Path,
    model: str,
    method: str,
    dataset: str,
    fraction: float,
    filler: str | None,
) -> Path:
    suffix = f"_filler_{filler}" if filler else ""
    return directory / (
        f"{model}__{method}_{dataset}_top{fraction}{suffix}_results.json"
    )


def load_records(
    directory: Path,
    model: str,
    method: str,
    dataset: str,
    filler: str | None,
) -> dict[float, dict[int, dict]] | None:
    by_fraction = {}
    for fraction in FRACTIONS:
        path = path_for(directory, model, method, dataset, fraction, filler)
        if not path.exists():
            return None
        payload = json.loads(path.read_text())["results"]
        by_fraction[fraction] = {int(row["index"]): row for row in payload}
    return by_fraction


def aopc(
    records: dict[float, dict[int, dict]], indices: list[int]
) -> np.ndarray:
    return np.asarray(
        [
            np.trapezoid(
                [records[fraction][index]["delta_log_prob"] for fraction in FRACTIONS],
                FRACTIONS,
            )
            for index in indices
        ]
    )


def bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, resamples: int = 10_000
) -> tuple[float, float]:
    means = np.empty(resamples)
    for start in range(0, resamples, 500):
        stop = min(start + 500, resamples)
        sampled = rng.integers(
            0, len(values), size=(stop - start, len(values))
        )
        means[start:stop] = values[sampled].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for dataset in ("lmbd1000", "IWSLT2017DE_EN"):
        for method in METHODS:
            filler = load_records(
                args.filler_dir,
                args.model_short,
                method,
                dataset,
                args.filler_name,
            )
            original = load_records(
                args.original_dir,
                args.model_short,
                method,
                dataset,
                None,
            )
            if filler is None or original is None:
                continue
            indices = sorted(
                set.intersection(
                    *(set(filler[fraction]) for fraction in FRACTIONS),
                    *(set(original[fraction]) for fraction in FRACTIONS),
                )
            )
            if not indices:
                continue
            filler_aopc = aopc(filler, indices)
            original_aopc = aopc(original, indices)
            difference = filler_aopc - original_aopc
            low, high = bootstrap_ci(difference, rng)
            rows.append(
                {
                    "dataset": dataset,
                    "method": METHOD_LABELS[method],
                    "n": len(indices),
                    "zero_aopc": original_aopc.mean(),
                    "filler_aopc": filler_aopc.mean(),
                    "difference": difference.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "kl_at_20pct": np.mean(
                        [
                            filler[0.2][index]["kl_divergence"]
                            for index in indices
                        ]
                    ),
                    "flip_rate_at_20pct": np.mean(
                        [
                            filler[0.2][index]["prediction_flipped"]
                            for index in indices
                        ]
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output.with_suffix(".csv"), index=False)
    lines = [
        f"# {args.model_short} filler-token AOPC summary",
        "",
        "Matched full-dataset prompts available at the time of summarization. "
        "More-negative AOPC is better; negative filler − zero differences favor filler replacement.",
        "",
    ]
    for dataset in ("lmbd1000", "IWSLT2017DE_EN"):
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Method | n | Zero AOPC | Filler AOPC | Filler − zero [95% CI] | KL@20% | Flip@20% |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in frame[frame["dataset"] == dataset].iterrows():
            lines.append(
                f"| {row['method']} | {int(row['n'])} | "
                f"{row['zero_aopc']:.3f} | {row['filler_aopc']:.3f} | "
                f"{row['difference']:+.3f} "
                f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | "
                f"{row['kl_at_20pct']:.3f} | "
                f"{100 * row['flip_rate_at_20pct']:.1f}% |"
            )
        lines.append("")
    args.output.write_text("\n".join(lines))
    print(f"Wrote {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
