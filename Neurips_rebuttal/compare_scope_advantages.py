#!/usr/bin/env python3
"""Compare best-Scope versus best-baseline AOPC gaps for zero and PAD."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
SCOPES = {"Semantic Scope", "Temperature Scope", "Fisher Scope"}
BASELINES = {"Integrated Gradients", "Input × Gradient"}


def best_method(group: pd.DataFrame, methods: set[str]) -> pd.Series:
    candidates = group[group["method"].isin(methods)].dropna(subset=["mean_aopc"])
    return candidates.loc[candidates["mean_aopc"].idxmin()]


def main() -> None:
    payload = json.loads((HERE / "aopc_reanalysis.json").read_text())
    zero = pd.DataFrame(payload["summary"])
    zero = zero[zero["subset"] == "all"]
    pad = pd.read_csv(HERE / "pad_table1.csv")

    rows = []
    cells = zero[["model", "dataset"]].drop_duplicates()
    for cell in cells.itertuples(index=False):
        zero_cell = zero[
            (zero["model"] == cell.model) & (zero["dataset"] == cell.dataset)
        ]
        pad_cell = pad[
            (pad["model"] == cell.model) & (pad["dataset"] == cell.dataset)
        ]
        if (
            pad_cell[pad_cell["method"].isin(SCOPES)]["mean_aopc"].notna().sum() == 0
            or pad_cell[pad_cell["method"].isin(BASELINES)]["mean_aopc"].notna().sum()
            == 0
        ):
            continue

        zero_scope = best_method(zero_cell, SCOPES)
        zero_baseline = best_method(zero_cell, BASELINES)
        pad_scope = best_method(pad_cell, SCOPES)
        pad_baseline = best_method(pad_cell, BASELINES)
        zero_gap = zero_scope["mean_aopc"] - zero_baseline["mean_aopc"]
        pad_gap = pad_scope["mean_aopc"] - pad_baseline["mean_aopc"]
        rows.append(
            {
                "model": cell.model,
                "dataset": cell.dataset,
                "zero_scope": zero_scope["method"],
                "zero_baseline": zero_baseline["method"],
                "zero_gap": zero_gap,
                "pad_scope": pad_scope["method"],
                "pad_baseline": pad_baseline["method"],
                "pad_gap": pad_gap,
                "gap_change": pad_gap - zero_gap,
            }
        )

    frame = pd.DataFrame(rows).sort_values(["model", "dataset"])
    frame.to_csv(HERE / "scope_advantage_comparison.csv", index=False)

    lines = [
        "# Scope advantage under zero and PAD ablation",
        "",
        "Gap = best Scope mean AOPC − best non-Scope mean AOPC; a more-negative "
        "gap indicates a larger Scope advantage.",
        "",
        "| Model | Dataset | Zero gap | PAD gap | Change |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.dataset} | {row.zero_gap:.3f} | "
            f"{row.pad_gap:.3f} | {row.gap_change:.3f} |"
        )
    lines.append("")
    output = "\n".join(lines)
    (HERE / "scope_advantage_comparison.md").write_text(output)
    print(output)


if __name__ == "__main__":
    main()
