#!/usr/bin/env python3
"""Sensitivity check: remove exact duplicate source rows and re-estimate main models."""
from __future__ import annotations

import argparse
from pathlib import Path

from paths import DATA_DIR, OUTPUT_DIR
import pandas as pd

from sensitivity_common import CANONICAL_EXPECTED, fit_main_models, prepare_analysis_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(DATA_DIR / "adzuna_sample.dta"))
    p.add_argument("--scores", default=str(DATA_DIR / "amenity_scores.csv"))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR / "sensitivity"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.scores, index_col=0)

    print("Preparing canonical sample with duplicates retained ...", flush=True)
    original, original_diag = prepare_analysis_data(args.input, scores, drop_exact_duplicates=False)
    print("Estimating canonical models ...", flush=True)
    original_results = fit_main_models(original)
    original_results["Duplicate_rule"] = "Retained"

    print("Preparing sample after removing exact duplicates ...", flush=True)
    dedup, dedup_diag = prepare_analysis_data(args.input, scores, drop_exact_duplicates=True)
    print("Estimating duplicate-exclusion models ...", flush=True)
    dedup_results = fit_main_models(dedup)
    dedup_results["Duplicate_rule"] = "Dropped exact duplicates"

    results = pd.concat([original_results, dedup_results], ignore_index=True)
    results.to_csv(out / "duplicate_sensitivity_results.csv", index=False)

    comparison = original_results[["Specification", "Estimate", "SE", "N"]].merge(
        dedup_results[["Specification", "Estimate", "SE", "N"]],
        on="Specification", suffixes=("_retained", "_dropped"),
    )
    comparison["Estimate_difference"] = comparison["Estimate_dropped"] - comparison["Estimate_retained"]
    comparison["Same_to_3_decimals"] = (
        comparison["Estimate_dropped"].round(3) == comparison["Estimate_retained"].round(3)
    )
    comparison.to_csv(out / "duplicate_sensitivity_comparison.csv", index=False)

    lines = [
        "EXACT-DUPLICATE SENSITIVITY",
        "=" * 60,
        f"Exact duplicate rows detected: {original_diag['exact_duplicate_rows']:,}",
        f"Raw rows: {original_diag['raw_rows']:,}",
        f"Rows after removal: {dedup_diag['rows_after_duplicate_rule']:,}",
        "",
        comparison.to_string(index=False),
        "",
        "Interpretation rule: the check passes if signs remain negative, p-values remain small,",
        "and the preferred coefficient is unchanged to three decimals or changes only trivially.",
    ]
    (out / "duplicate_sensitivity_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nOutputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
