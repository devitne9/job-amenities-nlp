#!/usr/bin/env python3
"""Shared data preparation and model helpers for the final thesis sensitivities."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyfixest as pf

CPI_DEFLATORS = {2017: 105.1 / 101.0, 2018: 105.1 / 103.3, 2019: 1.0}
CAT_COLS = ["education", "seniority", "experience", "type_contract", "time_contract"]
CANONICAL_CONTROLS = (
    "C(education) + C(seniority) + C(experience) + "
    "C(type_contract) + C(time_contract) + "
    "log_ad_length + range_posted + range_width"
)

CANONICAL_EXPECTED = {
    "baseline_beta": -0.187199,
    "firm_fe_beta": -0.097784,
    "baseline_n": 178_173,
    "firm_fe_n": 168_269,
}


def extract_description_body(df: pd.DataFrame) -> pd.Series:
    description = df["description"].astype("string")
    return description.str.split("│").str[1].fillna(description).fillna("").astype(str)


def prepare_analysis_data(
    data_path: str | Path,
    scores: pd.DataFrame,
    *,
    drop_exact_duplicates: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Prepare the canonical estimation sample while preserving original row indices."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    raw = pd.read_stata(data_path)
    raw_rows = len(raw)
    duplicate_mask = raw.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())

    if drop_exact_duplicates:
        raw = raw.loc[~duplicate_mask].copy()
    else:
        raw = raw.copy()

    if not scores.index.is_unique:
        raise ValueError("Score index contains duplicates.")
    if not raw.index.isin(scores.index).all():
        missing = raw.index[~raw.index.isin(scores.index)]
        raise ValueError(f"Scores missing for {len(missing):,} retained rows.")

    score_cols = [c for c in ["amenity_raw", "amenity_score"] if c in scores.columns]
    if "amenity_score" not in score_cols:
        raise ValueError("Scores must contain amenity_score.")
    raw = raw.join(scores[score_cols], how="left")
    if raw["amenity_score"].isna().any():
        raise ValueError("Missing amenity_score after index-aligned join.")

    raw["date_advertisement"] = pd.to_datetime(raw["date_advertisement"], errors="raise")
    raw["year"] = raw["date_advertisement"].dt.year
    raw["year_month"] = raw["date_advertisement"].dt.to_period("M").astype(str)

    raw["salary_real"] = raw["salary"] * raw["year"].map(CPI_DEFLATORS)
    raw["ln_wage"] = np.log(raw["salary_real"].where(raw["salary_real"] > 0))

    negotiable_text = raw["negotiable"].astype(str)
    labels = negotiable_text.dropna().unique().tolist()
    range_label = next((x for x in labels if "range" in x.lower()), None)
    if range_label is None:
        raise ValueError(f"Could not identify salary-range label. Found: {labels}")
    raw["range_posted"] = negotiable_text.eq(range_label).astype(int)

    valid_range = (
        raw["range_posted"].eq(1)
        & raw["max_salary"].notna()
        & raw["min_salary"].notna()
        & raw["salary"].gt(0)
    )
    raw["range_width"] = 0.0
    raw.loc[valid_range, "range_width"] = (
        (raw.loc[valid_range, "max_salary"] - raw.loc[valid_range, "min_salary"]).abs()
        / raw.loc[valid_range, "salary"]
    )

    raw["description_body"] = extract_description_body(raw)
    raw["log_ad_length"] = np.log(raw["description_body"].str.len().clip(lower=1))

    for col in CAT_COLS:
        raw[col] = raw[col].astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})

    raw["location_2d"] = raw["location_2d"].astype(str)
    raw["occ_reg_ym"] = (
        raw["occupation_code_3d"].astype(str)
        + "_" + raw["location_2d"]
        + "_" + raw["year_month"]
    )

    cell_counts = raw.groupby("occ_reg_ym", observed=False).size()
    valid_cells = cell_counts[cell_counts > 1].index
    prepared = raw.loc[raw["occ_reg_ym"].isin(valid_cells)].copy()

    diagnostics = {
        "raw_rows": raw_rows,
        "exact_duplicate_rows": duplicate_count,
        "rows_after_duplicate_rule": len(raw),
        "baseline_prepared_rows": len(prepared),
        "baseline_firms": int(prepared["company_id"].nunique()),
    }
    return prepared, diagnostics


def _tidy_row(model, term: str) -> pd.Series:
    tidy = model.tidy()
    if term in tidy.index:
        return tidy.loc[term]
    if "Coefficient" in tidy.columns:
        match = tidy.loc[tidy["Coefficient"] == term]
        if not match.empty:
            return match.iloc[0]
    raise KeyError(f"Term {term!r} not found. Index: {list(tidy.index)}")


def _model_n(model) -> int:
    value = getattr(model, "_N", None)
    if value is None:
        raise AttributeError("Could not retrieve model observation count (_N).")
    return int(value)


def _r2(model, attr: str) -> float:
    value = getattr(model, attr, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fit_main_models(df: pd.DataFrame, score_col: str = "amenity_score") -> pd.DataFrame:
    """Fit the baseline and preferred company-FE models using canonical controls."""
    required = [
        score_col, "ln_wage", "company_id", "occ_reg_ym",
        "log_ad_length", "range_posted", "range_width", *CAT_COLS,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing model columns: {missing}")

    clean = df.dropna(subset=required).copy()
    formulas = {
        "Baseline market-cell FE": (
            f"{score_col} ~ ln_wage + {CANONICAL_CONTROLS} | occ_reg_ym"
        ),
        "Preferred + company FE": (
            f"{score_col} ~ ln_wage + {CANONICAL_CONTROLS} | occ_reg_ym + company_id"
        ),
    }

    rows: list[dict[str, object]] = []
    for label, formula in formulas.items():
        model = pf.feols(formula, data=clean, vcov={"CRV1": "company_id"})
        row = _tidy_row(model, "ln_wage")
        beta = float(row["Estimate"])
        se = float(row["Std. Error"])
        p = float(row["Pr(>|t|)"])
        rows.append({
            "Specification": label,
            "Score": score_col,
            "Estimate": beta,
            "SE": se,
            "p_value": p,
            "CI_low": beta - 1.96 * se,
            "CI_high": beta + 1.96 * se,
            "N": _model_n(model),
            "R2": _r2(model, "_r2"),
            "Within_R2": _r2(model, "_r2_within"),
        })
    return pd.DataFrame(rows)
