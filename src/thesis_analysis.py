"""
thesis_analysis.py  v4.0
========================
Pipeline for: "Do Lower-Wage Job Postings Use More Amenity Language?"
Davit Melikjanyan, Utrecht University, 2026

REPRODUCIBILITY NOTE
--------------------
This pipeline reproduces results from amenity_scores.csv onward.
Score reconstruction is supplied separately in build_amenity_scores.py.
Archived comparisons show small numerical differences after standardisation;
see docs/reproducibility.md. The archived scores remain authoritative.

VALIDATION NOTE
---------------
The original rating file contained the amenity score and tertile
variables alongside the advertisement text. Therefore, the ratings
may not have been fully blinded and may be subject to model-information
bias. The correlations below describe agreement between the model and
potentially model-informed ratings. They do not constitute independent
blinded validation.

Inputs (private, under data/)
------------------------------------
    adzuna_sample.dta           main Adzuna dataset
    amenity_scores.csv          pre-computed SBERT scores (index-aligned)
    validation_sample_RATED.csv 200-posting rated validation sample

Outputs -> outputs/thesis/
-----------------------------
    00_preflight_report.txt
    01_environment.txt
    02_sample_construction.txt
    03_descriptive_stats.csv
    04_main_results.csv
    05_robustness.csv
    05b_sensitivity_endpoint_miss.csv
    05c_sensitivity_ordinal_controls.csv
    06_skill_heterogeneity.csv
    07_skill_slopes.csv
    08_posting_density.csv
    09_dictionary_dimensions.csv
    10_validation_results.txt
    11_economic_magnitudes.txt
    12_diagnostic_comparison.txt
    RESULTS_SUMMARY.txt
    fig_coefficient.png
    fig_skill.png
    fig_dimension.png
    fig_validation.png
    OUTPUT_MANIFEST.txt
"""

import sys
import os
import re as _re
import math
import hashlib
import warnings
import importlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, norm
from sklearn.metrics import cohen_kappa_score
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import DATA_DIR, OUTPUT_DIR as OUTPUT_ROOT

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
DATA_FILE       = DATA_DIR / "adzuna_sample.dta"
SCORES_FILE     = DATA_DIR / "amenity_scores.csv"
VALIDATION_FILE = DATA_DIR / "validation_sample_RATED.csv"
OUTPUT_DIR      = OUTPUT_ROOT / "thesis"

CPI_DEFLATORS = {2017: 105.1 / 101.0, 2018: 105.1 / 103.3, 2019: 1.0}

# Ordinal numeric maps — used ONLY for the sensitivity robustness check,
# not for the canonical specification.
SENIORITY_MAP     = {"Not mentioned": 0, "High seniority": 1,
                     "Medium seniority": 2, "Low seniority": 3}
EXPERIENCE_MAP    = {"Not mentioned": 0, "Experience required": 1,
                     "No experience needed": 2}
TYPE_CONTRACT_MAP = {"Not mentioned": 0, "Temporary job": 1, "Permanent job": 2}
TIME_CONTRACT_MAP = {"Not mentioned": 0, "Full-time job": 1, "Part-time job": 2}
EDUCATION_MAP     = {
    "None or not mentioned":       0,
    "GCSE (or equivalent)":        1,
    "A-level (or equivalent)":     2,
    "Bachelor's degree":           3,
    "Master's or Doctoral degree": 4,
}

# ── Expected values for diagnostic comparison (latest verified results) ──────
# Source: thesis_final_rerun.py, economic_magnitudes_exact.txt
# NOT hard-coded results — used only to flag discrepancies after running.
EXPECTED = {
    "n_baseline":      178_173,
    "n_firm_fe":       168_269,
    "n_firms":          16_332,
    "beta_baseline":  -0.187199,
    "beta_firm_fe":   -0.097784,
    "se_firm_fe":      0.014551,
    "ci_lo_firm_fe":  -0.126305,
    "ci_hi_firm_fe":  -0.069263,
    "arith_mean_wage": 31_496.0,
}

# Strict diagnostic tolerances from the archived analysis
TOL_COEF = 1e-5
TOL_SE   = 1e-5
TOL_CI   = 1e-5
TOL_N    = 0      # exact sample-size match required
TOL_FIRM = 0      # exact firm-count match required

def stars(p):
    """Thesis significance convention: *** p<0.01  ** p<0.05  * p<0.10"""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

# Palette matching the thesis presentation
C_INK   = "#1E1B3A"
C_AMBER = "#D97706"
C_GREY  = "#9CA3AF"
C_LITE  = "#FCD34D"

# ─────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
preflight_lines = []
manifest_lines  = []

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def log(msg):
    print(msg, flush=True)

def pf_note(msg, warn=False):
    tag = "WARNING" if warn else "OK"
    preflight_lines.append(f"[{tag}] {msg}")
    log(f"  PREFLIGHT {tag}: {msg}")

def abort(msg):
    pf_note(msg, warn=True)
    _write_preflight()
    raise SystemExit(f"\nABORTED: {msg}")

def _write_preflight():
    (OUTPUT_DIR / "00_preflight_report.txt").write_text(
        "\n".join(preflight_lines), encoding="utf-8")

def write(fname, text):
    p = OUTPUT_DIR / fname
    p.write_text(text, encoding="utf-8")
    manifest_lines.append(str(p))
    log(f"  -> Wrote {p}")

def save_csv(fname, df):
    p = OUTPUT_DIR / fname
    df.to_csv(p)
    manifest_lines.append(str(p))
    log(f"  -> Wrote {p}")

def save_fig(fname):
    p = OUTPUT_DIR / fname
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    manifest_lines.append(str(p))
    log(f"  -> Wrote {p}")

def section(title):
    bar = "=" * 68
    log(f"\n{bar}\n  {title}\n{bar}")

def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# ── pyfixest coefficient extraction ──────────────────────────────
def get_tidy_row(model, varname):
    """
    Robustly extract one coefficient row from a pyfixest model.
    Works whether tidy() stores variable names in the index or in
    a 'Coefficient' column (behaviour differs across pyfixest versions).
    Raises KeyError with diagnostic info if the variable is not found.
    """
    tidy = model.tidy()

    # Try index first (common in newer pyfixest)
    if varname in tidy.index:
        return tidy.loc[varname]

    # Try 'Coefficient' column (older pyfixest)
    if "Coefficient" in tidy.columns:
        matches = tidy.loc[tidy["Coefficient"] == varname]
        if not matches.empty:
            return matches.iloc[0]

    raise KeyError(
        f"Coefficient '{varname}' not found in tidy output. "
        f"Index={list(tidy.index)}, columns={list(tidy.columns)}"
    )

# ── Model N extraction ────────────────────────────────────────────
def get_model_n(model):
    """Return the actual estimation N recorded by the fitted model."""
    for attr in ("_N", "N", "nobs"):
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, (int, float)) and val > 0:
                return int(val)
    try:
        return int(model.resid().notna().sum())
    except Exception:
        return None

# ── R² extraction ─────────────────────────────────────────────────
def get_r2(model, kind="r2"):
    """
    Retrieve R² or within-R² from a pyfixest model.
    Prefers private attributes _r2 / _r2_within (pyfixest >= 0.18).
    Falls back to public methods, then summary-string parsing.
    Raises ValueError with pyfixest version info if unavailable.
    """
    preferred = {"r2": "_r2", "within_r2": "_r2_within"}
    attr = preferred.get(kind)
    if attr and hasattr(model, attr):
        val = getattr(model, attr)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            return round(float(val), 6)

    if hasattr(model, kind):
        try:
            val = getattr(model, kind)
            if callable(val): val = val()
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                return round(float(val), 6)
        except Exception:
            pass

    try:
        s = str(model.summary())
        pattern = r"Within R2" if kind == "within_r2" else r"R2\b"
        m = _re.search(pattern + r"[:\s]+([0-9.]+)", s)
        if m:
            return round(float(m.group(1)), 6)
    except Exception:
        pass

    try:
        pf_ver = pf.__version__
    except Exception:
        pf_ver = "unknown"
    raise ValueError(
        f"Could not retrieve '{kind}' from pyfixest model "
        f"(pyfixest version: {pf_ver}). "
        f"Model type: {type(model).__name__}. "
        f"R2-related attributes: {[a for a in dir(model) if 'r2' in a.lower()]}"
    )

# ── Coefficient row builder ───────────────────────────────────────
def coef_row(model, varname, spec_label, firm_fe_flag, formula_str, n_input=None):
    """
    Extract one coefficient row using get_tidy_row().
    Records actual model N (from pyfixest) vs input data N.
    """
    row_s = get_tidy_row(model, varname)

    # Column names differ slightly across pyfixest versions; try common variants
    def _get(series, *keys):
        for k in keys:
            if k in series.index:
                return series[k]
        raise KeyError(f"None of {keys} found in tidy row. Available: {list(series.index)}")

    est  = float(_get(row_s, "Estimate", "estimate"))
    se   = float(_get(row_s, "Std. Error", "std_error", "std error"))
    pval = float(_get(row_s, "Pr(>|t|)", "p_value", "pvalue"))
    ci_lo = est - 1.96 * se
    ci_hi = est + 1.96 * se

    actual_n = get_model_n(model)
    if n_input is not None and actual_n is not None and actual_n < n_input:
        log(f"    Sample reduction: input={n_input:,}  model={actual_n:,}  "
            f"removed by singleton handling={n_input - actual_n:,}")

    try:    r2  = get_r2(model, "r2")
    except ValueError as e:
        pf_note(str(e), warn=True); r2 = None

    try:    wr2 = get_r2(model, "within_r2")
    except ValueError as e:
        pf_note(str(e), warn=True); wr2 = None

    return {
        "Specification": spec_label,
        "Variable":      varname,
        "Estimate":      round(est,   6),
        "SE":            round(se,    6),
        "p_value":       round(pval,  8),
        "CI_low":        round(ci_lo, 6),
        "CI_high":       round(ci_hi, 6),
        "N_input":       n_input,
        "N_model":       actual_n,
        "R2":            r2,
        "Within_R2":     wr2,
        "Firm_FE":       "Yes" if firm_fe_flag else "No",
        "Stars":         stars(pval),
        "Formula":       formula_str,
    }

# ── Complete cleaning helper ──────────────────────────────────────
def clean_model(data, all_vars, label=""):
    """
    Drop rows with NaN in any variable needed for the regression.
    all_vars: explicit list of EVERY variable name the model uses
              (outcome, regressor, controls, FE identifiers).
              Do NOT pass formula strings; parse them before calling.

    Aborts if any variable is absent from the dataframe.
    Reports observation counts before and after dropping.
    Returns (cleaned_df, n_before, n_after).
    """
    # Resolve C(xxx) references to the underlying column name
    resolved = []
    for v in all_vars:
        m = _re.match(r"C\((\w+)\)", v)
        resolved.append(m.group(1) if m else v)

    missing_cols = [v for v in resolved if v not in data.columns]
    if missing_cols:
        abort(f"clean_model({label}): required columns missing from data: {missing_cols}")

    n_before = len(data)
    out = data.dropna(subset=resolved)
    n_after = len(out)
    if n_before != n_after:
        log(f"  clean_model({label}): dropped {n_before - n_after:,} rows with NaN "
            f"({n_before:,} -> {n_after:,})")
    return out, n_before, n_after


# ── Implied slope (delta method) ─────────────────────────────────
def implied_slope(weights, coef_series, vcov_df, var_names):
    """
    Compute a linear combination of coefficients and its SE via the
    delta method. p-values use a normal approximation (stated explicitly:
    appropriate for large samples with clustered SEs).
    """
    w = np.array([weights.get(v, 0.0) for v in var_names])
    c = coef_series.reindex(var_names).fillna(0.0).values

    if hasattr(vcov_df, "values"):
        try:
            V = vcov_df.reindex(index=var_names, columns=var_names).fillna(0.0).values
        except Exception:
            V = vcov_df.values
    else:
        V = np.array(vcov_df)

    est  = float(w @ c)
    var_ = float(w @ V @ w)
    se   = math.sqrt(max(0.0, var_))
    z    = est / se if se > 0 else float("nan")
    pval = float(2 * (1 - norm.cdf(abs(z)))) if not math.isnan(z) else float("nan")
    return est, se, pval


# ─────────────────────────────────────────────────────────────────
# STEP 0 — ENVIRONMENT SNAPSHOT
# ─────────────────────────────────────────────────────────────────
section("ENVIRONMENT SNAPSHOT")

env_lines = [
    f"Run timestamp: {datetime.now().isoformat()}",
    f"Python: {sys.version}",
]
for pkg in ["pandas", "numpy", "scipy", "sklearn", "pyfixest", "matplotlib"]:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "unknown")
    except Exception:
        ver = "NOT INSTALLED"
    env_lines.append(f"{pkg}: {ver}")

log("\n".join(f"  {l}" for l in env_lines))
write("01_environment.txt", "\n".join(env_lines))


# ─────────────────────────────────────────────────────────────────
# STEP 1 — PREFLIGHT: file existence + hashes
# ─────────────────────────────────────────────────────────────────
section("PREFLIGHT CHECKS")

for fpath in [DATA_FILE, SCORES_FILE, VALIDATION_FILE]:
    if not Path(fpath).exists():
        abort(f"{fpath} NOT FOUND — cannot continue.")
    pf_note(f"{fpath}  md5={file_md5(fpath)}")


# ─────────────────────────────────────────────────────────────────
# STEP 2 — LOAD DATA
# ─────────────────────────────────────────────────────────────────
section("LOADING DATA")

df = pd.read_stata(DATA_FILE)
log(f"  Raw rows: {len(df):,}  |  columns: {len(df.columns)}")

required_cols = [
    "date_advertisement", "salary", "max_salary", "min_salary",
    "negotiable", "description", "company_id", "occupation_code_3d",
    "location_2d", "seniority", "experience", "type_contract", "time_contract",
]
missing_req = [c for c in required_cols if c not in df.columns]
if missing_req:
    abort(f"Missing required columns: {missing_req}")
pf_note("All required columns present")

n_dup = df.duplicated().sum()
if n_dup > 0:
    pf_note(f"{n_dup:,} duplicate rows in source data", warn=True)
else:
    pf_note("No duplicate rows in source data")

HAS_EDUCATION = "education" in df.columns
if HAS_EDUCATION:
    edu_dtype = str(df["education"].dtype)
    edu_vals  = sorted(str(v) for v in df["education"].dropna().unique())
    pf_note(f"education: dtype={edu_dtype}  unique={edu_vals[:10]}")
else:
    pf_note("education column not found — excluded from controls", warn=True)

pf_note(f"negotiable unique values: {df['negotiable'].unique().tolist()}")

n_miss_firm = df["company_id"].isna().sum()
if n_miss_firm > 0:
    pf_note(f"{n_miss_firm:,} rows have missing company_id", warn=True)


# ─────────────────────────────────────────────────────────────────
# STEP 3 — DATE FEATURES
# ─────────────────────────────────────────────────────────────────
section("DATE FEATURES")

df["date_advertisement"] = pd.to_datetime(df["date_advertisement"])
df["year"]       = df["date_advertisement"].dt.year
df["quarter"]    = df["date_advertisement"].dt.to_period("Q").astype(str)
df["year_month"] = df["date_advertisement"].dt.to_period("M").astype(str)

years_found = sorted(df["year"].unique().tolist())
unexpected  = [y for y in years_found if y not in [2017, 2018, 2019]]
if unexpected:
    pf_note(f"Unexpected years in data: {unexpected}", warn=True)
else:
    pf_note(f"Years confirmed: {years_found}")


# ─────────────────────────────────────────────────────────────────
# STEP 4 — WAGE: CPI DEFLATION
# ─────────────────────────────────────────────────────────────────
section("WAGE CONSTRUCTION")

unmapped = [y for y in df["year"].unique() if y not in CPI_DEFLATORS]
if unmapped:
    abort(f"Years with no CPI deflator: {unmapped}")

n_nonpos = (df["salary"] <= 0).sum()
n_nullsal = df["salary"].isna().sum()
if n_nonpos > 0 or n_nullsal > 0:
    pf_note(f"salary: {n_nonpos:,} non-positive, {n_nullsal:,} null — "
            "will produce NaN ln_wage and be dropped at regression", warn=True)

df["salary_real"] = df["salary"] * df["year"].map(CPI_DEFLATORS)
df["ln_wage"]     = np.log(df["salary_real"].where(df["salary_real"] > 0))

arith_mean = df["salary_real"].mean()
geo_mean   = np.exp(df["ln_wage"].mean())
log(f"  Arithmetic mean real wage: GBP {arith_mean:,.2f}")
log(f"  Geometric  mean real wage: GBP {geo_mean:,.2f}")
pf_note(f"Arithmetic mean GBP {arith_mean:,.2f} | Geometric GBP {geo_mean:,.2f}")

if abs(arith_mean - EXPECTED["arith_mean_wage"]) > 500:
    pf_note(f"Arithmetic mean GBP {arith_mean:,.0f} differs from expected "
            f"GBP {EXPECTED['arith_mean_wage']:,.0f}", warn=True)


# ─────────────────────────────────────────────────────────────────
# STEP 5 — SALARY-RANGE CODING (CORRECTED)
# ─────────────────────────────────────────────────────────────────
section("SALARY-RANGE CODING")

actual_labels      = df["negotiable"].unique().tolist()
range_label_found  = next((v for v in actual_labels if "range"  in str(v).lower()), None)
unique_label_found = next((v for v in actual_labels if "unique" in str(v).lower()), None)

if range_label_found is None:
    abort(f"Cannot detect salary-range label in 'negotiable'. Found: {actual_labels}")

df["range_posted"] = (df["negotiable"] == range_label_found).astype(int)
pf_note(f"range_posted: {df['range_posted'].mean():.1%} using label '{range_label_found}'")

# Diagnostic: does salary equal midpoint for range postings?
range_rows_valid = df[
    (df["range_posted"] == 1)
    & df["max_salary"].notna()
    & df["min_salary"].notna()
].copy()
range_rows_valid["computed_midpoint"] = (
    (range_rows_valid["max_salary"] + range_rows_valid["min_salary"]) / 2
)
pct_exact = ((range_rows_valid["salary"] - range_rows_valid["computed_midpoint"]).abs() < 0.01).mean()
pf_note(f"salary == (max+min)/2: {pct_exact:.1%} of valid range postings")
if pct_exact < 0.95:
    pf_note("salary does NOT consistently equal the midpoint — "
            "range_width denominator may be wrong", warn=True)

range_mask       = df["range_posted"] == 1
valid_range      = (range_mask & df["max_salary"].notna()
                    & df["min_salary"].notna() & (df["salary"] > 0))
incomplete_range = range_mask & ~valid_range
n_reversed       = int((range_mask & df["max_salary"].notna() & df["min_salary"].notna()
                        & (df["max_salary"] < df["min_salary"])).sum())
n_valid_range    = int(valid_range.sum())
n_incomplete     = int(incomplete_range.sum())

df["range_width"]         = np.nan
df["range_endpoint_miss"] = 0

df.loc[valid_range, "range_width"] = (
    (df.loc[valid_range, "max_salary"] - df.loc[valid_range, "min_salary"]).abs()
    / df.loc[valid_range, "salary"]
)
df.loc[incomplete_range, "range_endpoint_miss"] = 1
df["range_width"] = df["range_width"].fillna(0.0)  # impute 0 for incomplete; matches original

pf_note(f"range_width: valid={n_valid_range:,}  "
        f"reversed={n_reversed:,}  incomplete->zero={n_incomplete:,}")

if unique_label_found:
    df["unique_salary"] = (df["negotiable"] == unique_label_found).astype(int)
else:
    df["unique_salary"] = (df["range_posted"] == 0).astype(int)
    pf_note("unique_salary: using range_posted==0 as fallback", warn=True)


# ─────────────────────────────────────────────────────────────────
# STEP 6 — AD TEXT + LENGTH
# ─────────────────────────────────────────────────────────────────
section("ADVERTISEMENT TEXT AND LENGTH")

df["description_body"] = (
    df["description"].str.split("│").str[1].fillna(df["description"])
)
df["log_ad_length"] = np.log(df["description_body"].str.len().clip(lower=1))


# ─────────────────────────────────────────────────────────────────
# STEP 7 — CONTROLS
#
# CANONICAL SPECIFICATION (matches verified thesis results):
# All five pre-built text-derived indicators are treated as
# categorical using C() syntax so that pyfixest creates proper
# dummy indicators. This is the specification that produces the
# canonical coefficients.
#
# SENSITIVITY (Step 15c): ordinal numeric encoding.
# ─────────────────────────────────────────────────────────────────
section("CONTROLS — CANONICAL CATEGORICAL SPECIFICATION")

# Keep original string/category values for C() encoding.
# Convert pandas Categorical -> str to avoid dtype issues.
cat_cols = ["education", "seniority", "experience", "type_contract", "time_contract"]
for col in cat_cols:
    if col not in df.columns:
        continue
    if hasattr(df[col], "cat") or df[col].dtype.name == "category":
        df[col] = df[col].astype(str)
    else:
        df[col] = df[col].astype(str)
    # Replace "nan" strings with NaN so they are dropped by clean_model
    df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

# Abort if education has unknown categories
if HAS_EDUCATION:
    edu_found   = set(df["education"].dropna().unique())
    edu_unknown = edu_found - set(EDUCATION_MAP.keys())
    if edu_unknown:
        abort(f"education: unrecognised categories {edu_unknown}. "
              f"Expected: {list(EDUCATION_MAP.keys())}")

# Canonical control string — C() creates dummies; pyfixest handles this natively
if HAS_EDUCATION:
    CANONICAL_CONTROLS = (
        "C(education) + C(seniority) + C(experience) + "
        "C(type_contract) + C(time_contract) + "
        "log_ad_length + range_posted + range_width"
    )
else:
    CANONICAL_CONTROLS = (
        "C(seniority) + C(experience) + "
        "C(type_contract) + C(time_contract) + "
        "log_ad_length + range_posted + range_width"
    )

# Canonical controls without range_posted / range_width
# (for unique-salary subsample where both are constant)
if HAS_EDUCATION:
    CANONICAL_CONTROLS_UNIQ = (
        "C(education) + C(seniority) + C(experience) + "
        "C(type_contract) + C(time_contract) + log_ad_length"
    )
else:
    CANONICAL_CONTROLS_UNIQ = (
        "C(seniority) + C(experience) + "
        "C(type_contract) + C(time_contract) + log_ad_length"
    )

# No text-derived controls (all five excluded — all are built from ad text)
NO_TEXT_CONTROLS = "log_ad_length + range_posted + range_width"

log(f"  CANONICAL_CONTROLS: {CANONICAL_CONTROLS}")
log(f"  NO_TEXT_CONTROLS:   {NO_TEXT_CONTROLS}")
pf_note(f"Canonical controls: {CANONICAL_CONTROLS}")

# --- Ordinal numeric controls (sensitivity only) ---
# Map the five categoricals to numeric scales for the sensitivity check.
# This is done in a copy so df remains clean for the canonical models.
def _make_numeric_controls(data):
    """Return a copy of data with the five categoricals mapped to ordinal numbers."""
    d = data.copy()
    for col, mapping in [
        ("seniority",     SENIORITY_MAP),
        ("experience",    EXPERIENCE_MAP),
        ("type_contract", TYPE_CONTRACT_MAP),
        ("time_contract", TIME_CONTRACT_MAP),
    ]:
        if col in d.columns:
            d[col] = d[col].map(mapping).fillna(0)
    if HAS_EDUCATION and "education" in d.columns:
        d["education"] = d["education"].map(EDUCATION_MAP).fillna(0)
    return d

if HAS_EDUCATION:
    ORDINAL_CONTROLS = (
        "education + seniority + experience + "
        "type_contract + time_contract + "
        "log_ad_length + range_posted + range_width"
    )
else:
    ORDINAL_CONTROLS = (
        "seniority + experience + "
        "type_contract + time_contract + "
        "log_ad_length + range_posted + range_width"
    )


# ─────────────────────────────────────────────────────────────────
# STEP 8 — LOAD AND ALIGN AMENITY SCORES
# ─────────────────────────────────────────────────────────────────
section("MERGING AMENITY SCORES — STRICT ALIGNMENT")

scores = pd.read_csv(SCORES_FILE, index_col=0)
log(f"  Scores shape: {scores.shape}  |  columns: {list(scores.columns)}")

for col in ["amenity_score", "amenity_raw"]:
    if col not in scores.columns:
        abort(f"amenity_scores.csv is missing expected column '{col}'")

if len(df) != len(scores):
    abort(f"Row count mismatch: data={len(df):,}, scores={len(scores):,}.")
if not df.index.equals(scores.index):
    abort("Index mismatch between data and scores — re-generate amenity_scores.csv.")
if scores.index.duplicated().sum() > 0:
    abort("amenity_scores.csv has duplicate index values.")

pf_note("Alignment checks passed: row count, index, no duplicates")
df = df.join(scores, how="left")
n_miss_score = df["amenity_score"].isna().sum()
if n_miss_score > 0:
    abort(f"{n_miss_score:,} missing amenity_score after merge.")
pf_note("amenity_score: 100% non-missing after merge")


# ─────────────────────────────────────────────────────────────────
# STEP 9 — ORDINAL AMENITY CATEGORY
# ─────────────────────────────────────────────────────────────────
section("ORDINAL AMENITY CATEGORY")

df["soc_year"] = df["occupation_code_3d"].astype(str) + "_" + df["year"].astype(str)

def safe_qcut(x):
    try:
        return pd.qcut(x, q=3, labels=[1, 2, 3], duplicates="drop")
    except ValueError:
        return pd.Series([2] * len(x), index=x.index)

df["amenity_cat"] = (
    df.groupby("soc_year")["amenity_raw"].transform(safe_qcut).astype(float)
)
log(f"  amenity_cat distribution:\n{df['amenity_cat'].value_counts().sort_index()}")


# ─────────────────────────────────────────────────────────────────
# STEP 10 — GEOGRAPHY VARIANTS
# ─────────────────────────────────────────────────────────────────
section("GEOGRAPHY AND TIME VARIANTS")

df["nuts1"]    = df["location_2d"].str[:3]
df["soc_2dig"] = (df["occupation_code_3d"] // 10).astype(str)  # 3-dig // 10 = 2-dig


# ─────────────────────────────────────────────────────────────────
# STEP 11 — BELOW-MARKET INDICATOR (FIRM-OCCUPATION LEVEL)
# ─────────────────────────────────────────────────────────────────
section("BELOW-MARKET INDICATOR (FIRM-OCCUPATION LEVEL)")

df["occ_reg_ym"] = (
    df["occupation_code_3d"].astype(str)
    + "_" + df["location_2d"]
    + "_" + df["year_month"]
)

n_miss_fe = df["occ_reg_ym"].isna().sum()
if n_miss_fe > 0:
    pf_note(f"{n_miss_fe:,} rows have missing occ_reg_ym", warn=True)

market_wage = (
    df.groupby("occ_reg_ym")["ln_wage"].mean().rename("market_wage_cell")
)
df = df.join(market_wage, on="occ_reg_ym")

firm_occ_wage = (
    df.groupby(["company_id", "occupation_code_3d"])["ln_wage"]
      .mean().rename("firm_avg_wage")
)
df = df.join(firm_occ_wage, on=["company_id", "occupation_code_3d"])

df["below_market_firm_occ"] = (df["firm_avg_wage"] < df["market_wage_cell"]).astype(int)
df["below_market_posting"]  = (df["ln_wage"]       < df["market_wage_cell"]).astype(int)

pf_note(f"below_market_firm_occ: {df['below_market_firm_occ'].mean():.1%} of postings")


# ─────────────────────────────────────────────────────────────────
# STEP 12 — SAMPLE RESTRICTION (DROP SINGLETONS)
# ─────────────────────────────────────────────────────────────────
section("SAMPLE RESTRICTION — DROP SINGLETON CELLS")

cell_counts = df.groupby("occ_reg_ym").size()
valid_cells = cell_counts[cell_counts > 1].index
df_reg      = df[df["occ_reg_ym"].isin(valid_cells)].copy()

n_base  = len(df_reg)
n_firms = df_reg["company_id"].nunique()
n_reversed_reg = int((df_reg["range_posted"].eq(1)
                      & df_reg["max_salary"].notna()
                      & df_reg["min_salary"].notna()
                      & (df_reg["max_salary"] < df_reg["min_salary"])).sum())
n_valid_range_reg = int((df_reg["range_posted"].eq(1)
                         & df_reg["max_salary"].notna()
                         & df_reg["min_salary"].notna()
                         & (df_reg["salary"] > 0)).sum())
n_incomplete_reg = int(df_reg["range_endpoint_miss"].sum())
log(f"  Singletons dropped: {len(df) - n_base:,}")
log(f"  Baseline sample:    {n_base:,}  (expected {EXPECTED['n_baseline']:,})")
log(f"  Unique firms:       {n_firms:,}  (expected {EXPECTED['n_firms']:,})")

if abs(n_base - EXPECTED["n_baseline"]) > 1000:
    pf_note(f"Baseline N={n_base:,} vs expected {EXPECTED['n_baseline']:,}", warn=True)
if abs(n_firms - EXPECTED["n_firms"]) > 500:
    pf_note(f"Firm count {n_firms:,} vs expected {EXPECTED['n_firms']:,}", warn=True)

write("02_sample_construction.txt",
    f"SAMPLE CONSTRUCTION\n{'='*60}\n"
    f"Raw observations:         {len(df):,}\n"
    f"Singleton cells dropped:  {len(df) - n_base:,}\n"
    f"Baseline sample:          {n_base:,}\n"
    f"Unique firms (baseline):  {n_firms:,}\n"
    f"Period:                   {df_reg['year'].min()} to {df_reg['year'].max()}\n"
    f"Salary-range share (baseline):       {df_reg['range_posted'].mean():.4%}\n"
    f"Firm-occ below-market (baseline):    {df_reg['below_market_firm_occ'].mean():.4%}\n"
    f"Posting-level below-mkt (baseline):  {df_reg['below_market_posting'].mean():.4%}\n"
    f"\n"
    f"RANGE DIAGNOSTICS — RAW SAMPLE\n"
    f"Missing range endpoints:  {n_incomplete:,}\n"
    f"Reversed endpoints:       {n_reversed:,}\n"
    f"Valid range postings:     {n_valid_range:,}\n"
    f"\n"
    f"RANGE DIAGNOSTICS — BASELINE SAMPLE\n"
    f"Missing range endpoints:  {n_incomplete_reg:,}\n"
    f"Reversed endpoints:       {n_reversed_reg:,}\n"
    f"Valid range postings:     {n_valid_range_reg:,}\n"
)


# ─────────────────────────────────────────────────────────────────
# STEP 13 — DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────
section("DESCRIPTIVE STATISTICS")

desc_vars = [
    "amenity_score", "salary_real", "ln_wage",
    "log_ad_length", "range_posted", "below_market_firm_occ", "range_width",
]
desc = df_reg[desc_vars].describe().T[["mean", "std", "min", "max", "count"]]
desc["count"] = desc["count"].astype(int)
save_csv("03_descriptive_stats.csv", desc.round(6))


# ─────────────────────────────────────────────────────────────────
# STEP 14 — MAIN REGRESSIONS (CANONICAL CATEGORICAL CONTROLS)
# ─────────────────────────────────────────────────────────────────
section("MAIN REGRESSIONS — CANONICAL CATEGORICAL CONTROLS")

results_main  = []
stored_models = {}
vcov = {"CRV1": "company_id"}

# Variables in every canonical model (for clean_model)
_cat_vars   = (["education"] if HAS_EDUCATION else []) + [
    "seniority", "experience", "type_contract", "time_contract"
]
_base_vars  = _cat_vars + ["log_ad_length", "range_posted", "range_width"]

def _run(label, formula, data, firm_fe, main_var, n_input=None):
    log(f"\n  {label}")
    m = pf.feols(formula, data=data, vcov=vcov)
    log(m.summary())
    row = coef_row(m, main_var, label, firm_fe, formula, n_input=n_input)
    stored_models[label] = m
    return row

# ── Spec 1: baseline (categorical controls, no firm FE)
_s1_vars = ["amenity_score", "ln_wage", "occ_reg_ym"] + _base_vars
d1, _, n1 = clean_model(df_reg, _s1_vars, "Spec1")
f1 = f"amenity_score ~ ln_wage + {CANONICAL_CONTROLS} | occ_reg_ym"
r1 = _run("ln(wage), market-cell FE", f1, d1, False, "ln_wage", n_input=n1)
results_main.append(r1)

# ── Spec 2: headline (categorical controls, + firm FE)
_s2_vars = ["amenity_score", "ln_wage", "occ_reg_ym", "company_id"] + _base_vars
d2, _, n2 = clean_model(df_reg, _s2_vars, "Spec2")
f2 = f"amenity_score ~ ln_wage + {CANONICAL_CONTROLS} | occ_reg_ym + company_id"
r2 = _run("ln(wage), + firm FE  [HEADLINE]", f2, d2, True, "ln_wage", n_input=n2)
results_main.append(r2)

if r2["N_model"] and abs(r2["N_model"] - EXPECTED["n_firm_fe"]) > 1000:
    pf_note(f"Headline N={r2['N_model']:,} vs expected {EXPECTED['n_firm_fe']:,}", warn=True)

# ── Spec 3: below-market, no firm FE
_s3_vars = ["amenity_score", "below_market_firm_occ", "occ_reg_ym"] + _base_vars
d3, _, n3 = clean_model(df_reg, _s3_vars, "Spec3")
f3 = f"amenity_score ~ below_market_firm_occ + {CANONICAL_CONTROLS} | occ_reg_ym"
results_main.append(_run("Below-market (firm-occ), market-cell FE",
                         f3, d3, False, "below_market_firm_occ", n_input=n3))

# ── Spec 4: below-market + firm FE
_s4_vars = ["amenity_score", "below_market_firm_occ",
             "occ_reg_ym", "company_id"] + _base_vars
d4, _, n4 = clean_model(df_reg, _s4_vars, "Spec4")
f4 = (f"amenity_score ~ below_market_firm_occ + {CANONICAL_CONTROLS} "
      f"| occ_reg_ym + company_id")
results_main.append(_run("Below-market (firm-occ), + firm FE",
                         f4, d4, True, "below_market_firm_occ", n_input=n4))

# ── Spec 5: ordinal 1-3 outcome (note: different DV — excluded from coef plot)
_s5_vars = ["amenity_cat", "ln_wage", "occ_reg_ym"] + _base_vars
d5, _, n5 = clean_model(df_reg.dropna(subset=["amenity_cat"]),
                        _s5_vars, "Spec5")
f5 = f"amenity_cat ~ ln_wage + {CANONICAL_CONTROLS} | occ_reg_ym"
results_main.append(_run("Ordinal 1-3 outcome, market-cell FE",
                         f5, d5, False, "ln_wage", n_input=n5))

df_main = pd.DataFrame(results_main)
save_csv("04_main_results.csv", df_main)


# ─────────────────────────────────────────────────────────────────
# STEP 15 — ROBUSTNESS
# ─────────────────────────────────────────────────────────────────
section("ROBUSTNESS CHECKS")

results_rob = []

# Rob A: coarser geography
df_reg["occ_nuts1_ym"] = (df_reg["occupation_code_3d"].astype(str)
                           + "_" + df_reg["nuts1"] + "_" + df_reg["year_month"])
_vA = ["amenity_score", "ln_wage", "occ_nuts1_ym"] + _base_vars
dA, _, nA = clean_model(df_reg, _vA, "RobA")
results_rob.append(_run("NUTS-1 region x occupation x month FE",
    f"amenity_score ~ ln_wage + {CANONICAL_CONTROLS} | occ_nuts1_ym",
    dA, False, "ln_wage", n_input=nA))

# Rob B: coarser time
df_reg["occ_reg_q"] = (df_reg["occupation_code_3d"].astype(str)
                        + "_" + df_reg["location_2d"] + "_" + df_reg["quarter"])
_vB = ["amenity_score", "ln_wage", "occ_reg_q"] + _base_vars
dB, _, nB = clean_model(df_reg, _vB, "RobB")
results_rob.append(_run("Occupation x region x quarter FE",
    f"amenity_score ~ ln_wage + {CANONICAL_CONTROLS} | occ_reg_q",
    dB, False, "ln_wage", n_input=nB))

# Rob C: coarser occupation
df_reg["soc2_reg_ym"] = (df_reg["soc_2dig"]
                          + "_" + df_reg["location_2d"] + "_" + df_reg["year_month"])
_vC = ["amenity_score", "ln_wage", "soc2_reg_ym"] + _base_vars
dC, _, nC = clean_model(df_reg, _vC, "RobC")
results_rob.append(_run("2-digit SOC x region x month FE",
    f"amenity_score ~ ln_wage + {CANONICAL_CONTROLS} | soc2_reg_ym",
    dC, False, "ln_wage", n_input=nC))

# Rob D: unique salary subsample (range_posted / range_width are constant here)
_uniq_base = (["education"] if HAS_EDUCATION else []) + [
    "seniority", "experience", "type_contract", "time_contract", "log_ad_length"
]
_vD = ["amenity_score", "ln_wage", "occ_reg_ym"] + _uniq_base
dD, _, nD = clean_model(df_reg[df_reg["unique_salary"] == 1].copy(), _vD, "RobD")
results_rob.append(_run("Unique-salary subsample",
    f"amenity_score ~ ln_wage + {CANONICAL_CONTROLS_UNIQ} | occ_reg_ym",
    dD, False, "ln_wage", n_input=nD))

# Rob E: firm FE, all text-derived controls excluded
_vE = ["amenity_score", "ln_wage", "occ_reg_ym", "company_id",
       "log_ad_length", "range_posted", "range_width"]
dE, _, nE = clean_model(df_reg, _vE, "RobE")
results_rob.append(_run("Firm FE, excluding ALL text-derived controls",
    f"amenity_score ~ ln_wage + {NO_TEXT_CONTROLS} | occ_reg_ym + company_id",
    dE, True, "ln_wage", n_input=nE))

# Rob F: posting-level below-market, firm FE (robustness only)
_vF = ["amenity_score", "below_market_posting",
       "occ_reg_ym", "company_id"] + _base_vars
dF, _, nF = clean_model(df_reg, _vF, "RobF")
results_rob.append(_run("Posting-level below-market, + firm FE",
    f"amenity_score ~ below_market_posting + {CANONICAL_CONTROLS} "
    f"| occ_reg_ym + company_id",
    dF, True, "below_market_posting", n_input=nF))

df_rob = pd.DataFrame(results_rob)
save_csv("05_robustness.csv", df_rob)


# ─────────────────────────────────────────────────────────────────
# STEP 15b — SENSITIVITY: range_endpoint_miss CONTROL
#   Runs in both baseline and firm-FE form
# ─────────────────────────────────────────────────────────────────
section("SENSITIVITY: range_endpoint_miss CONTROL")

controls_with_miss = CANONICAL_CONTROLS + " + range_endpoint_miss"
_v_em = ["amenity_score", "ln_wage", "occ_reg_ym",
          "range_endpoint_miss"] + _base_vars

# Baseline form
d_em1, _, nem1 = clean_model(df_reg, _v_em, "Sens_endp_miss_base")
m_em1 = pf.feols(
    f"amenity_score ~ ln_wage + {controls_with_miss} | occ_reg_ym",
    data=d_em1, vcov=vcov)
log(m_em1.summary())
em1_row = coef_row(m_em1, "ln_wage",
    "Sensitivity: +endpoint_miss, no firm FE", False,
    f"amenity_score ~ ln_wage + {controls_with_miss} | occ_reg_ym", n_input=nem1)

# Firm-FE form
_v_em2 = _v_em + ["company_id"]
d_em2, _, nem2 = clean_model(df_reg, _v_em2, "Sens_endp_miss_firmFE")
m_em2 = pf.feols(
    f"amenity_score ~ ln_wage + {controls_with_miss} | occ_reg_ym + company_id",
    data=d_em2, vcov=vcov)
log(m_em2.summary())
em2_row = coef_row(m_em2, "ln_wage",
    "Sensitivity: +endpoint_miss, + firm FE", True,
    f"amenity_score ~ ln_wage + {controls_with_miss} | occ_reg_ym + company_id",
    n_input=nem2)

bl_est  = r1["Estimate"]
hl_est  = r2["Estimate"]
for em_row, ref_est, ref_label in [(em1_row, bl_est, "baseline"),
                                    (em2_row, hl_est, "headline firm-FE")]:
    d = abs(em_row["Estimate"] - ref_est)
    pf_note(
        f"endpoint_miss sensitivity ({ref_label}): "
        f"beta={em_row['Estimate']:.4f} vs {ref_label} {ref_est:.4f}  diff={d:.4f}"
        + (" — MATERIAL" if d > 0.005 else " — negligible"),
        warn=(d > 0.005)
    )

save_csv("05b_sensitivity_endpoint_miss.csv", pd.DataFrame([em1_row, em2_row]))


# ─────────────────────────────────────────────────────────────────
# STEP 15c — SENSITIVITY: ORDINAL NUMERIC CONTROLS
#   Runs in both baseline and firm-FE form
# ─────────────────────────────────────────────────────────────────
section("SENSITIVITY: ORDINAL NUMERIC CONTROLS")

df_reg_ord = _make_numeric_controls(df_reg)
_v_ord = ["amenity_score", "ln_wage", "occ_reg_ym"] + [
    c.replace("C(", "").replace(")", "") for c in _cat_vars
] + ["log_ad_length", "range_posted", "range_width"]

# Baseline form
d_ord1, _, nord1 = clean_model(df_reg_ord, _v_ord, "Sens_ord_base")
m_ord1 = pf.feols(
    f"amenity_score ~ ln_wage + {ORDINAL_CONTROLS} | occ_reg_ym",
    data=d_ord1, vcov=vcov)
log(m_ord1.summary())
ord1_row = coef_row(m_ord1, "ln_wage",
    "Sensitivity: ordinal controls, no firm FE", False,
    f"amenity_score ~ ln_wage + {ORDINAL_CONTROLS} | occ_reg_ym", n_input=nord1)

# Firm-FE form
_v_ord2 = _v_ord + ["company_id"]
d_ord2, _, nord2 = clean_model(df_reg_ord, _v_ord2, "Sens_ord_firmFE")
m_ord2 = pf.feols(
    f"amenity_score ~ ln_wage + {ORDINAL_CONTROLS} | occ_reg_ym + company_id",
    data=d_ord2, vcov=vcov)
log(m_ord2.summary())
ord2_row = coef_row(m_ord2, "ln_wage",
    "Sensitivity: ordinal controls, + firm FE", True,
    f"amenity_score ~ ln_wage + {ORDINAL_CONTROLS} | occ_reg_ym + company_id",
    n_input=nord2)

for ord_row, ref_est, ref_label in [(ord1_row, bl_est, "baseline"),
                                     (ord2_row, hl_est, "headline firm-FE")]:
    d = abs(ord_row["Estimate"] - ref_est)
    pf_note(
        f"Ordinal-controls sensitivity ({ref_label}): "
        f"beta={ord_row['Estimate']:.4f} vs {ref_label} {ref_est:.4f}  diff={d:.4f}"
        + (" — MATERIAL" if d > 0.005 else " — negligible"),
        warn=(d > 0.005)
    )

save_csv("05c_sensitivity_ordinal_controls.csv", pd.DataFrame([ord1_row, ord2_row]))


# ─────────────────────────────────────────────────────────────────
# STEP 16 — SKILL HETEROGENEITY
# ─────────────────────────────────────────────────────────────────
section("SKILL HETEROGENEITY")

df_reg["soc_1dig"]   = (df_reg["occupation_code_3d"] // 100).astype(int)
df_reg["low_skill"]  = df_reg["soc_1dig"].isin([8, 9]).astype(int)
df_reg["high_skill"] = df_reg["soc_1dig"].isin([1, 2]).astype(int)

_v_skill = ["amenity_score", "ln_wage", "occ_reg_ym",
             "low_skill", "high_skill"] + _base_vars
d_skill, _, n_skill = clean_model(df_reg, _v_skill, "Skill")
f_skill = (f"amenity_score ~ ln_wage + ln_wage:low_skill + ln_wage:high_skill "
           f"+ {CANONICAL_CONTROLS} | occ_reg_ym")
m_skill = pf.feols(f_skill, data=d_skill, vcov=vcov)
stored_models["skill"] = m_skill
log(m_skill.summary())

# Extract interaction rows using get_tidy_row
rows_int = []
for var in ["ln_wage", "ln_wage:low_skill", "ln_wage:high_skill"]:
    try:
        row_s = get_tidy_row(m_skill, var)
        def _g(s, *keys):
            for k in keys:
                if k in s.index: return s[k]
            return float("nan")
        est_  = float(_g(row_s, "Estimate", "estimate"))
        se_   = float(_g(row_s, "Std. Error", "std_error", "std error"))
        pval_ = float(_g(row_s, "Pr(>|t|)", "p_value"))
        rows_int.append({
            "Term":    var,
            "Estimate": round(est_,  6),
            "SE":       round(se_,   6),
            "p_value":  round(pval_, 8),
            "Stars":    stars(pval_),
            "N":        get_model_n(m_skill),
        })
    except KeyError as e:
        pf_note(f"Skill model: {e}", warn=True)

df_skill_int = pd.DataFrame(rows_int)
save_csv("06_skill_heterogeneity.csv", df_skill_int)

# Implied group slopes from full covariance matrix
coef_s = m_skill.coef()
vcov_s = m_skill._vcov

if hasattr(vcov_s, "index"):
    var_names_s = list(vcov_s.index)
elif hasattr(coef_s, "index"):
    var_names_s = list(coef_s.index)
else:
    abort("Cannot determine variable ordering from skill model covariance matrix.")

rows_slopes = []
for label, w_dict in [
    ("Medium skill (SOC 3-7) — reference", {"ln_wage": 1.0}),
    ("Low skill (SOC 8-9)",                {"ln_wage": 1.0, "ln_wage:low_skill":  1.0}),
    ("High skill (SOC 1-2)",               {"ln_wage": 1.0, "ln_wage:high_skill": 1.0}),
]:
    try:
        est_, se_, pval_ = implied_slope(w_dict, coef_s, vcov_s, var_names_s)
    except Exception as e:
        pf_note(f"Slope compute failed for {label}: {e}", warn=True)
        est_, se_, pval_ = float("nan"), float("nan"), float("nan")
    rows_slopes.append({
        "Group":    label,
        "Estimate": round(est_,  6),
        "SE":       round(se_,   6),
        "p_value":  round(pval_, 8),
        "Stars":    stars(pval_) if not math.isnan(pval_) else "",
        "Note":     "delta-method SE; normal-approx. p-value",
    })

df_slopes = pd.DataFrame(rows_slopes)
log(f"\nImplied skill slopes:\n{df_slopes.to_string(index=False)}")
save_csv("07_skill_slopes.csv", df_slopes)


# ─────────────────────────────────────────────────────────────────
# STEP 17 — REGIONAL POSTING DENSITY
# ─────────────────────────────────────────────────────────────────
section("REGIONAL POSTING DENSITY (not labour-market tightness)")

# Canonical: obs-weighted median (~194) — matches thesis and slides.
# Sensitivity: unweighted cell median (~91).
postings_per_cell = (
    df_reg.groupby(["location_2d", "year_month"])
          .size()
          .reset_index(name="postings_in_cell")
)
df_reg = df_reg.merge(postings_per_cell, on=["location_2d", "year_month"], how="left")

median_obs_wt   = df_reg["postings_in_cell"].median()           # obs-weighted ~ 194
median_unweight = postings_per_cell["postings_in_cell"].median() # cell median ~ 91

log(f"  Obs-weighted median (canonical):  {median_obs_wt:.0f}")
log(f"  Unweighted cell median (sensit.): {median_unweight:.0f}")
pf_note(f"Posting density: obs-weighted canonical={median_obs_wt:.0f}, "
        f"unweighted sensitivity={median_unweight:.0f}")

if abs(median_obs_wt - 194) > 30:
    pf_note(f"Obs-weighted median {median_obs_wt:.0f} differs from expected ~194", warn=True)

df_reg["high_posting_density"] = (df_reg["postings_in_cell"] >= median_obs_wt ).astype(int)
df_reg["high_density_unweight"] = (df_reg["postings_in_cell"] >= median_unweight).astype(int)

_v_dens = ["amenity_score", "ln_wage", "occ_reg_ym",
            "high_posting_density", "high_density_unweight"] + _base_vars
d_dens, _, n_dens = clean_model(df_reg, _v_dens, "Density")

rows_dens = []
for label_d, int_var, cutoff in [
    ("Canonical (obs-weighted ~194)",    "ln_wage:high_posting_density",  median_obs_wt),
    ("Sensitivity (unweighted ~91)",     "ln_wage:high_density_unweight", median_unweight),
]:
    f_d = (f"amenity_score ~ ln_wage + ln_wage:{int_var.split(':')[1]} "
           f"+ {CANONICAL_CONTROLS} | occ_reg_ym")
    m_d = pf.feols(f_d, data=d_dens, vcov=vcov)
    stored_models[f"density_{label_d}"] = m_d
    log(m_d.summary())
    tidy_d = m_d.tidy()
    for var in ["ln_wage", int_var]:
        try:
            row_d = get_tidy_row(m_d, var)
            def _gd(s, *keys):
                for k in keys:
                    if k in s.index: return s[k]
                return float("nan")
            est_d  = float(_gd(row_d, "Estimate", "estimate"))
            se_d   = float(_gd(row_d, "Std. Error", "std_error", "std error"))
            pval_d = float(_gd(row_d, "Pr(>|t|)", "p_value"))
            rows_dens.append({
                "Analysis":      label_d,
                "Variable":      var,
                "Estimate":      round(est_d,  6),
                "SE":            round(se_d,   6),
                "p_value":       round(pval_d, 8),
                "Stars":         stars(pval_d),
                "N_model":       get_model_n(m_d),
                "Median_cutoff": cutoff,
            })
        except KeyError as e:
            pf_note(f"Density model ({label_d}) {var}: {e}", warn=True)

save_csv("08_posting_density.csv", pd.DataFrame(rows_dens))


# ─────────────────────────────────────────────────────────────────
# STEP 18 — DICTIONARY DIMENSION DECOMPOSITION
# ─────────────────────────────────────────────────────────────────
section("DICTIONARY-BASED DIMENSION DECOMPOSITION (supplementary)")

# KEYWORD PROVENANCE: retained from the archived thesis analysis.
# These substring dictionaries are separate from the SBERT reference sentences.
DIMENSION_KEYWORDS = {
    "Flexibility": [
        "flexible", "flexibility", "remote work", "work from home", "hybrid",
        "work-life balance", "work life balance", "flexi-time", "flexitime",
    ],
    "Culture": [
        "supportive", "inclusive", "collaborative", "friendly", "diverse",
        "team culture", "welcoming", "positive culture",
    ],
    "Meaning": [
        "meaningful", "rewarding", "make a difference", "impact", "purpose",
        "mission-driven", "social impact",
    ],
    "Development": [
        "career development", "professional development", "training opportunities",
        "mentoring", "mentorship", "growth opportunities", "progression",
        "career progression", "learning and development",
    ],
    "Perks": [
        "pension", "healthcare", "health insurance", "gym membership",
        "free lunch", "wellness", "bonus scheme", "generous benefits",
        "private medical", "life insurance",
    ],
}

text_lower = df_reg["description_body"].str.lower().fillna("")
rows_dim   = []

for dim, kws in DIMENSION_KEYWORDS.items():
    col  = f"dict_{dim.lower()}"
    colz = col + "_z"
    df_reg[col] = sum(
        text_lower.str.contains(kw, regex=False).astype(int) for kw in kws
    )
    df_reg[colz] = df_reg.groupby("soc_year")[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0
    )
    _v_dim = [colz, "ln_wage", "occ_reg_ym"] + _base_vars
    d_dim, _, n_dim = clean_model(
        df_reg.dropna(subset=[colz]), _v_dim, f"Dim_{dim}")
    f_dim = f"{colz} ~ ln_wage + {CANONICAL_CONTROLS} | occ_reg_ym"
    m_dim = pf.feols(f_dim, data=d_dim, vcov=vcov)
    stored_models[f"dim_{dim}"] = m_dim
    row_d = coef_row(m_dim, "ln_wage", dim, False, f_dim, n_input=n_dim)
    log(f"  {dim}: beta={row_d['Estimate']:.4f}  SE={row_d['SE']:.4f}  {row_d['Stars']}")
    rows_dim.append(row_d)

df_dim = pd.DataFrame(rows_dim)
save_csv("09_dictionary_dimensions.csv", df_dim)


# ─────────────────────────────────────────────────────────────────
# STEP 19 — VALIDATION
# ─────────────────────────────────────────────────────────────────
section("VALIDATION ANALYSIS")

ratings = pd.read_csv(VALIDATION_FILE, index_col=0)
log(f"  Validation shape: {ratings.shape}  |  columns: {list(ratings.columns)}")

for col in ["manual_rating", "amenity_raw", "tertile"]:
    if col not in ratings.columns:
        abort(f"Validation file missing column '{col}'")

r_p, p_p = pearsonr(ratings["manual_rating"], ratings["amenity_raw"])
r_s, p_s = spearmanr(ratings["manual_rating"], ratings["amenity_raw"])

tertile_map = {"low": 1, "medium": 2, "high": 3}
ratings["sbert_tertile"] = ratings["tertile"].map(tertile_map)
kappa_uw = cohen_kappa_score(ratings["manual_rating"], ratings["sbert_tertile"])
kappa_wt = cohen_kappa_score(
    ratings["manual_rating"], ratings["sbert_tertile"], weights="linear")

def fmt_p_value(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.4f}"

p_p_fmt = fmt_p_value(p_p)
p_s_fmt = fmt_p_value(p_s)

val_text = (
    f"VALIDATION RESULTS\n{'='*60}\n\n"
    f"CAVEAT — POSSIBLE MODEL-INFORMATION BIAS\n"
    f"The original rating file contained the amenity score and tertile\n"
    f"variables alongside the advertisement text. Therefore, the ratings\n"
    f"may not have been fully blinded and may be subject to model-information\n"
    f"bias. These statistics describe agreement between the model and\n"
    f"potentially model-informed ratings, not independent validation.\n\n"
    f"STATISTICS (N={len(ratings)}, stratified sample)\n"
    f"Pearson r:               {r_p:.3f}  ({p_p_fmt})\n"
    f"Spearman r:              {r_s:.3f}  ({p_s_fmt})\n"
    f"Kappa (unweighted):      {kappa_uw:.3f}\n"
    f"Kappa (linear-weighted): {kappa_wt:.3f}\n\n"
    f"MEAN SBERT SCORE BY MANUAL RATING CATEGORY\n"
    + ratings.groupby("manual_rating")["amenity_raw"]
              .agg(["mean", "std", "count"]).round(3).to_string()
    + "\n\nThe validation sample is stratified, not random. Correlations\n"
    + "describe this stratified sample and may not generalise to a random\n"
    + "population sample.\n"
)
write("10_validation_results.txt", val_text)


# ─────────────────────────────────────────────────────────────────
# STEP 20 — ECONOMIC MAGNITUDES
# ─────────────────────────────────────────────────────────────────
section("ECONOMIC MAGNITUDES")

hl   = df_main[df_main["Specification"].str.contains("HEADLINE")].iloc[0]
beta = hl["Estimate"]
se_  = hl["SE"]
ci_l = hl["CI_low"]
ci_h = hl["CI_high"]
ln_090 = math.log(0.90)
ln_110 = math.log(1.10)

write("11_economic_magnitudes.txt",
    f"ECONOMIC MAGNITUDES\n{'='*60}\n"
    f"Headline (firm-FE) estimate: {beta:.6f}  (SE={se_:.6f})\n"
    f"95% CI: [{ci_l:.6f}, {ci_h:.6f}]\n"
    f"Model N: {hl['N_model']}\n\n"
    f"Exact log calculations:\n"
    f"  10% wage DECREASE  ln(0.90) = {ln_090:.6f}:\n"
    f"    Amenity change = {beta * ln_090:.6f} SD "
    f"= {beta * ln_090 * 100:.2f}% of one SD  (more amenity language)\n\n"
    f"  10% wage INCREASE  ln(1.10) = {ln_110:.6f}:\n"
    f"    Amenity change = {beta * ln_110:.6f} SD "
    f"= {beta * ln_110 * 100:.2f}% of one SD  (less amenity language)\n\n"
    f"Wage means (regression sample):\n"
    f"  Arithmetic (HEADLINE):  GBP {df_reg['salary_real'].mean():,.2f}\n"
    f"  Geometric (exp of log): GBP {math.exp(df_reg['ln_wage'].mean()):,.2f}\n"
    f"  Report ARITHMETIC as the headline figure.\n"
)


# ─────────────────────────────────────────────────────────────────
# STEP 21 — DIAGNOSTIC COMPARISON (strict tolerances)
# ─────────────────────────────────────────────────────────────────
section("DIAGNOSTIC COMPARISON — STRICT TOLERANCES")

hl_row = df_main[df_main["Specification"].str.contains("HEADLINE")].iloc[0]
bl_row = df_main.iloc[0]

diag_checks = [
    ("Baseline beta",    bl_row["Estimate"],            EXPECTED["beta_baseline"],  TOL_COEF, "coef"),
    ("Firm-FE beta",     hl_row["Estimate"],            EXPECTED["beta_firm_fe"],   TOL_COEF, "coef"),
    ("Firm-FE SE",       hl_row["SE"],                  EXPECTED["se_firm_fe"],     TOL_SE,   "se"),
    ("Firm-FE CI low",   hl_row["CI_low"],              EXPECTED["ci_lo_firm_fe"],  TOL_CI,   "ci"),
    ("Firm-FE CI high",  hl_row["CI_high"],             EXPECTED["ci_hi_firm_fe"],  TOL_CI,   "ci"),
    ("Baseline N",       n_base,                        EXPECTED["n_baseline"],     TOL_N,    "n"),
    ("Firm-FE N",        hl_row["N_model"] or 0,        EXPECTED["n_firm_fe"],      TOL_N,    "n"),
    ("Firms",            n_firms,                       EXPECTED["n_firms"],        TOL_FIRM, "n"),
    ("Arith mean wage",  df_reg["salary_real"].mean(),  EXPECTED["arith_mean_wage"], 200,     "wage"),
]

diag_lines = [f"DIAGNOSTIC COMPARISON (strict tolerances)\n{'='*60}",
              f"  coef tol={TOL_COEF}  SE tol={TOL_SE}  CI tol={TOL_CI}  "
              f"N tol={TOL_N}  firm tol={TOL_FIRM}\n"]
all_pass = True
for name, actual, expected, tol, kind in diag_checks:
    actual_v = actual if actual is not None else float("nan")
    diff = abs(actual_v - expected)
    ok   = diff <= tol
    tag  = "OK     " if ok else "DIFFERS"
    if not ok:
        all_pass = False
    diag_lines.append(
        f"  [{tag}] {name:<22}  actual={actual_v!r:<14}  "
        f"expected={expected!r:<14}  diff={diff:.2e}  tol={tol:.0e}"
    )

diag_lines.append("")
if all_pass:
    diag_lines.append("REPLICATION PASSED")
else:
    diag_lines.append("REPLICATION DID NOT MATCH — DO NOT UPDATE THESIS OR SLIDES")

diag_text = "\n".join(diag_lines)
log(f"\n{diag_text}")
write("12_diagnostic_comparison.txt", diag_text)
if not all_pass:
    pf_note("Replication did not match — see 12_diagnostic_comparison.txt", warn=True)


# ─────────────────────────────────────────────────────────────────
# STEP 22 — FIGURES (from stored result objects, not re-runs)
# ─────────────────────────────────────────────────────────────────
section("GENERATING FIGURES FROM STORED RESULTS")

# ── Fig 1: coefficient plot ───────────────────────────────────────
# Continuous ln(wage) outcome only; ordinal excluded (different DV scale).
coef_rows_fig = [
    r for r in results_main + results_rob
    if r["Variable"] == "ln_wage"
    and "Ordinal" not in r["Specification"]
]

fig, ax = plt.subplots(figsize=(9.0, max(4.0, 0.62 * len(coef_rows_fig))))
y = np.arange(len(coef_rows_fig))[::-1]
for i, r in enumerate(coef_rows_fig):
    is_hl = "HEADLINE" in r["Specification"]
    c = C_AMBER if is_hl else C_INK
    if is_hl:
        ax.axhspan(y[i] - 0.44, y[i] + 0.44, color=C_LITE, alpha=0.3, zorder=0)
    ax.plot([r["CI_low"], r["CI_high"]], [y[i], y[i]], color=c, lw=1.6, zorder=2)
    ax.plot(r["Estimate"], y[i], "o", color=c, ms=8, zorder=3, mec=C_INK, mew=0.6)
    ax.text(r["CI_high"] + 0.007, y[i], f"{r['Estimate']:.3f}",
            va="center", fontsize=9.5, color=c,
            fontweight="bold" if is_hl else "normal")

labels = [r["Specification"].replace(" [HEADLINE]", "").replace(", market-cell FE", "")
          for r in coef_rows_fig]
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.axvline(0, color=C_GREY, lw=1, ls="--")
ax.set_xlabel("Coefficient on ln(wage)  (SD of amenity score, continuous outcome)",
              fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig("fig_coefficient.png")

# ── Fig 2: skill slopes — signed, with 95% CI ────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.0))
x = np.arange(len(df_slopes))
slope_labels = df_slopes["Group"].str.replace(" — reference", "").tolist()
slope_ests   = df_slopes["Estimate"].tolist()
slope_ses    = df_slopes["SE"].tolist()
slope_ci_h   = [1.96 * s for s in slope_ses]
slope_colors = [C_INK if "Medium" in l else C_AMBER for l in slope_labels]

ax.bar(x, slope_ests, color=slope_colors, edgecolor=C_INK, lw=0.8, width=0.55,
       yerr=slope_ci_h, capsize=5, error_kw={"ecolor": C_GREY, "elinewidth": 1.2})
ax.axhline(0, color=C_GREY, lw=1, ls="--")
skill_label_y = []
for i, (est_, ci_h_) in enumerate(zip(slope_ests, slope_ci_h)):
    y_pos = est_ - ci_h_ - 0.020 if est_ < 0 else est_ + ci_h_ + 0.008
    skill_label_y.append(y_pos)
    ax.text(i, y_pos, f"{est_:.3f}", ha="center", fontsize=11, fontweight="bold")
skill_y_min = min([0] + [e - h for e, h in zip(slope_ests, slope_ci_h)] + skill_label_y)
skill_y_max = max([0] + [e + h for e, h in zip(slope_ests, slope_ci_h)] + skill_label_y)
skill_y_pad = max(0.03, (skill_y_max - skill_y_min) * 0.12)
ax.set_ylim(skill_y_min - skill_y_pad, skill_y_max + skill_y_pad)
ax.set_xticks(x)
ax.set_xticklabels(slope_labels, fontsize=10.5)
ax.set_ylabel("Log-wage slope β  (SD of amenity score per log-wage point)", fontsize=10.5)
ax.set_title("Implied skill-group slopes with 95% CI "
             "(delta method; normal approx. p-values)", fontsize=9.5, color=C_GREY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig("fig_skill.png")

# ── Fig 3: dictionary dimension decomposition — signed, with 95% CI ─
dim_ests  = df_dim["Estimate"].tolist()
dim_ses   = df_dim["SE"].tolist()
dim_pvals = df_dim["p_value"].tolist()
dim_labs  = df_dim["Specification"].tolist()
dim_ci_h  = [1.96 * s for s in dim_ses]
bar_colors = [C_INK if p < 0.05 else (C_AMBER if p < 0.10 else C_GREY) for p in dim_pvals]

fig, ax = plt.subplots(figsize=(8.5, 5.2))
y3 = np.arange(len(dim_labs))[::-1]
dim_label_x = []
for i, (est_, ci_h_, p, col) in enumerate(zip(dim_ests, dim_ci_h, dim_pvals, bar_colors)):
    ax.barh(y3[i], est_, color=col, edgecolor=C_INK, lw=0.7, height=0.6)
    ax.errorbar(est_, y3[i], xerr=ci_h_, fmt="none", ecolor=C_GREY, capsize=4, lw=1.2)
    tag = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "n.s."
    x_pos = est_ + ci_h_ + 0.005 if est_ < 0 else est_ - ci_h_ - 0.005
    dim_label_x.append(x_pos)
    ax.text(x_pos, y3[i], f"{est_:.3f} {tag}",
            va="center", ha="left" if est_ < 0 else "right", fontsize=10.5,
            fontweight="bold" if p < 0.05 else "normal")
ax.axvline(0, color=C_GREY, lw=1, ls="--")
dim_x_values = [0] + [e - h for e, h in zip(dim_ests, dim_ci_h)]
dim_x_values += [e + h for e, h in zip(dim_ests, dim_ci_h)] + dim_label_x
dim_x_min, dim_x_max = min(dim_x_values), max(dim_x_values)
dim_x_pad = max(0.02, (dim_x_max - dim_x_min) * 0.12)
ax.set_xlim(dim_x_min - dim_x_pad, dim_x_max + dim_x_pad)
ax.set_yticks(y3)
ax.set_yticklabels(dim_labs, fontsize=11)
ax.set_xlabel("Coefficient on ln(wage) with 95% CI  (dict. score per log-wage point)",
              fontsize=10.5)
ax.text(0.99, 0.02, "Dictionary-based (not SBERT) — supplementary, exploratory",
        transform=ax.transAxes, fontsize=8.5, ha="right", style="italic", color=C_GREY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig("fig_dimension.png")

# ── Fig 4: validation bar chart — ±1 SD, calculated p-values ─────
by_r = ratings.groupby("manual_rating")["amenity_raw"].agg(["mean", "std", "count"])
cats    = ["Rating 1\n(low)", "Rating 2\n(medium)", "Rating 3\n(high)"]
fcolors = [C_LITE, C_AMBER, C_INK]

fig, ax = plt.subplots(figsize=(6.6, 5.2))
ax.bar(np.arange(3), by_r["mean"].values, yerr=by_r["std"].values,
       capsize=5, color=fcolors, edgecolor=C_INK, lw=0.8,
       error_kw={"ecolor": C_GREY, "elinewidth": 1.2})
for i, (m_, n_) in enumerate(zip(by_r["mean"], by_r["count"])):
    ax.text(i, m_ + by_r["std"].iloc[i] + 0.013,
            f"mean {m_:.3f}\nn = {int(n_)}", ha="center", fontsize=9.5)
ax.text(0.03, 0.97,
        f"Pearson r = {r_p:.3f}  ({p_p_fmt})\n"
        f"Spearman r = {r_s:.3f}  ({p_s_fmt})",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=C_INK, lw=0.8))
ax.set_xticks(np.arange(3))
ax.set_xticklabels(cats)
ax.set_ylabel("Mean SBERT amenity score (raw)", fontsize=11)
ax.set_title("Error bars = ±1 SD within rating category", fontsize=9.5,
             color=C_GREY, style="italic")
ax.set_ylim(0, max(by_r["mean"].values) + max(by_r["std"].values) + 0.12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
save_fig("fig_validation.png")


# ─────────────────────────────────────────────────────────────────
# STEP 23 — RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────
section("WRITING RESULTS SUMMARY")

hl_s = df_main[df_main["Specification"].str.contains("HEADLINE")].iloc[0]

def fmt_row(r):
    return (f"  {r['Specification'][:54]:<54}  "
            f"beta={r['Estimate']:+.4f}  SE={r['SE']:.4f}  "
            f"{r['Stars']:<3}  N_model={r['N_model'] or 'n/a'}")

write("RESULTS_SUMMARY.txt",
    f"{'='*74}\n"
    f"THESIS RESULTS SUMMARY  (thesis_analysis.py v4.0)\n"
    f"{'='*74}\n\n"
    f"REPRODUCIBILITY NOTE\n"
    f"This pipeline reproduces results from amenity_scores.csv onward.\n"
    f"Separate SBERT reconstruction and its limitations are documented in docs/reproducibility.md.\n"
    f"Full raw-text-to-results reproducibility cannot be claimed.\n\n"
    f"VALIDATION NOTE\n"
    f"The original rating file contained the model score and tertile variables.\n"
    f"Ratings may not be fully blinded. See 10_validation_results.txt.\n\n"
    f"CANONICAL CONTROLS: {CANONICAL_CONTROLS}\n\n"
    f"Sample: {n_base:,} postings | {n_firms:,} firms | "
    f"{df_reg['year'].min()}-{df_reg['year'].max()}\n\n"
    f"{'─'*74}\nHEADLINE (Spec 2, ln(wage) + firm FE)\n{'─'*74}\n"
    f"beta = {hl_s['Estimate']:.6f}  SE={hl_s['SE']:.6f}  {hl_s['Stars']}\n"
    f"95% CI: [{hl_s['CI_low']:.6f}, {hl_s['CI_high']:.6f}]\n"
    f"p-value: {hl_s['p_value']:.2e}\n"
    f"Model N: {hl_s['N_model']}\n"
    f"Within R2: {hl_s['Within_R2']}\n\n"
    f"{'─'*74}\nMAIN SPECIFICATIONS\n{'─'*74}\n"
    + "\n".join(fmt_row(r) for r in results_main)
    + f"\n\n{'─'*74}\nROBUSTNESS\n{'─'*74}\n"
    + "\n".join(fmt_row(r) for r in results_rob)
    + f"\n\n{'─'*74}\nSKILL SLOPES (delta method; normal approx. p-values)\n{'─'*74}\n"
    + "\n".join(
        f"  {r['Group']:<40}  beta={r['Estimate']:+.4f}  SE={r['SE']:.4f}  {r['Stars']}"
        for _, r in df_slopes.iterrows()
    )
    + f"\n\n{'─'*74}\nDICTIONARY DIMENSIONS (keyword-based, not SBERT, supplementary)\n{'─'*74}\n"
    + "\n".join(
        f"  {r['Specification']:<14}  beta={r['Estimate']:+.4f}  SE={r['SE']:.4f}  {r['Stars']}"
        for _, r in df_dim.iterrows()
    )
    + f"\n\n{'─'*74}\nVALIDATION (potentially non-blinded)\n{'─'*74}\n"
    f"Pearson r={r_p:.3f}  Spearman r={r_s:.3f}  "
    f"Weighted kappa={kappa_wt:.3f}  N={len(ratings)}\n\n"
    f"{'─'*74}\nECONOMIC MAGNITUDE\n{'─'*74}\n"
    f"10% lower wage -> {beta * ln_090 * 100:.2f}% of one SD more amenity language (firm FE)\n"
    f"Arithmetic mean real wage: GBP {df_reg['salary_real'].mean():,.0f}\n\n"
    f"{'─'*74}\nSIGNIFICANCE CONVENTION\n{'─'*74}\n"
    f"*** p < 0.01  ** p < 0.05  * p < 0.10\n\n"
    f"{'─'*74}\nKEY DEFINITIONS\n{'─'*74}\n"
    f"below_market: FIRM-OCCUPATION level (firm avg wage vs cell mean)\n"
    f"range_posted: text label, not negotiable==1\n"
    f"range_width:  |max-min|/salary; zero-imputed for incomplete ranges\n"
    f"posting density: COUNTS per region-month, NOT unemployment\n"
    f"dimension scores: DICTIONARY keyword counts, NOT SBERT embeddings\n"
    f"wage headline: ARITHMETIC mean (GBP {df_reg['salary_real'].mean():,.0f})\n"
    f"{'='*74}\n"
)


# ─────────────────────────────────────────────────────────────────
# STEP 24 — MANIFEST + FINAL STATUS
# ─────────────────────────────────────────────────────────────────
_write_preflight()

manifest_text = (
    f"OUTPUT MANIFEST — {datetime.now().isoformat()}\n" + "=" * 60 + "\n"
    + "\n".join(manifest_lines)
)
(OUTPUT_DIR / "OUTPUT_MANIFEST.txt").write_text(manifest_text, encoding="utf-8")

section("FINAL STATUS")
if all_pass:
    log("\n" + "=" * 60)
    log("  REPLICATION PASSED")
    log("=" * 60 + "\n")
else:
    log("\n" + "=" * 60)
    log("  REPLICATION DID NOT MATCH — DO NOT UPDATE THESIS OR SLIDES")
    log("  See outputs/thesis/12_diagnostic_comparison.txt for details.")
    log("=" * 60 + "\n")

log(f"All outputs written to ./{OUTPUT_DIR}/")
for f in sorted(OUTPUT_DIR.iterdir()):
    log(f"  {f.name}")
