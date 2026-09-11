#!/usr/bin/env python3
"""
Rebuild the thesis amenity-language scores from raw Adzuna advertisement text.

Recovered methodology
---------------------
This script is a cleaned, submission-ready reconstruction of the SBERT code
recovered from the project's development chat. It preserves the recovered model,
15 amenity reference sentences, 6 placebo reference sentences, text extraction,
cosine-similarity difference, and SOC-by-year standardisation.

IMPORTANT
---------
The recovered code embeds each advertisement body as ONE document. It does not
split advertisements into sentences. Treat this implementation as confirmed only
if its raw scores reproduce the archived amenity_scores.csv. Use --reference to
produce an explicit comparison report, and do not overwrite the archived file
until the comparison succeeds.

Examples
--------
Quick provenance test on the first 1,000 advertisements (compares raw scores):
    python src/build_amenity_scores.py \
        --input data/adzuna_sample.dta \
        --output outputs/amenity_scores_test_1000.csv \
        --reference data/amenity_scores.csv \
        --limit 1000

Full rebuild:
    python src/build_amenity_scores.py \
        --input data/adzuna_sample.dta \
        --output outputs/amenity_scores_recreated.csv \
        --reference data/amenity_scores.csv
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from paths import DATA_DIR, OUTPUT_DIR

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

AMENITY_REFERENCES = [
    "We offer flexible working arrangements and remote work options.",
    "You can choose your own working hours to fit your lifestyle.",
    "We support work-life balance with flexible scheduling.",
    "Join our supportive and inclusive team culture.",
    "We foster a collaborative and friendly work environment.",
    "Our team values diversity and mutual respect.",
    "Make a real difference in people's lives through your work.",
    "This role offers meaningful and rewarding work.",
    "You will have the opportunity to create positive social impact.",
    "We invest in your professional growth and career development.",
    "Ongoing training and development opportunities are provided.",
    "You will receive mentoring and support to advance your career.",
    "We offer a generous benefits package including healthcare and pension.",
    "Enjoy perks such as gym membership, free lunches, and team events.",
    "Competitive salary plus excellent benefits and bonuses.",
]

PLACEBO_REFERENCES = [
    "Please attach your CV and cover letter to your application.",
    "Applications close on the date stated in this advertisement.",
    "Candidates must be eligible to work in the United Kingdom.",
    "Please submit your application via the link below.",
    "Only shortlisted candidates will be contacted.",
    "The closing date for applications is listed above.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild SBERT amenity-language scores and compare them with an archived score file."
    )
    parser.add_argument("--input", default=str(DATA_DIR / "adzuna_sample.dta"), help="Input Stata dataset.")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "amenity_scores_recreated.csv"),
        help="Output CSV. The archived amenity_scores.csv is never overwritten by default.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional archived amenity_scores.csv for numerical comparison.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--device",
        default=None,
        help="SentenceTransformer device, e.g. cpu, mps, or cuda. Default: automatic.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of leading rows for a quick provenance test. Compare amenity_raw only when using a limit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the selected output file. Never point this at the archived score file before validation.",
    )
    return parser.parse_args()


def extract_description_body(df: pd.DataFrame) -> pd.Series:
    if "description" not in df.columns:
        raise KeyError("Input data must contain a 'description' column.")
    description = df["description"].astype("string")
    body = description.str.split("│").str[1].fillna(description).fillna("")
    return body.astype(str)


def encode_centroid(
    model: SentenceTransformer,
    references: list[str],
    batch_size: int,
) -> np.ndarray:
    embeddings = model.encode(
        references,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.mean(axis=0, keepdims=True)


def score_documents(
    model: SentenceTransformer,
    texts: list[str],
    amenity_centroid: np.ndarray,
    placebo_centroid: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    output = np.empty(len(texts), dtype=np.float64)
    total = len(texts)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        embeddings = model.encode(
            texts[start:end],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        amenity_similarity = cosine_similarity(embeddings, amenity_centroid).ravel()
        placebo_similarity = cosine_similarity(embeddings, placebo_centroid).ravel()
        output[start:end] = amenity_similarity - placebo_similarity

        if start == 0 or end == total or start % 20_000 == 0:
            print(f"Scored {end:,}/{total:,} advertisements ({100 * end / total:.1f}%).")

    return output


def standardise_within_soc_year(df: pd.DataFrame) -> pd.Series:
    required = {"occupation_code_3d", "year", "amenity_raw"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Cannot standardise scores; missing columns: {sorted(missing)}")

    soc_year = df["occupation_code_3d"].astype(str) + "_" + df["year"].astype(str)

    def z_score(values: pd.Series) -> pd.Series:
        sd = values.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(values)), index=values.index)
        return (values - values.mean()) / sd

    return df.groupby(soc_year, observed=False)["amenity_raw"].transform(z_score)


def compare_with_reference(
    recreated: pd.DataFrame,
    reference_path: Path,
    limited_run: bool,
) -> dict[str, object]:
    reference = pd.read_csv(reference_path, index_col=0)
    required = {"amenity_raw", "amenity_score"}
    missing = required.difference(reference.columns)
    if missing:
        raise ValueError(f"Reference file is missing columns: {sorted(missing)}")

    common_index = recreated.index.intersection(reference.index)
    if len(common_index) == 0:
        raise ValueError("Recreated and reference files have no common index values.")

    report: dict[str, object] = {
        "reference_file": str(reference_path),
        "recreated_rows": int(len(recreated)),
        "reference_rows": int(len(reference)),
        "common_rows": int(len(common_index)),
        "limited_run": bool(limited_run),
    }

    variables = ["amenity_raw"] if limited_run else ["amenity_raw", "amenity_score"]
    for variable in variables:
        old = pd.to_numeric(reference.loc[common_index, variable], errors="coerce")
        new = pd.to_numeric(recreated.loc[common_index, variable], errors="coerce")
        valid = old.notna() & new.notna()
        difference = (old[valid] - new[valid]).abs()
        report[variable] = {
            "n_compared": int(valid.sum()),
            "correlation": float(old[valid].corr(new[valid])),
            "mean_absolute_difference": float(difference.mean()),
            "maximum_absolute_difference": float(difference.max()),
            "allclose_atol_1e-6": bool(
                np.allclose(old[valid].to_numpy(), new[valid].to_numpy(), rtol=0, atol=1e-6)
            ),
        }

    return report


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Choose another name or use --overwrite."
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")

    print(f"Loading {input_path} ...")
    df = pd.read_stata(input_path)
    if "date_advertisement" not in df.columns:
        raise KeyError("Input data must contain 'date_advertisement'.")
    if "occupation_code_3d" not in df.columns:
        raise KeyError("Input data must contain 'occupation_code_3d'.")

    df["date_advertisement"] = pd.to_datetime(df["date_advertisement"], errors="raise")
    df["year"] = df["date_advertisement"].dt.year
    df["description_body"] = extract_description_body(df)

    if args.limit is not None:
        df = df.iloc[: args.limit].copy()
        print(f"Quick-test mode: using the first {len(df):,} rows.")

    print(f"Loading model {MODEL_NAME!r} ...")
    model_kwargs = {"device": args.device} if args.device else {}
    model = SentenceTransformer(MODEL_NAME, **model_kwargs)
    print(f"Model max sequence length: {model.max_seq_length} tokens.")

    print("Embedding reference sentences ...")
    amenity_centroid = encode_centroid(model, AMENITY_REFERENCES, args.batch_size)
    placebo_centroid = encode_centroid(model, PLACEBO_REFERENCES, args.batch_size)

    print(f"Scoring {len(df):,} advertisement bodies ...")
    df["amenity_raw"] = score_documents(
        model=model,
        texts=df["description_body"].tolist(),
        amenity_centroid=amenity_centroid,
        placebo_centroid=placebo_centroid,
        batch_size=args.batch_size,
    )

    if args.limit is None:
        df["amenity_score"] = standardise_within_soc_year(df)
    else:
        # A subset cannot reproduce full-sample SOC-year z-scores. The raw-score
        # comparison is still sufficient for a quick provenance test.
        df["amenity_score"] = np.nan

    recreated = df[["amenity_raw", "amenity_score"]].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recreated.to_csv(output_path, index=True)
    print(f"Saved recreated scores to {output_path.resolve()}")

    metadata = {
        "model": MODEL_NAME,
        "model_max_sequence_length": int(model.max_seq_length),
        "scoring_unit": "entire advertisement body as one model input",
        "amenity_reference_count": len(AMENITY_REFERENCES),
        "placebo_reference_count": len(PLACEBO_REFERENCES),
        "standardisation": "within 3-digit SOC x calendar year; pandas sample SD (ddof=1)",
        "input_rows": int(len(df)),
        "batch_size": int(args.batch_size),
        "device": str(model.device),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to {metadata_path.resolve()}")

    if args.reference:
        reference_path = Path(args.reference)
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference score file not found: {reference_path}")
        report = compare_with_reference(
            recreated=recreated,
            reference_path=reference_path,
            limited_run=args.limit is not None,
        )
        report_path = output_path.with_suffix(".comparison.json")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("\nComparison with archived scores:")
        print(json.dumps(report, indent=2))
        print(f"Saved comparison report to {report_path.resolve()}")


if __name__ == "__main__":
    main()
