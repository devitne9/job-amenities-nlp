#!/usr/bin/env python3
"""
Sensitivity check for SBERT truncation.

Constructs an alternative score from the first 120 and last 120 model tokens of
an advertisement (or the complete text when it has <=240 tokens), preserving
information from benefits sections that may occur at the end of long ads. It
then standardises the alternative raw score within 3-digit SOC x year and
re-estimates the baseline and preferred company-FE models.
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

from build_amenity_scores import AMENITY_REFERENCES, PLACEBO_REFERENCES
from paths import DATA_DIR, OUTPUT_DIR

from sensitivity_common import CANONICAL_EXPECTED, extract_description_body, fit_main_models, prepare_analysis_data

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(DATA_DIR / "adzuna_sample.dta"))
    p.add_argument("--original-scores", default=str(DATA_DIR / "amenity_scores.csv"))
    p.add_argument("--amenity-refs", default=None, help="Optional local CSV with a text column; default: archived amenity anchors in code.")
    p.add_argument("--placebo-refs", default=None, help="Optional local CSV with a text column; default: archived placebo anchors in code.")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR / "sensitivity"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--head-tokens", type=int, default=120)
    p.add_argument("--tail-tokens", type=int, default=120)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_references(path: str | Path) -> list[str]:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError(f"Reference file {path} must contain a 'text' column.")
    values = df["text"].dropna().astype(str).tolist()
    if not values:
        raise ValueError(f"No references found in {path}.")
    return values


def make_head_tail_texts(tokenizer, texts: list[str], head_n: int, tail_n: int) -> tuple[list[str], int]:
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]

    transformed: list[str] = []
    n_truncated = 0
    for ids in encoded:
        if len(ids) > head_n + tail_n:
            ids = ids[:head_n] + ids[-tail_n:]
            n_truncated += 1
        transformed.append(
            tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        )
    return transformed, n_truncated


def score_head_tail(
    model: SentenceTransformer,
    texts: pd.Series,
    amenity_centroid: np.ndarray,
    placebo_centroid: np.ndarray,
    batch_size: int,
    head_n: int,
    tail_n: int,
) -> tuple[np.ndarray, int]:
    values = texts.fillna("").astype(str).tolist()
    scores = np.empty(len(values), dtype=np.float64)
    total_truncated = 0

    for start in range(0, len(values), batch_size):
        end = min(start + batch_size, len(values))
        transformed, n_truncated = make_head_tail_texts(
            model.tokenizer, values[start:end], head_n, tail_n
        )
        total_truncated += n_truncated
        embeddings = model.encode(
            transformed,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        sim_a = cosine_similarity(embeddings, amenity_centroid).ravel()
        sim_p = cosine_similarity(embeddings, placebo_centroid).ravel()
        scores[start:end] = sim_a - sim_p

        if start == 0 or end == len(values) or start % 20_000 == 0:
            print(f"Scored {end:,}/{len(values):,} ads ({100*end/len(values):.1f}%).", flush=True)

    return scores, total_truncated


def zscore_within_soc_year(df: pd.DataFrame, raw_col: str) -> pd.Series:
    group = df["occupation_code_3d"].astype(str) + "_" + df["year"].astype(str)

    def z(x: pd.Series) -> pd.Series:
        sd = x.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - x.mean()) / sd

    return df.groupby(group, observed=False)[raw_col].transform(z)


def main() -> None:
    args = parse_args()
    if args.head_tokens <= 0 or args.tail_tokens <= 0:
        raise ValueError("Head and tail token counts must be positive.")
    if args.head_tokens + args.tail_tokens > 250:
        raise ValueError("Use at most 250 content tokens to leave room for model special tokens.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    score_path = out / "amenity_scores_head_tail.csv"

    print(f"Loading {args.input} ...", flush=True)
    raw = pd.read_stata(args.input)
    raw["date_advertisement"] = pd.to_datetime(raw["date_advertisement"], errors="raise")
    raw["year"] = raw["date_advertisement"].dt.year
    raw["description_body"] = extract_description_body(raw)

    amenity_refs = load_references(args.amenity_refs) if args.amenity_refs else AMENITY_REFERENCES
    placebo_refs = load_references(args.placebo_refs) if args.placebo_refs else PLACEBO_REFERENCES

    print(f"Loading {MODEL_NAME} on {args.device} ...", flush=True)
    model = SentenceTransformer(MODEL_NAME, device=args.device)
    print(f"Model max sequence length: {model.max_seq_length}", flush=True)

    amenity_emb = model.encode(amenity_refs, show_progress_bar=False, convert_to_numpy=True)
    placebo_emb = model.encode(placebo_refs, show_progress_bar=False, convert_to_numpy=True)
    amenity_centroid = amenity_emb.mean(axis=0, keepdims=True)
    placebo_centroid = placebo_emb.mean(axis=0, keepdims=True)

    print(
        f"Constructing head-tail score: first {args.head_tokens} + last {args.tail_tokens} tokens ...",
        flush=True,
    )
    raw["amenity_raw_head_tail"], n_truncated = score_head_tail(
        model, raw["description_body"], amenity_centroid, placebo_centroid,
        args.batch_size, args.head_tokens, args.tail_tokens,
    )
    raw["amenity_score_head_tail"] = zscore_within_soc_year(raw, "amenity_raw_head_tail")
    raw[["amenity_raw_head_tail", "amenity_score_head_tail"]].to_csv(score_path, index=True)

    original_scores = pd.read_csv(args.original_scores, index_col=0)
    if not raw.index.equals(original_scores.index):
        raise ValueError("Original score index does not exactly match raw-data index.")

    correlations = {
        "raw_score_correlation": float(raw["amenity_raw_head_tail"].corr(original_scores["amenity_raw"])),
        "standardised_score_correlation": float(raw["amenity_score_head_tail"].corr(original_scores["amenity_score"])),
        "ads_longer_than_240_model_tokens": int(n_truncated),
        "share_longer_than_240_model_tokens": float(n_truncated / len(raw)),
    }

    head_tail_scores = pd.DataFrame(
        {
            "amenity_raw": raw["amenity_raw_head_tail"],
            "amenity_score": raw["amenity_score_head_tail"],
        },
        index=raw.index,
    )
    prepared, diagnostics = prepare_analysis_data(args.input, head_tail_scores)
    results = fit_main_models(prepared, score_col="amenity_score")
    results["Difference_from_canonical"] = [
        results.loc[0, "Estimate"] - CANONICAL_EXPECTED["baseline_beta"],
        results.loc[1, "Estimate"] - CANONICAL_EXPECTED["firm_fe_beta"],
    ]
    results.to_csv(out / "head_tail_sensitivity_results.csv", index=False)

    metadata = {
        "model": MODEL_NAME,
        "model_max_sequence_length": int(model.max_seq_length),
        "head_tokens": args.head_tokens,
        "tail_tokens": args.tail_tokens,
        "batch_size": args.batch_size,
        "device": str(model.device),
        "input_rows": len(raw),
        **correlations,
        **diagnostics,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (out / "head_tail_sensitivity_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    lines = [
        "HEAD-TAIL TRUNCATION SENSITIVITY",
        "=" * 60,
        f"Window: first {args.head_tokens} + last {args.tail_tokens} model tokens",
        f"Ads requiring head-tail truncation: {n_truncated:,} ({n_truncated/len(raw):.2%})",
        f"Correlation with original raw score: {correlations['raw_score_correlation']:.6f}",
        f"Correlation with original standardised score: {correlations['standardised_score_correlation']:.6f}",
        "",
        results.to_string(index=False),
        "",
        "Interpretation rule: the sensitivity is reassuring if both coefficients remain negative,",
        "statistically precise, and substantively close enough to support the same conclusion.",
    ]
    (out / "head_tail_sensitivity_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nOutputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
