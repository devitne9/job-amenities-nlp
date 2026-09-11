# Methodology implemented in the code

This document describes the supplied scripts. The public preparation preserves
their analytical expressions, regression formulas, defaults affecting estimation,
and numerical diagnostic targets.

## Text and score construction

`src/build_amenity_scores.py` uses `sentence-transformers/all-MiniLM-L6-v2`.
The saved reconstruction metadata records a maximum sequence length of 256 tokens.

The literal extraction expression is `description.str.split("│").str[1]`, falling
back to the original description when that segment is missing. It selects the
second segment: if several separators occur, later segments are not concatenated.
The score reconstruction fills missing text with an empty string. The main analysis
retains its original missing-text handling for the length control.

The script averages embeddings of the 15 authored amenity anchors into one centroid
and of the six procedural anchors into another. Both lists are retained verbatim
in the scoring module; they are model inputs, not sampled job advertisements.
The original auxiliary CSV lists were checked for exact equality, including order,
before the head–tail script was changed to share the lists in code.

For advertisement embedding `e`, amenity centroid `a`, and procedural centroid `p`:

```text
amenity_raw = cosine_similarity(e, a) - cosine_similarity(e, p)
amenity_score = (amenity_raw - cell_mean) / cell_sample_standard_deviation
```

Standardisation uses three-digit occupation × calendar-year cells and `ddof=1`.
Cells with zero or non-finite standard deviations receive zeros. Each advertisement
body is one model input, with tokenizer truncation. There is no sentence splitting,
fine-tuning, supervised classifier, or sentence-level aggregation in these scripts.
The reconstruction's default batch size is 512.

## Wages, controls, and sample

Dates produce calendar year, year-month, and quarter. Salary is converted to the
2019 price basis using the preserved factors `105.1 / 101.0` for 2017,
`105.1 / 103.3` for 2018, and `1.0` for 2019. The regressor is the natural log of
positive real salary. No new trimming, imputation, or annualisation is introduced.

The salary-range flag uses the supplied `negotiable` category whose label contains
“range”. Width is `abs(max_salary - min_salary) / salary` for valid range postings,
with zero filling otherwise. Missing endpoints also produce a separate indicator
used only in a sensitivity specification. The unique-salary sample uses the
category containing “unique”, with the original non-range fallback.

Canonical controls are:

```text
C(education) + C(seniority) + C(experience)
+ C(type_contract) + C(time_contract)
+ log_ad_length + range_posted + range_width
```

`log_ad_length` is the log character count, clipped to at least one character.
The five categorical indicators arrive pre-built in the input dataset; their
original extraction process is not implemented here. The main script permits
education to be absent, but that fallback is not the canonical reported model.
The sensitivity helper requires all five columns.

Scores are joined using original row indices. The main analysis requires equal
row counts, exact index equality, a unique score index, and no missing standardised
scores. It forms `occ_reg_ym` from three-digit occupation, `location_2d`, and
year-month, then excludes cells containing only one posting. Each model drops
missing required variables; further fixed-effect singleton handling is left to
PyFixest exactly as in the source. Source duplicates are retained in the main run.

## Main estimation

The dependent variable is `amenity_score`; log wage is a regressor.

```text
Baseline:
amenity_score ~ ln_wage + canonical_controls | occ_reg_ym

Preferred:
amenity_score ~ ln_wage + canonical_controls | occ_reg_ym + company_id
```

Both use `pyfixest.feols` with `vcov={"CRV1": "company_id"}`. The script constructs
reported 95% intervals as coefficient ± 1.96 × standard error. It does not alter
PyFixest's default singleton settings. The significance convention is
`*** p < 0.01`, `** p < 0.05`, `* p < 0.10`.

Additional main specifications replace log wage with a below-market indicator,
or replace the dependent variable with within-occupation-year raw-score tertiles.
The principal below-market indicator compares the firm's mean log wage within an
occupation with the posting's occupation–region–month mean. A separate posting-level
indicator is used in robustness checks. Tertile construction retains its original
fallback to the middle category if quantile cutting fails.

## Supporting and sensitivity analyses

| Analysis | Implementation |
| --- | --- |
| Alternative market cells | Coarser region, quarter, or two-digit occupation fixed effects, using the already restricted baseline sample |
| Salary and controls | Unique-salary sample; omit the five text-derived categorical controls; add missing-endpoint indicator; ordinal encoding of the five controls |
| Skill heterogeneity | Log-wage interactions for low skill (SOC 8–9) and high skill (SOC 1–2), relative to SOC 3–7; market-cell FE; implied slopes use the full covariance matrix and delta-method SEs |
| Posting density | Region-month posting counts split at the observation-weighted median; unweighted cell-median sensitivity; this is not an unemployment or labour-market-tightness measure |
| Amenity dimensions | Sum of keyword-presence indicators from literal substring searches, standardised within occupation-year in the restricted sample; separate market-cell FE regressions; these are dictionary scores, not SBERT dimensions |
| Rating agreement | Pearson and Spearman correlations; unweighted and linearly weighted Cohen's kappa, using the supplied 200-posting rated file |
| Exact duplicates | Compare retained rows with `duplicated(keep="first")` exclusion before rebuilding the sample; retain original indices and archived scores |
| Head–tail truncation | Tokenise without special tokens, retain first 120 + last 120 tokens when longer than 240, decode and re-encode; shorter inputs also undergo decode/re-encode; default batch size 256 and CPU; same centroids and occupation-year standardisation |

The head–tail script retains optional local reference-CSV overrides. Defaults use
the identical authored anchor lists in code. Changes to anchors, token windows,
or input order would be new analytical choices, not replication of the defaults.

## Interpretation boundaries

The outcome measures language, not verified benefits or realised workplace quality.
The design is observational. Fixed effects and controls do not establish causality,
worker preferences, or intentional employer compensation strategies. The manual
ratings are stratified and may have been informed by displayed model scores and
tertiles; they do not provide independent blinded validation.
