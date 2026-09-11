# Private input contract

No data files or example records are supplied. These requirements are inferred
from the actual loading and validation code. Do not reorder, deduplicate, or reset
indices before the main analysis: the archived scores align to the original rows.

Create an ignored `data/` directory only when you have authorised inputs.

## `data/adzuna_sample.dta`

Read by `pandas.read_stata`, including its default handling of Stata value labels.
No Stata installation or additional Stata packages are used.

| Column | Role and expected representation |
| --- | --- |
| `description` | Private advertisement text; original separator handling is preserved |
| `date_advertisement` | Values convertible by pandas to dates; canonical years 2017–2019 |
| `occupation_code_3d` | Numeric three-digit occupation code; also used with integer division to form broader groups |
| `location_2d` | Region code usable as a string; first three characters form the coarser-region variant |
| `company_id` | Company identifier for fixed effects, clustering, and wage grouping |
| `salary`, `min_salary`, `max_salary` | Numeric offered salary and range endpoints; the script uses the supplied salary basis |
| `negotiable` | Labelled salary-format category; labels containing “range” and “unique” are detected by the code |
| `education` | Canonical categories: `None or not mentioned`, `GCSE (or equivalent)`, `A-level (or equivalent)`, `Bachelor's degree`, `Master's or Doctoral degree` |
| `seniority` | Categories expected by the ordinal sensitivity: `Not mentioned`, `High seniority`, `Medium seniority`, `Low seniority` |
| `experience` | `Not mentioned`, `Experience required`, `No experience needed` |
| `type_contract` | `Not mentioned`, `Temporary job`, `Permanent job` |
| `time_contract` | `Not mentioned`, `Full-time job`, `Part-time job` |

The main script validates education categories. Its ordinal sensitivity maps
unknown or missing control categories to zero using the archived logic; this
behaviour is preserved, not a recommendation for new data. Other source columns
can affect the exact-duplicate check, which compares complete source rows.

## `data/amenity_scores.csv`

The first column is the saved original row index (`index_col=0`). Required score
columns are `amenity_raw` and `amenity_score`. For the main analysis the score index
must be unique and exactly equal, in order and length, to the raw-data index.
The archived file covers the entire 200,000-row source sample before restrictions.
Scores are private observation-level data even when no advertisement text is present.

Reconstruction writes a separate file in `outputs/`. It does not automatically
promote reconstructed scores to the authoritative archived input.

## `data/validation_sample_RATED.csv`

Read with `index_col=0`. Required columns are `manual_rating`, `amenity_raw`, and
`tertile`. Ratings use 1, 2, and 3; tertile labels are `low`, `medium`, and `high`.
The archived validation uses 200 stratified postings. Advertisement text and other
private columns in the original file are not required for the validation formulas
and must not be published. Do not substitute invented ratings or observations.

## Local outputs

| Location | Contents |
| --- | --- |
| `outputs/amenity_scores_recreated.csv` | Reconstructed row-level scores |
| `outputs/*.metadata.json`, `outputs/*.comparison.json` | Reconstruction environment and numerical comparison reports |
| `outputs/thesis/` | Main and robustness tables, sample diagnostics, rating agreement, economic magnitudes, four figures, and manifest |
| `outputs/sensitivity/` | Duplicate comparisons, head–tail row-level scores, regression summaries, and metadata |

All output directories are ignored. Logs can contain local input paths and source
category labels. An aggregate-looking filename is not evidence that a file is safe
to publish; review content and image metadata before explicitly selecting outputs.
