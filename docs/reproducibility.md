# Reproducibility and provenance

## What this release supports

The repository makes the implemented methods and code reviewable. It does not
contain the private raw dataset, archived observation-level scores, or rated
validation sample needed to rerun the complete analysis. The raw Stata file was
also absent from the supplied private code package.

The main analysis is intentionally retained as a sequential research script.
Importing `thesis_analysis.py` executes it; run it as a script only after providing
authorised inputs. Splitting its stateful stages or changing model settings was
avoided to preserve the research calculations.

## SBERT reconstruction status

The supplied reconstruction describes itself as recovered from project-development
materials. Earlier analysis comments said the scoring code and anchors were missing;
the package now includes a reconstruction, the anchor lists, and comparison metadata.
The public documentation resolves that inconsistency without claiming bitwise identity.

The archived full comparison covers all 200,000 rows:

| Quantity | Maximum absolute difference | Agreement at absolute tolerance 1e−6, zero relative tolerance |
| --- | ---: | --- |
| Raw score | 5.191052627473436e−7 | Pass |
| Standardised score | 7.64905534023086e−6 | Fail |

Both archived correlations exceed 0.999999999999. The differences are small but
nonzero, and the standardised scores do not meet the stated 1e−6 comparison test.
The archive's replication PDF attributes the differences to floating-point
computation; this release has not independently established their cause.
The main pipeline therefore continues to load the archived `amenity_scores.csv`.

For an initial check with authorised inputs:

```bash
python src/build_amenity_scores.py \
  --reference data/amenity_scores.csv \
  --output outputs/amenity_scores_test_1000.csv \
  --limit 1000 --device cpu
```

A limited run writes missing standardised scores and compares raw scores only:
subsets cannot reproduce full-sample occupation-year standardisation.
After reviewing that report, use the full command in the README. The reconstruction
refuses to overwrite an existing selected output unless `--overwrite` is supplied.
Do not point that flag at an archived input. The main and sensitivity scripts
retain their original behaviour of replacing their generated outputs on reruns.

## Environment evidence

`requirements.txt` contains the actual third-party imports plus PyTorch, the
Sentence-Transformers runtime backend. Its version bounds are retained from the
source package; they are not a lockfile or a guarantee of identical results.
No `pip freeze` was used. There is no statsmodels, seaborn, R, Jupyter, or Stata
program dependency in the supplied workflow.

The source README reports verification with Python 3.12.13 and the replication PDF
says Python 3.12. Saved main-analysis and scoring records instead show Python 3.13.2.
This inconsistency remains documented rather than selecting an unsupported claim.

| Package | Saved main-analysis environment | Saved scoring environment, where recorded |
| --- | --- | --- |
| pandas | 3.0.3 | 2.3.3 |
| NumPy | 2.4.6 | 2.5.0 |
| SciPy | 1.17.1 | Not recorded |
| scikit-learn | 1.9.0 | Not recorded |
| PyFixest | 0.50.1 | Not recorded |
| Matplotlib | 3.10.9 | Not recorded |

The archived SBERT record uses CPU, batch size 512, and model maximum sequence
length 256. The supplied scripts name the model but do not pin a model revision or
record every dependency version. PyFixest defaults and private result attributes
are used by the original scripts. These are limits on exact future replication.

## Verification during public preparation

Verification covers Python syntax, CLI loading, path configuration, anchor equality,
source-level analytical preservation, image content and metadata, Git exclusions,
and public-file/history inspection. It uses no copied data or synthetic data sample.
The archived regressions, embeddings, and figures were not recomputed.

The fixed numerical targets in `thesis_analysis.py` and `sensitivity_common.py` are
diagnostic comparisons, not substituted regression outputs. A completed process
does not itself prove replication: inspect `outputs/thesis/12_diagnostic_comparison.txt`
and the score-comparison reports. The original main script logs a failed numerical
comparison without necessarily returning a failing process exit code.
