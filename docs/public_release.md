# Public release notes

## Scope and file mapping

The private source package was inspected read-only. It contains five Python scripts,
20 CSV files, four PNG figures, a replication PDF, and supporting reports and
documentation: 48 files in total. No notebooks, R scripts, Stata programs, temporary
files, caches, credentials files, or Git repository were present in that package.
The referenced raw Stata dataset was not supplied. Files referenced as earlier
development versions were not found and are not presented as included code.

| Private source | Public destination | Preparation |
| --- | --- | --- |
| `thesis_analysis.py` | `src/thesis_analysis.py` | Repository-relative input/output configuration; updated obsolete scoring-provenance text and editorial comments |
| `sbert_replication_package/00_build_amenity_scores.py` | `src/build_amenity_scores.py` | Descriptive name; safe default paths; output-directory creation; updated examples; removed unused typing import |
| `final_sensitivity_package/01_duplicate_sensitivity.py` | `src/duplicate_sensitivity.py` | Descriptive name and safe default paths |
| `final_sensitivity_package/02_head_tail_sensitivity.py` | `src/head_tail_sensitivity.py` | Descriptive name, safe default paths, shared authored reference lists; optional local CSV overrides preserved |
| `final_sensitivity_package/sensitivity_common.py` | `src/sensitivity_common.py` | Copied unchanged |
| `thesis_outputs/fig_coefficient.png` | `figures/coefficient_estimates.png` | Copied unchanged after review |
| `thesis_outputs/fig_skill.png` | `figures/skill_group_slopes.png` | Copied unchanged after review |

New files provide path configuration, documentation, the MIT code license,
Git exclusions, and a standard-library public-release guard. Research progress
messages and numerical diagnostics remain because they help review long runs.
The large sequential analysis was not aggressively refactored.

## Exclusions and review decisions

All 20 CSV files are excluded, including raw-score archives, reconstructed and
head–tail scores, validation advertisements and ratings, aggregate result tables,
and the two auxiliary anchor lists. The latter exactly duplicate the authored
lists already in Python; their strings and ordering were verified before sharing
those lists between the scoring scripts. No sampled advertisements or invented
dataset were added.

The replication PDF, generated text reports, JSON metadata, manifests, source
READMEs, and two unselected figures are also excluded. Relevant aggregate findings
are documented with source filenames in `results.md`. Some older documentation
said SBERT code was missing; later reconstruction evidence resolves that statement
only with the numerical qualifications in `reproducibility.md`.

Unresolved research provenance concerns are the inconsistent recorded Python
environments, the unpinned model revision, and the nonzero score-reconstruction
differences. They are documented limitations, not evidence of a privacy leak.
No file with uncertain privacy content was selected for publication.

## Privacy controls

`.gitignore` excludes private input/output directories, tabular and database formats,
serialized arrays and models, archives, office documents, environment files,
credential files, local configuration, caches, virtual environments, build artifacts,
and editor/OS files. The two public PNGs are explicitly reviewed copies; generated
figures remain in ignored output folders.

`scripts/check_public_release.py` inspects non-ignored public files, staged Git blobs,
or reachable history. It restricts public types and locations, rejects symlinks and
files over 1 MB, checks Python syntax, scans common credential and personal-path
patterns, and rejects unexpected PNG metadata. It reports issue labels without
printing matched secrets. It does not certify arbitrary text or pixels as safe;
manual review remains necessary, and new file types require an explicit review.

## Verification evidence

- Every private text/code/CSV/JSON file was scanned locally for credential patterns,
  URLs, database connections, and personal paths without printing matches. The
  replication PDF was extracted and reviewed locally. No secrets were found.
- All four source figures were visually inspected. The two selected PNGs contain
  aggregate estimates, no advertisement records or company IDs, and only Matplotlib
  software and image-resolution metadata.
- A normalised syntax-tree comparison passed for all five copied scripts. The
  explicitly allowed differences were documentation, path configuration, directory
  creation, imports supporting those changes, and loading identical anchor lists.
  Analytical expressions, sample rules, numerical settings, and model formulas
  remain unchanged.
- All Python source files compile. All three command-line parsers load successfully
  with `--help`, without loading data or downloading model weights.
- The release guard was checked against forbidden file types, oversized content,
  personal paths, credential tokens, unreviewed notebooks, and invalid images.
  Eighteen representative private/cache paths were confirmed ignored by Git.
- Documentation links resolve. The original README-only Git history was inspected.
  All 48 private file hashes were checked unchanged after preparation, and the
  private project still has no Git directory.

Full estimation and embedding reconstruction were not rerun because the required
raw dataset is absent. Existing archive diagnostics are clearly labelled as archived
evidence. Git index checks and manual staged-diff review are performed before the
release commit; neither datasets nor analysis runtime outputs belong in that commit.
