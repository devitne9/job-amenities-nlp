# Archived results and qualifications

These values were transcribed from the private package's aggregate reports and
cross-checked against its code and replication PDF. No dataset or result CSV is
distributed. The models were not rerun during public repository preparation.

## Main association

| Specification | Log-wage coefficient | Company-clustered SE | Model observations |
| --- | ---: | ---: | ---: |
| Occupation–region–month fixed effects | −0.187199 | 0.024847 | 178,173 |
| Add company fixed effects | −0.097784 | 0.014551 | 168,269 |

The preferred model's 95% interval is [−0.126305, −0.069263], calculated as
coefficient ± 1.96 × SE. Its dependent variable is the standardised amenity score.
The baseline sample contains 16,332 distinct companies; this is not a separately
verified count of firms in the preferred model's final estimation sample.

The archived economic-magnitude calculation is `−0.097784 × ln(0.90)`, or about
0.0103 score standard deviations for a 10% lower offered wage. This is an
association in language, not a wage penalty or the value of actual amenities.

## Sensitivity and measurement limitations

Removing 111 exact duplicate source rows changes the preferred coefficient from
−0.097784 to −0.097670, with 168,153 model observations. This does not change the
rounded three-decimal estimate.

The head–tail alternative yields a baseline coefficient of −0.156291 and a
preferred coefficient of −0.054363 (SE 0.016187; 95% interval
[−0.086090, −0.022637]). The sign remains negative, but the preferred magnitude is
substantially smaller. The alternative standardised score correlates 0.648015 with
the archived score; 78.68% of source advertisements exceed its 240-token window.
These results make text-window sensitivity a material limitation.

In the 200-posting stratified rating sample, Pearson correlation is 0.306,
Spearman correlation is 0.324, and linearly weighted kappa is 0.200.
The rating file included model scores and tertiles, so agreement may reflect
model-information bias. These statistics do not establish blinded validation
or classifier accuracy.

## Included figures

| Public file | Archived source | Content |
| --- | --- | --- |
| `figures/coefficient_estimates.png` | `thesis_outputs/fig_coefficient.png` | Log-wage coefficients and 95% intervals from main and robustness specifications |
| `figures/skill_group_slopes.png` | `thesis_outputs/fig_skill.png` | Implied skill-group slopes and 95% intervals from the market-cell FE interaction model |

Both images were copied unchanged after visual and metadata review. They contain
model-level estimates, not individual observations, text, or company identifiers.
The skill plot is a separate specification without company fixed effects; its
group slopes should not be described as estimates from the preferred model.

The dictionary-dimension and rating-agreement figures were reviewed but omitted
to keep the public selection small. No new charts were generated.

## Evidence within the private archive

Source reports: `thesis_outputs/RESULTS_SUMMARY.txt`,
`thesis_outputs/12_diagnostic_comparison.txt`,
`final_sensitivity_outputs/duplicate_sensitivity_report.txt`, and
`final_sensitivity_outputs/head_tail_sensitivity_report.txt`.
These filenames document provenance; they are not public download links.
The main diagnostic report records a passing comparison against the embedded
numerical targets. That is an archived check, not a new replication claim.
