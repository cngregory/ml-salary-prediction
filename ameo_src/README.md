# AMEO: Does Big Five personality add salary-prediction value?

This standalone extension leaves your synthetic-data scripts unchanged. It tests incremental predictive value, without assuming personality must help. Big Five personality is a related construct, not an interchangeable measurement of all soft skills.

## Run from your existing project

1. Copy `ameo_pipeline.py` into your project root.
2. Download **train.xlsx** and **data_description_doc.xlsx** from https://zenodo.org/records/45735 and put them in `data/ameo/`. Keep the source files unchanged. The script uses labeled train.xlsx and creates its own holdout; do not combine it with the challenge test.xlsx.
3. Install dependencies and run:

```bash
python -m pip install 'numpy>=1.24' 'pandas>=2.0' 'scikit-learn>=1.2' 'openpyxl>=3.1'
python ameo_pipeline.py --data data/ameo/train.xlsx --out results/ameo --threshold-inr 300000
```

Annual Salary is CTC in INR, not US dollars. INR 300,000 (3 lakh/year) is a fixed illustrative cutoff chosen before model fitting, not an optimized economic definition of high salary. Confirm this choice with your team before the definitive experiment. Changing it changes the question. Do not tune the cutoff based on holdout scores. All displayed SVM scores are decision margins, not probabilities. Default model decision thresholds are used without holdout tuning.

## Comparisons

| Set | Predictors | Purpose |
| --- | --- | --- |
| A | 10th %, 12th %, college GPA | Academic reference |
| B | A + college tier, city tier, graduation year, degree, specialization, state | Academic and education context |
| B + Big Five | B + five personality scores | Personality gain without aptitude controls |
| C | B + English, Logical, Quant and optional technical/domain scores | Stronger non-personality reference |
| C + Big Five | C + five personality scores | Primary controlled personality comparison |

The decisive comparisons are **B + Big Five minus B** and **C + Big Five minus C**, for the same classifier and split. Field/specialization and all background variables stay fixed within each pair. AMEO does not provide equivalents of your synthetic internship/project/certification variables, so assessment scores are a distinct feature domain, not substitutes for work experience.

## Method and cleaning

- Logistic regression, depth-5 random forest (200 trees), and RBF SVM; fixed settings, no tuning.
- Shared 80/20 stratified split, seed 42; identical five training-only CV folds for every comparison.
- Imputation, numeric standardization, and categorical encoding are fit within each training fold. The holdout never fits preprocessing or selects the full-model family.
- Source spelling is preserved: `collegeGPA`, `nueroticism`, `openess_to_experience`. Common corrected spellings are accepted as aliases.
- Only optional assessment `-1` values become missing. Negative Big Five values remain valid standardized scores. No global `replace(-1, ...)` is used.
- Nonpositive/invalid salaries are excluded and counted. Academic scores <=0 are missing. GPA values >0 and <=10 are multiplied by ten, a documented approximation for mixed self-reported scales; original values are not overwritten in the source. This cannot make relative and absolute grading systems equivalent.
- Exact duplicate rows are removed; remaining missing or repeated IDs stop execution, preventing candidate overlap between partitions. Different IDs with similar profiles remain separate observations.
- Identifiers, job title, job city, joining/leaving dates, DOB and gender are excluded. College ID is excluded; institutional generalization is not established by this candidate-level split. College tier is a supplied aggregate assessment proxy, not a pure academic grade.
- Missingness indicators are included for numeric fields; optional test nonparticipation can carry information itself. Joint personality permutation also moves its missingness pattern.

## Output files

- `ablation_results.csv`: CV means/standard deviations and holdout F1, AUC, accuracy, balanced accuracy.
- `bigfive_paired_deltas.csv`: same-model personality gains and paired class-stratified bootstrap 95% percentile intervals (1,000 draws).
- `cv_fold_scores.csv`: paired fold results; fold standard deviations are descriptive, not confidence intervals or independent-sample tests.
- `group_permutation_importance.csv`: holdout AUC loss when groups are jointly shuffled, plus individual personality traits. Larger positive loss means the fitted model relied more on that group. Negative values are possible. Permutation standard deviation is not a sampling confidence interval.
- `holdout_predictions.csv`: IDs, annual salaries, predictions, scores, FP/FN flags for error review.
- `split_manifest.csv`: reproducible candidate partitions.
- `run_metadata.json`: source hash, cleaning/missingness counts, cutoff, versions, class balance, majority-class dummy baseline and selected full-model family.

Group importance is measured for the full-model family with highest **training CV AUC**. It is not Gini importance and is not a percentage of salary explained. Group sizes differ, correlated predictors can substitute for one another, and group importances do not add up to 100%.

## Reading the result

Start with C + Big Five minus C. Positive F1/AUC deltas mean personality improved these predictions. If a bootstrap interval crosses zero, this holdout does not clearly distinguish improvement from deterioration. Intervals condition on this split and these fitted models; they do not capture retraining or model-selection uncertainty and are not multiple-comparison-adjusted tests. All holdout comparisons are exploratory; avoid repeatedly changing models after inspecting them.

A useful report sentence is: “Adding Big Five scores changed holdout AUC by ___ and F1 by ___ relative to the same model with academic, context, and assessment features.” This measures association and predictive utility, not a causal salary effect. AMEO surveyed AMCAT takers about employment outcomes; response selection and unmeasured variables limit generalization. It is an Indian engineering-graduate sample, not evidence about all graduates or today's US salaries.

Do not compare raw Gini percentages against the synthetic model to claim personality matters more in one population. Compare matched ablation gains, use comparable importance methods, and describe dataset/measurement differences. A continuous log-salary regression sensitivity analysis and college-disjoint validation would strengthen follow-up work; this script keeps classification aligned with your existing project.

## Corrections to the synthetic write-up

- Your Set C jointly adds soft skills, networking AND field of study. Existing gains do not isolate personality or soft skills alone.
- 0.9615 to 0.9714 is **0.99 percentage points** of F1 (about 1.03% relative).
- Identical observed predictors with different salary labels do not prove generator noise. They can reflect omitted predictors or stochastic outcomes. Call these inconsistent observed outcomes unless generation rules confirm noise.
- A strong academic baseline is a reference point, not a predictive ceiling.

## Verified run

`verified_run/` contains outputs generated here using the original Zenodo train.xlsx and the default cutoff/seed. This is a completed exploratory run, not a forecast that personality should improve performance. To reproduce, use the source hash and package versions in run_metadata.json. No raw workbook is bundled. The source data dictionary was inspected to verify annual CTC units, mixed GPA scales, personality standardization, and optional assessment missing codes.

Source: Nisar, Srikant, Aggarwal. AMEO – Aspiring Minds' Employment Outcomes 2015. DOI: https://doi.org/10.5281/zenodo.45735
