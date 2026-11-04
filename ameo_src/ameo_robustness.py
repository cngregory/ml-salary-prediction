"""AMEO robustness checks: does adding Big Five improve salary prediction?

Companion to ameo_pipeline.py. It reuses the same cleaning and feature sets and adds:

  1. Repeated stratified CV (default 10x5) on all labeled rows. Each Big Five
     contrast is paired fold by fold and tested with the Nadeau-Bengio corrected
     resampled t-test (the plain t-test is too optimistic when folds overlap).
  2. Nested logistic regression with a likelihood-ratio test for the 5 Big Five
     terms, plus delta AIC/BIC, McFadden pseudo-R2 and odds ratios per 1 SD.
  3. Log-loss and Brier score in addition to AUC and F1.
  4. Sensitivity to the salary definition: three pre-declared alternative
     cutoffs (salary quantiles) and a continuous log-salary regression
     (delta CV R2, robust HC3 joint Wald test, % salary change per 1 SD).

Note: the 80/20 holdout from ameo_pipeline.py has already been looked at, so
this script re-estimates on all rows with resampling instead of a single split.

Usage (from ameo_src/):
    python ameo_robustness.py --data train.xlsx --out results/robustness
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_validate
from sklearn.svm import SVC
import statsmodels
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ameo_pipeline import BIG5, SETS, load_data, pipeline  # noqa: E402

PAIRS = [('B_academic_context', 'B_plus_BigFive'),
         ('C_academic_context_assessments', 'C_plus_BigFive')]
PRIMARY_PAIR = PAIRS[1]
# Lower is better for these; all deltas are reported as (with Big Five) - (without).
LOWER_IS_BETTER = {'log_loss', 'brier', 'mae'}


def corrected_ttest(diffs, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test on paired fold deltas."""
    diffs = np.asarray(diffs, dtype=float)
    j = len(diffs)
    mean, var = diffs.mean(), diffs.var(ddof=1)
    se = np.sqrt((1 / j + n_test / n_train) * var)
    if se == 0:
        return mean, mean, mean, 1.0
    t = mean / se
    half = stats.t.ppf(0.975, j - 1) * se
    return mean, mean - half, mean + half, 2 * stats.t.sf(abs(t), j - 1)


def cv_scores(X, y, models, set_names, splitter, scoring_for, n_jobs, label):
    rows = []
    for name, model in models.items():
        scoring = scoring_for(model)
        for s in set_names:
            nums, cats = SETS[s]
            t0 = time.time()
            cv = cross_validate(pipeline(model, nums, cats), X, y, cv=splitter,
                                scoring=scoring, n_jobs=n_jobs, error_score='raise')
            print(f'  [{label}] {name}: {s} ({time.time() - t0:.0f}s)', flush=True)
            for i in range(len(cv['fit_time'])):
                row = {'outcome': label, 'model': name, 'set': s, 'split': i}
                for m in scoring:
                    v = cv[f'test_{m}'][i]
                    row[m] = -v if m in LOWER_IS_BETTER else v  # undo sklearn's neg_ sign
                rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(scores, pairs, n_train, n_test):
    rows = []
    metrics = [c for c in scores.columns if c not in ('outcome', 'model', 'set', 'split')]
    for (label, model), g in scores.groupby(['outcome', 'model'], sort=False):
        for base, full in pairs:
            b = g[g.set == base].set_index('split')
            f = g[g.set == full].set_index('split')
            if b.empty or f.empty:
                continue
            for m in metrics:
                if b[m].isna().all():
                    continue
                d = (f[m] - b[m]).to_numpy()
                mean, lo, hi, p = corrected_ttest(d, n_train, n_test)
                improved = (d < 0) if m in LOWER_IS_BETTER else (d > 0)
                rows.append({'outcome': label, 'model': model, 'contrast': f'{full} minus {base}',
                             'metric': m, 'better_if': 'lower' if m in LOWER_IS_BETTER else 'higher',
                             'mean_without_big5': b[m].mean(), 'mean_with_big5': f[m].mean(),
                             'delta': mean, 'ci95_low': lo, 'ci95_high': hi,
                             'p_corrected': p, 'share_splits_improved': improved.mean()})
    return pd.DataFrame(rows)


def design_matrix(df, nums, cats, min_count=30):
    """Inference design: median-impute + missing flags, z-scored numerics, rare categories pooled."""
    parts = []
    for c in nums:
        col = df[c]
        if col.isna().any():
            parts.append(col.isna().astype(float).rename(f'{c}_missing'))
        col = col.fillna(col.median())
        sd = col.std()
        parts.append(((col - col.mean()) / sd if sd > 0 else col * 0).rename(c))
    for c in cats:
        col = df[c].fillna('missing')
        col = col.where(col.map(col.value_counts()) >= min_count, 'other')
        dummies = pd.get_dummies(col, prefix=c, dtype=float)
        reference = f'{c}_{col.value_counts().idxmax()}'  # most common level = reference
        parts.append(dummies.drop(columns=reference))
    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.std() > 0]
    # Binary columns with very few 1s (e.g. GraduationYear_missing for one person) cause
    # (quasi-)separation in an unpenalized logit; treat them like rare categories.
    sparse = [c for c in X.columns if X[c].isin([0, 1]).all() and min(X[c].sum(), (1 - X[c]).sum()) < min_count]
    X = X.drop(columns=sparse)
    # Big Five last so that any collinear column dropped is a control, never a trait.
    X = X[[c for c in X.columns if c not in BIG5] + [c for c in BIG5 if c in X.columns]]
    # Drop (near-)collinear controls, e.g. Degree=mca vs Specialization=computer application,
    # which otherwise leave the likelihood flat and the optimizer unconverged.
    keep = []
    for c in X.columns:
        A = np.column_stack([np.ones(len(X))] + [X[k].to_numpy() for k in keep])
        v = X[c].to_numpy()
        resid = v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
        if resid.var() / v.var() > 1e-2:   # R^2 on kept columns below 0.99
            keep.append(c)
    return sm.add_constant(X[keep]), sparse + [c for c in X.columns if c not in keep]


def nested_designs(df, pair):
    """Full design for the with-Big-Five set; base = same columns minus Big Five (strictly nested)."""
    full_set = pair[1]
    X_full, dropped = design_matrix(df, *SETS[full_set])
    big5_cols = [c for c in X_full.columns if c in BIG5 or c.replace('_missing', '') in BIG5]
    return X_full.drop(columns=big5_cols), X_full, big5_cols, dropped


def nested_logit(df, y, pair, label):
    X_base, X_full, big5_cols, dropped = nested_designs(df, pair)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rb = sm.Logit(y, X_base).fit(disp=0, maxiter=500)
        rf = sm.Logit(y, X_full).fit(disp=0, maxiter=500)
    lr = 2 * (rf.llf - rb.llf)
    dof = X_full.shape[1] - X_base.shape[1]
    test = {'outcome': label, 'contrast': f'{pair[1]} vs {pair[0]}', 'n': len(y),
            'positive_rate': float(y.mean()), 'lr_chi2': lr, 'df': dof,
            'p_value': stats.chi2.sf(lr, dof),
            'aic_without': rb.aic, 'aic_with': rf.aic, 'delta_aic': rf.aic - rb.aic,
            'bic_without': rb.bic, 'bic_with': rf.bic, 'delta_bic': rf.bic - rb.bic,
            'pseudo_r2_without': rb.prsquared, 'pseudo_r2_with': rf.prsquared,
            'delta_pseudo_r2': rf.prsquared - rb.prsquared,
            'converged': bool(rb.mle_retvals['converged'] and rf.mle_retvals['converged']),
            'collinear_controls_dropped': ';'.join(dropped)}
    ci = rf.conf_int()
    ors = [{'outcome': label, 'trait': c, 'odds_ratio_per_sd': np.exp(rf.params[c]),
            'or_ci95_low': np.exp(ci.loc[c, 0]), 'or_ci95_high': np.exp(ci.loc[c, 1]),
            'p_value_wald': rf.pvalues[c]} for c in big5_cols] if pair == PRIMARY_PAIR else []
    return test, ors


def nested_ols(df, log_salary, pair):
    X_base, X_full, big5_cols, dropped = nested_designs(df, pair)
    rb = sm.OLS(log_salary, X_base).fit(cov_type='HC3')
    rf = sm.OLS(log_salary, X_full).fit(cov_type='HC3')
    R = np.zeros((len(big5_cols), X_full.shape[1]))
    for i, c in enumerate(big5_cols):
        R[i, X_full.columns.get_loc(c)] = 1
    wald = rf.wald_test(R, scalar=True, use_f=True)
    test = {'outcome': 'log_salary', 'contrast': f'{pair[1]} vs {pair[0]}', 'n': len(log_salary),
            'r2_without': rb.rsquared, 'r2_with': rf.rsquared, 'delta_r2': rf.rsquared - rb.rsquared,
            'adj_r2_without': rb.rsquared_adj, 'adj_r2_with': rf.rsquared_adj,
            'robust_wald_F': float(wald.statistic), 'df': len(big5_cols),
            'p_value': float(wald.pvalue), 'delta_aic': rf.aic - rb.aic, 'delta_bic': rf.bic - rb.bic,
            'collinear_controls_dropped': ';'.join(dropped)}
    ci = rf.conf_int()
    coefs = [{'trait': c, 'coef_log_salary_per_sd': rf.params[c],
              'pct_salary_change_per_sd': 100 * np.expm1(rf.params[c]),
              'pct_ci95_low': 100 * np.expm1(ci.loc[c, 0]), 'pct_ci95_high': 100 * np.expm1(ci.loc[c, 1]),
              'p_value_hc3': rf.pvalues[c]} for c in big5_cols] if pair == PRIMARY_PAIR else []
    return test, coefs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, help='Labeled Zenodo train.xlsx')
    parser.add_argument('--out', default='results/robustness')
    parser.add_argument('--threshold-inr', type=float, default=300000,
                        help='Primary cutoff (same as ameo_pipeline.py); INR 300000 is the sample median')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-jobs', type=int, default=-1)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()

    df, audit = load_data(args.data)
    n = len(df)
    n_test = n // args.folds
    n_train = n - n_test
    seed = args.seed

    clf_models = {'LogisticRegression': LogisticRegression(max_iter=3000, random_state=seed),
                  'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=5, n_jobs=1,
                                                         random_state=seed),
                  'SVM_RBF': SVC(random_state=seed)}

    def clf_scoring(model):
        s = {'auc': 'roc_auc', 'f1': 'f1'}
        if hasattr(model, 'predict_proba'):  # SVC without probability=True has no calibrated scores
            s.update({'log_loss': 'neg_log_loss', 'brier': 'neg_brier_score'})
        return s

    # Pre-declared alternative cutoffs (salary quantiles), fixed before looking at any result.
    s = df.Salary
    cutoffs = {f'ge_{int(args.threshold_inr)}_primary': args.threshold_inr,
               f'not_bottom25_ge_{int(s.quantile(.25))}': s.quantile(.25),
               f'top25_ge_{int(s.quantile(.75))}': s.quantile(.75),
               f'top10_ge_{int(s.quantile(.90))}': s.quantile(.90)}

    all_sets = [x for pair in PAIRS for x in pair]
    score_frames, tests, ors = [], [], []
    for i, (label, cut) in enumerate(cutoffs.items()):
        y = (df.Salary >= cut).astype(int)
        print(f'\nCutoff {label}: positive rate {y.mean():.3f}', flush=True)
        splitter = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=seed)
        if i == 0:   # primary outcome: every model, both contrasts
            models, sets = clf_models, all_sets
        else:        # sensitivity: probabilistic models, primary contrast only
            models = {k: v for k, v in clf_models.items() if k != 'SVM_RBF'}
            sets = list(PRIMARY_PAIR)
        score_frames.append(cv_scores(df, y, models, sets, splitter, clf_scoring, args.n_jobs, label))
        for pair in (PAIRS if i == 0 else [PRIMARY_PAIR]):
            t, o = nested_logit(df, y, pair, label)
            tests.append(t)
            ors.extend(o)

    clf_scores = pd.concat(score_frames, ignore_index=True)
    clf_deltas = paired_deltas(clf_scores, PAIRS, n_train, n_test)

    print('\nContinuous outcome: log(annual CTC)', flush=True)
    log_salary = np.log(df.Salary)
    reg_models = {'LinearRegression': LinearRegression(),
                  'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=5, n_jobs=1,
                                                        random_state=seed)}
    reg_splitter = RepeatedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=seed)
    reg_scores = cv_scores(df, log_salary, reg_models, all_sets, reg_splitter,
                           lambda m: {'r2': 'r2', 'mae': 'neg_mean_absolute_error'},
                           args.n_jobs, 'log_salary')
    reg_deltas = paired_deltas(reg_scores, PAIRS, n_train, n_test)
    ols_tests, ols_coefs = [], []
    for pair in PAIRS:
        t, c = nested_ols(df, log_salary, pair)
        ols_tests.append(t)
        ols_coefs.extend(c)

    pd.concat([clf_scores, reg_scores], ignore_index=True).to_csv(out / 'repeated_cv_scores.csv', index=False)
    pd.concat([clf_deltas, reg_deltas], ignore_index=True).to_csv(out / 'repeated_cv_deltas.csv', index=False)
    pd.DataFrame(tests).to_csv(out / 'nested_logit_tests.csv', index=False)
    pd.DataFrame(ors).to_csv(out / 'bigfive_odds_ratios.csv', index=False)
    pd.DataFrame(ols_tests).to_csv(out / 'log_salary_nested_tests.csv', index=False)
    pd.DataFrame(ols_coefs).to_csv(out / 'bigfive_log_salary_coefs.csv', index=False)
    meta = {'rows': n, 'cleaning_audit': {k: v for k, v in audit.items() if k != 'missing_by_feature'},
            'cutoffs_inr': {k: float(v) for k, v in cutoffs.items()},
            'positive_rates': {k: float((df.Salary >= v).mean()) for k, v in cutoffs.items()},
            'cv': f'{args.repeats}x{args.folds} repeated (stratified for classification)',
            'seed': seed, 'ttest': 'Nadeau-Bengio corrected resampled t-test, 95% CI',
            'runtime_seconds': round(time.time() - start),
            'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                         'pandas': pd.__version__, 'sklearn': sklearn.__version__,
                         'scipy': scipy.__version__, 'statsmodels': statsmodels.__version__},
            'argv': sys.argv[1:]}
    (out / 'robustness_metadata.json').write_text(json.dumps(meta, indent=2))

    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)
    cols = ['outcome', 'model', 'metric', 'delta', 'ci95_low', 'ci95_high', 'p_corrected', 'share_splits_improved']
    primary = pd.concat([clf_deltas, reg_deltas])
    print('\n=== Repeated-CV paired deltas, C + Big Five minus C ===')
    print(primary[primary.contrast.str.startswith('C_plus')][cols].to_string(index=False))
    print('\n=== Nested logistic LR tests ===')
    print(pd.DataFrame(tests)[['outcome', 'contrast', 'lr_chi2', 'df', 'p_value', 'delta_aic',
                               'delta_bic', 'delta_pseudo_r2']].to_string(index=False))
    print('\n=== Log-salary OLS (HC3) ===')
    print(pd.DataFrame(ols_tests)[['contrast', 'delta_r2', 'robust_wald_F', 'p_value',
                                   'delta_aic', 'delta_bic']].to_string(index=False))
    print(f'\nSaved to {out}  ({meta["runtime_seconds"]}s)')


if __name__ == '__main__':
    main()
