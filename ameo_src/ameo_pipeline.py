"""AMEO paired Big Five ablation. See README.md before interpreting results."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ACADEMIC = ['10percentage', '12percentage', 'collegeGPA']
CONTEXT_NUM = ['CollegeTier', 'CollegeCityTier', 'GraduationYear']
CONTEXT_CAT = ['Degree', 'Specialization', 'CollegeState']
ASSESSMENTS = ['English', 'Logical', 'Quant', 'Domain', 'ComputerProgramming',
               'ElectronicsAndSemicon', 'ComputerScience', 'MechanicalEngg',
               'ElectricalEngg', 'TelecomEngg', 'CivilEngg']
BIG5 = ['conscientiousness', 'agreeableness', 'extraversion',
        'nueroticism', 'openess_to_experience']  # Original source spellings.
SETS = {
    'A_academic': (ACADEMIC, []),
    'B_academic_context': (ACADEMIC + CONTEXT_NUM, CONTEXT_CAT),
    'B_plus_BigFive': (ACADEMIC + CONTEXT_NUM + BIG5, CONTEXT_CAT),
    'C_academic_context_assessments': (ACADEMIC + CONTEXT_NUM + ASSESSMENTS, CONTEXT_CAT),
    'C_plus_BigFive': (ACADEMIC + CONTEXT_NUM + ASSESSMENTS + BIG5, CONTEXT_CAT),
}


def load_data(path):
    df = pd.read_csv(path) if Path(path).suffix.lower() == '.csv' else pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()
    aliases = {'CollegeGPA': 'collegeGPA', 'neuroticism': 'nueroticism',
               'openness_to_experience': 'openess_to_experience'}
    df = df.rename(columns={k: v for k, v in aliases.items() if v not in df.columns})
    required = ['ID', 'Salary'] + list(dict.fromkeys(
        c for nums, cats in SETS.values() for c in nums + cats))
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}. Use labeled Zenodo train.xlsx, not test.xlsx.')
    df = df.drop_duplicates().copy()
    if df.ID.isna().any() or df.ID.duplicated().any():
        raise ValueError('Missing/repeated candidate IDs. Resolve before splitting.')
    df['Salary'] = pd.to_numeric(df.Salary, errors='coerce')
    valid = np.isfinite(df.Salary) & (df.Salary > 0)
    audit = {'input_after_exact_dedup': len(df), 'invalid_salary_rows_removed': int((~valid).sum())}
    df = df.loc[valid].reset_index(drop=True)
    numeric = list(dict.fromkeys(c for nums, _ in SETS.values() for c in nums))
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    # Negative personality z-scores are VALID. Only optional assessment -1 is missing.
    for c in ASSESSMENTS:
        df[c] = df[c].mask(df[c] == -1)
    for c in ACADEMIC:
        df[c] = df[c].mask(df[c] <= 0)
    # Approximate conversion of self-reported 10-point GPAs; do not divide percentages.
    gpa_mask = df.collegeGPA.le(10) & df.collegeGPA.gt(0)
    audit['gpa_rows_multiplied_by_10'] = int(gpa_mask.sum())
    df.loc[gpa_mask, 'collegeGPA'] *= 10
    for c in CONTEXT_NUM:
        df[c] = df[c].mask(df[c] < 0)
    df['GraduationYear'] = df.GraduationYear.mask(df.GraduationYear == 0)
    for c in CONTEXT_CAT:
        df[c] = df[c].map(lambda v: str(v).strip().lower() if pd.notna(v) else np.nan)
        df[c] = df[c].replace({'': np.nan, '-1': np.nan})
    audit['retained_rows'] = len(df)
    audit['missing_by_feature'] = df[numeric + CONTEXT_CAT].isna().sum().to_dict()
    return df, audit


def pipeline(model, nums, cats):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True,
                                                keep_empty_features=True)),
                        ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent',
                                                    keep_empty_features=True)),
                            ('encode', OneHotEncoder(handle_unknown='ignore'))])
    parts = [('numeric', numeric, nums)]
    if cats:
        parts.append(('categorical', categorical, cats))
    return Pipeline([('preprocess', ColumnTransformer(parts)), ('model', clone(model))])


def score_vector(pipe, X):
    return (pipe.predict_proba(X)[:, 1] if hasattr(pipe, 'predict_proba')
            else pipe.decision_function(X))


def metrics(y, pred, score):
    return {'accuracy': accuracy_score(y, pred),
            'balanced_accuracy': balanced_accuracy_score(y, pred),
            'f1': f1_score(y, pred, zero_division=0), 'auc': roc_auc_score(y, score)}


def bootstrap_delta(y, base, full, seed, repeats):
    """Paired, class-stratified holdout bootstrap; fixed fitted models, not refit uncertainty."""
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y == v) for v in [0, 1]]
    draws = []
    for _ in range(repeats):
        ix = np.concatenate([rng.choice(c, len(c), replace=True) for c in classes])
        draws.append([f1_score(y[ix], full[0][ix], zero_division=0)
                      - f1_score(y[ix], base[0][ix], zero_division=0),
                      roc_auc_score(y[ix], full[1][ix]) - roc_auc_score(y[ix], base[1][ix])])
    return np.quantile(draws, [.025, .975], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Labeled Zenodo train.xlsx or matching CSV')
    parser.add_argument('--out', default='results/ameo')
    parser.add_argument('--threshold-inr', type=float, default=300000,
                        help='Predeclared annual CTC cutoff; default INR 300000, NOT USD')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--permutations', type=int, default=30)
    args = parser.parse_args()
    if args.threshold_inr <= 0 or args.bootstrap < 1 or args.permutations < 1:
        parser.error('Threshold and repeat counts must be positive.')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, audit = load_data(args.data)
    y = (df.Salary >= args.threshold_inr).astype(int)
    if y.nunique() != 2 or y.value_counts().min() < 10:
        raise ValueError('Cutoff leaves insufficient members of both classes for stratified CV.')
    tr, te = train_test_split(np.arange(len(df)), test_size=.2, random_state=args.seed, stratify=y)
    Xtr, Xte, ytr, yte = df.iloc[tr], df.iloc[te], y.iloc[tr], y.iloc[te]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=args.seed).split(Xtr, ytr))
    models = {'LogisticRegression': LogisticRegression(max_iter=3000, random_state=args.seed),
              'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=5,
                                                     n_jobs=1, random_state=args.seed),
              'SVM_RBF': SVC(random_state=args.seed)}
    rows, cvrows, predictions, fitted, saved = [], [], [], {}, {}
    for name, model in models.items():
        for setname, (nums, cats) in SETS.items():
            print(f'{name}: {setname}', flush=True)
            pipe = pipeline(model, nums, cats)
            cv = cross_validate(pipe, Xtr, ytr, cv=folds,
                                scoring={'f1': 'f1', 'auc': 'roc_auc', 'accuracy': 'accuracy'},
                                error_score='raise', n_jobs=1)
            for fold in range(5):
                cvrows.append({'model': name, 'set': setname, 'fold': fold,
                               **{m: cv[f'test_{m}'][fold] for m in ['f1', 'auc', 'accuracy']}})
            pipe.fit(Xtr, ytr)
            pred, score = pipe.predict(Xte), score_vector(pipe, Xte)
            rows.append({'model': name, 'set': setname,
                         **{f'cv_{m}_{stat}': getattr(cv[f'test_{m}'], stat)()
                            for m in ['f1', 'auc', 'accuracy'] for stat in ['mean', 'std']},
                         **{f'test_{k}': v for k, v in metrics(yte, pred, score).items()}})
            fitted[name, setname] = pipe
            saved[name, setname] = (pred, score)
            for idx, p, s in zip(te, pred, score):
                predictions.append({'ID': df.iloc[idx].ID, 'model': name, 'set': setname,
                                    'salary_inr': df.iloc[idx].Salary, 'actual': int(y.iloc[idx]),
                                    'predicted': int(p), 'score': float(s),
                                    'error': 'FP' if p > y.iloc[idx] else 'FN' if p < y.iloc[idx] else ''})
    results = pd.DataFrame(rows)
    results.to_csv(out / 'ablation_results.csv', index=False)
    pd.DataFrame(cvrows).to_csv(out / 'cv_fold_scores.csv', index=False)
    pd.DataFrame(predictions).to_csv(out / 'holdout_predictions.csv', index=False)
    deltas = []
    for name in models:
        for base in ['B_academic_context', 'C_academic_context_assessments']:
            full = 'B_plus_BigFive' if base.startswith('B') else 'C_plus_BigFive'
            ci = bootstrap_delta(yte.to_numpy(), saved[name, base], saved[name, full],
                                 args.seed, args.bootstrap)
            b = results.query('model == @name and set == @base').iloc[0]
            f = results.query('model == @name and set == @full').iloc[0]
            deltas.append({'model': name, 'contrast': full + ' minus ' + base,
                           'delta_cv_f1': f.cv_f1_mean - b.cv_f1_mean,
                           'delta_cv_auc': f.cv_auc_mean - b.cv_auc_mean,
                           'delta_test_f1': f.test_f1 - b.test_f1,
                           'delta_test_auc': f.test_auc - b.test_auc,
                           'f1_ci_low': ci[0, 0], 'f1_ci_high': ci[1, 0],
                           'auc_ci_low': ci[0, 1], 'auc_ci_high': ci[1, 1]})
    pd.DataFrame(deltas).to_csv(out / 'bigfive_paired_deltas.csv', index=False)
    # Choose the full-model family by TRAIN CV AUC, never by holdout performance.
    winner = results.query("set == 'C_plus_BigFive'").sort_values('cv_auc_mean', ascending=False).iloc[0].model
    best = fitted[winner, 'C_plus_BigFive']
    original_auc = roc_auc_score(yte, score_vector(best, Xte))
    rng = np.random.default_rng(args.seed)
    groups = {'academic': ACADEMIC, 'context': CONTEXT_NUM + CONTEXT_CAT,
              'assessments': ASSESSMENTS, 'BigFive_joint': BIG5,
              **{c: [c] for c in BIG5}}
    importance = []
    for group, cols in groups.items():
        drops = []
        for _ in range(args.permutations):
            shuffled = Xte.copy()
            # Same row permutation preserves within-group relationships.
            shuffled[cols] = Xte[cols].iloc[rng.permutation(len(Xte))].to_numpy()
            drops.append(original_auc - roc_auc_score(yte, score_vector(best, shuffled)))
        importance.append({'model': winner, 'group': group, 'auc_drop_mean': np.mean(drops),
                           'permutation_std': np.std(drops)})
    pd.DataFrame(importance).to_csv(out / 'group_permutation_importance.csv', index=False)
    dummy = DummyClassifier(strategy='prior').fit(np.zeros((len(tr), 1)), ytr)
    audit.update({'source': 'https://zenodo.org/records/45735',
                  'sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                  'threshold_inr': args.threshold_inr, 'seed': args.seed,
                  'train_rows': len(tr), 'test_rows': len(te),
                  'train_positive_rate': float(ytr.mean()), 'test_positive_rate': float(yte.mean()),
                  'full_model_selected_by_train_cv_auc': winner,
                  'dummy_test': metrics(yte, dummy.predict(np.zeros((len(te), 1))),
                                        dummy.predict_proba(np.zeros((len(te), 1)))[:, 1]),
                  'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                               'pandas': pd.__version__, 'sklearn': sklearn.__version__},
                  'argv': sys.argv[1:]})
    pd.DataFrame({'ID': df.ID, 'split': np.where(np.isin(np.arange(len(df)), tr), 'train', 'test')}).to_csv(out / 'split_manifest.csv', index=False)
    (out / 'run_metadata.json').write_text(json.dumps(audit, indent=2))
    print(results[['model', 'set', 'cv_f1_mean', 'cv_auc_mean', 'test_f1', 'test_auc']].to_string(index=False))
    print('\nBig Five incremental value:\n', pd.DataFrame(deltas).to_string(index=False))
    print(f'\nSaved outputs to {out}')


if __name__ == '__main__':
    main()
