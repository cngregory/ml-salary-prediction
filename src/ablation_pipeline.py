import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

SEED = 42


def run_ablation_study(data_path: str):
    """
    Step 3: Systematic Feature Ablation across Academic, Experiential,
    and Soft-Skill feature sets.

    Prevents data leakage by fitting transformers strictly inside Pipeline
    objects during 5-fold CV.
    """
    df = pd.read_csv(data_path)
    df['High_Salary'] = (df['Starting_Salary'] >= 75000).astype(int)

    # Define Feature Subsets
    academic_num = ['High_School_GPA', 'SAT_Score', 'University_GPA']
    experiential_num = [
        'Internships_Completed', 'Projects_Completed', 'Certifications'
    ]
    softskills_num = ['Soft_Skills_Score', 'Networking_Score']
    categorical_cols = ['Field_of_Study']

    feature_sets = {
        'Set A (Academic Only)': (academic_num, []),
        'Set B (Academic + Experiential)': (
            academic_num + experiential_num, []
        ),
        'Set C (Full Model: Acad + Exp + SoftSkills)': (
            academic_num + experiential_num + softskills_num, categorical_cols
        ),
    }

    models = {
        'Logistic Regression (L2)': LogisticRegression(
            random_state=SEED, max_iter=1000
        ),
        'Random Forest': RandomForestClassifier(
            random_state=SEED, n_estimators=100, max_depth=5
        ),
        'SVM (RBF Kernel)': SVC(random_state=SEED, probability=True),
    }

    # Stratified Train-Test Split (matching baseline split)
    y = df['High_Salary']
    X_full = df.drop(columns=['Starting_Salary', 'High_Salary', 'Student_ID'])
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=0.20, random_state=SEED, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []

    print("================================================================================")
    print(" RUNNING FEATURE ABLATION STUDY (5-FOLD CV) ")
    print("================================================================================\n")

    for model_name, clf in models.items():
        for set_name, (num_cols, cat_cols) in feature_sets.items():
            # Construct preprocessing column transformer
            transformers = [('num', StandardScaler(), num_cols)]
            if cat_cols:
                transformers.append((
                    'cat',
                    OneHotEncoder(drop='first', handle_unknown='ignore'),
                    cat_cols,
                ))

            preprocessor = ColumnTransformer(transformers=transformers)
            pipeline = Pipeline([('prep', preprocessor), ('clf', clf)])

            # Select relevant columns
            cols_to_keep = num_cols + cat_cols
            X_tr = X_train_full[cols_to_keep]
            X_te = X_test_full[cols_to_keep]

            # 5-Fold Cross Validation
            cv_scores = cross_validate(
                pipeline, X_tr, y_train, cv=cv,
                scoring=['f1', 'roc_auc', 'accuracy'],
            )
            cv_f1_mean = cv_scores['test_f1'].mean()
            cv_auc_mean = cv_scores['test_roc_auc'].mean()

            # Evaluate on Holdout Test Set
            pipeline.fit(X_tr, y_train)
            y_pred = pipeline.predict(X_te)
            y_proba = pipeline.predict_proba(X_te)[:, 1]

            test_f1 = f1_score(y_test, y_pred)
            test_auc = roc_auc_score(y_test, y_proba)

            results.append({
                'Model': model_name,
                'Feature Set': set_name,
                'CV Mean F1': round(cv_f1_mean, 4),
                'CV Mean AUC': round(cv_auc_mean, 4),
                'Test F1': round(test_f1, 4),
                'Test AUC': round(test_auc, 4),
            })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "education_career_success.csv")
    run_ablation_study(DATA_PATH)