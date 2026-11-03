import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

# Set global random seed for reproducibility.
SEED = 42


def load_and_preprocess_data(data_path: str):
    """Ingest data, construct High_Salary (>= $75k), and split 80/20."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}. Please check file location."
        )

    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully. Total records: {len(df)}")

    # Target variable: >= $75,000 versus < $75,000.
    df['High_Salary'] = (df['Starting_Salary'] >= 75000).astype(int)

    # Academic features for the baseline model.
    academic_features = ['High_School_GPA', 'SAT_Score', 'University_GPA']
    X = df[academic_features]
    y = df['High_Salary']

    # Stratification preserves class proportions; the test set stays untouched.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    print(f"Train set size: {X_train.shape} samples")
    print(f"Test set size: {X_test.shape} samples (Holdout)")
    print(
        f"Target Distribution (Train): "
        f"{y_train.value_counts(normalize=True).to_dict()}\n"
    )
    return X_train, X_test, y_train, y_test


def evaluate_academic_baseline(X_train, X_test, y_train, y_test):
    """Evaluate academic logistic regression with 5-fold CV and a holdout set."""
    # Fit the scaler only on each CV training fold to prevent data leakage.
    baseline_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=SEED)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    print("--- 1. RUNNING 5-FOLD CROSS-VALIDATION ON TRAINING SET ---")
    cv_results = cross_validate(
        baseline_pipeline, X_train, y_train, cv=cv, scoring=scoring_metrics
    )

    for metric in scoring_metrics:
        mean_val = cv_results[f'test_{metric}'].mean()
        std_val = cv_results[f'test_{metric}'].std()
        print(f"CV Mean {metric.upper():<10}: {mean_val:.4f} (+/- {std_val:.4f})")

    # Train on the full training set and evaluate on the holdout test set.
    baseline_pipeline.fit(X_train, y_train)
    y_pred = baseline_pipeline.predict(X_test)
    y_proba = baseline_pipeline.predict_proba(X_test)[:, 1]

    print("\n--- 2. HOLDOUT TEST SET PERFORMANCE (BASELINE TO BEAT) ---")
    print(f"Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test F1-Score : {f1_score(y_test, y_pred):.4f}")
    print(f"Test ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['< $75k', '>= $75k']))
    return baseline_pipeline

if __name__ == "__main__":
    # Update this path to match your local CSV location.
    DATA_PATH = os.path.join("data", "education_career_success.csv")

    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)
    baseline_model = evaluate_academic_baseline(X_train, X_test, y_train, y_test)