import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SEED = 42


def run_error_analysis(data_path: str):
    df = pd.read_csv(data_path)
    df['High_Salary'] = (df['Starting_Salary'] >= 75000).astype(int)

    academic_num = ['High_School_GPA', 'SAT_Score', 'University_GPA']
    experiential_num = [
        'Internships_Completed', 'Projects_Completed', 'Certifications'
    ]
    softskills_num = ['Soft_Skills_Score', 'Networking_Score']
    categorical_cols = ['Field_of_Study']

    num_cols = academic_num + experiential_num + softskills_num
    all_features = num_cols + categorical_cols

    y = df['High_Salary']
    X = df[all_features]
    student_ids = df['Student_ID']

    # Stratified Split (matching exact indices)
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, student_ids, test_size=0.20, random_state=SEED, stratify=y
    )

    # Preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        (
            'cat',
            OneHotEncoder(drop='first', handle_unknown='ignore'),
            categorical_cols,
        ),
    ])

    # Fit Random Forest Full Model
    rf_pipeline = Pipeline([
        ('prep', preprocessor),
        (
            'clf',
            RandomForestClassifier(
                random_state=SEED, n_estimators=100, max_depth=5
            ),
        ),
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred = rf_pipeline.predict(X_test)
    y_proba = rf_pipeline.predict_proba(X_test)[:, 1]

    # 1. EXTRACT MISCLASSIFIED RECORDS
    test_analysis_df = X_test.copy()
    test_analysis_df['Student_ID'] = ids_test
    test_analysis_df['Actual_High_Salary'] = y_test
    test_analysis_df['Predicted_High_Salary'] = y_pred
    test_analysis_df['Predicted_Probability'] = np.round(y_proba, 4)

    misclassified = test_analysis_df[
        test_analysis_df['Actual_High_Salary']
        != test_analysis_df['Predicted_High_Salary']
    ]

    print("================================================================================")
    print(
        f" STEP 4: ERROR ANALYSIS — MISCLASSIFIED TEST RECORDS "
        f"({len(misclassified)} total)"
    )
    print("================================================================================\n")

    if len(misclassified) > 0:
        print(misclassified[[
            'Student_ID',
            'University_GPA',
            'SAT_Score',
            'Soft_Skills_Score',
            'Internships_Completed',
            'Actual_High_Salary',
            'Predicted_High_Salary',
            'Predicted_Probability',
        ]].to_string(index=False))
    else:
        print("No misclassified records found on the test set!")

    # 2. FEATURE IMPORTANCE RANKING
    ohe_categories = (
        rf_pipeline.named_steps['prep']
        .named_transformers_['cat']
        .get_feature_names_out(categorical_cols)
    )
    feature_names = num_cols + list(ohe_categories)
    importances = rf_pipeline.named_steps['clf'].feature_importances_

    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
    })
    fi_df = fi_df.sort_values(
        by='Importance', ascending=False
    ).reset_index(drop=True)

    print("\n================================================================================")
    print(" RANDOM FOREST FEATURE IMPORTANCE RANKING ")
    print("================================================================================\n")
    print(fi_df.to_string(index=False))

    return misclassified, fi_df


if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "education_career_success.csv")
    run_error_analysis(DATA_PATH)