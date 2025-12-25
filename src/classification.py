import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_PATH = "data/processed/diabetes_with_clusters.csv"
RESULTS_PATH = "report/classification_results.csv"

FEATURES = [
    "Age",
    "BMI",
    "HighBP",
    "HighChol",
    "GenHlth",
    "PhysHlth",
    "Cluster_KMeans"
]

TARGETS = {
    "HeartDisease": "HeartDiseaseorAttack",
    "Stroke": "Stroke",
    "Diabetes": "Diabetes_012"
}

MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        random_state=42
    ),
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc"
}


def main():
    print("======================================")
    print(" Loading dataset")
    print("======================================")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")
    print("Columns:", list(df.columns))

    X = df[FEATURES]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []

    print("\n======================================")
    print(" Starting supervised learning phase")
    print("======================================")

    for complication, target in TARGETS.items():
        print(f"\n>>> Target complication: {complication}")
        y = df[target]

        print(f"Target distribution:\n{y.value_counts(normalize=True)}")

        for model_name, model in MODELS.items():
            print(f"\n  Training model: {model_name}")
            print("  Cross-validation started...")

            scores = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=SCORING,
                n_jobs=1
            )

            print("  Cross-validation completed")

            results.append({
                "Complication": complication,
                "Model": model_name,
                "Accuracy": scores["test_accuracy"].mean(),
                "Precision": scores["test_precision"].mean(),
                "Recall": scores["test_recall"].mean(),
                "ROC_AUC": scores["test_roc_auc"].mean()
            })

            print(
                f"  Results → "
                f"Accuracy={scores['test_accuracy'].mean():.3f}, "
                f"Recall={scores['test_recall'].mean():.3f}, "
                f"ROC_AUC={scores['test_roc_auc'].mean():.3f}"
            )

    print("\n======================================")
    print(" Saving results")
    print("======================================")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("Classification completed successfully")
    print("\nFinal Results (sorted by ROC-AUC):")
    print(
        results_df.sort_values(
            by=["Complication", "ROC_AUC"],
            ascending=[True, False]
        )
    )


if __name__ == "__main__":
    main()
