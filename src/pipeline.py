# src/integrated_pipeline.py

import pandas as pd
import numpy as np
import joblib

from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =========================
# Paths
# =========================
PROCESSED_DATA_PATH = "data/processed/diabetes_processed.csv"
FINAL_DATA_PATH     = "data/processed/diabetes_with_clusters.csv"
MODEL_DIR           = "models/"

# =========================
# Features
# =========================
CLUSTER_FEATURES = [
    "Age", "BMI", "HighBP", "HighChol", "GenHlth", "PhysHlth"
]

CLASSIFICATION_FEATURES = CLUSTER_FEATURES + ["Cluster_KMeans"]

TARGETS = {
    "Diabetes": "Diabetes_012",
    "HeartDisease": "HeartDiseaseorAttack",
    "Stroke": "Stroke"
}

# =========================
# 5.1 Clustering Integration
# =========================
def run_clustering(df):
    print("🔹 Running KMeans clustering")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CLUSTER_FEATURES])

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )
    df["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)

    # Save models
    joblib.dump(kmeans, MODEL_DIR + "kmeans_model.joblib")
    joblib.dump(scaler, MODEL_DIR + "cluster_scaler.joblib")

    print("✅ Clustering completed and saved")

    return df

# =========================
# 5.2 Supervised + Tuning
# =========================
def run_supervised(df):
    results = []

    X = df[CLASSIFICATION_FEATURES]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5]
    }

    for task, target in TARGETS.items():
        print(f"\n🔹 Training for target: {task}")

        y = df[target]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            ))
        ])

        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=3,                      # بدل 5
            scoring="roc_auc_ovr",     # لأن multiclass
            n_jobs=1,                  # مهم جدًا
            verbose=1
        )


        grid.fit(X, y)

        print(f"✅ Best ROC-AUC: {grid.best_score_:.3f}")
        print(f"✅ Best Params: {grid.best_params_}")

        # Save best model
        model_path = MODEL_DIR + f"best_model_{task}.joblib"
        joblib.dump(grid.best_estimator_, model_path)

        results.append({
            "Task": task,
            "Best_ROC_AUC": grid.best_score_
        })

    return pd.DataFrame(results)

# =========================
# Main Pipeline
# =========================
def main():
    print("======================================")
    print(" Integrated Pipeline Started")
    print("======================================")

    df = pd.read_csv(PROCESSED_DATA_PATH)

    # 5.1 Clustering
    df = run_clustering(df)

    # Save final dataset
    df.to_csv(FINAL_DATA_PATH, index=False)
    print(f"📁 Dataset saved: {FINAL_DATA_PATH}")

    # 5.2 + 5.3 Supervised learning
    results = run_supervised(df)

    print("\n======================================")
    print(" Final Integrated Results")
    print("======================================")
    print(results)

    print("\n✅ All models saved successfully")

if __name__ == "__main__":
    main()
