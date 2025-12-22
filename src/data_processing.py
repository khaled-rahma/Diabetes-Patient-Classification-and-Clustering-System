import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =========================
# Paths
# =========================
RAW_DATA_PATH = r"C:\Users\khaled-rahma\Desktop\test\Diabetes-Patient-Classification-and-Clustering-System\data\raw\diabetes_012.csv"


PROCESSED_DATA_PATH = r"C:\Users\khaled-rahma\Desktop\test\Diabetes-Patient-Classification-and-Clustering-System\data\processed\diabetes_processed.csv"
CLUSTER_DATA_PATH   = r"C:\Users\khaled-rahma\Desktop\test\Diabetes-Patient-Classification-and-Clustering-System\data\processed\diabetes_clustering.csv"
TRAIN_DATA_PATH     = r"C:\Users\khaled-rahma\Desktop\test\Diabetes-Patient-Classification-and-Clustering-System\data\processed\train.csv"
TEST_DATA_PATH      = r"C:\Users\khaled-rahma\Desktop\test\Diabetes-Patient-Classification-and-Clustering-System\data\processed\test.csv"


def handle_missing_values(df):
    missing_ratio = df.isnull().mean()
    print("Missing values ratio:\n", missing_ratio[missing_ratio > 0])
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["int64", "float64"]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df


# =========================
# 2.2 Outliers using IQR
# =========================
def cap_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower, upper)
    return df


def handle_outliers(df):
    outlier_columns = ["BMI", "PhysHlth", "MentHlth"]
    for col in outlier_columns:
        df = cap_outliers_iqr(df, col)
    return df


# =========================
# 2.3 Encoding & Scaling
# =========================
def scale_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler


# =========================
# Main Pipeline
# =========================
def main():
    # Load data
    df = pd.read_csv(RAW_DATA_PATH)

    # 2.1 Missing values
    df = handle_missing_values(df)

    # 2.2 Outliers
    df = handle_outliers(df)

    # Features for clustering (Unsupervised)
    clustering_features = [
        "Age",
        "BMI",
        "HighBP",
        "HighChol",
        "GenHlth",
        "PhysHlth"
    ]

    # 2.3 Scaling
    df_scaled = df.copy()
    df_scaled, scaler = scale_features(df_scaled, clustering_features)

    # Save processed full dataset (for supervised learning)
    df_scaled.to_csv(PROCESSED_DATA_PATH, index=False)

    # Save clustering-only dataset (for K-means)
    df_scaled[clustering_features].to_csv(CLUSTER_DATA_PATH, index=False)

    # =========================
    # 2.4 Train / Test Split
    # =========================
    X = df_scaled.drop("Diabetes_012", axis=1)
    y = df_scaled["Diabetes_012"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_df = X_train.copy()
    train_df["Diabetes_012"] = y_train

    test_df = X_test.copy()
    test_df["Diabetes_012"] = y_test

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print("✅ Data processing completed successfully")


if __name__ == "__main__":
    main()