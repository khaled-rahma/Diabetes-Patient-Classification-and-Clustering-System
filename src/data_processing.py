import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_DATA_PATH = "data/raw/data.csv"
PROCESSED_DATA_PATH = "data/processed/diabetes_processed.csv"
CLUSTER_DATA_PATH = "data/processed/diabetes_clustering.csv"


def cap_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower, upper)
    return df


def main():
    df = pd.read_csv(RAW_DATA_PATH)

    outlier_columns = ["BMI", "PhysHlth", "MentHlth"]
    for col in outlier_columns:
        df = cap_outliers_iqr(df, col)

    clustering_features = [
        "Age",
        "BMI",
        "HighBP",
        "HighChol",
        "GenHlth",
        "PhysHlth"
    ]

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[clustering_features] = scaler.fit_transform(
        df_scaled[clustering_features]
    )

    df_scaled.to_csv(PROCESSED_DATA_PATH, index=False)
    df_scaled[clustering_features].to_csv(CLUSTER_DATA_PATH, index=False)


if __name__ == "__main__":
    main()
