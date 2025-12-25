import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

INPUT_PATH = "data/processed/diabetes_processed.csv"
OUTPUT_PATH = "data/processed/diabetes_with_clusters.csv"

CLUSTER_FEATURES = [
    "Age",
    "BMI",
    "HighBP",
    "HighChol",
    "GenHlth",
    "PhysHlth"
]

def main():
    df = pd.read_csv(INPUT_PATH)

    X = df[CLUSTER_FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42)
    df["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)

    df.to_csv(OUTPUT_PATH, index=False)
    print("Clusters added successfully")

if __name__ == "__main__":
    main()
