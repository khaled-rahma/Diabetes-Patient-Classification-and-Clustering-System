
# app/main_streamlit_clustering_full_with_index_pca.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
from st_aggrid import AgGrid, GridOptionsBuilder
from sklearn.decomposition import PCA

# =========================
# Paths
# =========================
CLUSTER_DATA_PATH = r"C:\Users\khaled-rahma\Desktop\test\Diabetes-Patient-Classification-and-Clustering-System\data\processed\diabetes_clustering.csv"

st.set_page_config(page_title="Diabetes Patient Clustering", layout="wide")

st.title("🩺 Diabetes Patients Clustering Dashboard")
st.markdown("""
هذا التطبيق يعرض نتائج **التجميع غير الموجه للمرضى** باستخدام K-means مع Elbow Method تلقائيًا،
ويعرض المجموعات ثنائي الأبعاد باستخدام PCA.
""")

# =========================
# Load data
# =========================
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(CLUSTER_DATA_PATH)

# =========================
# Select features for clustering
# =========================
clustering_features = ["Age", "BMI", "HighBP", "HighChol", "PhysHlth"]
X = df[clustering_features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# Determine best K using Elbow Method
# =========================
inertia = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

knee = KneeLocator(K_range, inertia, curve="convex", direction="decreasing")
best_k = knee.knee

st.subheader("Elbow Method")
fig, ax = plt.subplots(figsize=(8,5))

# رسم كل النقاط مع marker
ax.plot(K_range, inertia, marker='o', linestyle='-', color='blue', label='Inertia')

# خط عمودي عند best_k
ax.axvline(best_k, color='red', linestyle='--', label=f'Best K = {best_k}')

# تظليل المنطقة بعد best_k لإظهار الاستقرار
ax.fill_between(K_range, inertia, max(inertia), where=[k>=best_k for k in K_range],
                color='red', alpha=0.1)

ax.set_xlabel("Number of clusters (K)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method for Optimal K")
ax.legend()
st.pyplot(fig)

st.success(f"✅ العدد المثالي للمجموعات تم تحديده تلقائيًا: {best_k}")

# =========================
# Apply K-means clustering
# =========================
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

# =========================
# Assign cluster labels
# =========================
cluster_names = {
    0: "Healthy / Low Risk",
    1: "Overweight / Prediabetic",
    2: "Young / High BMI",
    3: "Elderly / Hypertensive",
    4: "Diabetic / High Metabolic Risk"
}
df['Cluster_Label'] = df['Cluster_KMeans'].map(cluster_names)

# =========================
# Cluster Overview
# =========================
st.subheader("Cluster Counts")
st.dataframe(df['Cluster_Label'].value_counts())

st.subheader("Cluster Summary Statistics")
st.dataframe(df.groupby('Cluster_Label')[clustering_features].mean().round(2))

# =========================
# Boxplots for each feature
# =========================
st.subheader("Feature Distributions per Cluster")
sample_df = df.sample(n=10000, random_state=42) if len(df) > 20000 else df
for feature in clustering_features:
    fig, ax = plt.subplots(figsize=(7,5))
    sns.boxplot(x='Cluster_Label', y=feature, data=sample_df, palette="Set2", ax=ax)
    ax.set_title(f"{feature} distribution per Cluster")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    st.pyplot(fig)

# =========================
# Bar chart of mean values per cluster
# =========================
st.subheader("Mean Feature Values per Cluster")
cluster_means = df.groupby('Cluster_Label')[clustering_features].mean()
fig, ax = plt.subplots(figsize=(10,6))
cluster_means.plot(kind='bar', colormap='Set3', ax=ax)
ax.set_ylabel("Standardized Value")
ax.set_title("Average Feature Values per Cluster")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
st.pyplot(fig)

# =========================
# PCA 2D Visualization
# =========================
st.subheader("PCA 2D Projection of Clusters")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:,0]
df['PCA2'] = X_pca[:,1]

fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(
    x='PCA1', y='PCA2',
    hue='Cluster_Label',
    palette='Set2',
    data=df,
    alpha=0.7,
    s=50
)
ax.set_title("2D PCA Projection of Clusters")
st.pyplot(fig)

# =========================
# Filter by Cluster with full table using AgGrid + Index
# =========================
st.subheader("Filter Patients by Cluster")
selected_cluster = st.selectbox("اختر مجموعة لعرض المرضى فيها:", df['Cluster_Label'].unique())

filtered_df = df[df['Cluster_Label'] == selected_cluster].copy()
filtered_df.insert(0, "Index", range(1, len(filtered_df)+1))

st.write(f"عدد المرضى في المجموعة '{selected_cluster}': {len(filtered_df)}")

gb = GridOptionsBuilder.from_dataframe(filtered_df)
gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=1000)
gb.configure_default_column(filterable=True, sortable=True)
gridOptions = gb.build()

AgGrid(filtered_df, gridOptions=gridOptions, height=600, fit_columns_on_grid_load=True)

# =========================
# Export filtered cluster
# =========================
st.markdown("### تصدير بيانات المجموعة المحددة")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ تحميل CSV",
    data=csv,
    file_name=f"{selected_cluster.replace(' ', '_')}_patients.csv",
    mime="text/csv"
)

st.success("✅ يمكنك الآن تصدير بيانات أي مجموعة بسهولة")
