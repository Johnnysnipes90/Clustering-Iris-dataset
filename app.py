import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from scipy.cluster.hierarchy import dendrogram, linkage

# Sidebar Layout
st.sidebar.title("Iris Clustering Dashboard")
dataset_option = st.sidebar.selectbox("Select Dataset", ["Iris Dataset", "Upload Your Dataset"])
clean_duplicates = st.sidebar.checkbox("Remove Duplicates", value=True)
remove_outliers = st.sidebar.checkbox("Remove Outliers (IQR Method)", value=True)
scaling_method = st.sidebar.selectbox("Scaling Method", ["StandardScaler", "MinMaxScaler", "None"])
clustering_method = st.sidebar.selectbox("Clustering Algorithm", ["K-Means", "Hierarchical"])
linkage_method = st.sidebar.selectbox("Linkage Method (Hierarchical)", ["ward", "single", "complete", "average"])
n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3)

# Load Dataset
if dataset_option == "Iris Dataset":
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()

# Data Cleaning
if clean_duplicates:
    df = df.drop_duplicates()

if remove_outliers:
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Sidebar Data Preview
st.sidebar.write("Dataset Preview:")
st.sidebar.dataframe(df.head())

# Scale the Data
numeric_cols = df.select_dtypes(include=['number']).columns
if scaling_method == "StandardScaler":
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])
elif scaling_method == "MinMaxScaler":
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])
else:
    X_scaled = df[numeric_cols].values

# Clustering
if clustering_method == "K-Means":
    model = KMeans(n_clusters=n_clusters, random_state=42)
elif clustering_method == "Hierarchical":
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)

df['cluster'] = model.fit_predict(X_scaled)

# Silhouette Score
silhouette_avg = silhouette_score(X_scaled, df['cluster'])
st.sidebar.write(f"Silhouette Score: {silhouette_avg:.3f}")

# Tabs for Main Sections
tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "Clustering Results", "Visualizations", "Model Management"])

# Dataset Overview Tab
with tab1:
    st.header("Dataset Overview")
    st.dataframe(df)
    st.write("Basic Statistics:")
    st.write(df.describe())

# Clustering Results Tab
with tab2:
    st.header("Clustering Results")
    st.write(f"Clustering Method: {clustering_method}")
    st.write(f"Number of Clusters: {n_clusters}")
    st.write("Cluster Assignments:")
    st.dataframe(df[['cluster'] + numeric_cols.tolist()])

# Visualizations Tab
with tab3:
    st.header("Visualizations")
    # PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis', s=50)
    plt.title(f"PCA - {clustering_method} Clustering Results")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Cluster")
    st.pyplot(plt)

    # Dendrogram for Hierarchical Clustering
    if clustering_method == "Hierarchical":
        linkage_matrix = linkage(X_scaled, method=linkage_method)
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix)
        plt.title(f"Dendrogram ({linkage_method.capitalize()} Linkage)")
        plt.xlabel("Samples")
        plt.ylabel("Distance")
        st.pyplot(plt)

# Model Management Tab
with tab4:
    st.header("Model Management")
    if st.button("Save Model"):
        joblib.dump(model, f"{clustering_method.lower()}_model.pkl")
        st.success(f"{clustering_method} model saved successfully as {clustering_method.lower()}_model.pkl.")
    uploaded_model = st.file_uploader("Load a Saved Model", type=["pkl"])
    if uploaded_model is not None:
        loaded_model = joblib.load(uploaded_model)
        st.success("Model loaded successfully!")
