#!/usr/bin/env python3
"""SmartCart customer segmentation pipeline.

Run from any working directory with: python src/smartcart_customer_segmentation.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score

RANDOM_STATE = 42
CHOSEN_K = 4  # baseline retained from the original minor project; not claimed to be optimal
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "smartcart_customers.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Input shape: {df.shape}; missing values: {int(df.isna().sum().sum())}; duplicate rows: {int(df.duplicated().sum())}")
    income_missing = int(df["Income"].isna().sum())
    income_median = float(df["Income"].median())
    df["Income"] = df["Income"].fillna(income_median)
    print(f"Income imputed: {income_missing}; median: {income_median:.2f}")

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    if df["Dt_Customer"].isna().any():
        raise ValueError("Unable to parse one or more Dt_Customer values.")
    df["Age"] = 2026 - df["Year_Birth"]
    df["Customer_Tenure_Days"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days
    spending_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df["Total_Spending"] = df[spending_cols].sum(axis=1)
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    df["Education"] = df["Education"].replace({"Basic":"Undergraduate", "2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})
    df["Living_With"] = df["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Single":"Alone", "Divorced":"Alone", "Widow":"Alone", "Absurd":"Alone", "YOLO":"Alone"})

    drop_cols = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer", *spending_cols]
    cleaned = df.drop(columns=drop_cols).copy()

    # Inspect the unfiltered distributions before removing any records.
    outlier_cols = ["Income", "Recency", "Age", "Total_Spending", "Total_Children"]
    fig, axes = plt.subplots(len(outlier_cols), 1, figsize=(10, 12))
    for ax, col in zip(axes, outlier_cols):
        sns.boxplot(x=cleaned[col], ax=ax)
        ax.set_title(f"Before filtering — {col}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "outlier_boxplots_before.png", dpi=160)
    plt.close(fig)

    before = len(cleaned)
    age_removed = int((cleaned["Age"] >= 90).sum())
    income_removed = int((cleaned["Income"] >= 600_000).sum())
    cleaned = cleaned[cleaned["Age"] < 90].copy()
    cleaned = cleaned[cleaned["Income"] < 600_000].copy()
    print(f"Rows before filters: {before}; age-rule count: {age_removed}; income-rule count: {income_removed}; retained: {len(cleaned)}")

    corr = cleaned.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size":7}, ax=ax)
    ax.set_title("SmartCart Numeric Feature Correlation Heatmap"); fig.tight_layout()
    fig.savefig(FIGURES_DIR / "numeric_correlation_heatmap.png", dpi=180); plt.close(fig)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().rename("correlation").to_frame()
    pairs["absolute_correlation"] = pairs["correlation"].abs()
    pairs.sort_values("absolute_correlation", ascending=False).head(15).to_csv(TABLES_DIR / "strong_correlations.csv")

    cat_cols = ["Education", "Living_With"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc = encoder.fit_transform(cleaned[cat_cols])
    enc_df = pd.DataFrame(enc, columns=encoder.get_feature_names_out(cat_cols), index=cleaned.index)
    post_cluster = cleaned[["Response", "Complain"]].copy() if {"Response", "Complain"}.issubset(cleaned.columns) else None
    features = pd.concat([cleaned.drop(columns=cat_cols + [c for c in ["Response", "Complain"] if c in cleaned.columns]), enc_df], axis=1)
    if features.isna().any().any():
        raise ValueError("Feature matrix still contains missing values.")
    X_scaled = StandardScaler().fit_transform(features)
    X_pca = PCA(n_components=3, random_state=RANDOM_STATE).fit_transform(X_scaled)

    fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], s=18, alpha=.65)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3"); ax.set_title("SmartCart Customers — PCA 3D Projection")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "pca_3d_projection.png", dpi=180); plt.close(fig)

    wcss, sils, ks = [], [], list(range(2, 11))
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_pca)
        wcss.append(km.inertia_)
        if k >= 2: sils.append(silhouette_score(X_pca, km.labels_))
    fig, ax = plt.subplots(figsize=(9, 5)); ax.plot(range(1,11), wcss, marker="o")
    ax.set_xlabel("K"); ax.set_ylabel("WCSS / inertia"); ax.set_title("Elbow Method"); ax.set_xticks(range(1,11)); fig.tight_layout()
    fig.savefig(FIGURES_DIR / "elbow_method.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,5)); ax.plot(ks, sils, marker="o")
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette score"); ax.set_title("Silhouette Score by K"); ax.set_xticks(ks); fig.tight_layout()
    fig.savefig(FIGURES_DIR / "silhouette_scores.png", dpi=160); plt.close(fig)
    pd.DataFrame({"k":ks,"silhouette_score":sils,"wcss":wcss[1:]}).to_csv(TABLES_DIR / "k_selection_diagnostics.csv", index=False)
    best_k = ks[int(np.argmax(sils))]
    print(f"Best silhouette among tested K=2..10: K={best_k}, score={max(sils):.4f}; K={CHOSEN_K} remains baseline for comparison.")

    km = KMeans(n_clusters=CHOSEN_K, random_state=RANDOM_STATE, n_init=10)
    labels_km = km.fit_predict(X_pca)
    ag = AgglomerativeClustering(n_clusters=CHOSEN_K, linkage="ward")
    labels_ag = ag.fit_predict(X_pca)
    def metric(x, labels, name):
        return {"algorithm":name,"clusters":len(np.unique(labels)),"silhouette":silhouette_score(x,labels),"calinski_harabasz":calinski_harabasz_score(x,labels),"davies_bouldin":davies_bouldin_score(x,labels)}
    metrics = pd.DataFrame([metric(X_pca, labels_km, "K-Means on PCA"),metric(X_pca, labels_ag, "Agglomerative on PCA")])
    metrics.to_csv(TABLES_DIR / "clustering_metrics.csv", index=False); print(metrics.to_string(index=False))

    # Sensitivity check: compare the same algorithms on PCA space and all standardized features.
    sensitivity_rows = []
    for representation_name, representation in [
        ("PCA (3 components)", X_pca),
        ("Full standardized features", X_scaled),
    ]:
        km_labels = KMeans(n_clusters=CHOSEN_K, random_state=RANDOM_STATE, n_init=10).fit_predict(representation)
        agg_labels = AgglomerativeClustering(n_clusters=CHOSEN_K, linkage="ward").fit_predict(representation)
        sensitivity_rows.append(metric(representation, km_labels, f"K-Means - {representation_name}"))
        sensitivity_rows.append(metric(representation, agg_labels, f"Agglomerative - {representation_name}"))
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(TABLES_DIR / "representation_sensitivity_metrics.csv", index=False)
    print("Representation sensitivity (same K across feature spaces):\\n", sensitivity.to_string(index=False))

    # Small seed check for K-Means label consistency; ARI=1 means identical partitions.
    seed_labels = {
        seed: KMeans(n_clusters=CHOSEN_K, random_state=seed, n_init=10).fit_predict(X_pca)
        for seed in (42, 43, 44)
    }
    stability_rows = []
    for seed in (43, 44):
        stability_rows.append({
            "reference_seed": 42,
            "comparison_seed": seed,
            "adjusted_rand_index": adjusted_rand_score(seed_labels[42], seed_labels[seed]),
        })
    pd.DataFrame(stability_rows).to_csv(TABLES_DIR / "kmeans_seed_stability.csv", index=False)

    base = cleaned.drop(columns=[c for c in ["Response", "Complain"] if c in cleaned.columns]).copy()
    base["KMeans_Cluster"] = labels_km; base["Agglomerative_Cluster"] = labels_ag
    counts = pd.DataFrame({"KMeans":base["KMeans_Cluster"].value_counts().sort_index(),"Agglomerative":base["Agglomerative_Cluster"].value_counts().sort_index()}).fillna(0).astype(int)
    counts.to_csv(TABLES_DIR / "cluster_sizes.csv"); print("Cluster sizes:\n", counts)
    profile_cols = [c for c in ["Age","Income","Recency","Total_Spending","Total_Children","Customer_Tenure_Days","NumDealsPurchases","NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumWebVisitsMonth"] if c in base.columns]
    base.groupby("KMeans_Cluster")[profile_cols].mean().round(2).to_csv(TABLES_DIR / "kmeans_cluster_mean_profile.csv")
    base.groupby("KMeans_Cluster")[profile_cols].median().round(2).to_csv(TABLES_DIR / "kmeans_cluster_median_profile.csv")
    base.groupby("Agglomerative_Cluster")[profile_cols].mean().round(2).to_csv(TABLES_DIR / "agglomerative_cluster_mean_profile.csv")

    fig, axes = plt.subplots(1,2,figsize=(12,4))
    sns.countplot(data=base,x="KMeans_Cluster",hue="KMeans_Cluster",legend=False,ax=axes[0]); axes[0].set_title("K-Means cluster sizes")
    sns.countplot(data=base,x="Agglomerative_Cluster",hue="Agglomerative_Cluster",legend=False,ax=axes[1]); axes[1].set_title("Agglomerative cluster sizes")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "cluster_size_comparison.png",dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    sns.scatterplot(data=base,x="Total_Spending",y="Income",hue="KMeans_Cluster",palette="tab10",alpha=.75,ax=axes[0]); axes[0].set_title("Income vs spending — K-Means")
    sns.scatterplot(data=base,x="Total_Spending",y="Income",hue="Agglomerative_Cluster",palette="tab10",alpha=.75,ax=axes[1]); axes[1].set_title("Income vs spending — Agglomerative")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "income_spending_by_cluster.png",dpi=180); plt.close(fig)
    for labels, name, filename in [(labels_km,"K-Means","kmeans_clusters_pca.png"),(labels_ag,"Agglomerative","agglomerative_clusters_pca.png")]:
        fig=plt.figure(figsize=(9,7)); ax=fig.add_subplot(111,projection="3d")
        sc=ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2],c=labels,cmap="viridis",s=20,alpha=.7)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3"); ax.set_title(f"{name} clusters (K={CHOSEN_K})"); fig.colorbar(sc,ax=ax,pad=.1,label="Cluster")
        fig.tight_layout(); fig.savefig(FIGURES_DIR / filename,dpi=180); plt.close(fig)
    if post_cluster is not None:
        audit=post_cluster.copy(); audit["KMeans_Cluster"]=labels_km
        pd.concat([audit.groupby("KMeans_Cluster")["Response"].mean().rename("response_rate"),audit.groupby("KMeans_Cluster")["Complain"].mean().rename("complaint_rate")],axis=1).round(3).to_csv(TABLES_DIR / "post_cluster_response_complaint_audit.csv")
    base.to_csv(TABLES_DIR / "customer_cluster_assignments.csv",index=False)
    print(f"Done. {len(base)} customers; outputs: {TABLES_DIR}")


if __name__ == "__main__":
    main()
