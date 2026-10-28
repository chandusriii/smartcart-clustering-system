# SmartCart Customer Segmentation System

**Minor Project | Unsupervised Machine Learning | Customer Analytics**

## Abstract

This project explores customer segmentation for SmartCart, an e-commerce platform whose customer data contains demographic information, purchase behaviour, channel activity, and selected engagement indicators. The workflow combines data-quality checks, feature engineering, missing-value imputation, feature selection, outlier inspection, correlation analysis, one-hot encoding, standardization, PCA, K-selection diagnostics, K-Means, Agglomerative Clustering, and profile summaries. The purpose is to discover interpretable patterns that can help formulate customer engagement hypotheses, not to claim a validated churn predictor or proven marketing uplift.

## 1. Problem statement

Generic marketing strategies can overlook meaningful differences between customers. SmartCart needs an exploratory way to group customer records by spending, shopping-channel use, income, household characteristics, and engagement. This minor project tests whether unsupervised clustering provides useful customer profiles that can be reviewed and validated by a business team.

## 2. Objectives

- Inspect the dataset and summarize missingness, data types, unique values, and duplicates.
- Handle missing values and engineer interpretable customer-level features.
- Identify irrelevant, redundant, or identifier columns before distance-based analysis.
- Inspect outliers and examine numeric relationships with a correlation heatmap.
- Encode categorical columns and standardize the clustering feature matrix.
- Use PCA to visualize high-dimensional customers in a smaller space.
- Examine candidate K values using the Elbow method and Silhouette score.
- Compare K-Means and Agglomerative Clustering with internal metrics.
- Profile cluster composition in original units and discuss limitations.

## 3. Dataset overview

The provided CSV contains **2,240 rows and 22 columns**. Each row represents a customer. The attributes include birth year, education, marital status, household children, income, enrollment date, days since last purchase, product-category spending, deal purchases, website/catalog/store purchases, monthly website visits, complaint indicator, and response indicator.

Initial inspection found **24 missing values in `Income` (1.07%)**, no missing values in other inspected columns, **no duplicate full rows**, and a unique customer `ID`. The ID is removed from model features because it is a record identifier rather than a behavioural measurement.

## 4. Data preparation and feature engineering

### 4.1 Missing income values

The 24 missing `Income` observations were filled with the median income of **51,381.50**. Median imputation is less sensitive than mean imputation to extreme values. The number of imputed values is logged so the operation remains auditable.

### 4.2 Engineered features

The pipeline creates:

- `Age`: `2026 - Year_Birth`.
- `Customer_Tenure_Days`: latest enrollment date in the supplied dataset minus the customer's `Dt_Customer`.
- `Total_Spending`: sum of the six product-category spending columns.
- `Total_Children`: sum of `Kidhome` and `Teenhome`.
- `Living_With`: consolidated living-status category derived from marital status.

Education levels are also consolidated. The reference year 2026 is a project configuration and should be updated if the analysis is reused for a different time period.

### 4.3 Column selection

The pipeline drops `ID`, raw birth year, original marital status, original child-count components, enrollment date, and individual spending category columns after derived features are calculated. This avoids allowing identifiers or duplicated representations to unnecessarily dominate the distance calculations. `Response` and `Complain` are excluded from the clustering feature matrix and kept only for an optional descriptive audit after clustering.

### 4.4 Outlier inspection and filters

Boxplots are generated for income, recency, age, total spending, and number of children. The baseline then applies the rules `Age < 90` and `Income < 600000`. In the executed run, 3 rows met the age rule and 1 row met the income rule; sequential filtering retained **2,236 customers** and removed 4 records in total. These thresholds are explicit project assumptions, not universal statistical definitions of outliers. Alternative IQR-based or robust approaches should be compared as a sensitivity check.

## 5. Exploratory analysis

A numeric correlation heatmap is saved to `outputs/figures/numeric_correlation_heatmap.png`. The strongest observed positive relationships included:

| Feature pair | Pearson correlation |
|---|---:|
| Income and Total_Spending | 0.789 |
| NumCatalogPurchases and Total_Spending | 0.778 |
| Income and NumCatalogPurchases | 0.694 |
| Total_Spending and NumStorePurchases | 0.675 |
| Income and NumWebVisitsMonth | -0.646 |

These values describe linear association in the retained data; they do not establish causation. Some purchase-channel features overlap with overall spending behaviour.

## 6. Encoding, scaling, and PCA

`Education` and `Living_With` are one-hot encoded. The resulting 16-feature matrix is standardized with `StandardScaler`, which makes numeric inputs comparable for distance-based methods. PCA is then used to create a three-component representation for visualization and the baseline clustering comparison.

The explained variance ratios for PC1, PC2, and PC3 were approximately 25.73%, 12.58%, and 11.57%. Together they explain **49.89%** of the variance in the standardized feature matrix. This is useful for visualization, but almost half the variance is outside these three components. Comparing clustering on the full standardized matrix with clustering in PCA space is therefore an important next validation step.

## 7. Choosing the number of clusters

The Elbow method records within-cluster sum of squares (WCSS/inertia) for K=1 through K=10. Silhouette scores are calculated for K=2 through K=10. The observed silhouette values were:

| K | Silhouette score |
|---:|---:|
| 2 | 0.3732 |
| 3 | 0.3762 |
| 4 | 0.4092 |
| 5 | 0.4218 |
| 6 | 0.4177 |
| 7 | 0.4377 |
| 8 | 0.4493 |
| 9 | 0.4491 |
| 10 | 0.4496 |

The highest silhouette score within this tested range was at **K=10**, but that alone does not mean ten clusters are the best business segmentation. The baseline comparison preserves **K=4** to reproduce the original project's four-cluster setup while comparing algorithms. Consequently, four clusters should be treated as an initial baseline—not as the proven optimum. The Elbow curve, cluster size balance, interpretability, and stability should inform the final selection.

## 8. Clustering algorithms and baseline results

Both algorithms are fitted on the same three-component PCA representation with K=4. K-Means uses a fixed random state for reproducibility and `n_init=10`; Agglomerative Clustering uses Ward linkage. The results are:

| Algorithm | K | Silhouette (higher is better) | Calinski–Harabasz (higher is better) | Davies–Bouldin (lower is better) |
|---|---:|---:|---:|---:|
| K-Means on PCA | 4 | 0.4092 | 1430.79 | 1.0363 |
| Agglomerative on PCA | 4 | 0.3610 | 1222.03 | 0.9144 |

K-Means has the higher silhouette and Calinski–Harabasz scores in this run, while Agglomerative Clustering has the lower Davies–Bouldin score. These metrics do not all favour one method, so model choice should also consider the original-scale cluster profiles and whether the segments are stable and actionable.

### Cluster sizes

| Cluster ID | K-Means customers | Agglomerative customers |
|---:|---:|---:|
| 0 | 822 | 795 |
| 1 | 620 | 455 |
| 2 | 351 | 477 |
| 3 | 443 | 509 |

Cluster IDs are arbitrary labels. The number `0` is not inherently lower or worse than `3`.

## 9. Cluster characterization

The notebook exports cluster means and medians using original-scale variables, including age, income, recency, total spending, children, customer tenure, purchase counts, and website visits. It also plots income against total spending by cluster and calculates post-clustering response and complaint rates separately from the clustering feature matrix.

These tables should be the basis for descriptive segment names. A label such as “higher-spending, multichannel customers” should only be assigned after profile statistics support it. `Response` and `Complain` are examined after clustering and do not train the clustering models. Their rates are descriptive and cannot be interpreted as causal effects or validated response predictions.

Key K-Means mean-profile contrasts in the baseline include:

- Clusters 1 and 2 have average income around 71,000 and average spending around 1,167–1,193.
- Clusters 0 and 3 have average income around 37,000 and average spending around 168–169.
- Higher-spending groups also show more catalog/store activity and fewer monthly website visits on average than the lower-spending groups.

Clusters 0 and 3, and clusters 1 and 2, have similar averages across several included variables. This calls for investigating whether K=4 creates sufficiently distinct profiles rather than assigning strong marketing names immediately.

## 10. Conclusion

The SmartCart project implements an end-to-end unsupervised machine learning workflow using data inspection, median imputation, feature engineering, feature selection, outlier filtering, correlation analysis, one-hot encoding, standardization, PCA, Elbow and Silhouette diagnostics, K-Means, Agglomerative Clustering, and cluster profiling. The baseline provides useful exploratory customer groupings and reproducible artefacts.

The results do **not** yet prove the best cluster count, cluster stability, churn propensity, causal marketing lift, or business impact. Before presenting the project as finalized, compare full-dimensional clustering with PCA-based clustering, check stability, review cluster profiles, validate dataset redistribution permissions, and align the final model choice with the actual segmentation objective.

## 11. Limitations and future work

1. Compare clustering on the full standardized feature matrix with PCA-based clustering.
2. Check K-Means stability across multiple seeds or resampled data.
3. Compare the chosen age/income cut-offs with alternative outlier strategies.
4. Investigate whether clusters that have very similar profiles should be merged or represented differently.
5. Validate segment usefulness with business stakeholders and, where appropriate, controlled experiments.
6. Verify source licensing and privacy implications before public dataset redistribution.

## 12. Reproducibility

Install dependencies with `python -m pip install -r requirements.txt`, then run `notebooks/smartcart_customer_segmentation.ipynb` from top to bottom. The baseline script is available at `src/smartcart_customer_segmentation.py`. Figures and tables are written under `outputs/`. Confirm generated outputs correspond to the configuration used for the final submission.
