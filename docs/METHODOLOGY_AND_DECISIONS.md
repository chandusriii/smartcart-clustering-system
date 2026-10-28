# Methodology and Implementation Decisions

## Dataset quality
The supplied CSV contains 2,240 rows and 22 columns. Initial inspection found 24 missing `Income` values (1.07%) and no duplicate complete rows. The customer `ID` is unique and is excluded from clustering features.

## Missing values
Missing `Income` values are imputed using the median (`51,381.50` in the supplied data). The number of imputed values is recorded before transformation. All other columns were complete in the initial inspection.

## Feature engineering
- `Age = 2026 - Year_Birth` (reference year is configurable; update it when adapting the analysis to another period).
- `Customer_Tenure_Days = latest enrollment date in this dataset - Dt_Customer`, which provides a repeatable tenure measure for this fixed dataset.
- `Total_Spending` is the sum of the six product-category spending attributes.
- `Total_Children` is the sum of `Kidhome` and `Teenhome`.
- Education levels and living-status categories are consolidated before one-hot encoding.

## Column selection
The pipeline removes the customer identifier and raw columns represented by engineered variables: `Year_Birth`, `Marital_Status`, `Kidhome`, `Teenhome`, `Dt_Customer`, and the six individual category-spending fields. It retains `Response` and `Complain` for optional post-clustering descriptive checks only; neither field enters the feature matrix.

## Outlier inspection and filtering
Boxplots are generated for `Income`, `Recency`, `Age`, `Total_Spending`, and `Total_Children`. The baseline applies the project rules `Age < 90` and `Income < 600000`, retaining 2,236 of 2,240 records in the executed run. These are explicit project-specific cut-offs, not universal statistical definitions of outliers. Consider comparing with IQR/robust methods and a domain review.

## Correlation analysis
A numeric correlation heatmap and sorted list of strong pairwise correlations are produced. In this run, income and total spending correlate at approximately 0.789, and catalog purchases and total spending at approximately 0.778. Such correlations are descriptive; they do not prove causality and may indicate overlapping behavioural signals.

## Encoding, scaling, and PCA
`Education` and `Living_With` are one-hot encoded. `StandardScaler` standardizes the model features before PCA. The first three principal components explain approximately 49.89% of variance. PCA is useful for visualization and is used in the baseline model comparison, but it discards information; a full-scaled-feature comparison is an important sensitivity check.

## K selection and clustering
The notebook evaluates WCSS/inertia for K=1–10 and silhouette scores for K=2–10. The highest silhouette in that candidate range occurred at K=10 (approximately 0.4496), while the baseline retains K=4 to reproduce the original project's four-segment comparison. Therefore, K=4 is a baseline choice, not the demonstrated global optimum. Use the elbow shape, profile interpretability, segment sizes, stability, and business context together when selecting K.

K-Means and Ward-linkage Agglomerative Clustering are compared on the same 3-component PCA representation using silhouette, Calinski–Harabasz, and Davies–Bouldin metrics. These are internal geometric metrics, not evidence of marketing impact.

## Profiling and reporting
Cluster counts, income/spending plots, original-scale means and medians, and a post-clustering `Response`/`Complain` audit are exported. Cluster IDs have no intrinsic ordering. Assign segment names only after examining the actual summary statistics; do not label a group as churn-prone without appropriate outcome data and validation.

## Recommended follow-up checks
1. Compare clustering on all standardized features and on PCA components.
2. Evaluate K with the diagnostics and inspect whether smaller segments are actionable.
3. Check cluster stability across random seeds or resampled samples.
4. Compare the chosen cut-offs with IQR/robust methods and document their effect.
5. Verify dataset reuse/redistribution rights before publishing the CSV publicly.
