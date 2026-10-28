# SmartCart Customer Segmentation System

**Minor Project | Unsupervised Machine Learning | Customer Analytics**

## Abstract
This project explores customer segmentation for SmartCart using unsupervised machine learning. It includes data-quality checks, missing-value imputation, feature engineering, feature selection, outlier inspection, correlation analysis, one-hot encoding, standardization, PCA, K-selection diagnostics, K-Means and Agglomerative Clustering, and original-scale cluster profiling. The results are exploratory and should be validated before operational marketing decisions.

## Problem and objective
SmartCart's generic customer engagement strategy may miss differences in purchasing and engagement behaviour. The objective is to group customers based on demographic, spending, and channel-activity features, then summarize the resulting segments. This is not a supervised churn prediction project and does not prove that marketing actions based on segments will improve results.

## Dataset
The supplied CSV contains 2,240 customer rows and 22 columns. It includes demographics, income, enrollment date, days since last purchase, spending across six product categories, purchases by channel, monthly website visits, a complaint indicator, and a response indicator. Initial inspection found 24 missing `Income` entries (1.07%), no missing values in other inspected columns, no duplicate complete rows, and unique customer IDs. Customer ID is excluded from model features.

## Preprocessing and feature engineering
Missing income values were filled with the median, 51,381.50. The pipeline derives `Age = 2026 - Year_Birth`, customer tenure in days relative to the latest enrollment date in the dataset, total spending across six product categories, and total children in the household. Education and living-status categories are consolidated. The raw ID, birth year, marital status, household child components, enrollment date, and product-spending columns are dropped after their information is represented by derived features. `Response` and `Complain` are excluded from clustering features and held out for optional descriptive analysis.

Boxplots are generated for income, recency, age, total spending, and number of children. The baseline applies the project rules `Age < 90` and `Income < 600000`. Four rows were removed by the sequential filters, leaving 2,236 customers. These cut-offs are assumptions, not universal outlier definitions; alternative IQR/robust rules should be compared.

## Correlation analysis
The strongest observed pairwise correlations in the filtered numeric features included income–total spending (0.789), catalog purchases–total spending (0.778), income–catalog purchases (0.694), total spending–store purchases (0.675), and income–monthly website visits (-0.646). Correlation describes association and does not establish causation.

## Encoding, scaling, and PCA
Education and living status are one-hot encoded, then the resulting 16-feature matrix is standardized with `StandardScaler`. PCA reduces this representation to three components for visualization and the baseline clustering comparison. PC1 explains 25.73%, PC2 12.58%, and PC3 11.57%, for a cumulative 49.89%. Because these components preserve only about half of the variance, clustering on the full standardized features should also be evaluated.

## Selecting K
The Elbow diagnostic records inertia for K=1–10. Silhouette scores for K=2–10 were 0.3732, 0.3762, 0.4092, 0.4218, 0.4177, 0.4377, 0.4493, 0.4491, and 0.4496, respectively. The highest tested silhouette was at K=10, but the baseline retains K=4 to match the original four-cluster comparison. K=4 is therefore a baseline, not a demonstrated optimum. The elbow curve, profile usefulness, segment sizes, and stability should guide a final choice.

## Baseline algorithm comparison
Both methods were fitted on the same three-component PCA representation with K=4.

| Algorithm | Silhouette ↑ | Calinski–Harabasz ↑ | Davies–Bouldin ↓ |
|---|---:|---:|---:|
| K-Means | 0.4092 | 1430.79 | 1.0363 |
| Agglomerative (Ward linkage) | 0.3610 | 1222.03 | 0.9144 |

K-Means has higher silhouette and Calinski–Harabasz scores in this run, while Agglomerative Clustering has the lower Davies–Bouldin score. The metrics disagree, so the model choice should also consider original-scale profiles and stability.

| Cluster ID | K-Means size | Agglomerative size |
|---:|---:|---:|
| 0 | 822 | 795 |
| 1 | 620 | 455 |
| 2 | 351 | 477 |
| 3 | 443 | 509 |

Cluster IDs are arbitrary. K-Means clusters 1 and 2 have mean income around 71,000 and mean spending around 1,167–1,193, while clusters 0 and 3 have mean income around 37,000 and spending around 168–169. The paired groups appear similar on these summary variables, so more analysis is needed to determine if four segments are meaningfully distinct. The notebook exports mean/median profiles, segment counts, income-versus-spending charts, assignments, and an optional post-clustering response/complaint audit.

## Conclusion and limitations
The project demonstrates the core stages of an unsupervised segmentation workflow. Its current output is an exploratory segmentation hypothesis, not validated churn prediction, proven marketing lift, or evidence of causality. Future work should compare full-dimensional and PCA-based clustering, check stability over seeds/resamples, compare outlier-handling alternatives, verify whether similar clusters should be merged, and validate segment usefulness with stakeholders. Confirm the dataset's source license and redistribution permissions before public sharing.

## Reproducibility
Install dependencies with `python -m pip install -r requirements.txt`. Run `python src/smartcart_customer_segmentation.py` or execute `notebooks/smartcart_customer_segmentation.ipynb` from top to bottom. Generated figures and tables are stored under `outputs/`.
