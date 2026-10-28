# SmartCart Customer Segmentation System

An unsupervised machine learning minor project to discover customer segments from SmartCart customer demographics, purchasing behaviour, and engagement activity.

## Problem statement

SmartCart has customer records containing demographic attributes, purchase amounts, purchase-channel frequency, and engagement indicators. A single generic marketing strategy may overlook meaningful differences between customer groups. This project explores customer segmentation with unsupervised learning so that groups can be described and used as starting points for personalised marketing analysis.

## Project objectives

- Inspect and validate the customer dataset.
- Handle missing values and examine duplicates and outliers.
- Engineer interpretable features such as age, customer tenure, total spending, and household children.
- Remove identifiers and features that are inappropriate for distance-based clustering.
- Explore feature relationships with descriptive statistics and a correlation heatmap.
- Encode categorical values and scale numerical features.
- Use PCA for dimensionality reduction and visual exploration.
- Use the Elbow method and Silhouette score to investigate candidate values of K.
- Compare K-Means and Agglomerative Clustering using internal clustering metrics.
- Profile clusters using original-scale business features and document limitations.

## Workflow

1. **Understand the data** — inspect dimensions, types, distributions, missing values, and duplicates.
2. **Clean and engineer features** — impute missing income values, derive customer-level features, and review filtering decisions.
3. **Explore relationships** — review distributions, outlier behaviour, and correlations.
4. **Prepare features** — encode categories, scale numeric columns, and keep customer IDs out of model inputs.
5. **Reduce dimensions** — use PCA for exploratory visualization and optionally compare with clustering on the full scaled feature matrix.
6. **Select clustering settings** — compare WCSS and silhouette scores across candidate K values instead of assuming a fixed K is optimal.
7. **Fit and evaluate** — compare K-Means and hierarchical clustering using silhouette, Calinski–Harabasz, and Davies–Bouldin scores.
8. **Profile segments** — examine counts and original-unit summaries before assigning understandable names.

## Repository structure

```text
smartcart-clustering-system/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── smartcart_customers.csv
├── notebooks/
│   ├── smartcart_customer_segmentation.ipynb
│   └── smartcart_original_notebook.ipynb
├── src/
│   └── smartcart_customer_segmentation.py
├── docs/
│   ├── PROJECT_REPORT.md
│   └── METHODOLOGY_AND_DECISIONS.md
├── reports/
│   └── SmartCart_Customer_Segmentation_Report.pdf
└── outputs/
    ├── figures/
    └── tables/
```

> Note: this tree describes the intended full project layout. Add the corresponding files and folders to the repository before treating the project as fully published.

## Setup

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
```

Activate the environment, then install requirements:

```bash
python -m pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open `notebooks/smartcart_customer_segmentation.ipynb` and run the cells from top to bottom. If using the Python script, run it from the repository root so relative paths resolve as expected.

## Interpreting results responsibly

Cluster labels are arbitrary identifiers, not rankings. A cluster should only be described as high-spending, discount-oriented, highly engaged, or at-risk after its actual summary statistics support that description. Silhouette and other internal metrics quantify geometric structure; they do not demonstrate marketing lift or customer-retention impact. Compare results on PCA-transformed and full scaled feature spaces before making a final methodological choice.

## Dataset and privacy

The project uses the SmartCart customer dataset supplied for this minor project. Before publishing the CSV publicly, verify the source dataset's redistribution terms and confirm that it contains no personal or restricted information. If redistribution is not permitted, keep the data out of the public repository and provide documented instructions for obtaining it separately.

## References

- [Scikit-learn: Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Scikit-learn: K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Scikit-learn: Agglomerative Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [Scikit-learn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Scikit-learn: silhouette score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

## Project status

This repository's README establishes the intended organization and workflow. The project should be considered complete only after the notebook, source code, report, dependencies, and any permitted dataset files are committed and verified to run from a clean environment.
