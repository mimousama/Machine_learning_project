# Introduction to Machine Learning - Age and Income project


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# 1. LOAD THE DATA
# --------------------------------------------------

data = pd.read_csv("AgeIncome.csv")

print("First five rows:")
print(data.head())

print("\nDataset information:")
data.info()

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate rows:", data.duplicated().sum())

print("\nBasic statistics:")
print(data.describe())


# --------------------------------------------------
# 2. SELECT THE FEATURES
# --------------------------------------------------

# This project uses Age and Income.
# If your CSV uses different column names, change these two names.

age_column = "Age"
income_column = "Income"

# Keep only Age and Income for the clustering model.
data = data[[age_column, income_column]].copy()

# Convert values to numbers.
# Text that cannot be converted becomes missing.
data[age_column] = pd.to_numeric(data[age_column], errors="coerce")
data[income_column] = pd.to_numeric(data[income_column], errors="coerce")


# --------------------------------------------------
# 3. CLEAN THE DATA
# --------------------------------------------------

# Replace missing values with the median of each column.
data[age_column] = data[age_column].fillna(data[age_column].median())
data[income_column] = data[income_column].fillna(data[income_column].median())

# Remove duplicate rows.
data = data.drop_duplicates()

print("\nData after cleaning:")
data.info()

print("\nMissing values after cleaning:")
print(data.isnull().sum())


# --------------------------------------------------
# 4. EXPLORE THE DATA WITH GRAPHS
# --------------------------------------------------

plt.figure(figsize=(6, 4))
plt.hist(data[age_column], bins=10, edgecolor="black")
plt.title("Age distribution")
plt.xlabel("Age")
plt.ylabel("Number of people")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(data[income_column], bins=10, edgecolor="black")
plt.title("Income distribution")
plt.xlabel("Income")
plt.ylabel("Number of people")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(data[age_column], data[income_column])
plt.title("Age vs Income")
plt.xlabel("Age")
plt.ylabel("Income")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 5. PREPARE THE DATA FOR K-MEANS
# --------------------------------------------------

# Age and Income use different scales.
# StandardScaler puts them on a comparable scale.

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)


# --------------------------------------------------
# 6. CREATE THE FIRST K-MEANS MODEL (K = 3)
# --------------------------------------------------

kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_3.fit(scaled_data)

data["Cluster_K3"] = kmeans_3.labels_

print("\nNumber of people in each cluster (K=3):")
print(data["Cluster_K3"].value_counts().sort_index())

print("\nAverage Age and Income in each cluster (K=3):")
print(data.groupby("Cluster_K3")[[age_column, income_column]].mean())


# Show the clusters.
plt.figure(figsize=(7, 5))
plt.scatter(
    data[age_column],
    data[income_column],
    c=data["Cluster_K3"],
    cmap="viridis"
)

plt.title("K-Means clustering (K=3)")
plt.xlabel("Age")
plt.ylabel("Income")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 7. FIND A GOOD NUMBER OF CLUSTERS
# --------------------------------------------------

# We test different values of K.
# Inertia is used for the Elbow Method.
# Silhouette Score measures how well separated the clusters are.

inertia_values = []
silhouette_values = []

# Test K values from 2 to 9, depending on dataset size.
k_values = range(2, min(10, len(data)))

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_data)

    inertia_values.append(model.inertia_)
    silhouette_values.append(silhouette_score(scaled_data, labels))


# Elbow graph
plt.figure(figsize=(6, 4))
plt.plot(list(k_values), inertia_values, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.tight_layout()
plt.show()


# Silhouette graph
plt.figure(figsize=(6, 4))
plt.plot(list(k_values), silhouette_values, marker="o")
plt.title("Silhouette Score")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette score")
plt.xticks(list(k_values))
plt.tight_layout()
plt.show()


# The highest Silhouette Score gives our suggested K.
best_k = list(k_values)[silhouette_values.index(max(silhouette_values))]

print("\nBest K according to the Silhouette Score:", best_k)
print("Best Silhouette Score:", max(silhouette_values))


# --------------------------------------------------
# 8. COMPARE K=3 WITH THE SUGGESTED K
# --------------------------------------------------

best_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
best_labels = best_model.fit_predict(scaled_data)

data["Cluster_BestK"] = best_labels

print("\nAverage Age and Income for K=3:")
print(data.groupby("Cluster_K3")[[age_column, income_column]].mean())

print("\nAverage Age and Income for the suggested K:")
print(data.groupby("Cluster_BestK")[[age_column, income_column]].mean())

print("\nCluster sizes for K=3:")
print(data["Cluster_K3"].value_counts().sort_index())

print("\nCluster sizes for the suggested K:")
print(data["Cluster_BestK"].value_counts().sort_index())


# --------------------------------------------------
# 9. SAVE THE RESULTS
# --------------------------------------------------

data.to_csv("AgeIncome_Cluster_Results.csv", index=False)

print("\nThe results were saved in AgeIncome_Cluster_Results.csv")


# --------------------------------------------------
# 10. PROJECT CONCLUSION
# --------------------------------------------------

# K-Means is an unsupervised learning algorithm.
# We do not have a target column, so we do not use a train/test split.
# The model groups people according to similarities in Age and Income.
# We compare K=3 with the K suggested by the Silhouette Score.
# Accuracy and a confusion matrix are not appropriate because there are
# no known correct cluster labels in this dataset.
