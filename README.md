# Age and Income Analysis – Machine Learning Project

**Course:** I&C SCI_X427.09 – Introduction to Machine Learning  
**Semester:** Fall 2026  
**Dataset:** AgeIncome_PRJ.csv  
**Algorithm:** K-Means Clustering  
**Machine Learning Type:** Unsupervised Learning

## 1. Project Description

The goal of this project is to analyze the AgeIncome_PRJ.csv dataset using the K-Means Clustering algorithm.

K-Means is an unsupervised machine learning algorithm. It groups data into clusters based on similarities between observations.

In this project, we use age and income to identify different groups of people with similar characteristics.

## 2. Project Objectives

The main objectives are:

- Explore and understand the dataset.
- Clean and prepare the data.
- Handle missing values and duplicate records.
- Apply feature scaling.
- Implement the K-Means algorithm.
- Find the optimal number of clusters.
- Evaluate the model using the Silhouette Score.
- Compare the results using K = 3 and the optimal K.
- Visualize and interpret the results.

## 3. Technologies and Libraries

This project uses Python and the following libraries:

- **Pandas:** Data manipulation and analysis.
- **NumPy:** Numerical calculations.
- **Matplotlib:** Data visualization.
- **Seaborn:** Statistical visualizations.
- **Scikit-learn:** Machine learning algorithms and evaluation.

## 4. Data Preprocessing

Before applying K-Means, we perform several preprocessing steps:

1. Load the dataset.
2. Explore the data structure.
3. Check for missing values.
4. Remove duplicate records if necessary.
5. Select relevant numerical features.
6. Handle missing values if needed.
7. Apply StandardScaler to scale the features.

Feature scaling is important because K-Means uses Euclidean distance to group similar observations.

## 5. K-Means Clustering

We first apply K-Means with K = 3.

Each observation is assigned to a cluster based on its similarity to the cluster centers.

We then use two methods to find the optimal number of clusters:

- **Elbow Method:** Helps identify the appropriate number of clusters by examining the inertia.
- **Silhouette Score:** Measures how well the observations fit within their assigned clusters.

## 6. Model Evaluation

We evaluate the clustering results using:

- Inertia (WCSS).
- Silhouette Coefficient.
- Cluster sizes.
- Cluster centers.

We also compare the results obtained with K = 3 and the optimal number of clusters.

Since K-Means is an unsupervised learning algorithm, accuracy and confusion matrix are not directly applicable without actual reference labels.

## 7. Project Structure

The project contains the following files:

```text
AgeIncome_ML_Project/
│
├── AgeIncome_PRJ.csv
├── main.py
│
├── AgeIncome_Project_Results/
│   ├── clustered_data.csv
│   ├── cluster_statistics.csv
│   ├── elbow_method.png
│   ├── silhouette_score.png
│   ├── cluster_visualization.png
│   └── project_summary.txt
│
└── README.md
```

## 8. Installation

Install the required Python libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 9. How to Run the Project

1. Download or clone the project.
2. Make sure the CSV dataset is in the correct folder.
3. Open the project in VS Code or Jupyter Notebook.
4. Install the required libraries.
5. Run the Python script:

```bash
python main.py
```

6. Check the `AgeIncome_Project_Results` folder to view the generated results and graphs.

## 10. Expected Results

At the end of the project, we expect to:

- Identify groups of people with similar ages and incomes.
- Determine an appropriate number of clusters.
- Understand the characteristics of each cluster.
- Compare different clustering configurations.
- Discover useful patterns in the dataset.

## 11. Conclusion

This project demonstrates how K-Means Clustering can be used to analyze age and income data.

By applying data preprocessing, clustering, and evaluation techniques, we can discover groups of people with similar characteristics.

The results help us better understand the dataset and demonstrate the practical application of unsupervised machine learning.

---

**Project developed for I&C SCI_X427.09 – Introduction to Machine Learning.**

**Group Members:**
- Student 1: [Full Name]
- Student 2: [Full Name]
