# Demographic Segmentation Study: Age & Income K-Means Clustering

## Executive Summary
This study applies **K-Means clustering** to a dataset of $N = 30$ individuals across two continuous demographic variables: **Age** and **Income**. Exploratory data analysis reveals a strong positive linear relationship between age and income ($r = 0.687$), accompanied by substantial income dispersion among older cohorts. 

While the **Elbow Method** identifies a macro-segmentation inflection point at $K = 3$ clusters, **Silhouette Analysis** reaches a global maximum at $K = 6$ ($\approx 0.523$). Selecting $K = 6$ captures distinct socioeconomic sub-segments—most notably partitioning older working-age adults into low-, moderate-, upper-middle-, and peak-income tiers.

---

## Repository Assets
* **Visualizations:**
  * `01_age_distribution.png` — Univariate distribution and KDE of Age
  * `02_income_distribution.png` — Univariate distribution and KDE of Income
  * `03_age_vs_income.png` — Bivariate scatter plot illustrating income fanning
  * `04_elbow_method.png` — Within-Cluster Sum of Squares (Inertia) across $K \in [1, 10]$
  * `05_silhouette_scores.png` — Silhouette coefficient evaluation across $K \in [2, 10]$
  * `06_clusters_k3.png` — Baseline K-Means cluster assignments ($K = 3$)
  * `07_clusters_optimal_k.png` — Optimal K-Means cluster assignments ($K = 6$)
  * `08_k3_vs_optimal.png` — Side-by-side structural comparison of $K = 3$ vs. $K = 6$
* **Data Outputs:**
  * `clustered_age_income_data.csv` — Observation-level records with assigned cluster labels
  * `cluster_statistics.csv` — Summary centroid and dispersion metrics per cluster

---

## 1. Dataset Overview & Exploratory Data Analysis (EDA)
The dataset contains 30 complete observations spanning working-age adults from early career (age 22) to retirement age (age 66).

| Metric | Age (Years) | Income ($) |
| :--- | :---: | :---: |
| **Count ($N$)** | 30 | 30 |
| **Mean** | 45.97 | $28,011.90 |
| **Standard Deviation** | 13.04 | $13,703.34 |
| **Minimum** | 22.00 | $8,877.00 |
| **25th Percentile ($Q_1$)** | 37.25 | $18,166.25 |
| **Median (50th / $Q_2$)** | 47.00 | $24,241.00 |
| **75th Percentile ($Q_3$)** | 56.75 | $35,923.75 |
| **Maximum** | 66.00 | $59,804.00 |

### Key Distributional Patterns
* **Age Distribution (`01_age_distribution.png`):** Slightly left-skewed ($\text{mean } 45.97 < \text{median } 47.00$), with the highest density falling between ages **37 and 60** (19 of 30 observations) and a smaller early-career cohort between ages **22 and 31**.
* **Income Distribution (`02_income_distribution.png`):** Heavily right-skewed ($\text{mean } \$28,011.90 > \text{median } \$24,241.00$). A prominent modal peak occurs between **$\$17,500$ and $\$26,000$** (40% of the sample), while a right tail extends up to $\$59,804$.
* **Bivariate Relationship (`03_age_vs_income.png`):** Age and Income exhibit a strong positive correlation ($r = 0.687$) with pronounced **heteroscedasticity** (fanning variance). Individuals under age 40 cluster tightly between $\$8,877$ and $\$30,085$, whereas individuals aged 45+ diverge sharply across earnings brackets ranging from $\$15,736$ to $\$59,804$.

---

## 2. Clustering Methodology & Model Selection
To uncover natural demographic groupings in the standardized Age–Income feature space, K-Means clustering was evaluated across $K \in [1, 10]$ using two validation criteria:

1. **Elbow Method (`04_elbow_method.png`):** Inertia drops sharply from $K = 1$ ($\approx 60.0$) to $K = 2$ ($\approx 25.2$) and $K = 3$ ($\approx 14.1$), after which marginal variance reduction flattens—indicating **$K = 3$** as the coarse inflection point.
2. **Silhouette Analysis (`05_silhouette_scores.png`):** Measuring intra-cluster cohesion relative to nearest-cluster separation, the average silhouette score dips at $K = 3$ ($\approx 0.453$) and $K = 4$ ($\approx 0.451$) before rising to a global peak at **$K = 6$ ($\approx 0.523$)**.

> **Selection Rationale:** While $K = 3$ provides high-level macro-buckets, it merges distinct career stages and income tiers. $K = 6$ achieves stronger geometric separation and produces actionable demographic personas.

---

## 3. Macro-Segmentation Analysis ($K = 3$)
The baseline $K = 3$ solution (`06_clusters_k3.png`) partitions the sample into three broad groups:

| Cluster | Count | Share (%) | Mean Age | Median Age | Mean Income | Median Income | Archetype Profile |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 8 | 26.67% | 57.88 | 57.0 | $47,729.13 | $45,805.00 | **Older, High-Income Earners** |
| **1** | 12 | 40.00% | 32.58 | 36.5 | $20,363.58 | $20,244.50 | **Younger to Early-Mid Career, Lower-Moderate Income** |
| **2** | 10 | 33.33% | 52.50 | 51.5 | $21,416.10 | $22,511.00 | **Older, Stagnant / Lower-Moderate Income** |

### Limitations of $K = 3$
* **Broad Age Blending:** Cluster 1 spans an 18-year window (ages 22–40), grouping entry-level workers together with established mid-career professionals.
* **Upper-Tier Compression:** Cluster 0 merges upper-middle earners ($\$37.9\text{k}–\$41.0\text{k}$) with peak senior earners ($\$50.6\text{k}–\$59.8\text{k}$).

---

## 4. Optimal Granular Segmentation ($K = 6$)
Selecting $K = 6$ (`07_clusters_optimal_k.png`) resolves the intra-cluster variance of the $K = 3$ model by splitting each broad life stage into distinct socioeconomic tiers:

| Cluster | Count | Share (%) | Mean Age | Median Age | Mean Income | Median Income | Segment Label & Characteristics |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2** | 4 | 13.33% | 23.50 | 22.5 | $14,357.75 | $14,089.50 | **Entry-Level Youth:** Youngest cohort (ages 22–27) with entry-level earnings ($\$8,877–\$20,375$). |
| **3** | 8 | 26.67% | 37.13 | 37.5 | $23,366.50 | $22,432.50 | **Core Mid-Career Workforce:** Largest segment (ages 31–40) earning steady moderate incomes ($\$17,730–\$30,085$). |
| **5** | 5 | 16.67% | 47.60 | 48.0 | $17,432.80 | $16,575.00 | **Middle-Aged Low Earners:** Ages 44–51 experiencing low earnings relative to age ($\$15,736–\$20,810$). |
| **0** | 5 | 16.67% | 57.40 | 58.0 | $25,399.40 | $24,947.00 | **Pre-Retirement Moderate Earners:** Ages 52–62 maintaining modest, stable incomes ($\$24,212–\$26,909$). |
| **4** | 4 | 13.33% | 53.25 | 55.0 | $39,591.75 | $39,731.50 | **Upper-Middle Professionals:** Ages 46–57 achieving solid mid-to-upper earnings ($\$37,870–\$41,034$). |
| **1** | 4 | 13.33% | 62.50 | 63.5 | $55,866.50 | $56,543.00 | **Senior Peak Earners:** Oldest, most affluent cohort (ages 57–66) earning top-decile incomes ($\$50,576–\$59,804$). |

### Structural Evolution: $K = 3 \rightarrow K = 6$ (`08_k3_vs_optimal.png`)
* **Younger Cohort Split ($K=3$ Cluster 1 $\rightarrow$ $K=6$ Clusters 2 & 3):** Separates early-20s entry-level individuals (mean income $\$14.4\text{k}$) from established 30s professionals (mean income $\$23.4\text{k}$).
* **Older Lower-Income Split ($K=3$ Cluster 2 $\rightarrow$ $K=6$ Clusters 5 & 0):** Distinguishes 40s low-income earners (mean income $\$17.4\text{k}$) from 50s–60s moderate-income earners (mean income $\$25.4\text{k}$).
* **Older Affluent Split ($K=3$ Cluster 0 $\rightarrow$ $K=6$ Clusters 4 & 1):** Separates upper-middle professionals around age 53 (mean income $\$39.6\text{k}$) from peak-earning seniors around age 63 (mean income $\$55.9\text{k}$).

---

## 5. Conclusions & Future Work
* **Diverging Earnings Trajectories:** Although income correlates strongly with age ($r = 0.687$), age alone does not guarantee upward mobility. Among individuals aged 50+ ($n = 15$), one-third (Cluster 0) remain near the dataset median income ($\sim\$25\text{k}$), whereas two-thirds advance into upper-middle (Cluster 4) or peak (Cluster 1) income tiers.
* **Recommended Segmentation:** Adopt **$K = 6$** for downstream persona modeling, targeted outreach, or financial product segmentation where distinguishing career stage and income tier matters. Use **$K = 3$** solely for high-level executive summaries.
* **Limitations & Next Steps:** Because $N = 30$ yields cluster sizes of 4–8 observations under $K = 6$, future iterations should validate cluster stability on a larger sample size and incorporate additional covariates (e.g., education level, industry, household size, or savings rate).