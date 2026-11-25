"""import section"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
INITIAL_K = 3
MAX_K = 10
OUTPUT_DIRECTORY = "AgeIncome_Project_Results"


def parse_arguments() -> argparse.Namespace:
    """Read an optional CSV path; relative paths are resolved beside this script."""
    parser = argparse.ArgumentParser(
        description="Explore Age/Income data and perform reproducible K-Means clustering."
    )
    parser.add_argument(
        "--input",
        help="Path to the input CSV. Defaults to AgeIncome_PRJ.csv, then AgeIncome.csv.",
    )
    return parser.parse_args()


def resolve_input_path(input_argument: Optional[str]) -> Path:
    """Find the requested CSV or a supported default filename."""
    script_directory = Path(__file__).resolve().parent
    if input_argument:
        candidate = Path(input_argument).expanduser()
        if not candidate.is_absolute():
            candidate = script_directory / candidate
        if not candidate.is_file():
            raise FileNotFoundError("CSV file not found: {}".format(candidate))
        return candidate

    for filename in ("AgeIncome_PRJ.csv", "AgeIncome.csv"):
        candidate = script_directory / filename
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No input CSV found beside main.py. Expected AgeIncome_PRJ.csv "
        "or AgeIncome.csv; alternatively pass --input <path>."
    )


def normalize_column_name(column: object) -> str:
    """Normalize a column name for robust, case-insensitive matching."""
    return re.sub(r"[^a-z0-9]", "", str(column).strip().lower())


def select_features(data: pd.DataFrame) -> Tuple[str, str]:
    """Choose actual age and income columns, with a numeric-column fallback."""
    normalized_columns = {
        normalize_column_name(column): column for column in data.columns
    }
    age_aliases = ("age", "ageyears")
    income_aliases = ("income", "annualincome", "yearlyincome", "salary")
    age_column = next(
        (normalized_columns[name] for name in age_aliases if name in normalized_columns),
        None,
    )
    income_column = next(
        (
            normalized_columns[name]
            for name in income_aliases
            if name in normalized_columns
        ),
        None,
    )

    if age_column is not None and income_column is not None:
        return str(age_column), str(income_column)

    numeric_columns = list(data.select_dtypes(include=[np.number]).columns)
    if len(numeric_columns) < 2:
        raise ValueError(
            "K-Means requires two usable numeric feature columns. "
            "Columns found: {}".format(list(data.columns))
        )
    print(
        "Warning: Age/Income names were not both detected. "
        "Using the first two numeric columns present: {} and {}.".format(
            numeric_columns[0], numeric_columns[1]
        )
    )
    return str(numeric_columns[0]), str(numeric_columns[1])


def inspect_data(data: pd.DataFrame) -> None:
    """Print the requested data overview and descriptive statistics."""
    print("\n" + "=" * 72)
    print("1. DATA IMPORT AND EXPLORATION")
    print("=" * 72)
    print("\nFirst rows:\n{}".format(data.head()))
    print("\nLast rows:\n{}".format(data.tail()))
    print("\nShape: {} rows x {} columns".format(data.shape[0], data.shape[1]))
    print("\nColumn names and data types:\n{}".format(data.dtypes))
    print("\nMissing values by column:\n{}".format(data.isna().sum()))
    print("\nDuplicate rows: {}".format(int(data.duplicated().sum())))
    print("\nDescriptive statistics:\n{}".format(data.describe(include="all").T))


def save_distribution_plots(data: pd.DataFrame, features: Sequence[str], output: Path) -> None:
    """Create and save histograms for the two selected features."""
    sns.set_theme(style="whitegrid", context="notebook")
    for column, filename, color in (
        (features[0], "01_age_distribution.png", "#2878B5"),
        (features[1], "02_income_distribution.png", "#E07A2D"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        sns.histplot(data=data, x=column, kde=True, color=color, ax=axis)
        axis.set_title("{} distribution".format(column))
        axis.set_xlabel(column)
        axis.set_ylabel("Number of observations")
        figure.tight_layout()
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=data,
        x=features[0],
        y=features[1],
        color="#2878B5",
        s=65,
        edgecolor="white",
        linewidth=0.5,
        ax=axis,
    )
    axis.set_title("{} vs {}".format(features[0], features[1]))
    axis.set_xlabel(features[0])
    axis.set_ylabel(features[1])
    figure.tight_layout()
    figure.savefig(output / "03_age_vs_income.png", dpi=160)
    plt.close(figure)


def prepare_data(
    original_data: pd.DataFrame, features: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, Dict[str, int], List[str]]:
    """Run the seven numbered preprocessing steps and preserve raw-unit values."""
    print("\n" + "=" * 72)
    print("3. DATA PREPROCESSING")
    print("=" * 72)

    # Step 1: identify missing values before changing the data.
    print("\nStep 1 - Check missing values in the selected features.")
    working_data = original_data.copy()
    feature_data = working_data.loc[:, list(features)].copy()
    print(feature_data.isna().sum().to_string())

    # Step 2: coerce selected inputs to numeric and impute missing values by median.
    print("\nStep 2 - Convert selected values to numeric and impute missing values.")
    for column in features:
        feature_data[column] = pd.to_numeric(feature_data[column], errors="coerce")
    invalid_or_missing = feature_data.isna().sum()
    for column in features:
        median = feature_data[column].median()
        if pd.isna(median):
            raise ValueError(
                "Column {!r} has no valid numeric values to impute.".format(column)
            )
        feature_data[column] = feature_data[column].fillna(median)
        working_data[column] = feature_data[column]
    print("Values imputed by column:\n{}".format(invalid_or_missing.to_string()))

    # Step 3: remove duplicate observations consistently from features and export data.
    print("\nStep 3 - Remove duplicate rows, if any.")
    duplicate_mask = working_data.duplicated()
    removed_duplicates = int(duplicate_mask.sum())
    if removed_duplicates:
        working_data = working_data.loc[~duplicate_mask].copy()
        feature_data = feature_data.loc[working_data.index].copy()
    print("Duplicate rows removed: {}".format(removed_duplicates))

    # Step 4: keep only the relevant two features for clustering.
    print("\nStep 4 - Select clustering features: {}".format(list(features)))
    feature_data = working_data.loc[:, list(features)].copy()

    # Step 5: ensure the selected features are numeric.
    print("\nStep 5 - Verify numeric feature types.")
    feature_data = feature_data.apply(pd.to_numeric, errors="coerce")
    if feature_data.isna().any().any():
        raise ValueError("Selected features still contain missing or non-numeric data.")
    if not np.isfinite(feature_data.to_numpy(dtype=float)).all():
        raise ValueError("Selected features contain infinite values.")
    if len(feature_data) < 3:
        raise ValueError("At least three usable observations are required.")
    if any(feature_data[column].nunique() < 2 for column in features):
        raise ValueError("Each selected feature must contain at least two distinct values.")
    print(feature_data.dtypes.to_string())

    # Step 6: report IQR outliers but retain them; legitimate high/low incomes
    # should not be deleted automatically without evidence that they are errors.
    print("\nStep 6 - Detect and explain treatment of potential outliers (IQR rule).")
    outlier_counts: Dict[str, int] = {}
    for column in features:
        q1 = feature_data[column].quantile(0.25)
        q3 = feature_data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        count = int(
            ((feature_data[column] < lower_bound) | (feature_data[column] > upper_bound)).sum()
        )
        outlier_counts[column] = count
        print(
            "{}: {} potential outlier(s); bounds [{:.2f}, {:.2f}]. "
            "Retained for analysis.".format(column, count, lower_bound, upper_bound)
        )

    # Step 7: standardize because Euclidean distance is sensitive to feature scale.
    print("\nStep 7 - Standardize the features with StandardScaler.")
    print(
        "Scaling is essential for Euclidean-distance K-Means: Income is measured "
        "in much larger numbers than Age. The unscaled feature would dominate "
        "distance calculations. The unscaled data is retained for interpretation."
    )
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(feature_data)
    scaled_data = pd.DataFrame(
        scaled_values, columns=list(features), index=feature_data.index
    )

    return working_data, scaled_data, scaler, outlier_counts, list(features)


def fit_kmeans(scaled_data: pd.DataFrame, k: int) -> KMeans:
    """Fit a reproducible K-Means estimator."""
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    model.fit(scaled_data)
    return model


def cluster_statistics(
    data_with_labels: pd.DataFrame, features: Sequence[str]
) -> pd.DataFrame:
    """Summarize cluster sizes and means in the original feature units."""
    grouped = data_with_labels.groupby("Cluster")[list(features)]
    means = grouped.mean().add_prefix("Mean_")
    medians = grouped.median().add_prefix("Median_")
    counts = grouped.size().rename("Count")
    result = pd.concat([counts, means, medians], axis=1)
    result["Percentage"] = (result["Count"] / len(data_with_labels) * 100).round(2)
    return result


def plot_clusters(
    data_with_labels: pd.DataFrame,
    centers: pd.DataFrame,
    features: Sequence[str],
    title: str,
    filename: str,
    output: Path,
) -> None:
    """Plot original-unit observations and their inverse-scaled centroids."""
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=data_with_labels,
        x=features[0],
        y=features[1],
        hue="Cluster",
        palette="tab10",
        s=70,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
        ax=axis,
    )
    axis.scatter(
        centers[features[0]],
        centers[features[1]],
        marker="X",
        s=250,
        c="black",
        edgecolors="white",
        linewidths=1.2,
        label="Centroids",
        zorder=5,
    )
    axis.set_title(title)
    axis.set_xlabel(features[0])
    axis.set_ylabel(features[1])
    axis.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(output / filename, dpi=160, bbox_inches="tight")
    plt.close(figure)


def choose_elbow(inertias: Dict[int, float]) -> int:
    """Estimate the elbow as the point farthest from the end-point line."""
    ks = np.array(list(inertias.keys()), dtype=float)
    values = np.array(list(inertias.values()), dtype=float)
    if len(ks) < 3:
        return int(ks[-1])

    x = (ks - ks.min()) / (ks.max() - ks.min())
    y_range = values.max() - values.min()
    y = (values - values.min()) / y_range if y_range else np.zeros_like(values)
    line_y = y[0] + (y[-1] - y[0]) * x
    return int(ks[np.argmax(np.abs(y - line_y))])


def evaluate_cluster_counts(
    scaled_data: pd.DataFrame, output: Path
) -> Tuple[Dict[int, float], Dict[int, float], int, int]:
    """Compute inertia and silhouette values and save both model-selection plots."""
    print("\n" + "=" * 72)
    print("5. CHOOSING THE NUMBER OF CLUSTERS")
    print("=" * 72)
    max_k = min(MAX_K, len(scaled_data) - 1)
    if max_k < 3:
        raise ValueError("At least four observations are needed for K selection.")

    inertias: Dict[int, float] = {}
    silhouette_scores: Dict[int, float] = {}
    for k in range(1, max_k + 1):
        model = fit_kmeans(scaled_data, k)
        inertias[k] = float(model.inertia_)
        if k >= 2:
            labels = model.labels_
            if 1 < len(np.unique(labels)) < len(scaled_data):
                silhouette_scores[k] = float(
                    silhouette_score(scaled_data, labels, metric="euclidean")
                )

    elbow_k = choose_elbow(inertias)
    if not silhouette_scores:
        raise ValueError("Silhouette scores could not be calculated for this dataset.")
    silhouette_k = max(silhouette_scores, key=silhouette_scores.get)

    print("\nInertia (WCSS) by K:")
    for k, score in inertias.items():
        print("  K={}: {:.4f}".format(k, score))
    print("\nSilhouette scores:")
    for k, score in silhouette_scores.items():
        print("  K={}: {:.4f}".format(k, score))
    print(
        "\nElbow heuristic suggests K={}. It estimates a bend in the inertia "
        "curve; the elbow can be subjective.".format(elbow_k)
    )
    print(
        "Best tested silhouette score suggests K={} ({:.4f}).".format(
            silhouette_k, silhouette_scores[silhouette_k]
        )
    )
    print(
        "The silhouette coefficient ranges from -1 to +1: close to +1 means "
        "well-separated clusters, near 0 means overlap, and a negative score "
        "can indicate observations assigned to the wrong cluster."
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(list(inertias.keys()), list(inertias.values()), marker="o", color="#2878B5")
    axis.axvline(elbow_k, color="#E07A2D", linestyle="--", label="Estimated elbow K")
    axis.set_title("Elbow method")
    axis.set_xlabel("Number of clusters (K)")
    axis.set_ylabel("Inertia (within-cluster sum of squares)")
    axis.set_xticks(list(inertias.keys()))
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "04_elbow_method.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        list(silhouette_scores.keys()),
        list(silhouette_scores.values()),
        marker="o",
        color="#32865C",
    )
    axis.axvline(
        silhouette_k, color="#E07A2D", linestyle="--", label="Best silhouette K"
    )
    axis.set_title("Silhouette score by number of clusters")
    axis.set_xlabel("Number of clusters (K)")
    axis.set_ylabel("Silhouette score")
    axis.set_xticks(list(silhouette_scores.keys()))
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "05_silhouette_scores.png", dpi=160)
    plt.close(figure)

    return inertias, silhouette_scores, elbow_k, silhouette_k


def build_summary(
    row_count: int,
    features: Sequence[str],
    preprocessing: Sequence[str],
    outlier_counts: Dict[str, int],
    elbow_k: int,
    optimal_k: int,
    silhouette_scores: Dict[int, float],
    inertias: Dict[int, float],
    initial_stats: pd.DataFrame,
    optimal_stats: pd.DataFrame,
) -> str:
    """Create an academic, accessible report for console and text-file output."""
    optimal_score = silhouette_scores[optimal_k]
    lines = [
        "AGE AND INCOME K-MEANS PROJECT SUMMARY",
        "=" * 48,
        "Observations analyzed: {}".format(row_count),
        "Features used: {}".format(", ".join(features)),
        "",
        "Machine-learning type and objective",
        "K-Means is an unsupervised learning algorithm. For this exercise, the "
        "rows are treated as unlabeled: Age and Income are used to find homogeneous "
        "groups of people. There is no known target label. A supervised "
        "X_train/X_test/y_train/y_test split is therefore not normally required; "
        "clustering evaluates structure in the data rather than predicting a "
        "held-out labeled target.",
        "",
        "Preprocessing completed",
    ]
    lines.extend("- {}".format(item) for item in preprocessing)
    lines.append(
        "- Potential IQR outliers retained (not removed): {}".format(
            ", ".join("{}={}".format(name, count) for name, count in outlier_counts.items())
        )
    )
    lines.extend(
        [
            "",
            "Cluster-count selection",
            "Estimated elbow K: {} (an approximate, visually subjective bend).".format(
                elbow_k
            ),
            "Selected K: {} (highest tested silhouette score, {:.4f}).".format(
                optimal_k, optimal_score
            ),
            "The elbow method measures the decrease in within-cluster inertia as "
            "K increases. Silhouette balances within-cluster cohesion and "
            "between-cluster separation. These methods can disagree; the selected "
            "K is a data-driven proposal, not a universal truth.",
            "",
            "K=3 versus selected K",
            "K=3 inertia: {:.4f}; silhouette: {}.".format(
                inertias[INITIAL_K],
                "{:.4f}".format(silhouette_scores[INITIAL_K])
                if INITIAL_K in silhouette_scores
                else "not available",
            ),
            "Selected K={} inertia: {:.4f}; silhouette: {:.4f}.".format(
                optimal_k, inertias[optimal_k], optimal_score
            ),
            "K=3 cluster profile:\n{}".format(initial_stats.to_string()),
            "",
            "Selected-K cluster profile:\n{}".format(optimal_stats.to_string()),
            "",
            "Interpretation and limitations",
            "Cluster numbers are arbitrary identifiers, not rankings. Compare the "
            "mean Age and Income values to describe each group. K-Means is "
            "distance-based and sensitive to scaling, outliers, and the chosen K. "
            "Because there are no reference labels, K-Means does not directly "
            "provide accuracy or a confusion matrix; no accuracy value is invented.",
            "Recommendations: describe each segment using its original-unit "
            "centroid and size; compare the K=3 and selected-K profiles; and "
            "validate whether the segments are useful for the project's real "
            "question before making decisions about people.",
        ]
    )
    return "\n".join(lines)


def run_analysis(csv_path: Path) -> None:
    """Run exploration, preprocessing, modeling, evaluation, plots, and exports."""
    try:
        raw_data = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as error:
        raise ValueError("The CSV file is empty: {}".format(csv_path)) from error
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        raise ValueError("Could not parse CSV file {}: {}".format(csv_path, error)) from error

    if raw_data.empty or len(raw_data.columns) == 0:
        raise ValueError("The CSV contains no observations or columns.")

    inspect_data(raw_data)
    features = select_features(raw_data)
    print("\nSelected variables: {}".format(list(features)))
    print(
        "These two numeric variables represent the requested segmentation "
        "dimensions directly: age and income. K-Means can group observations "
        "using their scaled Euclidean distances."
    )

    output_directory = csv_path.parent / OUTPUT_DIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)

    clean_data, scaled_data, scaler, outlier_counts, features = prepare_data(
        raw_data, features
    )
    clean_feature_data = clean_data.loc[:, list(features)].copy()
    save_distribution_plots(clean_feature_data, features, output_directory)

    print("\n" + "=" * 72)
    print("4. INITIAL K-MEANS MODEL (K=3)")
    print("=" * 72)
    initial_model = fit_kmeans(scaled_data, INITIAL_K)
    initial_data = clean_data.copy()
    initial_data["Cluster"] = initial_model.labels_
    initial_centers = pd.DataFrame(
        scaler.inverse_transform(initial_model.cluster_centers_),
        columns=list(features),
    )
    initial_stats = cluster_statistics(initial_data, features)
    print("\nCluster counts:\n{}".format(initial_data["Cluster"].value_counts().sort_index()))
    print("\nCentroids in original units:\n{}".format(initial_centers.to_string(index=False)))
    print("\nCluster characteristics:\n{}".format(initial_stats.to_string()))
    plot_clusters(
        initial_data,
        initial_centers,
        features,
        "K-Means clusters (K=3)",
        "06_clusters_k3.png",
        output_directory,
    )

    inertias, silhouette_scores, elbow_k, optimal_k = evaluate_cluster_counts(
        scaled_data, output_directory
    )

    print("\n" + "=" * 72)
    print("6. COMPARISON: K=3 AND SELECTED K")
    print("=" * 72)
    optimal_model = fit_kmeans(scaled_data, optimal_k)
    optimal_data = clean_data.copy()
    optimal_data["Cluster"] = optimal_model.labels_
    optimal_centers = pd.DataFrame(
        scaler.inverse_transform(optimal_model.cluster_centers_),
        columns=list(features),
    )
    optimal_stats = cluster_statistics(optimal_data, features)
    print(
        "K=3: inertia={:.4f}, silhouette={:.4f}".format(
            inertias[INITIAL_K], silhouette_scores[INITIAL_K]
        )
    )
    print(
        "K={}: inertia={:.4f}, silhouette={:.4f}".format(
            optimal_k, inertias[optimal_k], silhouette_scores[optimal_k]
        )
    )
    print("\nK=3 cluster sizes:\n{}".format(initial_stats["Count"].to_string()))
    print(
        "\nK={} cluster sizes:\n{}".format(
            optimal_k, optimal_stats["Count"].to_string()
        )
    )
    if optimal_k == INITIAL_K:
        print(
            "\nThe selected K is 3, so both models have the same K. "
            "Re-fitting with the same random_state should reproduce the same clustering."
        )
    else:
        print(
            "\nThe cluster assignments and centroids can change because the selected "
            "model uses a different number of groups. Compare the original-unit "
            "centroids and sizes above; cluster IDs do not correspond across models."
        )
    print(
        "No accuracy score or confusion matrix is reported because no ground-truth "
        "cluster labels exist for this unsupervised exercise."
    )
    plot_clusters(
        optimal_data,
        optimal_centers,
        features,
        "K-Means clusters (selected K={})".format(optimal_k),
        "07_clusters_optimal_k.png",
        output_directory,
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, dataset, centers, k, subtitle in (
        (axes[0], initial_data, initial_centers, INITIAL_K, "K=3"),
        (axes[1], optimal_data, optimal_centers, optimal_k, "Selected K={}".format(optimal_k)),
    ):
        sns.scatterplot(
            data=dataset,
            x=features[0],
            y=features[1],
            hue="Cluster",
            palette="tab10",
            s=55,
            alpha=0.8,
            legend=False,
            ax=axis,
        )
        axis.scatter(
            centers[features[0]],
            centers[features[1]],
            marker="X",
            s=180,
            c="black",
            edgecolors="white",
            linewidths=1,
            label="Centroids",
        )
        axis.set_title(subtitle)
        axis.set_xlabel(features[0])
        axis.set_ylabel(features[1])
        axis.legend()
    figure.suptitle("Cluster comparison", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_directory / "08_k3_vs_optimal.png", dpi=160)
    plt.close(figure)

    # The exported rows retain every original column and the original feature units.
    initial_export = clean_data.copy()
    initial_export["Cluster_K3"] = initial_model.labels_
    initial_export["Cluster_Optimal"] = optimal_model.labels_
    initial_export.to_csv(output_directory / "clustered_age_income_data.csv", index=False)
    optimal_stats.to_csv(output_directory / "cluster_statistics.csv", index=True)

    preprocessing = [
        "Checked selected-feature missing values; missing/non-numeric values were converted and median-imputed when needed.",
        "Removed exact duplicate rows when present ({} removed).".format(
            len(raw_data) - len(clean_data)
        ),
        "Selected the actual CSV columns {} and {}.".format(features[0], features[1]),
        "Verified selected features are numeric, finite, and contain enough observations.",
        "Detected IQR outliers and retained them; no observation was removed solely for being an outlier.",
        "Standardized features with StandardScaler; original values were retained for interpretation.",
    ]
    summary = build_summary(
        row_count=len(clean_data),
        features=features,
        preprocessing=preprocessing,
        outlier_counts=outlier_counts,
        elbow_k=elbow_k,
        optimal_k=optimal_k,
        silhouette_scores=silhouette_scores,
        inertias=inertias,
        initial_stats=initial_stats,
        optimal_stats=optimal_stats,
    )
    summary_path = output_directory / "results_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print("\n" + "=" * 72)
    print("7. AUTOMATIC SUMMARY")
    print("=" * 72)
    print(summary)
    print("\nResults saved in: {}".format(output_directory.resolve()))


def main() -> int:
    """Program entry point with clear, user-facing error messages."""
    arguments = parse_arguments()
    try:
        csv_path = resolve_input_path(arguments.input)
        print("Loading dataset: {}".format(csv_path))
        run_analysis(csv_path)
    except (FileNotFoundError, ValueError, OSError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())