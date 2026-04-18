from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "mall_customers.csv"
OUTPUT_DIR = ROOT / "outputs"
SUMMARY_PATH = OUTPUT_DIR / "customer_segmentation_summary.txt"
ELBOW_PATH = OUTPUT_DIR / "customer_segmentation_elbow.png"
CLUSTER_PATH = OUTPUT_DIR / "customer_segmentation_clusters.png"

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")


def assign_segment_name(age: float, income: float, score: float) -> str:
    if income >= 70 and score >= 70:
        return "High-Value Spenders"
    if income >= 70 and score < 40:
        return "Affluent Cautious Shoppers"
    if income < 40 and score >= 60:
        return "Budget Enthusiasts"
    if income < 40 and score < 40:
        return "Low-Engagement Customers"
    if age < 35 and score >= 50:
        return "Young Active Shoppers"
    return "Balanced Mainstream Customers"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    sns.set_theme(style="whitegrid")
    df = pd.read_csv(DATA_PATH)
    feature_columns = [
        "Annual Income (k$)",
        "Spending Score (1-100)",
    ]
    features = df[feature_columns]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    evaluation_rows: list[dict[str, float]] = []
    best_k = None
    best_score = -1.0
    best_model = None

    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = model.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, labels)
        evaluation_rows.append({"k": k, "silhouette_score": score})
        if score > best_score:
            best_k = k
            best_score = score
            best_model = model

    if best_model is None or best_k is None:
        raise RuntimeError("Unable to fit a clustering model.")

    df["Cluster"] = best_model.labels_

    profile = (
        df.groupby("Cluster")
        .agg(
            customer_count=("CustomerID", "count"),
            avg_age=("Age", "mean"),
            avg_income_k=("Annual Income (k$)", "mean"),
            avg_spending_score=("Spending Score (1-100)", "mean"),
            dominant_gender=("Genre", lambda values: values.mode().iat[0]),
        )
        .reset_index()
        .sort_values(["avg_income_k", "avg_spending_score"], ascending=[False, False])
        .reset_index(drop=True)
    )
    profile["segment_name"] = profile.apply(
        lambda row: assign_segment_name(
            row["avg_age"], row["avg_income_k"], row["avg_spending_score"]
        ),
        axis=1,
    )

    # Preserve readable cluster labels in the chart legend.
    cluster_name_map = {
        int(row.Cluster): f"{row.segment_name} (Cluster {int(row.Cluster)})"
        for row in profile.itertuples()
    }
    df["Segment Label"] = df["Cluster"].map(cluster_name_map)

    evaluation_df = pd.DataFrame(evaluation_rows)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=evaluation_df, x="k", y="silhouette_score", marker="o", linewidth=2.5)
    plt.title("Silhouette Scores by Cluster Count")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.xticks(range(2, 9))
    plt.tight_layout()
    plt.savefig(ELBOW_PATH, dpi=220)
    plt.close()

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Segment Label",
        palette="tab10",
        s=85,
        alpha=0.85,
    )
    centers = scaler.inverse_transform(best_model.cluster_centers_)
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=320,
        color="black",
        label="Cluster Centers",
    )
    plt.title("Customer Segments Based on Income and Spending")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(CLUSTER_PATH, dpi=220, bbox_inches="tight")
    plt.close()

    summary_lines = [
        "Customer Segmentation Project Summary",
        "=" * 40,
        f"Dataset: {DATA_PATH.name}",
        f"Rows: {len(df)}",
        f"Selected cluster count (best silhouette score): {best_k}",
        f"Best silhouette score: {best_score:.4f}",
        "",
        "Cluster profiles:",
    ]

    for row in profile.itertuples():
        summary_lines.extend(
            [
                (
                    f"- Cluster {int(row.Cluster)} | {row.segment_name}: "
                    f"{int(row.customer_count)} customers, "
                    f"avg age {row.avg_age:.1f}, "
                    f"avg income {row.avg_income_k:.1f}k, "
                    f"avg spending score {row.avg_spending_score:.1f}, "
                    f"dominant gender {row.dominant_gender}"
                )
            ]
        )

    SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"\nSaved silhouette chart to: {ELBOW_PATH}")
    print(f"Saved cluster chart to: {CLUSTER_PATH}")
    print(f"Saved written summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
