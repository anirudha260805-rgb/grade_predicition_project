"""Cluster student scores and map clusters to letter grades."""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = "Student_Performance.csv"
MODEL_PATH = "student_performance_clustering_model.joblib"
MAPPING_PATH = "cluster_to_grade_mapping.joblib"


def _load_grade_artifacts():
    """Load the saved clustering pipeline and its cluster-to-grade mapping."""
    return joblib.load(MODEL_PATH), joblib.load(MAPPING_PATH)


def predict_final_grade(overall_score):
    """Predict a letter grade using the saved KMeans model and cluster mapping.

    Since evaluation only has an overall score, select the cluster whose
    centroid is closest on the score feature; the grade feature is unknown.
    """
    score = float(overall_score)
    if not (-float("inf") < score < float("inf")):
        raise ValueError("Overall score must be a finite number.")

    model, mapping = _load_grade_artifacts()
    feature_names = list(model.feature_names_in_)
    score_index = feature_names.index("overall_score")
    scaler = model.named_steps["scaler"]
    kmeans = model.named_steps["regressor"]
    scaled_score = (score - scaler.mean_[score_index]) / scaler.scale_[score_index]
    cluster = int(
        abs(kmeans.cluster_centers_[:, score_index] - scaled_score).argmin()
    )
    return str(mapping[cluster]).upper()


def train_model():
    """Fit, save, and plot the KMeans clustering model and grade mapping."""
    # Encode the letter grades for numerical clustering.
    ovr = pd.read_csv(DATA_PATH)
    grade_values = ovr["final_grade"].map(
        {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}
    )
    X = pd.DataFrame(
        {
            "final_grade": grade_values,
            "overall_score": ovr["overall_score"],
        }
    )

    # Standardize the two plotting features before fitting the six-cluster model.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", KMeans(n_clusters=6, random_state=42)),
        ]
    )
    ovr["cluster"] = model.fit_predict(X)
    cluster_labels = ovr["cluster"].to_numpy()

    # Silhouette measures how well-separated the clusters are.
    scaled_X = model.named_steps["scaler"].transform(X)
    silhouette = silhouette_score(scaled_X, cluster_labels)
    print(f"Silhouette score : {silhouette:.3f}")

    train_clusters = model.named_steps["regressor"].labels_
    mapping = (
        pd.DataFrame(
            {"cluster": train_clusters, "final_grade": ovr["final_grade"]}
        )
        .groupby("cluster")["final_grade"]
        .agg(lambda grades: grades.mode()[0])
        .to_dict()
    )

    joblib.dump(model, MODEL_PATH)
    joblib.dump(mapping, MAPPING_PATH)

    # Plot each record and the centroids after converting them back to original units.
    plot_x = "final_grade"
    plot_y = "overall_score"
    centers_scaled = model.named_steps["regressor"].cluster_centers_
    centers_original = model.named_steps["scaler"].inverse_transform(centers_scaled)

    plt.scatter(
        X[plot_x],
        X[plot_y],
        c=cluster_labels,
        cmap="tab10",
        s=30,
        alpha=0.6,
        label="Students",
    )
    plt.scatter(
        centers_original[:, 0],
        centers_original[:, 1],
        s=220,
        c="red",
        marker="X",
        edgecolors="black",
        label="Cluster centroids",
    )
    plt.xticks(range(6), ["a", "b", "c", "d", "e", "f"])
    plt.xlabel("final_grade")
    plt.ylabel("overall_score")
    plt.title("Student Performance Clusters")
    plt.colorbar(label="Cluster")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_model()
