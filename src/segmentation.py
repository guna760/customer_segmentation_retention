import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from .data_preprocessing import SEGMENT_FEATURES


def train_segmentation(
    df,
    n_clusters=4
):

    # -------------------------------------------------
    # Select features
    # -------------------------------------------------

    X = df[SEGMENT_FEATURES].copy()

    # -------------------------------------------------
    # Scaling
    # -------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -------------------------------------------------
    # K-Means
    # -------------------------------------------------

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(
        X_scaled
    )

    # Add cluster to dataframe
    df["segment"] = clusters

    # -------------------------------------------------
    # Silhouette Score
    # -------------------------------------------------

    silhouette = silhouette_score(
        X_scaled,
        clusters
    )

    print(
        f"Segmentation Silhouette Score: "
        f"{silhouette:.4f}"
    )

    # -------------------------------------------------
    # Save models
    # -------------------------------------------------

    joblib.dump(
        scaler,
        "models/scaler_segmentation.pkl"
    )

    joblib.dump(
        kmeans,
        "models/kmeans_model.pkl"
    )

    return df