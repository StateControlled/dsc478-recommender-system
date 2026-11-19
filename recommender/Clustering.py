import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from dataclasses import dataclass


RANDOM_STATE=42
NUM_CLUSTERS = 4
DO_RANGE = False


@dataclass
class ClusterModel:
    """
    Represents users grouped by taste, with one book-similarity matrix per group.

    labels:     cluster of every user (row order of the rating matrix)
    combined:   cluster - blended books x books similarity matrix used to predict for that cluster's users
    item_mean:  cluster - mean rating of every book among the cluster's users
    """

    labels: np.ndarray
    global_sim: np.ndarray
    combined: dict[int, np.ndarray]
    item_mean: dict[int, np.ndarray]
    blend: float
    silhouette: dict[int, float] | None = None

    def similarity_for(self, user: int) -> np.ndarray:
        return self.combined[int(self.labels[user])]


#######################################################################################################################
#
# Clustering
#
#######################################################################################################################


def user_features(ratings: np.ndarray,
                  center: bool=True,
                  n_components: int=10) -> np.ndarray:
    """
    Turns the users x books rating matrix into a clustered feature matrix

    :param ratings:         users x books matrix
    :param center:          center
    :param n_components:    project the users onto this many SVD components before clustering.

    :return: users x features matrix
    """

    result: np.ndarray = np.asarray(ratings, dtype=np.float64)

    # subtract each user's mean rating from their rated entries
    if center:
        mask = result > 0
        means = result.sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
        result = np.where(mask, result - means[:, None], 0.0)

    # project onto SVD components
    # SVD = Singular Value Decomposition
    if n_components < min(result.shape):
        result = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE).fit_transform(result)

    return result


def cluster_users(ratings: np.ndarray,
                  n_clusters: int=NUM_CLUSTERS,
                  seed: int=RANDOM_STATE,
                  center: bool=True) -> tuple[np.ndarray, KMeans]:
    """
    Clusters users by taste with k-means.

    :return: the cluster label of every user (row order of `ratings`), and the fitted KMeans model
    """

    features: np.ndarray = user_features(ratings, center)
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, max_iter=300, random_state=seed)
    # Compute cluster centers and predict cluster index for each sample
    labels = km.fit_predict(features)

    return labels, km


def choose_k(ratings: np.ndarray,
             k_values=range(2, 9),
             seed: int=RANDOM_STATE,
             center: bool=True) -> tuple[int, dict[int, float]]:
    """
    Picks the number of clusters with the best silhouette score.

    :return: the best k and the silhouette score of every k
    """

    features: np.ndarray = user_features(ratings, center)
    scores: dict[int, float] = {}

    # k values 2 - 8 by default.
    for k in k_values:
        if k >= len(features):
            break

        labels = KMeans(n_clusters=k, init="k-means++", n_init=5, max_iter=300, random_state=seed).fit_predict(features)

        if len(np.unique(labels)) < 2:
            continue

        scores[k] = float(silhouette_score(features, labels, sample_size=min(len(features), 2000), random_state=seed))

    if not scores:
        raise ValueError("Not enough users to choose a number of clusters.")

    # higher = better separated clusters
    return max(scores, key=scores.get), scores
