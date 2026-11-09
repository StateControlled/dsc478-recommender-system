import numpy as np
from collections import Counter

# Adapted from Assignment 2

EUCLIDEAN_DISTANCE = 0
COSINE_SIMILARITY = 1

def measure_to_str(measure: int) -> str:
    if measure == EUCLIDEAN_DISTANCE:
        return "Euclidean Distance"
    else:
        return "Cosine Similarity"


def _calc_euclidean_distance(query: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Calculates Euclidean distance between two vectors.
    :param query: the query vector.
    :param data:  the training data matrix.
    :return: a vector
    """
    radicand: np.ndarray = ((data - query) ** 2).sum(axis=1)
    return np.sqrt(radicand)


def _calc_cosine_similarity(query: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    The distance measure will be the inverse of Cosine similarity.
    Calculates the vector norm for each row in DATA as well as the norm for the QUERY vector, computes cosine similarity, and returns the inverse.
    :param query: the query vector.
    :param data:  the training data matrix.
    :return: a vector
    """
    norm_data = np.array([np.linalg.norm(data[i]) for i in range(len(data))])
    norm_query = np.linalg.norm(query)

    denominator = norm_data * norm_query

    cosine_sim: np.ndarray = np.dot(data, query) / denominator

    cosine_sim[np.isinf(cosine_sim)] = 0  # handle divide by zero errors

    return 1 - cosine_sim  # return inverse


def _calculate_distances(query: np.ndarray, data: np.ndarray, measure: int) -> np.ndarray:
    """
    Calculates the distances using either Euclidean distance or cosine similarity.
    :param query:   the query vector.
    :param data:    the training data matrix.
    :param measure: 0 for Euclidean distance, 1 for Cosine Similarity.
    :return:        a matrix containing (unsorted) distance measures from the query vector.
    """
    if measure == EUCLIDEAN_DISTANCE:
        _distances: np.ndarray = _calc_euclidean_distance(query, data)
    elif measure == COSINE_SIMILARITY:
        _distances = _calc_cosine_similarity(query, data)
    else:
        raise ValueError(f"Unknown measure: {measure}")

    return _distances


def knn(query: np.ndarray, data: np.ndarray, k: int, measure: int) -> tuple[np.ndarray, np.ndarray]:
    """
    K-Nearest-Neighbor - returns the indices of the k nearest neighbors.
    :param query:   the query vector.
    :param data:    the training data matrix.
    :param k:       the number of similar items to find.
    :param measure: 0 for Euclidean distance, 1 for Cosine Similarity.
    :return:        the indices of the top K neighbors.
    """

    _distances = _calculate_distances(query, data, measure)

    indices: np.ndarray = np.argsort(_distances)  # sorting
    # return the indices of K nearest neighbors
    return indices[:k], _distances


def knn_classify(query: np.ndarray, data: np.ndarray, categories: np.ndarray, k: int, measure: int) -> tuple[int, Counter]:
    """
    K-Nearest-Neighbor classifier method. Returns the predicted class for the given query and labels for top k similar items.
    :param query:       the instance to be classified.
    :param data:        the training data matrix.
    :param categories:  the training labels.
    :param k:           the number of similar items to use for classification.
    :param measure:     0 for Euclidean distance, 1 for Cosine Similarity.
    :return:            the predicted class for the instance
    """

    # get K nearest neighbors
    neighbor_indices, distances = knn(query, data, k, measure)

    # get labels of closest documents
    neigh_labels: np.ndarray = categories[neighbor_indices]

    # tally results
    counter: Counter = Counter(neigh_labels)

    predicted_label: int = counter.most_common(1)[0][0]
    return predicted_label, counter
