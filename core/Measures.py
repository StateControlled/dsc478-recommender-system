import numpy as np
from numpy import linalg as la

def euclidean_similarity(inA, inB) -> float:
    """
    Euclidean Similarity.
    :param inA:
    :param inB:
    :return:
    """

    # la.norm is the length of the difference vector, so similarity falls as the two ratings diverge.
    return 1.0 / (1.0 + la.norm(inA - inB))


def pearson_similarity(inA, inB) -> float:
    """
    Pearson Similarity.
    :param inA:
    :param inB:
    :return:
    """

    # A correlation cannot be estimated from 1 or 2 points, so report no similarity instead of guessing.
    if len(inA) < 3:
        return 0.0

    # corrcoef returns a 2x2 matrix; [0][1] is the correlation, rescaled here from [-1,1] to [0,1].
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar = 0)[0][1]


def cosine_similarity(inA, inB) -> float:
    """
    Cosine Similarity.
    :param inA:
    :param inB:
    :return:
    """

    # inA.T * inB is the dot product of the two columns, as a 1x1 matrix.
    numerator = float((inA.T * inB)[0,0])
    denominator = la.norm(inA) * la.norm(inB)

    end = numerator / denominator

    # The cosine of the angle between the two vectors, rescaled from [-1,1] to [0,1].
    return 0.5 + 0.5 * end
