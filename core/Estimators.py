import numpy as np
from numpy import linalg as la

def standard_estimator(dataMat: np.matrix, user: int, simMeas, item: int) -> int | float:
    """
    standEst() estimates the rating a user would give an item, from the ratings that user has
    already given. Each of those ratings is weighted by how similar its item is to the target
    item, where similarity is measured over the users who rated both items, and the weighted
    average is the estimate. This is the standard implementation of item-based collaborative
    filtering.

    :param dataMat: the user-item ratings matrix, with 0 for a missing rating
    :param user:    is the row index of the user whose rating is being predicted
    :param simMeas: is the similarity function to use
    :param item:    is the column index of the item whose rating is being predicted
    :return: Returns the predicted rating, or 0 when the item cannot be compared with anything the user has rated
    """

    n = np.shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    data=np.asmatrix(dataMat)

    for j in range(n):
        userRating = data[user,j]
        if userRating == 0:
            continue

        # The rows where both items were rated, which is where the two items can be compared.
        overLap = np.nonzero(np.logical_and(data[:,item]>0, data[:,j]>0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(data[overLap,item], data[overLap,j])

        #print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating

    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal


def svd_estimator(dataMat: np.matrix, user: int, simMeas, item: int) -> int | float:
    """
    svdEst() estimates the same rating, but compares items in a reduced space rather than on the
    raw ratings. The ratings matrix is factored with the SVD and each item is described by its
    first k dimensions in the reduced dimension space. So every pair of items can be compared
    even when few users have rated both.
    k = 4 is hard-coded here for simplicity. Ideally, the SVD decomposition should be performed
    outside this function and the transformed items passed in along with the ratings matrix,
    rather than recomputed on every call.

    :param dataMat: the user-item ratings matrix, with 0 for a missing rating
    :param user:    the row index of the user whose rating is being predicted
    :param simMeas: the similarity function to use
    :param item:    the column index of the item whose rating is being predicted
    :return:        the predicted rating, or 0 when no similarity could be computed
    """

    n = np.shape(dataMat)[1]
    k = 4 #number of dimension for SVD
    simTotal = 0.0; ratSimTotal = 0.0
    data=np.asmatrix(dataMat)
    # Factor the ratings into U, Sigma and VT, so that the items can be described by k dimensions.
    U,Sigma,VT = la.svd(data)
    Sig_k = np.asmatrix(np.eye(k)*Sigma[:k]) #arrange Sig_k into a diagonal matrix
    xformedItems = data.T * U[:,:k] * Sig_k.I  #create transformed items

    for j in range(n):
        userRating = data[user,j]
        if userRating == 0 or j==item:
            continue

        similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
        #print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating

    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal
