import argparse
import os
import sys

import pandas as pd
import numpy as np

from core import Common

from IPython.core.display_functions import display


def cross_validate_user(user_vectors: np.matrix,
                        similarity_matrix: np.ndarray,
                        user: int,
                        k: int,
                        test_ratio: float,
                        seed=None) -> tuple[float, int]:
    """
    The cross_validate_user function performs evaluation on a single user based on
    the test_ratio: with test_ratio = 0.2, a randomly selected 20 percent of rated
    items by the user are withheld as test data. The remaining part of the user
    profile is used as input for the estimation functions to predict the withheld
    ratings and compute the error for this user.

    The seed argument controls the random selection of the withheld items. With
    seed=None the selection is different on every call, as before. With an integer
    seed, the same user and the same test_ratio always withhold the same items, so
    the evaluation is reproducible. The user id is added to the seed internally, so
    the same seed value can be passed for every user and each one still withholds a
    different part of its profile.

    :param user_vectors:        the user-item ratings matrix, with 0 for a missing rating
    :param user:                the row index of the user to evaluate
    :param k:                   for top k
    :param test_ratio:          the share of this user's ratings to withhold as test data
    :param similarity_matrix:   the static similarity matrix
    :param seed:                fixes the random selection when it is an integer; see the note above
    :return: the total absolute error over this user's withheld ratings, and the number of test cases they make up.
    """

    data_matrix = np.array(user_vectors)

    # The user id is added to the seed so that every user draws a different test sample.
    if seed is not None:
        np.random.seed(seed + user)

    number_of_items = np.shape(data_matrix)[1]

    # The column indices of the items this user has actually rated.
    rated_items_by_user = np.array([i for i in range(number_of_items) if data_matrix[user, i] > 0])
    test_size = int(test_ratio * len(rated_items_by_user))

    # Draw test_size positions within that list, at random and with replacement.
    test_indices = np.random.randint(0, len(rated_items_by_user), test_size)

    # Turn those positions into the item ids they refer to.
    withheld_items = rated_items_by_user[test_indices]
    original_user_profile = np.copy(data_matrix[user])

    # Zero the withheld items so that they are not used in the rating estimation below.
    data_matrix[user, withheld_items] = 0
    error_u = 0.0
    count_u = len(withheld_items)

    # Compute absolute error for user u over all test items
    for item in withheld_items:
        # Estimate rating on the withheld item
        estimated_score = rating_prediction(data_matrix, user, similarity_matrix, item, k)

        if estimated_score is not None:
            error_u = error_u + abs(estimated_score - original_user_profile[item])
        else:  # treat None as "zero"
            error_u = error_u + original_user_profile[item]

    # Now restore ratings of the withheld items to the user profile
    for item in withheld_items:
        data_matrix[user, item] = original_user_profile[item]

    # Return sum of absolute errors and the count of test cases for this user
    # Note that these will have to be accumulated for each user to compute MAE
    return error_u, count_u


def test(user_vectors: np.matrix, similarity_matrix: np.ndarray, test_ratio: float, k: int, seed=None) -> float:
    """
    :param user_vectors:     the user-item ratings matrix, with 0 for a missing rating
    :param test_ratio:  the share of this user's ratings to withhold as test data
    :param k:           for top k matches
    :param seed:        passed straight through to cross_validate_user
    :return: the mean absolute error over the whole data set
    """

    total_error: float = 0.0
    total_cases: int = 0

    # iterate over all users
    for user_id in range(np.shape(user_vectors)[0]):
        err, count = cross_validate_user(user_vectors, similarity_matrix, user_id, k, test_ratio, seed)

        total_error += err
        total_cases += count

    return total_error / total_cases  # MAE


def rating_prediction(user_vectors: np.matrix, user: int, similarity_matrix: np.ndarray, item: int,
                      k: int = 10) -> float | None:
    """
    A prediction function. Returns the predicted rating based on top k most similar items. If the user has not rated any of the top k items, returns None.

    :param user_vectors:             the user-item ratings matrix
    :param user:                target user
    :param item:                target item
    :param k:                   top k items
    :param similarity_matrix:   similarity matrix computed in training
    :return:                    the predicted rating or None if the user has rated none of the top k most similar items
    """

    # user_row = dataMat[user]
    # print(user_row)

    user_row = np.asarray(user_vectors)[user]

    similarities = similarity_matrix[item].copy()
    similarities[item] = -np.inf  # exclude the item itself
    # top k most similar items
    _temp = np.argsort(-similarities)
    neighbors = _temp[:k]

    rated: list = [j for j in neighbors if user_row[j] > 0]  # filter out unrated items

    # if there are no items in the list, then the user has not rated any of the top k items
    if not rated:
        return None

    weighted_average = similarity_matrix[item, rated]
    # print(weighted_average)
    if weighted_average.sum() == 0:
        return None

    # predicted rating
    return np.dot(weighted_average, user_row[rated]) / weighted_average.sum()


def recommend(user_vectors: np.matrix, user: int, sim_matrix: np.ndarray, N: int = 3, k: int = 10) -> str | list:
    """
    recommend() turns a prediction function into a top-N recommender. It calls the estimator once
    for every item the user has not yet rated, then returns the N items with the highest predicted
    ratings.

    :param user_vectors: the user-item ratings matrix, with 0 for a missing rating
    :param user:    the row index of the user to recommend to
    :param sim_matrix:  a static similarity matrix
    :param N:       how many recommendations to return
    :param k:       for top k recommendations

    :return:    a list of (item, predicted rating) pairs, highest first, or a message if the
                user has rated every item.
    """

    unrated_items = np.nonzero(user_vectors[user, :].A == 0)[1]  # find unrated items
    if len(unrated_items) == 0:
        return 'you rated everything'

    item_scores: list[tuple] = []
    for joke in unrated_items:
        estimated_score = rating_prediction(user_vectors, user, sim_matrix, joke, k)
        if estimated_score is None:
            estimated_score = 0
        item_scores.append((joke, estimated_score))

    # Sort the (item, score) pairs by score, highest first, and keep the top N.
    return sorted(item_scores, key=lambda jj: jj[1], reverse=True)[:N]


#######################################################################################################################


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prediction function.")
    ap.add_argument("--book-genre-map", default="data/processed/book_genre_map.csv")
    ap.add_argument("--user-data", default="data/processed/user_vectors.csv")
    # ap.add_argument("--out", default="data/processed/", help="Output directory")
    ap.add_argument("--save-output", action=argparse.BooleanOptionalAction, help="Whether to save outputs to disk")
    ap.add_argument("--list-user-ids", action=argparse.BooleanOptionalAction, help="Print list of user ids")
    ap.add_argument("--list-book-ids", action=argparse.BooleanOptionalAction, help="Print list of book ids")
    ap.add_argument("--predict", type=int, nargs=2, help="Usage: --predict [user_id] [book_id]")

    args = ap.parse_args(args=argv)
    save_output: bool = args.save_output

    book_genre_map: pd.DataFrame = pd.read_csv(args.book_genre_map)
    user_vectors: pd.DataFrame = pd.read_csv(args.user_data)

    DATA_MATRIX: np.matrix = np.asmatrix(user_vectors)


    display(book_genre_map.head(20))
    display(user_vectors.head(20))

    return 0


if __name__ == "__main__":
    pd.options.display.float_format = '{:,.2f}'.format

    # pd.set_option("display.max_rows", None)
    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.width", None)

    sys.exit(main())