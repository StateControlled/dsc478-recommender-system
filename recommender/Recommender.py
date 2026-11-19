import argparse
import os
import sys

from recommender import Clustering

import numpy as np
import pandas as pd

from IPython.core.display_functions import display


#######################################################################################################################
#
# SIMILARITY
#
#######################################################################################################################


def build_similarity_matrix(ratings: np.ndarray | np.matrix, min_overlap: int=3) -> np.ndarray:
    """
    Constructs a similarity matrix of books x books.

    :param ratings:     users x books rating matrix
    :param min_overlap: minimum number of common books rated
    :return:            books x books similarity matrix
    """

    r: np.ndarray = np.asarray(ratings, dtype=np.float64)
    m: np.ndarray = (r > 0).astype(np.float64)
    r2 = r * r

    n = m.T @ m         # number of users who rated both i and j
    sum_x = r.T @ m     # sum of ratings of i over those users
    sum_xx = r2.T @ m   # sum of squared ratings of i over those users
    sum_xy = r.T @ r    # sum of r_i * r_j (zero unless both rated)
    sum_y = sum_x.T     # transpose of sum_x
    sum_yy = sum_xx.T   # transpose of sum_xx

    cov: np.ndarray = n * sum_xy - sum_x * sum_y
    var_x = np.clip(n * sum_xx - sum_x ** 2, 0, None)
    var_y = np.clip(n * sum_yy - sum_y ** 2, 0, None)
    den = np.sqrt(var_x * var_y)

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / den

    valid = np.isfinite(corr) & (den > 0) & (n >= min_overlap)
    similarity_matrix_result = np.where(valid, 0.5 + 0.5 * np.clip(corr, -1.0, 1.0), 0.0)
    np.fill_diagonal(similarity_matrix_result, 0.0)

    return similarity_matrix_result


#######################################################################################################################
#
# CLUSTERING
#
#######################################################################################################################


def fit_cluster_model(ratings: np.ndarray | np.matrix,
                      n_clusters: int,
                      seed: int = 42,
                      blend: float = 0.5,
                      min_overlap: int = 5) -> Clustering.ClusterModel:
    """
    Clusters the users and builds the per-cluster similarity matrices.

    :param ratings:      users x books matrix, 0 = missing
    :param n_clusters:   number of clusters; a negative number means "choose by silhouette score"
    :param seed:         random seed for k-means
    :param blend:        weight of the cluster similarity against the global one, in [0, 1]. 0 = ignore clusters
    :param min_overlap:  minimum number of co-raters inside a cluster for a pair's cluster similarity to count
    """

    r = np.asarray(ratings, dtype=np.float64)
    scores = None
    if n_clusters < 0:
        n_clusters, scores = Clustering.choose_k(r, seed=seed)

    labels, _ = Clustering.cluster_users(r, n_clusters, seed)
    g = build_similarity_matrix(r)

    combined, item_mean = {}, {}
    for c in np.unique(labels):
        members = r[labels == c]

        cluster_sim = build_similarity_matrix(members, min_overlap)
        combined[int(c)] = np.where(cluster_sim > 0, blend * cluster_sim + (1 - blend) * g, g)

        counts = (members > 0).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            item_mean[int(c)] = np.where(counts > 0, members.sum(axis=0) / counts, np.nan)

    return Clustering.ClusterModel(labels, g, combined, item_mean, blend, scores)


#######################################################################################################################
#
# PREDICTION
#
#######################################################################################################################


def _predict_from_row(user_row: np.ndarray, item: int, item_similarities: np.ndarray, k: int) -> float | None:
    """
    Item-based estimator. Works on a single user's rating row so that callers never need to copy the whole matrix.

    The k books that the user has rated and that are most similar to the book are the neighbors; the prediction is the
    similarity-weighted average of the user's ratings of those neighbors.

    :return: predicted rating, or None if the user has no rated book with a positive similarity to the given book
    """

    rated = np.flatnonzero(user_row > 0)
    rated = rated[rated != item]  # never use the item itself, so that it can be predicted for a rated book
    if rated.size == 0:
        return None

    sims = item_similarities[rated]
    positive = sims > 0  # a similarity of 0 means "no information"
    rated, sims = rated[positive], sims[positive]
    if rated.size == 0:
        return None

    if rated.size > k:
        top = np.argsort(-sims, kind="stable")[:k]
        rated, sims = rated[top], sims[top]

    return float(np.dot(sims, user_row[rated]) / sims.sum())


def rating_prediction(user_vectors: np.matrix | np.ndarray,
                      user: int,
                      item: int,
                      similarity_matrix: np.ndarray,
                      k: int=10) -> float | None:
    """
    Predicts the rating a user would give a given book (item), from the user's ratings of the k most similar books they have read.

    :param user_vectors:        the user x book ratings matrix
    :param user:                row index of the target user
    :param item:                column index of the target book
    :param similarity_matrix:   books x books similarity matrix computed in training
    :param k:                   number of neighboring books
    :return:                    the predicted rating, or None if no prediction can be made
    """

    user_row = np.asarray(user_vectors)[user]
    item_similarities = similarity_matrix[item]
    return _predict_from_row(user_row, item, item_similarities, k)


def recommend(user_vectors: np.matrix | np.ndarray,
              user: int,
              sim_matrix: np.ndarray,
              num_recommend: int = 3,
              neighbors: int = 10) -> str | list[tuple[int, float]]:
    """
    Estimates a rating for every book the user has not rated and returns the top results with the highest predicted rating.
    Books for which no prediction can be made are skipped.

    :param user_vectors:    the user x book ratings matrix
    :param user:            row index of the user to recommend to
    :param sim_matrix:      books x books similarity matrix
    :param num_recommend:   how many recommendations to return
    :param neighbors:       number of neighboring books used for each prediction

    :return:    a list of book column index, predicted rating (highest rated first), or a message if the user has rated every book.
    """

    user_row: np.ndarray = np.asarray(user_vectors)[user]


    unrated_items: np.ndarray = np.flatnonzero(user_row == 0)
    if unrated_items.size == 0:
        return "you rated everything"

    item_scores: list[tuple[int, float]] = []
    for itm in unrated_items:
        estimated = _predict_from_row(user_row, itm, sim_matrix[itm], neighbors)

        if estimated is not None:
            item_scores.append((int(itm), estimated))

    return sorted(item_scores, key=lambda pair: pair[1], reverse=True)[:num_recommend]


#######################################################################################################################
#
# EVALUATION
#
#######################################################################################################################


def _cross_validate_user(user_vectors: np.matrix | np.ndarray,
                         similarity_matrix: np.ndarray,
                         user: int,
                         k: int,
                         test_ratio: float,
                         seed=42) -> tuple[float, int]:
    """
    Evaluates a single user. With test_ratio = 0.2, a random 20 percent of the user's rated books are withheld, the
    ratings of the withheld books are predicted from the rest of the profile, and the absolute errors are summed.

    With an integer seed the selection is reproducible; the user id is added to the seed internally so that the same
    seed can be passed for every user and each one still withholds a different part of its profile. With seed=None the
    selection differs on every call.

    The books are drawn WITHOUT replacement.

    Where no prediction can be made, the user's mean rating over the kept part of the profile is used. (Using 0, as
    this function used to, adds a full rating's worth of error for every miss and makes MAE depend on coverage.)

    :param user_vectors:        the user-item ratings matrix, 0 = missing
    :param similarity_matrix:   the static similarity matrix
    :param user:                row index of the user to evaluate
    :param k:                   number of neighboring books
    :param test_ratio:          share of this user's ratings to withhold
    :param seed:                random seed
    :return: total absolute error over the withheld ratings, and the number of test cases
    """

    original_profile = np.array(np.asarray(user_vectors)[user], dtype=np.float64)  # single row
    rng = np.random.default_rng(seed + user)

    rated_items = np.flatnonzero(original_profile > 0)
    test_size = min(int(test_ratio * len(rated_items)), len(rated_items) - 1)  # always keep at least one rating

    if test_size <= 0:
        return 0.0, 0

    withheld_items = rng.choice(rated_items, size=test_size, replace=False)

    profile = original_profile.copy()
    profile[withheld_items] = 0
    user_mean = profile[profile > 0].mean()

    error_u = 0.0
    for item in withheld_items:
        estimated = _predict_from_row(profile, item, similarity_matrix[item], k)
        if estimated is None:
            estimated = user_mean
        error_u += abs(estimated - original_profile[item])

    return error_u, int(withheld_items.shape[0])


def _test(user_vectors: np.matrix | np.ndarray,
          similarity_matrix: np.ndarray,
          test_ratio: float,
          k: int,
          seed=42) -> float:
    """
    Per-user hold-out evaluation against a STATIC similarity matrix.

    Caution: if the similarity matrix was built from all ratings, the withheld ratings already influenced it, so this
    MAE is optimistic. Use evaluate() for a leak-free number.

    :return: the mean absolute error (MAE) over the whole data set
    """

    ratings = np.asarray(user_vectors)
    similarity_matrix = np.asarray(similarity_matrix)
    total_error: float = 0.0
    total_cases: int = 0

    for user_id in range(ratings.shape[0]):
        err, count = _cross_validate_user(ratings, similarity_matrix, user_id, k, test_ratio, seed)
        total_error += err
        total_cases += count

    return total_error / total_cases if total_cases else float("nan")


def evaluate(user_vectors: np.matrix | np.ndarray,
             test_ratio: float=0.2,
             k: int=10,
             seed: int | None=42,
             similarity_matrix: np.ndarray | None=None,
             n_clusters: int=0,
             cluster_blend: float=0.5) -> dict[str, float]:
    """
    Evaluation of the item-based predictor.

    For every user, test_ratio of their ratings are withheld (without replacement, at least one rating is kept). Then
    the item-item similarities are built from training ratings only, so that nothing about the test ratings
    leaks into the model, and every withheld rating is predicted.

    Predictions fall back to the user's training mean, then to the global training mean, when the model cannot say
    anything. 'coverage' reports how often the item-based estimate itself was available.

    Two baselines are scored on the same test set so the numbers mean something:
        user_mean - predict each user's average training rating
        item_mean - predict each book's average training rating (user mean when the book has no training ratings)

    :param user_vectors:        users x books rating matrix, 0 = missing
    :param test_ratio:          share of each user's ratings withheld
    :param k:                   number of neighboring books
    :param seed:                random seed (None = different every run)
    :param similarity_matrix:   pass a pre-built matrix to evaluate against it instead of rebuilding. This leaks the
                                test ratings into the model and is only useful for comparison.
    :param n_clusters:          if not 0, also evaluate cluster-aware filtering on the same split (negative = choose k
                                by silhouette). The users are clustered on the TRAINING ratings only. Its predictions
                                fall back to the cluster's mean rating for the book, then to the user's mean.
    :param cluster_blend:       weight of the cluster similarity against the global one
    :return: dict with mae, rmse, coverage, n_test, and mae/rmse of both baselines (plus cluster_* keys when
             n_clusters is not 0)
    """

    ratings = np.asarray(user_vectors, dtype=np.float64)
    rng = np.random.default_rng(seed)

    train = ratings.copy()
    test: list[tuple[int, int, float]] = []

    for u in range(ratings.shape[0]):
        rated = np.flatnonzero(ratings[u] > 0)
        size = min(int(test_ratio * len(rated)), len(rated) - 1)
        if size <= 0:
            continue
        held = rng.choice(rated, size=size, replace=False)
        train[u, held] = 0
        test.extend((u, int(i), ratings[u, i]) for i in held)

    if not test:
        raise ValueError("No test cases: users have too few ratings for this test_ratio.")

    sim = build_similarity_matrix(train) if similarity_matrix is None else np.asarray(similarity_matrix)

    model = None
    if n_clusters != 0:
        model = fit_cluster_model(train, n_clusters, seed=42 if seed is None else seed, blend=cluster_blend)

    # training statistics for the fallbacks and baselines
    train_mask = train > 0
    global_mean = train[train_mask].mean()
    user_counts = train_mask.sum(axis=1)
    user_means = np.where(user_counts > 0, train.sum(axis=1) / np.maximum(user_counts, 1), global_mean)
    item_counts = train_mask.sum(axis=0)
    item_sums = train.sum(axis=0)

    actual, pred_cf, pred_user, pred_item, pred_cl = [], [], [], [], []
    covered = cl_covered = 0

    for u, i, rating in test:
        estimate = _predict_from_row(train[u], i, sim[i], k)
        if estimate is not None:
            covered += 1
        else:
            estimate = user_means[u]

        if model is not None:
            label = int(model.labels[u])
            cl_estimate = _predict_from_row(train[u], i, model.combined[label][i], k)
            if cl_estimate is not None:
                cl_covered += 1
            else:
                cl_mean = model.item_mean[label][i]
                cl_estimate = cl_mean if np.isfinite(cl_mean) else user_means[u]
            pred_cl.append(cl_estimate)

        actual.append(rating)
        pred_cf.append(estimate)
        pred_user.append(user_means[u])
        pred_item.append(item_sums[i] / item_counts[i] if item_counts[i] > 0 else user_means[u])

    actual = np.array(actual)

    def mae(p):
        return float(np.mean(np.abs(np.array(p) - actual)))

    def rmse(p):
        return float(np.sqrt(np.mean((np.array(p) - actual) ** 2)))

    result = {
        "mae": mae(pred_cf),
        "rmse": rmse(pred_cf),
        "coverage": covered / len(test),
        "n_test": len(test),
        "user_mean_mae": mae(pred_user),
        "user_mean_rmse": rmse(pred_user),
        "item_mean_mae": mae(pred_item),
        "item_mean_rmse": rmse(pred_item),
    }

    if model is not None:
        result.update({
            "cluster_mae": mae(pred_cl),
            "cluster_rmse": rmse(pred_cl),
            "cluster_coverage": cl_covered / len(test),
            "n_clusters": len(model.combined),
            "cluster_sizes": np.bincount(model.labels).tolist(),
        })

    return result


#######################################################################################################################


def _load_optional(path: str, **kwargs) -> pd.DataFrame | None:
    return pd.read_csv(path, **kwargs) if os.path.exists(path) else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Item-based book recommender: prediction, recommendation, evaluation.")

    ap.add_argument("--predict-for", action="store_true", help="Predict a user's rating for a book")
    ap.add_argument("--user", "-u", type=int, help="User row index (see --list-user-ids)")
    ap.add_argument("--book", "-b", type=int, help="Book column index (see --list-book-ids)")

    ap.add_argument("--recommend", action="store_true", help="Recommend top N books for a user")
    ap.add_argument("--top-n", "-n", type=int, default=5, help="Number of recommendations to return (default 5)")

    ap.add_argument("--k", type=int, default=10, help="Number of neighbouring books per prediction (default 10)")

    ap.add_argument("--book-genre-map", default="data/processed/book_genre_map.csv", help="Path to book-genre map")
    ap.add_argument("--book-ids", default="data/processed/book_id_map.csv", help="Path to book index-id map")
    ap.add_argument("--user-data", default="data/processed/user_vectors.csv", help="Path to user x book rating matrix")
    ap.add_argument("--users", default="data/processed/user_id_map.csv", help="Path to user_id-index map")

    ap.add_argument("--list-user-ids", action=argparse.BooleanOptionalAction, help="Print list of user ids")
    ap.add_argument("--list-book-ids", action=argparse.BooleanOptionalAction, help="Print list of book ids")
    ap.add_argument("--print-matrix", action=argparse.BooleanOptionalAction, help="Prints the user x book rating matrix")

    ap.add_argument("--evaluate", action=argparse.BooleanOptionalAction, help="Run hold-out evaluation (MAE / RMSE)")
    ap.add_argument("--test-ratio", type=float, default=0.2, help="Share of each user's ratings withheld (default 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for the hold-out split (default 42)")
    ap.add_argument("--clusters", type=int, default=-1,
                    help="Cluster users by taste and use cluster-specific similarities. N = number of clusters, "
                         "-1 = pick by silhouette score, 0 = off (default)")
    ap.add_argument("--cluster-blend", type=float, default=0.5,
                    help="Weight of the cluster similarity against the global one, 0-1 (default 0.5)")
    ap.add_argument("--cluster-info", action="store_true", help="Describe the clusters (needs --clusters)")

    args = ap.parse_args(args=argv)

    user_id: int = args.user
    predict_for = args.predict_for
    args_recommend = args.recommend
    do_cluster_info = args.cluster_info
    num_clusters = args.clusters


    # load data
    book_genre_map: pd.DataFrame = pd.read_csv(args.book_genre_map, index_col=0)
    user_id_map: pd.DataFrame = pd.read_csv(args.users, index_col=0)
    book_id_map: pd.DataFrame | None = _load_optional(args.book_ids, index_col=0)

    user_vectors: pd.DataFrame = pd.read_csv(args.user_data, header=None, index_col=None)  # vectors, no headers
    ratings: np.ndarray = user_vectors.to_numpy(dtype=int)
    n_users, n_books = ratings.shape

    similarity: np.ndarray | None = None


    if do_cluster_info and num_clusters == 0:
        ap.error("'--cluster-info' requires '--clusters' (a number, or -1 to choose automatically)")

    model: Clustering.ClusterModel | None = None

    if num_clusters != 0 and (predict_for or args_recommend or do_cluster_info):
        model = fit_cluster_model(ratings, num_clusters, seed=args.seed, blend=args.cluster_blend)

        if model.silhouette:
            print("Silhouette score by k: " + ", ".join(f"k={k}: {v:.3f}" for k, v in model.silhouette.items()))
            print(f"Using {len(model.combined)} clusters.")

    def book_label(idx: int) -> str:
        if book_id_map is None:
            return f"book index {idx}"

        bid = book_id_map["book_id"].iloc[idx]
        genres = book_genre_map.loc[book_genre_map["book_id"] == bid, "genre"].tolist()

        return f"book index {idx} (Goodreads id {bid}{', ' + '/'.join(genres) if genres else ''})"

    if do_cluster_info:
        print(f"Clusters ({len(model.combined)}):")
        counts = (ratings > 0).sum(axis=1)

        for c in sorted(model.combined):
            members = model.labels == c
            means = model.item_mean[c]
            raters = (ratings[members] > 0).sum(axis=0)
            ranked = [i for i in np.argsort(-np.nan_to_num(means, nan=0.0)) if raters[i] >= 5][:3]
            avg = ratings[members][ratings[members] > 0].mean()

            print(f"  cluster {c}: {int(members.sum())} users, avg {counts[members].mean():.0f} ratings each, "
                  f"mean rating {avg:.2f}")

            for i in ranked:
                print(f"      top: {book_label(int(i))}  cluster mean {means[i]:.2f} ({int(raters[i])} raters)")

    if args.list_user_ids:
        print("User IDs:")
        display(user_id_map)

    if args.list_book_ids:
        print("Book IDs:")
        display(book_id_map)

    if args.print_matrix:
        print("User Vectors:")
        display(user_vectors)


    if predict_for:
        if user_id is None or args.book is None:
            ap.error("'--predict-for' requires both '--user' and '--book'; provide a user index and a book index")

        if not (0 <= user_id < n_users) or not (0 <= args.book < n_books):
            ap.error(f"user must be in [0, {n_users - 1}] and book in [0, {n_books - 1}]")

        print(f"Predicted rating for {book_label(args.book)} by user {user_id}")
        sim_for_user = model.similarity_for(user_id)

        if sim_for_user is None:
            raise Exception("Similarities did non populate!")

        if model is not None:
            print(f"(user {user_id} is in cluster {int(model.labels[user_id])})")

        rating = rating_prediction(ratings, user_id, args.book, sim_for_user, args.k)

        print("Rating: " + ("no prediction possible (user has rated no similar book)" if rating is None else f"{rating:.2f}"))

        actual = int(ratings[user_id, args.book])

        if actual > 0:
            print(f"(The user actually rated it {actual}; the prediction ignores that rating.)")

    if args_recommend:
        if user_id is None:
            ap.error("'--recommend' requires '--user'; provide a user index")

        if not (0 <= user_id < n_users):
            ap.error(f"user must be in [0, {n_users - 1}]")

        sim_for_user = similarity if model is None else model.similarity_for(user_id)
        if model is not None:
            print(f"(user {user_id} is in cluster {int(model.labels[user_id])})")

        top_recs = recommend(ratings, user_id, sim_for_user, args.top_n, args.k)

        if isinstance(top_recs, str):
            print(top_recs)
        elif not top_recs:
            print("No recommendations: no unread book is similar to anything this user has rated.")
        else:
            print(f"Top {len(top_recs)} recommendations for user {user_id}:")
            for rank, (idx, score) in enumerate(top_recs, start=1):
                print(f"  {rank}. {book_label(idx)}  predicted rating {score:.2f}")

    if args.evaluate:
        sim_arg = None

        print(f"Evaluating: test ratio={args.test_ratio}, k={args.k}")

        res = evaluate(ratings, args.test_ratio, args.k, args.seed, sim_arg, num_clusters, args.cluster_blend)

        print(f"Test ratings:        {res['n_test']}")
        print(f"Coverage:            {res['coverage']:.1%} (predictions made by the item-based model itself)")
        print(f"{'':20} {'MAE':>7} {'RMSE':>7}")
        print(f"{'Item-based CF':20} {res['mae']:7.3f} {res['rmse']:7.3f}")

        if "cluster_mae" in res:
            print(f"{'Cluster-aware CF':20} {res['cluster_mae']:7.3f} {res['cluster_rmse']:7.3f}"
                  f"   ({res['n_clusters']} clusters, sizes {res['cluster_sizes']}, "
                  f"blend {args.cluster_blend}, coverage {res['cluster_coverage']:.1%})")
        print(f"{'Baseline: user mean':20} {res['user_mean_mae']:7.3f} {res['user_mean_rmse']:7.3f}")
        print(f"{'Baseline: item mean':20} {res['item_mean_mae']:7.3f} {res['item_mean_rmse']:7.3f}")

    return 0


if __name__ == "__main__":
    pd.options.display.float_format = '{:,.2f}'.format
    sys.exit(main())
