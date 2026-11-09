###############################################################

# Some code is adapted from
# https://github.com/MengtingWan/goodreads
# according to the terms of the Apache License 2.0 http://www.apache.org/licenses/

###############################################################

import pandas as pd

import argparse
import gzip
import json
import os
import sys

from core import Common

pd.options.display.float_format = '{:,}'.format

def count_interactions(file_name: str) -> tuple[int, int, int, int, int]:
    """
    Extracts some basic statistics from interaction files. Counts the number of times a given book appears on users' shelves, the number
    of ratings, the number of reviews, and the number of unique users associated with this book.
    :param file_name: the file name.
    :return: number of shelves, number of reads, number of ratings, number of reviews, number of unique users.
    """

    num_shelves, num_reads, num_ratings, num_reviews = 0, 0, 0, 0
    user_set: set = set()

    with gzip.open(file_name) as file:
        for ln in file:
            d_json = json.loads(ln)

            num_shelves += 1
            num_reads += int(d_json["is_read"])
            num_ratings += int(d_json["rating"] > 0)
            num_reviews += int(d_json["review_text_incomplete"] != "")
            user_set.add(d_json["user_id"])

    print(f"Counted {num_shelves} shelves, {num_reads} reads, {num_ratings} ratings, and {num_reviews} reviews in file '{file_name}'")

    return num_shelves, num_reads, num_ratings, num_reviews, len(user_set)


def count_lines(file_name: str) -> int:
    """
    Counts the number of records in the input file.
    :param file_name:   the file name.
    :return:    the number of records.
    """

    count: int = 0

    with gzip.open(file_name) as file:
        for _ in file:
            count += 1

    return count


def count_all_genres(input_dir: str, genre_list: list[str]) -> pd.DataFrame:
    """
    Calculate statistics for each data.
    :param input_dir:   the file name.
    :param genre_list: a list of the data names.
    :return:
    """

    res: list[list[int]] = []
    for genre in genre_list:
        books: str = os.path.join(input_dir, f"goodreads_books_{genre}.json.gz")

        n_book: int = count_lines(books)
        print(f"Counted {n_book} entries in file '{books}'")

        interactions: str = os.path.join(input_dir, f"goodreads_interactions_{genre}.json.gz")
        n_shelve, n_read, n_rate, n_review, n_user = count_interactions(interactions)

        res.append([n_book, n_user, n_shelve, n_read, n_rate, n_review])

    stats_by_genre: pd.DataFrame = pd.DataFrame(res, dtype=float, columns=["# book", "# user", "# shelve", "# read", "# rate", "# review"], index=genre_list)

    return stats_by_genre


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Downloads target data files from the web and saves to local disk.")
    ap.add_argument("--genres", default="data-download/targets.txt", help="A list of genres.")
    ap.add_argument("--input", default="data/archive", help="Source directory containing data to analyze.")
    ap.add_argument("--out", default="data/results/", help="Output directory")

    args = ap.parse_args(args=argv)

    genres: str = args.genres
    in_dir: str = args.input
    out_dir: str = args.out

    genre_list = Common.read_file(genres)

    stats_by_genre: pd.DataFrame = count_all_genres(in_dir, genre_list)

    for label in ["# shelve", "# read", "# rate", "# review"]:
        stats_by_genre[label + '/' + "book"] = stats_by_genre[label] / stats_by_genre["# book"]
        stats_by_genre[label + '/' + "user"] = stats_by_genre[label] / stats_by_genre["# user"]

    print()
    print("Basic statistics:")
    stats = stats_by_genre.round(2).T
    print(stats)

    # Examine how many books each user rates on average; what are the most-rated books; what are the most rated genres

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save_path = os.path.join(out_dir, "stats.csv")
    stats.to_csv(save_path, encoding='utf-8')
    print(f"\nSaved basic results to '{save_path}'")
    print("Done!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
