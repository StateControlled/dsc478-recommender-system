import argparse
import os
import sys

import pandas as pd

from core import Common

from IPython.display import display

# columns in interactions
# "user_id",
# "book_id",
# "review_id",
# "is_read",
# "rating",
# "review_text_incomplete",
# "date_added",
# "date_updated",
# "read_at",
# "started_at"

def get_reviews(input_dir: str, genre: str) -> pd.DataFrame:
    reviews: str = os.path.join(input_dir, f"goodreads_reviews_{genre}.json.gz")
    df: pd.DataFrame = pd.read_json(reviews, lines=True)
    return df

def get_books(input_dir: str, genre: str) -> pd.DataFrame:
    books: str = os.path.join(input_dir, f"goodreads_books_{genre}.json.gz")
    df: pd.DataFrame = pd.read_json(books, lines=True)
    return df

def get_interactions(input_dir: str, genre: str, drop_rows=None, max_rows: int = 10000) -> pd.DataFrame:
    """

    :param drop_rows: rows to be dropped from the DataFrame
    :param max_rows: the maximum number of rows to read from the file
    :param input_dir: directory containing the zip files.
    :param genre: the genre
    :return: a DataFrame containing interaction data that has had some preliminary cleaning.
    """
    # Interactions are the most important for what I want to do
    interactions: str = os.path.join(input_dir, f"goodreads_interactions_{genre}.json.gz")

    # each file consists of JSON objects, one on each line
    df: pd.DataFrame = pd.read_json(interactions, lines=True, nrows=max_rows)

    # drop unnecessary columns
    if drop_rows is None:
        drop_rows = ["review_id", "date_added", "date_updated", "read_at", "started_at"]

    df = df.drop(drop_rows, axis=1)

    # This column may contain an empty string ("") or a string (a review of a book).
    # I'm not concerned with the content of the review for this project, so I will
    # replace empty strings with 0 and non-empty strings with 1
    df["review_text_incomplete"] = df["review_text_incomplete"].ne("").astype(int)
    df = df.rename(columns={"review_text_incomplete": "is_reviewed"})

    # Convert from boolean to integer
    df["is_read"] = df["is_read"].astype(int)

    # rating == 0 means "read but not rated", drop rows where there is no rating
    df = df[df["rating"] > 0]

    # drop unread
    df = df[df["is_read"] > 0]

    # remove column because it is now always '1'
    df = df.drop(["is_read"], axis=1)

    return df.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Downloads target data files from the web and saves to local disk.")
    ap.add_argument("--genres", default="data-download/targets.txt", help="A list of genres.")
    ap.add_argument("--input", default="data/archive", help="Source directory containing data to analyze.")
    ap.add_argument("--out", default="data/raw/", help="Output directory")

    args = ap.parse_args(args=argv)

    genres: str = args.genres
    in_dir: str = args.input
    out_dir: str = args.out

    genre_list = Common.read_file(genres)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ################################################
    genre = "young_adult"  # "young_adult_short"
    frame: pd.DataFrame = get_interactions(in_dir, genre, max_rows=50000) # length = 34,919,255

    print("INTERACTIONS")
    display(frame.head(10))
    print()

    print(f"Dimensions: {frame.shape}")
    print(f"Columns: {frame.columns}")
    print()

    # The number of unique users
    unique_user_count: int = frame['user_id'].nunique()
    print(f"Unique users: {unique_user_count}")
    # The number of unique books (book id)
    unique_book_count: int = frame['book_id'].nunique()
    print(f"Unique books: {unique_book_count}")
    print()

    # The number of books each user has read
    books_per_user = frame.groupby('user_id')['book_id'].nunique()

    # The average number of books read
    avg_books_read: float = books_per_user.mean()
    print(f"Average books read: {avg_books_read:.2f}")
    print()

    # The number of books each user has read
    print("Books per user:")
    display(books_per_user.sort_values(ascending=False))
    print()

    # average rating per book
    avg_rating_per_book = frame.groupby('book_id')['rating'].mean()
    print("Average rating per book:")
    display(avg_rating_per_book.sort_values(ascending=False))
    print()

    ################################################
    # frame1: pd.DataFrame = get_books(in_dir, genre0)
    # file_name = os.path.join(out_dir, f"{genre0}_books.csv")
    # # frame.to_csv(file_name, index=False)

    ################################################
    # frame2: pd.DataFrame = get_reviews(in_dir, genre0)
    # file_name = os.path.join(out_dir, f"{genre0}_reviews.csv")
    # # frame.to_csv(file_name, index=False)

    ################################################

    print()
    print("-" * 32)
    print("Complete!")
    return 0

if __name__ == "__main__":
    pd.options.display.float_format = '{:,.5f}'.format

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    sys.exit(main())
