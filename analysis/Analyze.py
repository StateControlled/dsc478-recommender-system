import argparse
import os
import sys

import pandas as pd

from core import Common

from IPython.display import display


def save_to_disk(compiled_data: pd.DataFrame, user_vectors: pd.DataFrame, out_dir: str) -> None:
    """
    Save the dataframes to local disk.
    :param compiled_data:
    :param user_vectors:
    :param out_dir:         output directory
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    user_vectors.to_csv(os.path.join(out_dir, "user_vectors.csv"), index=False)
    compiled_data.to_csv(os.path.join(out_dir, "compiled_data.csv"), index=False)


def _print_info(frame: pd.DataFrame) -> None:
    """
    Do basic analysis of the items in each genre
    :param frame:   the DataFrame
    """

    print("Basic analysis.")
    print("-" * 32)
    print("Sample: First ten rows of data.")
    display(frame.head(10))
    print()

    print(f"Dimensions: {frame.shape}")
    print(f"Columns: {frame.columns}")

    # The number of unique users (user id)
    unique_user_count: int = frame['user_id'].nunique()
    print(f"Unique users: {unique_user_count}")

    # The number of unique books (book id)
    unique_book_count: int = frame['book_id'].nunique()
    print(f"Unique books: {unique_book_count}")

    # The number of unique genres in the data
    unique_genre_count: int = frame['genre'].nunique()
    print(f"Number of genres represented: {unique_genre_count}")
    print()

    # The number of books each user has read
    books_per_user = frame.groupby('user_id')['book_id'].nunique()

    # The average number of books read
    avg_books_read: float = books_per_user.mean()
    print(f"Average books read per user: {avg_books_read:.2f}")
    print()

    # The number of books each user has read
    print("Top ten users by number of books read:")
    display(books_per_user.sort_values(ascending=False).head(10))
    print()

    # average rating per book
    avg_rating_per_book = frame.groupby('book_id')['rating'].mean()
    print("Top 20 highest rated books:")
    display(avg_rating_per_book.sort_values(ascending=False).astype(int).head(20))
    print()
    print("-" * 32)


def _get_interactions(input_dir: str, genre: str, drop_columns=None, max_rows: int = 10000) -> pd.DataFrame:
    """
    The frame returns has columns 'user_id' (str, contains a user id as hex string), 'book_id' (int, contains a unique id for each book),
    'rating' (int, a rating 1-5 given to the book by the user), and 'is_reviewed' (int, 0 or 1; 1 if the user left a written review for the book)

    :param genre:           the genre
    :param input_dir:       directory containing the zip files
    :param drop_columns:    columns to be dropped from the DataFrame
    :param max_rows:        the maximum number of rows to read from the file
    :return: a DataFrame containing interaction data that has had some preliminary cleaning.
    """
    # Interactions are the most important for what I want to do
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
    interactions: str = os.path.join(input_dir, f"goodreads_interactions_{genre}.json.gz")

    # each file consists of JSON objects, one on each line
    df: pd.DataFrame = pd.read_json(interactions, lines=True, nrows=max_rows)

    # drop unnecessary columns
    if drop_columns is None:
        drop_columns = ["review_id", "date_added", "date_updated", "read_at", "started_at"]

    df = df.drop(drop_columns, axis=1)

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

    df["Genre"] = genre

    return df.reset_index(drop=True)


def initial_frame(genre: str,
                  in_dir: str,
                  max_rows_processed: int,
                  columns_to_drop: list[str],
                  print_intermediate: bool=False) -> tuple[str, pd.DataFrame]:
    """
    Returns the genre, the dataframe, number of unique users, and number of unique books.

    :param genre:               the genre
    :param in_dir:              directory containing the zip files
    :param max_rows_processed:  the maximum number of rows to read from the file
    :param columns_to_drop:     columns to be dropped from the DataFrame
    :param print_intermediate:  True to print extra information
    :return:
    """

    frame: pd.DataFrame = _get_interactions(in_dir, genre, drop_columns=columns_to_drop, max_rows=max_rows_processed)  # real length = 34,919,255

    if print_intermediate:
        _print_info(frame)

    return genre, frame


def analyze(genre_list: list[str], in_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    unnecessary_columns = ["review_id", "date_added", "date_updated", "read_at", "started_at"]

    temp_frame_list: list[pd.DataFrame] = []

    # get data for each genre to compile it later
    for book_genre in genre_list:
        genre, frame = initial_frame(book_genre,
                                     in_dir,
                                     columns_to_drop=unnecessary_columns,
                                     max_rows_processed=20000)

        temp_frame_list.append(frame)

    # this frame has all the combined data
    complete_frame: pd.DataFrame = pd.concat(temp_frame_list, ignore_index=True)
    _print_info(complete_frame)

    # create vector for each user
    # table format:

    #         book_0 book_1 book_2 book_3
    # user_0       4      5      2      1
    # user_1       0      3      1      0
    # user_2       0      0      5      4
    vectors = complete_frame.pivot_table(index='user_id', columns='book_id', values='rating', fill_value=0)
    vectors = vectors.astype(int)

    print("User Vectors")
    print("A set of vectors for each user. Each vector shows a single user's rating for every book. Unrated books have a rating of zero.")
    print(vectors.shape)
    print()

    display(vectors.head(10))
    print()

    return vectors, complete_frame


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Downloads target data files from the web and saves to local disk.")
    ap.add_argument("--genres", default="data-download/targets.txt", help="Path to a file containing a list of genres.")
    ap.add_argument("--input", default="data/archive", help="Source directory containing data to analyze.")
    ap.add_argument("--out", default="data/processed/", help="Output directory")

    args = ap.parse_args(args=argv)

    genres: str = args.genres
    in_dir: str = args.input
    out_dir: str = args.out

    genre_list = Common.read_file(genres)

    ################################################

    user_vectors, compiled_data = analyze(genre_list, in_dir)
    save_to_disk(compiled_data, user_vectors, out_dir)

    ################################################

    print("Complete!")
    return 0


if __name__ == "__main__":
    pd.options.display.float_format = '{:,.5f}'.format

    # pd.set_option("display.max_rows", None)
    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.width", None)

    sys.exit(main())
