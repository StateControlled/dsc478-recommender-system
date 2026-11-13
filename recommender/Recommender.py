import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from core import Common

from IPython.core.display_functions import display


def _analysis(frame: pd.DataFrame, book_ids: pd.DataFrame, user_ids: pd.DataFrame) -> tuple[int, int, int, float]:
    """
    :param frame:
    :param book_ids:
    :param user_ids:
    :return:
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

    unique_user_count: int = user_ids["user_id"].nunique()
    print(f"Unique users: {unique_user_count}")

    # The number of unique books (book id)
    unique_book_count: int = frame['book_id'].nunique()
    print(f"Unique books: {unique_book_count}")

    unique_book_count: int = book_ids["book_id"].nunique()
    print(f"Unique books: {unique_book_count}")

    # The number of unique genres in the data
    unique_genre_count: int = frame['genre'].nunique()
    print(f"Number of genres represented: {unique_genre_count}")
    print()

    # The number of books each user has read
    books_per_user = frame.groupby('user_id')['book_id'].nunique()

    # The average number of books read
    min_books = books_per_user.min()
    max_books = books_per_user.max()
    avg_books_read: float = books_per_user.mean()
    print(f"Min books read: {min_books}")
    print(f"Max books read: {max_books}")
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

    return unique_user_count, unique_book_count, unique_genre_count, avg_books_read


def _get_books_and_users(user_book_rating_matrix: pd.DataFrame, compiled_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separate unique user ids and unique book ids. Re-index the user vectors matrix to align

    :param user_book_rating_matrix: a matrix showing ratings for each book by user
    """

    print("Collecting unique users and books...")
    # user_ids = frame[['user_id']].drop_duplicates().reset_index(drop=True)
    user_ids = user_book_rating_matrix.index.to_frame().reset_index(drop=True)
    print(user_ids)
    print(type(user_ids))
    print()

    book_genre_map = compiled_data[['book_id', 'genre']].drop_duplicates().reset_index(drop=True)
    book_ids = user_book_rating_matrix.columns.to_frame().reset_index(drop=True)
    print(book_ids)
    print(type(book_ids))

    matrix = user_book_rating_matrix.copy()
    matrix.columns = book_ids.index
    matrix.index = user_ids.index

    return book_ids, user_ids, matrix, book_genre_map


def _second_pass(frame: pd.DataFrame, min_read: int) -> pd.DataFrame:
    """
    Once all the data is compiled, do a second pass to filter out users that do not meet requirements.

    :param frame:       the DataFrame containing all the compiled data
    :param min_read:    the minimum number of books read per user
    :return:
    """

    df_filtered = frame[frame.groupby('user_id')['book_id'].transform('nunique') >= min_read]

    return df_filtered


def _get_interactions(input_dir: str, genre: str, drop_columns: list[str] | None=None, max_rows: int=10000) -> pd.DataFrame:
    """
    The frame returns has columns 'user_id' (str, contains a user id as hex string), 'book_id' (int, contains a unique id for each book),
    'rating' (int, a rating 1-5 given to the book by the user), and 'is_reviewed' (int, 0 or 1; 1 if the user left a written review for the book)

    :param genre:           the genre.
    :param input_dir:       directory containing the zip files.
    :param drop_columns:    columns to be dropped from the DataFrame.
    :param max_rows:        the maximum number of rows to read from the file.
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

    df["genre"] = genre

    return df.reset_index(drop=True)


def _initial_frame(genre: str, in_dir: str, max_rows_processed: int, columns_to_drop: list[str]) -> pd.DataFrame:
    """
    Returns a DataFrame containing the raw data

    :param genre:               the genre
    :param in_dir:              directory containing the zip files
    :param max_rows_processed:  the maximum number of rows to read from the file
    :param columns_to_drop:     columns to be dropped from the DataFrame
    :return:
    """

    print(f"Gathering data for genre: {genre}...")
    frame: pd.DataFrame = _get_interactions(in_dir, genre, drop_columns=columns_to_drop, max_rows=max_rows_processed)  # real length = 34,919,255

    return frame


def compile_and_analyze(genre_list: list[str],
                        in_dir: str,
                        min_user_read: int,
                        max_rows_to_read: int=50000,
                        unnecessary_columns: list[str] | None=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # unnecessary_columns = ["review_id", "date_added", "date_updated", "read_at", "started_at"]

    if unnecessary_columns is None:
        unnecessary_columns: list[str] = ["review_id", "date_added", "date_updated", "read_at", "started_at"]

    # to compile complete list later
    temp_frame_list: list[pd.DataFrame] = []

    # read in the initial data from disk
    for book_genre in genre_list:
        frame = _initial_frame(book_genre, in_dir, columns_to_drop=unnecessary_columns, max_rows_processed=max_rows_to_read)

        temp_frame_list.append(frame)

    # this frame has all the combined data
    complete_frame: pd.DataFrame = pd.concat(temp_frame_list, ignore_index=True)
    complete_frame = _second_pass(complete_frame, min_user_read)

    # create vector for each user
    # table format:

    #         book_0 book_1 book_2 book_3
    # user_0       4      5      2      1
    # user_1       0      3      1      0
    # user_2       0      0      5      4
    vectors: pd.DataFrame = complete_frame.pivot_table(index='user_id', columns='book_id', values='rating', fill_value=0)
    """user x book table showing ratings for each book by user"""
    vectors = vectors.astype(int)

    # do additional processing, separate book ids and user ids
    book_ids, user_ids, user_vectors, book_genre_map = _get_books_and_users(vectors, complete_frame)

    user_count, book_count, genre_count, avg_books_read = _analysis(complete_frame, book_ids, user_ids)

    print("User Vectors")
    print("A set of vectors for each user. Each vector shows a single user's rating for every book. Unrated books have a rating of zero.")
    print(user_vectors.shape)
    print()

    display(user_vectors.head(10))
    print()

    return user_vectors, complete_frame, book_ids, user_ids, book_genre_map


#######################################################################################################################


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Downloads target data files from the web and saves to local disk.")
    ap.add_argument("--genres", default="data-download/test-targets.txt", help="Path to a file containing a list of genres.")
    ap.add_argument("--input", default="data/archive", help="Source directory containing data to analyze.")
    ap.add_argument("--out", default="data/processed/", help="Output directory")
    # ap.add_argument("--gen-tests", default=16, help="Number of test users to generate")
    ap.add_argument("--min-read", default=30, help="Minimum number of books read per user")
    # ap.add_argument("--max-read", default=2000, help="Maximum number of books read per user")
    ap.add_argument("--save-output", action=argparse.BooleanOptionalAction, default=True, help="Whether to save outputs to disk")

    args = ap.parse_args(args=argv)

    genres: str = args.genres
    in_dir: str = args.input
    out_dir: str = args.out
    # test_user_count: int = args.gen_tests
    save_output: bool = args.save_output
    min_read_books: int = args.min_read
    # max_read_books: int = args.max_read

    genre_list = Common.read_file(genres)

    ################################################

    user_vectors, compiled_data, book_ids, user_ids, book_genre_map = compile_and_analyze(genre_list, in_dir, min_read_books)

    # if test_user_count > 0:
    #     test_data = TestData.generate_test_data(test_user_count, avg_books_read, user_vectors, min_read=min_read_books, max_read=max_read_books)
    #     if save_output:
    #         save_to_disk(test_data, "test_data.csv", out_dir, True)

    if save_output:
        Common.save_to_disk(book_genre_map, filename="book_genre_map.csv",  out_dir=out_dir, index_=True,   header_=True)
        Common.save_to_disk(book_ids,       filename="book_id_map.csv",     out_dir=out_dir, index_=True,   header_=True)
        Common.save_to_disk(compiled_data,  filename="compiled_data.csv",   out_dir=out_dir, index_=False,  header_=True)
        Common.save_to_disk(user_ids,       filename="user_id_map.csv",     out_dir=out_dir, index_=True,   header_=True)
        Common.save_to_disk(user_vectors ,  filename="user_vectors.csv",    out_dir=out_dir, index_=False,  header_=False)

    ################################################

    # target_id = 512
    # TARGET = user_vectors[[target_id]]
    # USER_VECTORS = user_vectors.drop([target_id], axis=1)
    # X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(USER_VECTORS, TARGET, test_size=0.2, random_state=42)
    #
    # print(f"X_TRAIN: {X_TRAIN.shape}, X_TEST: {X_TEST.shape}")
    # print(f"Y_TRAIN: {Y_TRAIN.shape}, Y_TEST: {Y_TEST.shape}")

    ################################################

    print()
    print("Complete!")
    return 0


if __name__ == "__main__":
    pd.options.display.float_format = '{:,.2f}'.format

    # pd.set_option("display.max_rows", None)
    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.width", None)

    sys.exit(main())
