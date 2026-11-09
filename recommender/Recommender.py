import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from core import Common

from IPython.core.display_functions import display


def save_to_disk(data_frame_to_save: pd.DataFrame, filename: str, out_dir: str, write_row_names: bool) -> None:
    """
    Save the dataframes to local disk as a csv.

    :param data_frame_to_save:  the DataFrame to save to disk
    :param filename:            the name of the saved file
    :param out_dir:             output directory
    :param write_row_names:     corresponds to 'index' in DataFrame.to_csv()
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save_path = os.path.join(out_dir, filename)

    data_frame_to_save.to_csv(save_path, index=write_row_names)
    print(f"Saved file to {save_path}")


def _print_info(frame: pd.DataFrame) -> tuple[int, int, int, float]:
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
    min_books = books_per_user.min()
    max_books = books_per_user.max()
    avg_books_read: float = books_per_user.mean()
    print(f"Min books: {min_books}")
    print(f"Max books: {max_books}")
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


def _get_interactions(input_dir: str, genre: str, drop_columns=None, max_rows: int = 10000) -> pd.DataFrame:
    """
    The frame returns has columns 'user_id' (str, contains a user id as hex string), 'book_id' (int, contains a unique id for each book),
    'rating' (int, a rating 1-5 given to the book by the user), and 'is_reviewed' (int, 0 or 1; 1 if the user left a written review for the book)

    :param genre:           the genre.
    :param input_dir:       directory containing the zip files.
    :param drop_columns:    columns to be dropped from the DataFrame.
    :param max_rows:        the maximum number of rows to read from the file.
    :param min_read:        a minimum number of books, users with fewer read books will be dropped.
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


def _initial_frame(genre: str,
                   in_dir: str,
                   max_rows_processed: int,
                   columns_to_drop: list[str],
                   print_intermediate: bool=False) -> tuple[str, pd.DataFrame, pd.DataFrame]:
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

    book_ids = frame["book_id"]
    book_ids = book_ids.to_frame()
    book_ids["genre"] = genre

    if print_intermediate:
        _print_info(frame)

    return genre, frame, book_ids


def second_pass_filter(frame: pd.DataFrame, min_read: int) -> pd.DataFrame:
    """
    Once all the data is compiled, do a second pass to filter out users that do not meet requirements.

    :param frame:       the DataFrame containing all the compiled data
    :param min_read:    the minimum number of books read per user
    :return:
    """

    df_filtered = frame[frame.groupby('user_id')['book_id'].transform('nunique') >= min_read]

    return df_filtered


def _merge_genre_maps(book_genre_maps: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Merge the book_id x genre maps DataFrames and remove duplicates (exact duplicates across all rows).
    :param book_genre_maps:
    :return: a DataFrame
    """

    frame = pd.concat(book_genre_maps, ignore_index=True)

    return frame.drop_duplicates(keep='last')


def compile_and_analyze(genre_list: list[str], in_dir: str, min_user_read: int) -> tuple[pd.DataFrame, pd.DataFrame, int, int, int, float, pd.DataFrame]:
    unnecessary_columns = ["review_id", "date_added", "date_updated", "read_at", "started_at"]

    temp_frame_list: list[pd.DataFrame] = []
    book_genre_maps: list[pd.DataFrame] = []

    # get data for each genre to compile it later
    for book_genre in genre_list:
        genre, frame, book_genre_map = _initial_frame(book_genre,
                                      in_dir,
                                      columns_to_drop=unnecessary_columns,
                                      max_rows_processed=50000)

        temp_frame_list.append(frame)
        book_genre_maps.append(book_genre_map)

    # this frame has all the combined data
    complete_frame: pd.DataFrame = pd.concat(temp_frame_list, ignore_index=True)
    complete_frame = second_pass_filter(complete_frame, min_user_read)
    user_count, book_count, genre_count, avg_books_read = _print_info(complete_frame)

    master_book_genre_map = _merge_genre_maps(book_genre_maps)

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

    return vectors, complete_frame, user_count, book_count, genre_count, avg_books_read, master_book_genre_map


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Downloads target data files from the web and saves to local disk.")
    ap.add_argument("--genres", default="data-download/targets.txt", help="Path to a file containing a list of genres.")
    ap.add_argument("--input", default="data/archive", help="Source directory containing data to analyze.")
    ap.add_argument("--out", default="data/processed/", help="Output directory")
    # ap.add_argument("--gen-tests", default=16, help="Number of test users to generate")
    ap.add_argument("--min-read", default=30, help="Minimum number of books read per user")
    # ap.add_argument("--max-read", default=2000, help="Maximum number of books read per user")
    ap.add_argument("--save-output", action=argparse.BooleanOptionalAction, help="Whether to save outputs to disk")

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

    user_vectors, compiled_data, user_count, book_count, genre_count, avg_books_read, book_genre_map = compile_and_analyze(genre_list, in_dir, min_read_books)

    # if test_user_count > 0:
    #     test_data = TestData.generate_test_data(test_user_count, avg_books_read, user_vectors, min_read=min_read_books, max_read=max_read_books)
    #     if save_output:
    #         save_to_disk(test_data, "test_data.csv", out_dir, True)

    if save_output:
        save_to_disk(compiled_data, "compiled_data.csv", out_dir, False)
        save_to_disk(user_vectors , "user_vectors.csv" , out_dir, True)

    # always save map for future use
    save_to_disk(book_genre_map, f"book_genre_map.csv", out_dir, False)

    ################################################



    ################################################

    target_id = 36065344
    TARGET = user_vectors[[target_id]]
    USER_VECTORS = user_vectors.drop([target_id], axis=1)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(USER_VECTORS, TARGET, test_size=0.2, random_state=42)

    print(f"X_TRAIN: {X_TRAIN.shape}, X_TEST: {X_TEST.shape}")
    print(f"Y_TRAIN: {Y_TRAIN.shape}, Y_TEST: {Y_TEST.shape}")

    ################################################

    print("Complete!")
    return 0


if __name__ == "__main__":
    pd.options.display.float_format = '{:,.2f}'.format

    # pd.set_option("display.max_rows", None)
    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.width", None)

    sys.exit(main())
