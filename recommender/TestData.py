import pandas as pd
import random

from IPython.core.display_functions import display

def generate_test_data(users_to_generate: int,
                       average_read: float,
                       user_vectors: pd.DataFrame,
                       min_read: int,
                       max_read: int,
                       min_rating: int=1,
                       max_rating: int=5) -> pd.DataFrame:
    """
    Generates some random test user vectors.

    :param users_to_generate:   number of test users to generate
    :param average_read:        average number of books read per user
    :param user_vectors:        current user vectors
    :param min_read:
    :param max_read:
    :param max_rating:          minimum rating that can be generated
    :param min_rating:          maximum rating that can be generated
    :return: a DataFrame
    """

    random.seed(61324097) # set seed for consistent results

    all_book_ids = user_vectors.columns

    data_rows: list = []
    user_index: list = []

    # randomly choose how many books the test user has read,
    # and the rating they gave to the book.
    for i in range(users_to_generate):
        user_vec: list[int] = [0 for _ in range(len(all_book_ids))]

        if random.randint(min_read, max_read) > int(average_read):
            read_count = int(average_read)
        else:
            read_count = random.randint(min_read, max_read)

        for j in range(read_count):
            index_pos = random.randint(0, len(all_book_ids) - 1)
            rating = random.randint(min_rating, max_rating)
            user_vec[index_pos] = rating

        rows = dict(zip(all_book_ids, user_vec))
        user_index.append(f"test_user_{i}")

        data_rows.append(rows)

    print(f"Generated {users_to_generate} test users")
    test_frame: pd.DataFrame = pd.DataFrame(data_rows, columns=user_vectors.columns, index=user_index)
    test_frame.index.name = "user_id"
    test_frame.columns.name = "book_id"

    if test_frame.shape[0] > 20:
        display(test_frame.head(20))
    else:
        display(test_frame)

    return test_frame
