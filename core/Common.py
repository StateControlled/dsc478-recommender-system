# Contains some common utility methods.
import os

import pandas as pd

def read_file(file_name: str) -> list[str]:
    """
    Reads and returns the contents of a text file, stripping newlines and extra spaces first.

    :param file_name: the file to read.
    :return: a list of lines in the file.
    """

    res = []
    with open(file_name, 'r') as file:
        for line in file:
            name = line.strip()
            name = name.lower()
            res.append(name)

    return res


def merge_frames(frame_list: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge DataFrames and remove duplicates (exact duplicates across all rows).
    :param frame_list:
    :return: a DataFrame
    """

    frame = pd.concat(frame_list, ignore_index=True)

    return frame.drop_duplicates(keep='last')


def save_to_disk(data_frame_to_save: pd.DataFrame, filename: str, out_dir: str, index_: bool, header_: bool) -> None:
    """
    Save the dataframes to local disk as a csv.

    :param data_frame_to_save:  the DataFrame to save to disk
    :param filename:            the name of the saved file
    :param out_dir:             output directory
    :param index_:              corresponds to 'index' in DataFrame.to_csv()
    :param header:              corresponds to 'header' in DataFrame.to_csv()
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save_path = os.path.join(out_dir, filename)

    data_frame_to_save.to_csv(save_path, index=index_, header=header_)
    print(f"Saved file to {save_path}")
