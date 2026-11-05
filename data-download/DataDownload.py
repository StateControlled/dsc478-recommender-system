###############################################################

# Downloads target data files from the web and saves to local disk.
#
# Some code is adapted from
# https://github.com/MengtingWan/goodreads
# according to the terms of the Apache License 2.0 http://www.apache.org/licenses/

###############################################################

import pandas as pd

import argparse
import os
import requests
import sys

from core import Common

class DataObject:
    def __init__(self, genre: str, books_file: str, interactions_file: str, reviews_file: str):
        self.genre: str = genre
        self.books_file: str = books_file
        self.interactions_file: str = interactions_file
        self.reviews_file: str = reviews_file

    def __str__(self) -> str:
        return f"Genre:{self.genre} Books:{self.books_file} Interactions:{self.interactions_file} Reviews:{self.reviews_file}"


def _set_file_names(file_name: str) -> dict[str, str]:
    """
    Opens the given file, assumed to be dataset_names.csv, that contains a mapping of file names and the category
    assigned to them by the original research team.

    :param file_name:   dataset_names.csv
    """
    file_names: pd.DataFrame = pd.read_csv(file_name)
    # print(file_names)

    file_name_type_mapping: dict[str, str] = dict(zip(file_names["name"].values, file_names["type"].values))
    return file_name_type_mapping


def _set_file_urls(file_name_type_mapping: dict[str, str]) -> dict[str, str]:
    """
    Associates file names with the URL from which they can be downloaded.
    :param file_name_type_mapping: a mapping of file names and the category assigned to them by the original research team.
    :return:
    """

    file_name_url_mapping: dict[str, str] = {}

    for file_name in file_name_type_mapping:
        file_type = file_name_type_mapping[file_name]

        # Detailed information (Meta-Data of Books, Book Shelves, Book Reviews).
        # Not used for this project, but here so it could theoretically be incorporated.
        if file_type == "complete":
            url: str = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/" + file_name
            file_name_url_mapping[file_name] = url
        # Data filtered by Genre, listed under the 'By Genre' section on the data source website.
        elif file_type == "byGenre":
            url: str = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/" + file_name
            file_name_url_mapping[file_name] = url

    return file_name_url_mapping


def _define_filenames(genre_list_file: str) -> list[DataObject]:
    """
    Sets the filenames that will be downloaded later.

    :param genre_list_file: the file containing a list of genres.
    :return: a list of file names associated with the given genres.
    """

    temp: list[str] = Common.read_file(genre_list_file)
    res: list[DataObject] = []

    for line in temp:
        file_book: str = f"goodreads_books_{line}.json.gz"
        file_interaction: str = f"goodreads_interactions_{line}.json.gz"
        file_review: str = f"goodreads_reviews_{line}.json.gz"

        item = DataObject(line, file_book, file_interaction, file_review)

        res.append(item)

    return res


def _set_url_map(dataset_file: str) -> dict[str, str]:
    # read dataset_names.csv and associate the type of archive and the file name
    # this is used to pinpoint to directory from which to download the correct file
    file_name_type_mapping: dict[str, str] = _set_file_names(dataset_file)

    # a list of file names and the URL from which they can be downloaded
    file_name_url_mapping: dict[str, str] = _set_file_urls(file_name_type_mapping)

    return file_name_url_mapping


def _download_file(file_name: str, out_directory: str, file_name_url_mapping: dict[str, str]) -> int:
    """
    Downloads a file of the given name from the associated URL and saves to disk.
    :param file_name:               the file name
    :param out_directory:
    :param file_name_url_mapping:
    """

    print(f"Downloading '{file_name}' to {out_directory}")

    if file_name in file_name_url_mapping:
        url = file_name_url_mapping[file_name]

        with requests.get(url, stream=True) as req:
            req.raise_for_status()

            with open(out_directory, 'wb') as file:
                for chunk in req.iter_content(chunk_size=8192):
                    file.write(chunk)

        print(f"File '{file_name}' has been downloaded to '{out_directory}'!")
        return 1
    else:
        print(f"The data file '{file_name}' can not be found!")
        return 0


def _do_download(out_dir: str, target: str, file_name_url_mapping: dict[str, str]) -> int:
    output_path = os.path.join(out_dir, target)
    return _download_file(target, output_path, file_name_url_mapping)


def _download_mappings(mappings: str, out: str) -> None:
    temp: list[str] = Common.read_file(mappings)
    return


def main(argv: list[str] | None = None) -> int:
    """
    Downloads target data files from the web and saves to local disk.
    """

    argument_parser = argparse.ArgumentParser(description="Downloads target data files from the web and saves to local disk.")
    argument_parser.add_argument("--out", default="data/archive/", help="Output directory")
    argument_parser.add_argument("--genres", default="data-download/targets.txt", help="List of target genres to download data files for.")
    argument_parser.add_argument("--dataset_names", default="data-download/dataset_names.csv", help="List of dataset names.")
    argument_parser.add_argument("--maps", default="data-download/maps.csv", help="List of additional files that contain more information about the dataset.")

    args = argument_parser.parse_args(args=argv)

    out_dir: str = args.out
    target_list: str = args.genres
    dataset: str = args.dataset_names
    # mappings: str = args.maps

    # get list of items to download
    # targets is a list of file names (not URLs)
    targets: list[DataObject] = _define_filenames(target_list)
    # _download_mappings(mappings, "data/")

    print("Target files:")
    for t in targets:
        print(f"{t.books_file}")
        print(f"{t.interactions_file}")
        print(f"{t.reviews_file}")

    print()
    print("Downloading data...")

    file_name_url_mapping: dict[str, str] = _set_url_map(dataset)

    # create save directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # download data
    count: int = 0
    total: int = 0
    for target in targets:
        total += 3
        count += _do_download(out_dir, target.books_file, file_name_url_mapping)
        count += _do_download(out_dir, target.interactions_file, file_name_url_mapping)
        count += _do_download(out_dir, target.reviews_file, file_name_url_mapping)

    print()
    print(f"Downloaded {count} of {total} files and saved to '{out_dir}'")
    print("Download complete!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
