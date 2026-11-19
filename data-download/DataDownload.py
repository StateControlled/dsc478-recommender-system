###############################################################

# Downloads target data files from the web and saves to local disk.
#
# Some code is adapted from
# https://github.com/MengtingWan/goodreads
# according to the terms of the Apache License 2.0 http://www.apache.org/licenses/

#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
#
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
#
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS

###############################################################

import pandas as pd

import argparse
import os
import requests
import sys

from recommender import Common


class DataObject:
    # def __init__(self, genre: str, books_file: str, interactions_file: str, reviews_file: str):
    #     self.genre: str = genre
    #     self.books_file: str = books_file
    #     self.interactions_file: str = interactions_file
    #     self.reviews_file: str = reviews_file

    def __init__(self, genre: str, interactions_file: str):
        self.genre: str = genre
        self.interactions_file: str = interactions_file

    def __str__(self) -> str:
        # return f"Genre:{self.genre} Books:{self.books_file} Interactions:{self.interactions_file} Reviews:{self.reviews_file}"
        return f"Genre:{self.genre} Interactions:{self.interactions_file}"


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
        # Mappings for user ids or book ids.
        elif file_type == "book_id_map" or file_type == "user_id_map":
            url: str = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/" + file_name
            file_name_url_mapping[file_name] = url

    return file_name_url_mapping


def _define_filenames_maps(map_list_file: str) -> list[str]:
    temp: pd.DataFrame = pd.read_csv(map_list_file, index_col=False)

    return temp["name"].tolist()


def _define_filenames_genre(genre_list_file: str) -> list[DataObject]:
    """
    Sets the filenames that will be downloaded later.

    :param genre_list_file: the file containing a list of genres.
    :return: a list of file names associated with the given genres.
    """

    temp: list[str] = Common.read_file(genre_list_file)
    res: list[DataObject] = []

    for line in temp:
        # file_book: str = f"goodreads_books_{line}.json.gz"
        file_interaction: str = f"goodreads_interactions_{line}.json.gz"
        # file_review: str = f"goodreads_reviews_{line}.json.gz"

        # item = DataObject(line, file_book, file_interaction, file_review)
        item = DataObject(line, file_interaction)

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


def main(argv: list[str] | None = None) -> int:
    """
    Downloads target data files from the web and saves to local disk.
    """

    argument_parser = argparse.ArgumentParser(description="Downloads target data files from the web and saves to local disk.")
    argument_parser.add_argument("--out",           default="data/archive/",                    help="Output directory")
    argument_parser.add_argument("--genres",        default="data-download/targets.txt",        help="List of target genres to download data files for.")
    argument_parser.add_argument("--dataset_names", default="data-download/dataset_names.csv",  help="List of dataset names.")
    argument_parser.add_argument("--mapping_names", default="data-download/maps.csv",           help="List of additional files that contain more information about the dataset.")
    argument_parser.add_argument("--maps-out",      default="data/maps/",                       help="Output directory for book id and user id mappings.")

    args = argument_parser.parse_args(args=argv)

    out_dir: str = args.out
    target_list: str = args.genres
    dataset: str = args.dataset_names
    mappings: str = args.mapping_names
    mapping_out_dir: str = args.maps_out

    # get list of items to download
    # targets is a list of file names (not URLs)
    genre_targets: list[DataObject] = _define_filenames_genre(target_list)
    map_targets: list[str] = _define_filenames_maps(mappings)
    # _download_mappings(mappings, "data/")

    print("Target files:")
    for t in genre_targets:
        # print(f"{t.books_file}")
        print(f"{t.interactions_file}")
        # print(f"{t.reviews_file}")

    print()
    print("Downloading data...")

    genre_url_mapping: dict[str, str] = _set_url_map(dataset)
    mapping_url: dict[str, str] = _set_url_map(mappings)

    # create save directories
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(mapping_out_dir):
        os.makedirs(mapping_out_dir)

    # download data
    count: int = 0
    total: int = 0
    for target in genre_targets:
        total += 1
        # count += _do_download(out_dir, target.books_file, genre_url_mapping)
        count += _do_download(out_dir, target.interactions_file, genre_url_mapping)
        # count += _do_download(out_dir, target.reviews_file, genre_url_mapping)

    for mapping in map_targets:
        total += 1
        count += _do_download(mapping_out_dir, mapping, mapping_url)

    print()
    print(f"Downloaded {count} of {total} files.")
    print("Download complete!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
