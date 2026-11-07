# Contains some common utility methods.

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
