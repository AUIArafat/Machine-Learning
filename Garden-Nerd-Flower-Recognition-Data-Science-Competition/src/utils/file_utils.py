import os
import pickle
import csv


def get_file_name(file):
    """

    :param file: file name with path/extension
    :return: file name without path/extension
    """
    return os.path.splitext(os.path.basename(file))[0]


def get_file_extension(file):
    """

    :param file: file name with extension
    :return: file extension
    """
    return os.path.splitext(os.path.basename(file))[1]


def get_files_in_dir(path):
    """

    :param path: directory path
    :return: list of all files(absolute path) in the given directory(not recursive)
    """
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def write_pickle(obj, file_name):
    """
    write object to pickle file

    :param obj: dictionary object
    :param file_name: file name
    """
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def read_pickle(file_name):
    """
    read object from pickle file

    :param file_name: file name
    :return: object loaded from pickle
    """
    with open(file_name, 'rb') as f:
        obj = pickle.load(f)
        return obj


def read_csv_label(csv_file):
    """
    read class labels from csv file

    :param csv_file: csv file location, csv file may have multiple column but we are concerned
    with first two columns file name and category
    :return: dictionary containing class labels{file: class}
    """
    labels = {}
    with open(csv_file) as f:
        reader = csv.reader(f)
        for row in reader:
            labels[row[0]] = int(row[1])
    return labels


def write_csv_label(labels, csv_file):
    """
    write class labels to csv file

    :param labels: dictionary containing class labels{file: class}
    :param csv_file: csv file location
    """
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        for key, value in labels.items():
            writer.writerow([key, value])


def test_csv_label():
    # create random labels dictionary
    labels = {
        'a': 0,
        'b': 1,
        'c': 2,
        'd': 3,
        'e': 4,
        'f': 5
    }

    # write to csv
    csv_file = 'csv_label_test.csv'
    write_csv_label(labels, csv_file)

    # read from csv
    labels_read = read_csv_label(csv_file)
    print(labels_read)
    print(labels == labels_read)


def test_pickle():
    """

    {@true}
    test {@code write_pickle} and read_pickle functions
    """
    dictionary = {
        'a': 0,
        'b': 1,
        'c': 2,
        'd': 3
    }
    file = 'pickle_test'
    # write
    write_pickle(dictionary, file)

    # read back
    dictionary_read = read_pickle(file)
    print(dictionary_read)
    print(dictionary == dictionary_read)


if __name__ == "__main__":
    test_pickle()
