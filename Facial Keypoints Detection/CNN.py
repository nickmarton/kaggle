"""
Convolutional Neural Network for kaggle facial keypoints detection contest.

Note: Only the labels contain missing values in all of the data.
"""

from __future__ import division, print_function
import time
import logging
import numpy as np
import pandas as pd
import theano.tensor as T
from sklearn.cross_validation import train_test_split


def imputate(frame):
    """Deal with missing values in a DataFrame."""
    start_time = time.time()

    frame.fillna(-1, inplace=True)

    time_diff = time.time() - start_time
    logging.info("Imputation completed in " + str(time_diff) + " seconds")


def parse_data(train_file="training.csv", test_file="test.csv"):
    """
    Parse training and test data;
    split Image and labels and convert Image column to DataFrame.
    """

    start_time = time.time()

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    y_train = train.ix[:, :-1] / 255
    imputate(y_train)

    # Convert Image column in train into DataFrame
    pixel_columns = ["pixel" + str(i + 1) for i in range(96 * 96)]
    parsed_train_images = []
    for image in list(train["Image"]):
        parsed_train_images.append(map(lambda x: int(x) / 255, image.split()))
    X_train = pd.DataFrame(parsed_train_images, columns=pixel_columns)

    # Convert Image column in test into DataFrame, zipping their ImageId's too
    parsed_test_images = []
    for image_id, image in zip(list(test["ImageId"]), list(test["Image"])):
        parsed_test_images.append(
            [image_id] + map(lambda x: int(x) / 255, image.split()))
    X_test = pd.DataFrame(parsed_test_images,
                          columns=["ImageId"] + pixel_columns)

    X_train.to_csv("X_train.csv", index=False)
    y_train.to_csv("imputated_y_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)

    time_diff = time.time() - start_time
    logging.info("Parsing completed in " + str(time_diff) + " seconds")


def load_data(validation_size=None):
    """Load the data sets and optionally create a validation set."""
    start_time = time.time()

    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("imputated_y_train.csv")
    X_test = pd.read_csv("X_test.csv")

    # Drop ImageId on test frame
    X_test = X_test.ix[:, 1:]

    # Cast X_train into numpy array and reshape into form suitable for lasagne
    X_train_reshape = np.zeros((len(X_train), 1, 96, 96), dtype=np.float16)
    for i, row in enumerate(np.array(X_train)):
        X_train_reshape[i] = row.reshape(1, 96, 96)
    X_train = X_train_reshape

    # Convert y_train into numpy array
    y_train = np.array(y_train, dtype=np.float16)

    # Cast X_test into numpy array and reshape into form suitable for lasagne
    X_test_reshape = np.zeros((len(X_test), 1, 96, 96), dtype=np.float16)
    for i, row in enumerate(np.array(X_test)):
        X_test_reshape[i] = row.reshape(1, 96, 96)
    X_test = X_test_reshape

    if validation_size:
        try:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=validation_size)

            time_diff = time.time() - start_time
            logging.info("Loading completed in " + str(time_diff) + " seconds")

            return X_train, y_train, X_valid, y_valid, X_test
        except TypeError:
            pass
        except ValueError:
            pass

    time_diff = time.time() - start_time
    logging.info("Loading completed in " + str(time_diff) + " seconds")

    return X_train, y_train, X_test


def set_verbosity(verbose_level=3):
    """Set the level of verbosity of the Preprocessing."""
    if not type(verbose_level) == int:
        raise TypeError("verbose_level must be an int")

    if verbose_level < 0 or verbose_level > 4:
        raise ValueError("verbose_level must be between 0 and 4")

    verbosity = [
        logging.CRITICAL,
        logging.ERROR,
        logging.WARNING,
        logging.INFO,
        logging.DEBUG]

    logging.basicConfig(
        format='%(asctime)s:\t %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=verbosity[verbose_level])


def main():
    """."""

    set_verbosity(0)

    # parse_data()
    X_train, y_train, X_val, y_val, X_test = load_data(
        validation_size=0.33)

    # X_train, y_train, X_test = load_data()

    print (X_train.shape)
    print (y_train.shape)
    print (X_val.shape)
    print (y_val.shape)
    print (X_test.shape)

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # network = build_cnn(input_var)

if __name__ == "__main__":
    main()
