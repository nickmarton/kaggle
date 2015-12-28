"""
Convolutional Neural Network for kaggle facial keypoints detection contest.
"""

from __future__ import division, print_function
import pandas as pd
import numpy as np


def parse_data(train_file="training.csv", test_file="test.csv"):
    """
    Parse training and test data;
    split Image and labels and convert Image column to DataFrame.
    """

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    y_train = train.ix[:, :-1]

    # Convert Image column in train into DataFrame
    pixel_columns = ["pixel" + str(i + 1) for i in range(96 * 96)]
    parsed_train_images = []
    for image in list(train["Image"]):
        parsed_train_images.append(map(lambda x: int(x), image.split()))
    X_train = pd.DataFrame(parsed_train_images, columns=pixel_columns)

    # Convert Image column in test into DataFrame, zipping their ImageId's too
    parsed_test_images = []
    for image_id, image in zip(list(test["ImageId"]), list(test["Image"])):
        parsed_test_images.append(
            [image_id] + map(lambda x: int(x), image.split()))
    X_test = pd.DataFrame(parsed_test_images,
                          columns=["ImageId"] + pixel_columns)

    X_train.to_csv("X_train.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)


def load_data():
    """."""
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")
    X_test = pd.read_csv("X_test.csv")

    print (X_train)

def main():
    """."""
    # parse_data()
    load_data()

if __name__ == "__main__":
    main()
