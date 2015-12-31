"""Random forest model for Titanic dataset."""

from __future__ import print_function
import logging
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def set_verbosity(verbose_level=3):
    """Set the level of verbosity of the Preprocessing."""
    if not type(verbose_level) == int:
        raise TypeError("verbose_level must be an int")

    if verbose_level < 0 or verbose_level > 4:
        raise ValueError("verbose_level must be between 0 and 4")

    verbosity = [logging.CRITICAL,
                 logging.ERROR,
                 logging.WARNING,
                 logging.INFO,
                 logging.DEBUG]

    logging.basicConfig(
        format='%(asctime)s:\t %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=verbosity[verbose_level])


def ticket_transform(unique_column_vals):
    """Bucketize Parch."""

    def bucket_ticket(ticket):
        """Bucket an individual ticket."""
        try:
            bucket = int(ticket[0][0])
            if bucket == 1 or bucket == 2:
                return bucket
            else:
                return 3
        except ValueError:
            bucket = ord(ticket[0][0])
            if bucket == 80:
                return bucket
            else:
                return -1
            # return bucket

    column_map = {val: bucket_ticket(val) for val in unique_column_vals}
    return column_map


def sibsp_transform(unique_column_vals):
    """Bucketize SibSp."""

    def bucket_sibsp(sibsp):
        """Bucket an individual SipSp."""
        if sibsp > 2:
            return 3
        else:
            return sibsp
        return sibsp

    column_map = {val: bucket_sibsp(val) for val in unique_column_vals}
    return column_map


def name_transform(unique_column_vals):
    """Bucketize names."""

    def bucket_name(name):
        """Bucket an individual name."""

        for entry in (name.split()):
            if '.' in entry:
                title = entry
                break

        bucket_all_dead = ["Capt.", "Don.", "Jonkheer.", "Rev."]
        bucket_all_alive = ["Countess.", "Lady.", "Mlle.", "Mme.", "Ms.",
                            "Sir."]
        bucket_equal = ["Col.", "Major."]
        bucket_more_alive = ["Master.", "Miss.", "Mrs."]
        bucket_more_dead = ["Mr."]

        if title in bucket_all_dead:
            return 1
        elif title in bucket_all_alive:
            return 2
        elif title in bucket_equal:
            return 3
        elif title in bucket_more_alive:
            return 4
        elif title in bucket_more_dead:
            return 5
        else:
            logging.info("UNKNOWN TITLE: " + title)
            return 6

    column_map = {val: bucket_name(val) for val in unique_column_vals}
    return column_map


def default_transform(unique_column_vals):
    """
    Perform default transform on a given set of column values,
    i.e. map to unique integers.
    """

    if not all([type(val) is int for val in unique_column_vals]):
        column_map = {val: i for i, val in enumerate(unique_column_vals)}

    return column_map


def map_features(train, test, column_transforms):
    """
    Map the values that each feature takes on to unique ints in a given
    train and test DataFrame.
    """

    mapping = {}

    train_columns = np.array(train.columns)
    test_columns = np.array(test.columns)

    # Assert we're mapping the same columns to be safe
    assert np.array_equal(train_columns, test_columns)

    for column in train_columns:
        # Take union of values of columns in train and test then create
        # column mapping
        unique_column_vals = np.union1d(train[column].unique(),
                                        test[column].unique())

        if column not in column_transforms.keys():
            column_map = column_transforms["default"](unique_column_vals)
        else:
            column_map = column_transforms[column](unique_column_vals)
            mapping[column] = column_map

        for key, value in column_map.iteritems():
            train.loc[train[column] == key, column] = value
            test.loc[test[column] == key, column] = value

    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)


def load_data(columnn_transforms, train_file="train.csv", test_file="test.csv",
              val_size=None, drop_columns=None):
    """
    Load training and test set and optionally a validation set whose size is
    determined by val_size parameter. Also, optionally drop all columns
    in drop_columns if they occur in csv header.
    """

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # Try to drop columns provided if there are any
    if drop_columns:
        for column in drop_columns:
            if column in train.columns and column in test.columns:
                train.drop(column, axis=1, inplace=True)
                test.drop(column, axis=1, inplace=True)

    X_train = train.drop("Survived", axis=1)
    X_test = test

    map_features(X_train, X_test, columnn_transforms)

    y_train = pd.DataFrame(train["Survived"], columns=["Survived"])
    y_train = np.reshape(np.array(y_train, dtype=np.uint8), (len(y_train), ))

    # If validation size provided, split train set into train/validation sets
    if val_size:
        assert type(val_size) is int or type(val_size) is float
        X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                          y_train,
                                                          test_size=val_size)
        return X_train, y_train, X_val, y_val, X_test
    else:
        return X_train, y_train, X_test


def main():
    """Entry point."""
    set_verbosity(0)

    columns_to_drop = []
    column_transforms = {"default": default_transform,
                         "Name": name_transform,
                         "SibSp": sibsp_transform}#,
                         #"Ticket": ticket_transform}
    # X_train, y_train, X_val, y_val, X_test = load_data(
    #    column_transforms, val_size=.20, drop_columns=columns_to_drop)

    X_train, y_train, X_test = load_data(column_transforms,
                                         drop_columns=columns_to_drop)

    clf = RandomForestClassifier(n_estimators=20)
    clf.fit(X_train, y_train)

    '''
    y_pred = clf.predict(X_val)
    print (classification_report(y_val, y_pred))
    '''
    # '''
    y_pred = clf.predict(X_test)
    test_df = pd.DataFrame(y_pred, columns=["Survived"])
    test_df.index += 892
    test_df.index.name = "PassengerId"
    test_df.to_csv("Predictions.csv")
    # '''

if __name__ == "__main__":
    main()
