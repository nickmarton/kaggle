"""Support vector machine for Telstra Network Dsruptions."""

from __future__ import print_function
import logging
import numpy as np
import pandas as pd
from sklearn.svm import SVC


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


def main():
    """."""

    set_verbosity(3)

    train = pd.read_csv("parsed_train.csv")
    test = pd.read_csv("parsed_test.csv")

    y_train = train["fault_severity"]
    X_train = train.ix[:, 1:-1]
    X_test = test.ix[:, 1:]

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)

    clf = SVC(kernel='poly', degree=9, probability=True)
    logging.info("Beginning training")
    clf.fit(X_train, y_train)
    logging.info("Done training")

    print ()

    logging.info("Beginning prediction")
    predictions = pd.DataFrame(clf.predict_proba(X_test),
                               columns=["predict_0", "predict_1", "predict_2"])
    predictions = pd.concat([test.ix[:, 0], predictions], axis=1)
    logging.info("Done prediction; writing")
    predictions.to_csv("svm_predictions.csv", index=False)

if __name__ == "__main__":
    main()
