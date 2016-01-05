"""Parse data for San Francisco Crime Classification contest."""

from __future__ import print_function, division
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split


def map_categories(x, mapping):
    """Map a category to an integer."""
    return mapping[x]


def parse_time(x):
    """Extract hour, day, month, and year from time category."""
    from datetime import datetime
    DD = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    time = DD.hour
    day = DD.day
    month = DD.month
    year = DD.year
    return time, day, month, year


def get_season(x):
    """Extract summer from month values."""
    summer, fall, winter, spring = 0, 0, 0, 0

    if (x in [5, 6, 7]):
        summer = 1
    if (x in [8, 9, 10]):
        fall = 1
    if (x in [11, 0, 1]):
        winter = 1
    if (x in [2, 3, 4]):
        spring = 1
    return summer, fall, winter, spring


def parse_data(df, logodds):
    """Parse a given DataFrame into numeric form."""
    # Build log-odd address features
    address_features = df["Address"].apply(lambda x: logodds[x])
    address_features.columns = [
        "logodds" + str(x) for x in range(len(address_features.columns))]

    # Build temporal features
    df["Time"], df["Day"], df["Month"], df["Year"] = \
        zip(*df["Dates"].apply(parse_time))

    # Build one-hot features
    dummy_ranks_PD = pd.get_dummies(df['PdDistrict'], prefix='PD')
    dummy_ranks_DAY = pd.get_dummies(df["DayOfWeek"], prefix='DAY')

    # Build artificial feature for whether crime happened on intersection
    df["IsInterection"] = df["Address"].apply(
        lambda x: 1 if "/" in x else 0)

    # Drop processed columns
    df = df.drop("PdDistrict", axis=1)
    df = df.drop("DayOfWeek", axis=1)
    df = df.drop("Address", axis=1)
    df = df.drop("Dates", axis=1)

    # Join all features together into 1 frame
    feature_list = df.columns.tolist()

    features = df[feature_list]
    features = features.join(dummy_ranks_PD.ix[:, :])
    features = features.join(dummy_ranks_DAY.ix[:, :])
    features = features.join(address_features.ix[:, :])

    # Build feature for whether or not crime happened during middle of night
    features["Awake"] = features["Time"].apply(
        lambda x: 1 if (x == 0 or (x >= 8 and x <= 23)) else 0)

    # Build feature for season in which crime occured
    features["Summer"], features["Fall"], features["Winter"], features["Spring"] = zip(*features["Month"].apply(get_season))

    return features


def load_data(train_file="train.csv", test_file="test.csv",
              validation_size=None):
    """Load data from train and test files respectively."""
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # Scale X and Y features in train to 0 mean, unit variance
    xy_scaler = StandardScaler()
    xy_scaler.fit(train[["X", "Y"]])
    train[["X", "Y"]] = xy_scaler.transform(train[["X", "Y"]])
    train = train[abs(train["Y"]) < 100]
    train.index = range(len(train))

    # Scale X and Y in test too and remove outliers
    test[["X", "Y"]] = xy_scaler.transform(test[["X", "Y"]])
    # Set outliers to 0
    test["X"] = test["X"].apply(lambda x: 0 if abs(x) > 5 else x)
    test["Y"] = test["Y"].apply(lambda y: 0 if abs(y) > 5 else y)

    # Filter out attributes that occur in either the test or training set but
    # not both.
    good_attributes = list(
        set(train.columns) - ((set(train.columns)) ^ (set(test.columns))))
    X_train = train[good_attributes]
    y_train = train["Category"]

    X_test = test[good_attributes]

    # Create log-odds for addresses w.r.t. categories
    from copy import deepcopy
    train_addresses = sorted(train["Address"].unique())
    categories = sorted(train["Category"].unique())
    C_counts = train.groupby(["Category"]).size()
    A_C_counts = train.groupby(["Address", "Category"]).size()
    train_A_counts = train.groupby(["Address"]).size()
    logodds = {}
    logoddsPA = {}
    MIN_CAT_COUNTS = 2
    default_logodds = np.log(C_counts / len(train)) - np.log(1.0 - C_counts / len(train))

    # first make logodds for training set and parse X_train
    for addr in train_addresses:
        PA = train_A_counts[addr] / float(len(train))
        logoddsPA[addr] = np.log(PA) - np.log(1. - PA)
        logodds[addr] = deepcopy(default_logodds)
        for cat in A_C_counts[addr].keys():
            if (A_C_counts[addr][cat] > MIN_CAT_COUNTS) and A_C_counts[addr][cat] < train_A_counts[addr]:
                PA = A_C_counts[addr][cat] / float(train_A_counts[addr])
                logodds[addr][categories.index(cat)] = np.log(PA) - np.log(1.0 - PA)
        logodds[addr] = pd.Series(logodds[addr])
        logodds[addr].index = range(len(categories))

    X_train = parse_data(X_train, logodds)

    # second make logodds for testing set and parse X_test
    test_addresses = sorted(test["Address"].unique())
    test_A_counts = test.groupby("Address").size()
    only_test = set(test_addresses + train_addresses) - set(train_addresses)
    only_train = set(test_addresses + train_addresses) - set(test_addresses)
    test_and_train = set(test_addresses).intersection(train_addresses)

    for addr in only_test:
        PA = test_A_counts[addr] / float(len(test) + len(train))
        logoddsPA[addr] = np.log(PA) - np.log(1. - PA)
        logodds[addr] = deepcopy(default_logodds)
        logodds[addr].index = range(len(categories))
    for addr in test_and_train:
        PA = (train_A_counts[addr] + test_A_counts[addr]) / (len(test) + len(train))
        logoddsPA[addr] = np.log(PA) - np.log(1. - PA)

    X_test = parse_data(X_test, logodds)

    assert X_train.columns.tolist() == X_test.columns.tolist()
    columns = X_train.columns.tolist()
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train[columns] = scaler.transform(X_train)
    X_test[columns] = scaler.transform(X_test[columns])

    # category_map = {categories[i]: i + 1 for i in range(len(categories))}
    # y_train = y_train.apply(map_categories, args=(category_map,))

    if validation_size:
        try:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=validation_size)
            return X_train, y_train, X_valid, y_valid, X_test
        except TypeError:
            pass
        except ValueError:
            pass

    return X_train, y_train, X_test


def main():
    """."""
    X_train, y_train, X_valid, y_valid, X_test = load_data(validation_size=.2)
    print (np.array(X_train).shape)
    print (np.array(y_train).shape)
    print (np.array(X_valid).shape)
    print (np.array(y_valid).shape)
    print (np.array(X_test).shape)

if __name__ == "__main__":
    main()
