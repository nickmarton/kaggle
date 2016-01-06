"""Multilayer Perceptron for San Francisco Crime Classification contest."""

from __future__ import print_function
import numpy as np
from Parse import load_data, map_categories


def main():
    """."""

    # load the data without any validation
    X_train, y_train, X_test = load_data(train_file="X_train_parsed.csv",
                                         test_file="X_test_parsed.csv",
                                         label_file="y_train_parsed.csv",
                                         parse=False)

    categories = sorted(y_train["Category"].unique())
    category_map = {categories[i]: i + 1 for i in range(len(categories))}
    y_train = y_train["Category"].apply(map_categories, args=(category_map,))
    print (y_train)

if __name__ == "__main__":
    main()
