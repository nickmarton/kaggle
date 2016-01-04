"""Parse raw predictions into format acceptable by kaggle."""

from __future__ import division, print_function

import logging
import numpy as np
import pandas as pd

Y_MAX = 95.8089831215
Y_MIN = 3.82624305628


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


def load_csv_files(predictions_file="raw_predictions.csv",
                   lookup_file="IdLookupTable.csv"):
    """Load raw predictions and lookup table csv files."""
    # Get raw predictions and split predictions from ImageId's
    raw_predictions = pd.read_csv(predictions_file)
    image_ids = raw_predictions["ImageId"]
    raw_predictions = raw_predictions.ix[:, 1:]

    # Unscale predictions and add ImageId column back in
    unscaled_predictions = (
        (raw_predictions + 1) * (Y_MAX - Y_MIN) / 2) + Y_MIN
    unscaled_predictions["ImageId"] = image_ids

    output = []
    # Get lookup table
    lookup_table = pd.read_csv(lookup_file)
    for index, row in lookup_table.iterrows():
        row_id, image_id, label, loc = np.array(row)

        # Get predicted location corresponding to RowId
        out_loc = unscaled_predictions[
            unscaled_predictions["ImageId"] == image_id][label]
        output.append([row_id, np.array(out_loc)[0]])

        # Log some notion of how long we have left
        if index % 1000 == 0:
            logging.info(
                "{:.2f} percent done".format(index / len(lookup_table) * 100))

    out_df = pd.DataFrame(output, columns=["RowId", "Location"])
    return out_df


def main():
    """."""

    set_verbosity(3)

    out_df = load_csv_files("raw_predictions_195.csv")
    out_df.to_csv("Predictions_195.csv", index=False)

if __name__ == "__main__":
    main()
