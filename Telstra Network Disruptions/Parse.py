"""
Module to parse all the data provided for Telstra Network Disruption
competition. Additionally perform feature engineering.
"""

from __future__ import print_function
import numpy as np
import pandas as pd


def parse_single_attr(attribute):
    """Parse single attribute DataFrame."""
    # Create one hot version of event types frame
    single_attr_df = pd.read_csv(attribute + ".csv")

    one_hot_single_attrs = (single_attr_df[attribute].str.join("").str.get_dummies())
    del single_attr_df[attribute]
    single_attr_df = (pd.concat([single_attr_df, one_hot_single_attrs], axis=1))

    # compress frame by id
    compressed_array = []
    for name, single_attrs in single_attr_df.groupby(["id"]):
        attr_arrays = np.array(single_attrs)
        attr_array = attr_arrays[0]

        for other_array in attr_arrays[1:]:
            attr_array = np.add(attr_array, other_array)

        attr_array[0] /= len(attr_arrays)
        compressed_array.append(attr_array)

    columns = single_attr_df.columns.tolist()

    return pd.DataFrame(compressed_array, columns=columns)


def main():
    """."""

    event_type_df = parse_single_attr("event_type")
    resource_type_df = parse_single_attr("resource_type")
    # print (resource_type_df)


if __name__ == "__main__":
    main()
