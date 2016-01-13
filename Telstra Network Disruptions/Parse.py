"""
Module to parse all the data provided for Telstra Network Disruption
competition. Additionally perform feature engineering.
"""

from __future__ import print_function
import numpy as np
import pandas as pd


def parse_single_attr(attribute=None, frame=None, use_frame=False):
    """Parse single attribute DataFrame."""
    # Create one hot version of event types frame
    if use_frame:
        single_attr_df = frame
    else:
        single_attr_df = pd.read_csv(attribute + ".csv")

    if attribute == "volume":
        one_hot_single_attrs = single_attr_df[attribute].str.get_dummies()
        new_columns = [
            "volume_" + col for col in one_hot_single_attrs.columns.tolist()]
        one_hot_single_attrs.columns = new_columns
    else:
        one_hot_single_attrs = single_attr_df[attribute].str.join("").str.get_dummies()

    # print (one_hot_single_attrs)
    del single_attr_df[attribute]
    single_attr_df = pd.concat([single_attr_df, one_hot_single_attrs], axis=1)

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


def make_one_hot_data():
    """Convert all data to one_hot format."""
    event_type_df = parse_single_attr(attribute="event_type")
    resource_type_df = parse_single_attr(attribute="resource_type")
    severity_type_df = parse_single_attr(attribute="severity_type")

    log_feature_df = pd.read_csv("log_feature.csv")
    log_frame = log_feature_df[["id", "log_feature"]]
    volume_frame = log_feature_df[["id", "volume"]]
    feature_type_df = parse_single_attr(attribute="log_feature", frame=log_frame, use_frame=True)
    volume_type_df = (parse_single_attr(attribute="volume", frame=volume_frame, use_frame=True))

    total_frame = pd.merge(event_type_df, resource_type_df, on="id")
    total_frame = pd.merge(total_frame, severity_type_df, on="id")
    total_frame = pd.merge(total_frame, feature_type_df, on="id")
    total_frame = pd.merge(total_frame, volume_type_df, on="id")

    total_frame.to_csv("one_hot_data.csv", index=False)


def main():
    """."""

    # make_one_hot_data()
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    train_df["location"] = train_df["location"].apply(
        lambda x: int(x.split()[1]))
    test_df["location"] = test_df["location"].apply(
        lambda x: int(x.split()[1]))

    one_hot_df = pd.read_csv("one_hot_data.csv")

    parsed_train_df = pd.merge(one_hot_df, train_df, on="id")
    parsed_test_df = pd.merge(one_hot_df, test_df, on="id")

    parsed_train_df.to_csv("parsed_train.csv", index=False)
    parsed_test_df.to_csv("parsed_test.csv", index=False)

if __name__ == "__main__":
    main()
