"""Visualize data."""

from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def zipper(frame, *columns):
    """Zip given columns of a frame together."""
    return (np.array(frame[[col for col in columns]]))


def ticket_bucketizer(row):
    """."""
    try:
        bucket = int(row[0][0])
        if bucket == 1 or bucket == 2:
            return [bucket, row[1]]
        else:
            return [3, row[1]]
    except ValueError:
        bucket = ord(row[0][0])
        if bucket == 80:
            return [bucket, row[1]]
        else:
            return [-1, row[1]]


def replace_name_with_title(names):
    """Replace a given name with the respective title contained within."""
    for entry in (names[0].split()):
        if '.' in entry:
            title = entry
            break
    return [title, names[1]]


def plot(arr_2d, y_min=None, y_max=None, transform=None):
    """."""
    if transform:
        for i, row in enumerate(arr_2d):
            arr_2d[i] = (transform(row))

    from collections import Counter
    alive = arr_2d[arr_2d[:, 1] == 1]
    dead = arr_2d[arr_2d[:, 1] == 0]
    alive_counts = (Counter(alive[:, 0]))
    dead_counts = (Counter(dead[:, 0]))

    not_in_alive = []
    not_in_dead = []
    for key in alive_counts.keys():
        if key not in dead_counts.keys():
            not_in_dead.append(key)

    for key in dead_counts.keys():
        if key not in alive_counts.keys():
            not_in_alive.append(key)

    for key in not_in_dead:
        dead_counts[key] = 0
    for key in not_in_alive:
        alive_counts[key] = 0

    alive_nums, alive_labels = [], []
    alive_counts = sorted(alive_counts.iteritems(), key=lambda x: x[0])
    for pair in alive_counts:
        alive_labels.append(pair[0])
        alive_nums.append(pair[1])

    dead_nums, dead_labels = [], []
    dead_counts = sorted(dead_counts.iteritems(), key=lambda x: x[0])
    for pair in dead_counts:
        dead_labels.append(pair[0])
        dead_nums.append(pair[1])

    ind = (np.arange(len(alive_counts)))
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, alive_nums, width, color='r')
    rects2 = ax.bar(ind + width, dead_nums, width, color='b')

    ax.set_ylabel('Number of people')

    if y_min and y_max:
        ax.set_ylim(y_min, y_max)
    elif y_min:
        ax.set_ylim(bottom=y_min)
    elif y_max:
        ax.set_ylim(top=y_max)
    else:
        pass

    ax.set_xticks(ind + width)
    ax.set_xticklabels(alive_labels)

    ax.legend((rects1[0], rects2[0]), ('Alive', 'Dead'))

    plt.show()


def main():
    """."""
    X_train, X_test = pd.read_csv("train.csv"), pd.read_csv("test.csv")
    '''
    names = zipper(X_train, "Name", "Survived")
    plot(names, transform=replace_name_with_title)
    '''

    sex = zipper(X_train, "Ticket", "Survived")
    plot(sex, transform=ticket_bucketizer)

if __name__ == "__main__":
    main()
