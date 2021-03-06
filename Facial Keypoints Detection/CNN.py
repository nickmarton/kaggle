"""
Convolutional Neural Network for kaggle facial keypoints detection contest.

Note: Only the labels contain missing values in all of the data.
"""

# THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python CNN.py

from __future__ import division, print_function
import time
import theano
import lasagne
import logging
import numpy as np
import pandas as pd
import theano.tensor as T
from sklearn.cross_validation import train_test_split


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


def imputate(frame):
    """Deal with missing values in a DataFrame."""
    start_time = time.time()

    frame[frame.isnull().any(axis=1)].to_csv("train_incomplete.csv",
                                             index=False)
    frame.dropna(inplace=True)

    time_diff = time.time() - start_time
    logging.info("Imputation completed in " + str(time_diff) + " seconds")


def parse_data(train_file="training.csv", test_file="test.csv"):
    """
    Parse training and test data;
    split Image and labels and convert Image column to DataFrame.
    """

    start_time = time.time()

    train = pd.read_csv(train_file)
    imputate(train)

    test = pd.read_csv(test_file)

    # Get y_train then scale between [-1, 1]
    y_train = train.ix[:, :-1]
    y_max = np.max(np.array(y_train))
    y_min = np.min(np.array(y_train))
    y_train = (2 * (y_train - y_min) / (y_max - y_min)) - 1

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
    X_train_reshape = np.zeros((len(X_train), 1, 96, 96), dtype=np.float32)
    for i, row in enumerate(np.array(X_train)):
        X_train_reshape[i] = row.reshape(1, 96, 96)
    X_train = X_train_reshape

    # Convert y_train into numpy array
    y_train = np.array(y_train, dtype=np.float32)

    # Cast X_test into numpy array and reshape into form suitable for lasagne
    X_test_reshape = np.zeros((len(X_test), 1, 96, 96), dtype=np.float32)
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


def build_simple_cnn(input_var=None):
    """Build a simple CNN for Facial Keypoints Detection."""
    # INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC

    start_time = time.time()

    # Begin with input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 96, 96),
                                        input_var=input_var)

    # Add first [CONV -> RELU -> POOL]
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Add second [CONV -> RELU -> POOL]
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Add FC -> RELU, throw in dropout
    network = lasagne.layers.DropoutLayer(network, p=0.5)
    network = lasagne.layers.DenseLayer(
        network, num_units=256, nonlinearity=lasagne.nonlinearities.rectify)

    # Add output layer, throw in dropout; use identity fn for nonlinearity
    network = lasagne.layers.DropoutLayer(network, p=0.5)
    network = lasagne.layers.DenseLayer(
        network, num_units=30, nonlinearity=None)

    time_diff = time.time() - start_time
    logging.info("Built network in " + str(time_diff) + " seconds")

    return network


def build_complex_cnn(input_var=None):
    """Build a CNN specific to MNIST data."""

    start_time = time.time()

    # Start with input
    network = lasagne.layers.InputLayer(shape=(None, 1, 96, 96),
                                        input_var=input_var)

    # Add first [CONV -> RELU -> POOL]
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Add second [CONV -> RELU -> POOL]
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=64, filter_size=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Add third [CONV -> RELU -> POOL]
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=128, filter_size=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Add first [FC -> RELU], throw in dropout
    network = lasagne.layers.DropoutLayer(network, p=0.5)
    network = lasagne.layers.DenseLayer(
        network, num_units=500, nonlinearity=lasagne.nonlinearities.rectify)

    # Add second [FC -> RELU], throw in dropout
    network = lasagne.layers.DropoutLayer(network, p=0.5)
    network = lasagne.layers.DenseLayer(
        network, num_units=500, nonlinearity=lasagne.nonlinearities.rectify)

    # Add output layer, throw in dropout; use identity fn for nonlinearity
    network = lasagne.layers.DropoutLayer(network, p=0.5)
    network = lasagne.layers.DenseLayer(
        network, num_units=30, nonlinearity=None)

    time_diff = time.time() - start_time
    logging.info("Built network in " + str(time_diff) + " seconds")

    return network


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """Train via batches to avoid memory error."""
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def iterate_test_minibatches(inputs, batchsize, shuffle=False):
    """Train via batches to avoid memory error."""
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]


def main(verbose=3, num_epochs=1, batch_size=500):
    """."""

    set_verbosity(verbose)

    # parse_data()

    # X_train, y_train, X_val, y_val, X_test = load_data(
    #    validation_size=0.33)

    X_train, y_train, X_test = load_data()

    # Log array shapes for debugging
    logging.debug("X_train shape: " + str(X_train.shape))
    logging.debug("y_train shape: " + str(y_train.shape))
    try:
        logging.debug("X_val shape: " + str(X_val.shape))
        logging.debug("y_val shape: " + str(y_val.shape))
    except NameError:
        pass
    logging.debug("X_test shape: " + str(X_test.shape) + '\n')

    # Create tensors for X and y and build network
    input_var = T.ftensor4('inputs')
    target_var = T.fmatrix('targets')

    network = build_complex_cnn(input_var)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = np.sqrt(loss.mean())

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    '''
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = np.sqrt(test_loss.mean())

    val_fn = theano.function([input_var, target_var], test_loss)
    '''

    columns = ["left_eye_center_x", "left_eye_center_y",
               "right_eye_center_x", "right_eye_center_y",
               "left_eye_inner_corner_x", "left_eye_inner_corner_y",
               "left_eye_outer_corner_x", "left_eye_outer_corner_y",
               "right_eye_inner_corner_x", "right_eye_inner_corner_y",
               "right_eye_outer_corner_x", "right_eye_outer_corner_y",
               "left_eyebrow_inner_end_x", "left_eyebrow_inner_end_y",
               "left_eyebrow_outer_end_x", "left_eyebrow_outer_end_y",
               "right_eyebrow_inner_end_x", "right_eyebrow_inner_end_y",
               "right_eyebrow_outer_end_x", "right_eyebrow_outer_end_y",
               "nose_tip_x", "nose_tip_y",
               "mouth_left_corner_x", "mouth_left_corner_y",
               "mouth_right_corner_x", "mouth_right_corner_y",
               "mouth_center_top_lip_x", "mouth_center_top_lip_y",
               "mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y"]

    predict_function = theano.function([input_var], test_prediction)

    logging.info("Starting training...")
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train,
                                         y_train,
                                         batch_size,
                                         shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
            logging.debug("Done " + str(train_batches) + " batches")

        '''
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size,
                                         shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        '''

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        # print("  validation accuracy:\t\t{:.2f} %".format(
        #    val_acc / val_batches * 100))

        if (epoch + 1) % 1 == 0:
            logging.info("Intermediate prediction at epoch " + str(epoch + 1))
            start_time = time.time()

            frames, next_i = [], 0
            for i, batch in enumerate(iterate_test_minibatches(X_test, batch_size)):
                y_pred = predict_function(batch)
                df = pd.DataFrame(y_pred, columns=columns)
                df.index += (i * batch_size) + 1
                next_i = i + 1
                frames.append(df)

            # If batch_size doesn't divide length of X_test perfectly,
            # predict on remaining elements
            if (next_i * batch_size) < len(X_test):
                y_pred = predict_function(
                    X_test[slice((next_i * batch_size), len(X_test))])
                df = pd.DataFrame(y_pred, columns=columns)
                df.index += (next_i * batch_size) + 1
                frames.append(df)

            predictions = pd.concat(frames)
            predictions.index.name = "ImageId"
            predictions.to_csv("raw_predictions_" + str(epoch + 1) + ".csv")

            time_diff = time.time() - start_time
            logging.info(
                "Predictions complete in " + str(time_diff) + " seconds")

    # Do final prediction
    logging.info("Finished training; beginning prediction")
    start_time = time.time()

    frames, next_i = [], 0
    for i, batch in enumerate(iterate_test_minibatches(X_test, batch_size)):
        y_pred = predict_function(batch)
        df = pd.DataFrame(y_pred, columns=columns)
        df.index += (i * batch_size) + 1
        next_i = i + 1
        frames.append(df)

    # If batch_size doesn't divide length of X_test perfectly,
    # predict on remaining elements
    if (next_i * batch_size) < len(X_test):
        y_pred = predict_function(
            X_test[slice((next_i * batch_size), len(X_test))])
        df = pd.DataFrame(y_pred, columns=columns)
        df.index += (next_i * batch_size) + 1
        frames.append(df)

    predictions = pd.concat(frames)
    predictions.index.name = "ImageId"
    predictions.to_csv("raw_predictions_final.csv")

    time_diff = time.time() - start_time
    logging.info("Predictions complete in " + str(time_diff) + " seconds")


if __name__ == "__main__":
    main()
