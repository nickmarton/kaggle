"""Multilayer Perceptron for San Francisco Crime Classification contest."""

# THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python MLP.py

from __future__ import print_function
import time
import theano
import lasagne
import logging
import numpy as np
import pandas as pd
import theano.tensor as T
from Parse import load_data, map_categories


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


def build_network(input_var=None):
    """Build multilayer perceptron."""
    # Start with input
    network = lasagne.layers.InputLayer(shape=(None, 1, 1, 68),
                                        input_var=input_var)

    network = lasagne.layers.DropoutLayer(network, p=0.2)

    network = lasagne.layers.DenseLayer(
        network, num_units=1000, nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DropoutLayer(network, p=0.5)

    network = lasagne.layers.DenseLayer(
        network, num_units=1000, nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DropoutLayer(network, p=0.5)

    network = lasagne.layers.DenseLayer(
        network, num_units=39, nonlinearity=lasagne.nonlinearities.softmax)

    return network


def main(verbose=3, num_epochs=50, batch_size=10000):
    """."""

    set_verbosity(verbose)

    # load the data without any validation
    X_train, y_train, X_test = load_data(train_file="X_train_parsed.csv",
                                         test_file="X_test_parsed.csv",
                                         label_file="y_train_parsed.csv",
                                         parse=False)

    # Convert DataFrames into gpu capable numpy arrays
    X_train = np.array(X_train, dtype=np.float32)
    X_train = np.reshape(X_train, (len(X_train), 1, 1, 68))

    X_test = np.array(X_test, dtype=np.float32)
    X_test = np.reshape(X_test, (len(X_test), 1, 1, 68))

    categories = sorted(y_train["Category"].unique())
    category_map = {categories[i]: i for i in range(len(categories))}
    y_train = y_train["Category"].apply(map_categories, args=(category_map,))
    y_train = np.array(y_train, dtype=np.int32)

    # Create tensors for X and y and build network
    input_var = T.ftensor4('inputs')
    target_var = T.ivector('targets')

    network = build_network(input_var)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    predict_function = theano.function([input_var], test_prediction)

    print("Starting training...")
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size,
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
        logging.info("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        # print("  validation accuracy:\t\t{:.2f} %".format(
        #    val_acc / val_batches * 100))

        if (epoch + 1) % 10 == 0:
            logging.info("Intermediate prediction at epoch " + str(epoch + 1))
            frames, next_i = [], 0
            for i, batch in enumerate(iterate_test_minibatches(X_test, batch_size)):
                y_pred = predict_function(batch)
                df = pd.DataFrame(y_pred, columns=[c for c in categories])
                df.index += (i * batch_size)
                next_i = i + 1
                frames.append(df)

            # If batch_size doesn't divide length of X_test perfectly,
            # predict on remaining elements
            if (next_i * batch_size) < len(X_test):
                y_pred = predict_function(
                    X_test[slice((next_i * batch_size), len(X_test))])
                df = pd.DataFrame(y_pred, columns=[c for c in categories])
                df.index += (next_i * batch_size)
                frames.append(df)

            predictions = pd.concat(frames)
            predictions.index.name = "Id"
            predictions.to_csv("predictions_" + str(epoch + 1) + "epochs.csv")

    logging.info("Finished training; beginning final prediction")
    frames, next_i = [], 0
    for i, batch in enumerate(iterate_test_minibatches(X_test, batch_size)):
        y_pred = predict_function(batch)
        df = pd.DataFrame(y_pred, columns=[c for c in categories])
        df.index += (i * batch_size)
        next_i = i + 1
        frames.append(df)

    # If batch_size doesn't divide length of X_test perfectly,
    # predict on remaining elements
    if (next_i * batch_size) < len(X_test):
        y_pred = predict_function(
            X_test[slice((next_i * batch_size), len(X_test))])
        df = pd.DataFrame(y_pred, columns=[c for c in categories])
        df.index += (next_i * batch_size)
        frames.append(df)

    predictions = pd.concat(frames)
    predictions.index.name = "Id"
    predictions.to_csv("predictions_" + str(num_epochs) + "epochs.csv")


if __name__ == "__main__":
    main()
