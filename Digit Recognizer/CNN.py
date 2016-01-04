from __future__ import division, print_function
import time
import numpy as np
import pandas as pd
import theano.tensor as T
from sklearn.cross_validation import train_test_split
import theano
import lasagne


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


def load_data(
        train_file="train.csv", test_file="test.csv", validation_size=None):
    """
    Load training and test data; optionally create a validation set with size
    provided in validation_size parameter.
    """

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    y_train = pd.DataFrame(train["label"], columns=["label"])
    y_train = np.reshape(np.array(y_train, dtype=np.int32), (len(y_train), ))
    X_train = train.ix[:, 1:] / 255
    X_train_reshape = np.zeros((len(X_train), 1, 28, 28), dtype=np.float32)

    for i, row in enumerate(np.array(X_train)):
        X_train_reshape[i] = row.reshape(1, 28, 28)
    X_train = X_train_reshape

    X_test = test / 255
    X_test_reshape = np.zeros((len(X_test), 1, 28, 28), dtype=np.float32)

    for i, row in enumerate(np.array(X_test)):
        X_test_reshape[i] = row.reshape(1, 28, 28)
    X_test = X_test_reshape

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


def build_cnn(input_var=None):
    """Build a CNN specific to MNIST data."""

    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=70,
        filter_size=(6, 6), nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(
        network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=70,
        filter_size=(6, 6), nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(
        network, pool_size=(2, 2))

    network = lasagne.layers.DropoutLayer(network, p=0.5)

    network = lasagne.layers.DenseLayer(
        network, num_units=500,
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DropoutLayer(network, p=0.5)

    network = lasagne.layers.DenseLayer(
        network, num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network


def main():
    """."""
    num_epochs, batch_size = 350, 1000

    # X_train, y_train, X_val, y_val, X_test = load_data(
    #    validation_size=0.99)

    X_train, y_train, X_test = load_data()

    # Prepare Theano variables for inputs and targets
    input_var = T.ftensor4('inputs')
    target_var = T.ivector('targets')

    network = build_cnn(input_var)

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

    predict_function = theano.function([input_var], test_prediction)

    frames = []
    for i, batch in enumerate(iterate_test_minibatches(X_test, batch_size)):
        y_pred = predict_function(batch)
        df = pd.DataFrame(np.argmax(y_pred, axis=1), columns=["Label"])
        df.index += (i * batch_size) + 1
        frames.append(df)

    predictions = pd.concat(frames)
    predictions.index.name = "ImageId"
    predictions.to_csv("predictions_6x6filter_70kernel_2x2pool_500hidden_350epoch.csv")

if __name__ == "__main__":
    main()
