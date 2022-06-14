import numpy as np
from Req import LeakyReLUSoftmaxCCE

def simple_majority_voting(x_train, y_train, N, epochs, learning_rate, in_shape, out_shape, layer_sizes, x_test, y_test):
    mlps = [LeakyReLUSoftmaxCCE(in_shape, out_shape, layer_sizes) for _ in range(N)]
    test_predictions = []
    for mlp in mlps:
        mlp.adam_train(x_train, y_train, learning_rate, epochs, 0.9, 0.999, 1e-8, 32, x_train, y_train)
        test_predictions.append(mlp.predict(x_test))

    d = y_train.shape[1]
    predictions = sum([np.array([[int(i==np.argmax(y)) for i in range(d)] for y in test_prediction]) for test_prediction in test_predictions])
    classifications = np.argmax(predictions, axis=1)
    test_accuracy = (np.argmax(y_test, axis=1)==classifications).mean()

    return test_accuracy

def decorrelated_majority_voting(x_train, y_train, N, epochs, learning_rate, in_shape, out_shape, layer_sizes, x_test, y_test):
    mlps = [LeakyReLUSoftmaxCCE(in_shape, out_shape, layer_sizes) for _ in range(N)]
    test_predictions = []
    n = x_train.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    m = int(n/N)

    for i, mlp in enumerate(mlps):
        X = x_train[indices[m*i:m*(i+1)]]
        Y = y_train[indices[m*i:m*(i+1)]]

        mlp.adam_train(X, Y, learning_rate, epochs, 0.9, 0.999, 1e-8, 32, X, Y)
        test_predictions.append(mlp.predict(x_test))

    d = y_train.shape[1]
    predictions = sum([np.array([[int(i==np.argmax(y)) for i in range(d)] for y in test_prediction]) for test_prediction in test_predictions])
    classifications = np.argmax(predictions, axis=1)
    test_accuracy = (np.argmax(y_test, axis=1)==classifications).mean()

    return test_accuracy

def weighted_voting(x_train, y_train, N, beta, epochs, learning_rate, in_shape, out_shape, layer_sizes, x_test, y_test):
    mlps = [LeakyReLUSoftmaxCCE(in_shape, out_shape, layer_sizes) for _ in range(N)]
    weights = []
    test_predictions = []
    n = x_train.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    batch_indices = []
    complement_indices = []
    m = int(n/N)
    for i in range(N):
        batch_indices.append(indices[m*i:m*(i+1)])
        complement_indices.append(np.concatenate([indices[:m*i],indices[m*(i+1):]]))

    for num, mlp in enumerate(mlps):
        X = x_train[batch_indices[num]]
        Y = y_train[batch_indices[num]]

        mlp.adam_train(X, Y, learning_rate, epochs, 0.9, 0.999, 1e-8, 32, X, Y)
        Xc = x_train[complement_indices[num]]
        Yc = y_train[complement_indices[num]]

        weights.append(np.exp(-beta*mlp.loss(Xc, Yc)))
        test_predictions.append(mlp.predict(x_test))

    predictions = sum([w*p for w, p in zip(weights, test_predictions)])
    classifications = np.argmax(predictions, axis=1)
    test_accuracy = (np.argmax(y_test, axis=1)==classifications).mean()

    return test_accuracy
