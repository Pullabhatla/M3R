import numpy as np


class MLP:
    def __init__(self, in_shape, out_shape, layer_activation,
                 output_activation, layer_sizes):
        """Use Glorot initialisation"""
        i, j = np.prod(in_shape), layer_sizes[0]
        self.weights = [np.random.randn(i, j)*np.sqrt(2/(i+j))]
        self.biases = [np.zeros(j)]
        self.num_hidden_layers = len(layer_sizes)

        for i, j in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weights.append(np.random.randn(i, j)*np.sqrt(2/(i+j)))
            self.biases.append(np.zeros(j))

        i, j = layer_sizes[-1], out_shape
        self.weights.append(np.random.randn(i, j)*np.sqrt(2/(i+j)))
        self.biases.append(np.zeros(out_shape))

        self.layer_activation = layer_activation
        self.output_activation = output_activation

    def predict(self, X):
        a = X.reshape(X.shape[0], -1)

        layer_activation = self.layer_activation

        for W, b in zip(self.weights[:-1], self.biases[:-1]):  # noqa N806
            a = layer_activation(a@W + b)

        return self.output_activation(a@self.weights[-1] + self.biases[-1])

    def classify(self, X):
        return np.argmax(self.predict(X), axis=1)

    def accuracy(self, X, Y):
        return (np.argmax(Y, axis=1) == self.classify(X)).mean()

    def forward_pass(self, X):
        pre_act = [X.reshape(X.shape[0], -1)]
        post_act = [X.reshape(X.shape[0], -1)]

        layer_activation = self.layer_activation

        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            pre_act.append(pre_act[-1]@W + b)
            post_act.append(layer_activation(pre_act[-1]))

        pre_act.append(pre_act[-1]@self.weights[-1] + self.biases[-1])
        post_act.append(self.output_activation(pre_act[-1]))

        return pre_act, post_act


def leakyReLU(X):
    return np.where(X >= 0, X, X * 0.01)


def softmax(X):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(X - np.max(X, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


class LeakyReLUSoftmaxCCE(MLP):

    def __init__(self, in_shape, out_shape, layer_sizes):
        super().__init__(in_shape, out_shape, leakyReLU, softmax, layer_sizes)

    def loss(self, X, Y):
        return -np.mean(np.log(self.predict(X)[Y != 0]))

    def gd_train(self, x_train, y_train, learning_rate, epochs, x_test, y_test):
        return self.sgd_train(x_train, y_train, learning_rate, epochs,
                              x_train.shape[0], x_test, y_test)

    def sgd_train(self, x_train, y_train, learning_rate, epochs, batch_size,
                  x_test, y_test, momentum=0):
        layers = self.num_hidden_layers
        history = {'train_accuracy': [self.accuracy(x_train, y_train)],
                   'train_loss': [self.loss(x_train, y_train)],
                   'test_accuracy': [self.accuracy(x_test, y_test)],
                   'test_loss': [self.loss(x_test, y_test)]}
        n = y_train.shape[0]
        indices = np.arange(n)
        v_W = [np.zeros_like(W) for W in self.weights]
        v_b = [np.zeros_like(b) for b in self.biases]
        for epoch in range(epochs):
            np.random.shuffle(indices)
            batch_indices = []
            for i in range(0, n, batch_size):
                if i + batch_size <= n:
                    batch_indices.append(indices[i:i+batch_size])
                else:
                    batch_indices.append(indices[i:])
            batches = [(x_train[idx], y_train[idx])
                       for idx in batch_indices]

            for X, Y in batches:
                N = len(X)
                nabla_W = [None for W in self.weights]
                nabla_b = [None for b in self.biases]
                pre_act, post_act = self.forward_pass(X)

                delta = post_act[-1] - Y

                nabla_W[-1] = (post_act[-2].T@delta)/N
                nabla_b[-1] = delta.mean(axis=0)

                for i in range(2, layers+2):
                    delta = (np.where(pre_act[-i] >= 0, 1, 0.01)
                             * (delta@self.weights[-i+1].T))
                    nabla_W[-i] = (post_act[-i-1].T@delta)/N
                    nabla_b[-i] = delta.mean(axis=0)
                
                if momentum == 0:
                    self.weights = [W - learning_rate*nW for W, nW
                                    in zip(self.weights, nabla_W)]
                    self.biases = [b - learning_rate*nb for b, nb
                                in zip(self.biases, nabla_b)]
                
                else:
                    v_W = [(momentum*vW)+(learning_rate*nW) for vW, nW in zip(v_W, nabla_W)]
                    v_b = [(momentum*vb)+(learning_rate*nb) for vb, nb in zip(v_b, nabla_b)]
                    self.weights = [W - vW for W, vW in zip(self.weights, v_W)]
                    self.biases = [b - vb for b, vb in zip(self.biases, v_b)]

            history['train_accuracy'].append(self.accuracy(x_train, y_train))
            history['train_loss'].append(self.loss(x_train, y_train))
            history['test_accuracy'].append(self.accuracy(x_test, y_test))
            history['test_loss'].append(self.loss(x_test, y_test))

        return history

    def nesterov_sgd_train(self, x_train, y_train, learning_rate, epochs,
                           batch_size, x_test, y_test, momentum):
        layers = self.num_hidden_layers
        history = {'train_accuracy': [self.accuracy(x_train, y_train)],
                   'train_loss': [self.loss(x_train, y_train)],
                   'test_accuracy': [self.accuracy(x_test, y_test)],
                   'test_loss': [self.loss(x_test, y_test)]}
        n = y_train.shape[0]
        indices = np.arange(n)
        v_W = [np.zeros_like(W) for W in self.weights]
        v_b = [np.zeros_like(b) for b in self.biases]
        for epoch in range(epochs):
            np.random.shuffle(indices)
            batch_indices = []
            for i in range(0, n, batch_size):
                if i + batch_size <= n:
                    batch_indices.append(indices[i:i+batch_size])
                else:
                    batch_indices.append(indices[i:])
            batches = [(x_train[idx], y_train[idx])
                       for idx in batch_indices]

            for X, Y in batches:
                N = len(X)
                nabla_W = [None for W in self.weights]
                nabla_b = [None for b in self.biases]

                self.weights = [W - momentum*vW for W, vW in zip(self.weights, v_W)]
                self.biases = [b - momentum*vb for b, vb in zip(self.biases, v_b)]
                pre_act, post_act = self.forward_pass(X)

                delta = post_act[-1] - Y

                nabla_W[-1] = (post_act[-2].T@delta)/N
                nabla_b[-1] = delta.mean(axis=0)

                for i in range(2, layers+2):
                    delta = (np.where(pre_act[-i] >= 0, 1, 0.01)
                             * (delta@self.weights[-i+1].T))
                    nabla_W[-i] = (post_act[-i-1].T@delta)/N
                    nabla_b[-i] = delta.mean(axis=0)

                self.weights = [W - learning_rate*nW
                                for W, nW in zip(self.weights, nabla_W)]
                self.biases = [b - learning_rate*nb
                               for b, nb in zip(self.biases, nabla_b)]
                v_W = [momentum*vW+learning_rate*nW
                       for vW, nW in zip(v_W, nabla_W)]
                v_b = [momentum*vb+learning_rate*nb
                       for vb, nb in zip(v_b, nabla_b)]

            history['train_accuracy'].append(self.accuracy(x_train, y_train))
            history['train_loss'].append(self.loss(x_train, y_train))
            history['test_accuracy'].append(self.accuracy(x_test, y_test))
            history['test_loss'].append(self.loss(x_test, y_test))

        return history
