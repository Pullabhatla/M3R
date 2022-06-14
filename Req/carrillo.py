import numpy as np
from Req import LeakyReLUSoftmaxCCE

def carrillo_cbo(x_train, y_train, N, M, n_its, m, l, beta, eta, sigma_0, in_shape, out_shape, layers, x_test, y_test):
    n = y_train.shape[0]
    remainder = np.array([], dtype=int)
    particles = np.array([LeakyReLUSoftmaxCCE(in_shape, out_shape, layers) for _ in range(N)])
    X_bar = LeakyReLUSoftmaxCCE(in_shape, out_shape, layers)
    shapes_W = [W.shape for W in X_bar.weights]
    shapes_b = [b.shape for b in X_bar.biases]
    sigma = sigma_0

    history = {'train_accuracy': [X_bar.accuracy(x_train, y_train)],
                   'train_loss': [X_bar.loss(x_train, y_train)],
                   'test_accuracy': [X_bar.accuracy(x_test, y_test)],
                   'test_loss': [X_bar.loss(x_test, y_test)]}

    for k in range(n_its):
        indices = np.concatenate([np.array(range(N)), remainder])
        np.random.shuffle(indices)
        batch_indices = []
        for i in range(0, N, M):
            if i+M<= N:
                batch_indices.append(indices[i:i+M])
            else:
                remainder = indices[i:]
        particle_batches = [particles[idx] for idx in batch_indices]

        for batch in particle_batches:
            batch_indices = np.random.choice(np.arange(n), m, False)
            A_x = x_train[batch_indices]
            A_y = y_train[batch_indices]

            losses = np.array([X.loss(A_x, A_y) for X in batch])

            num = np.exp(-beta*losses)[:, None]
            den = np.sum(num)

            old_loss = X_bar.loss(A_x, A_y)

            X_bar.weights = (num*np.array([np.array(X.weights) for X in batch])).sum(axis=0)/den
            X_bar.biases = (num*np.array([np.array(X.biases) for X in batch])).sum(axis=0)/den

            Xb_w, Xb_b = X_bar.weights, X_bar.biases
            for X in particles:
                X_w, X_b = X.weights, X.biases
                X.weights = X_w - l*eta*(X_w-Xb_w) + sigma*np.sqrt(eta)*(X_w-Xb_w)*np.array([np.random.randn(*shape) for shape in shapes_W])
                X.biases = X_b - l*eta*(X_b-Xb_b) + sigma*np.sqrt(eta)*(X_b-Xb_b)*np.array([np.random.randn(*shape) for shape in shapes_b])

            if old_loss<=X_bar.loss(A_x, A_y):
                for X in particles:
                    X_w, X_b = X.weights, X.biases
                    X.weights = X_w + sigma*np.sqrt(eta)*np.array([np.random.randn(*shape) for shape in shapes_W])
                    X.biases = X_b + sigma*np.sqrt(eta)*np.array([np.random.randn(*shape) for shape in shapes_b])

        sigma = sigma_0/np.log(k+2)

        history['train_accuracy'].append(X_bar.accuracy(x_train, y_train))
        history['train_loss'].append(X_bar.loss(x_train, y_train))
        history['test_accuracy'].append(X_bar.accuracy(x_test, y_test))
        history['test_loss'].append(X_bar.loss(x_test, y_test))
    return X_bar, history
