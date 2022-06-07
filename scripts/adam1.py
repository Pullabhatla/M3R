import numpy as np
import matplotlib.pyplot as plt
from Req import LeakyReLUSoftmaxCCE, load_data

(x_train, y_train), (x_test, y_test) = load_data()

beta1s = np.arange(0.85, 0.95, 0.01)[:10]
beta2s = np.arange(0.99, 1, 0.001)[:10]

scores = np.empty((10, 10))
for i in range(10):
    for j in range(10):
        np.random.seed(1112001)
        mlp = LeakyReLUSoftmaxCCE((28, 28), 10, [16, 16, 16])
        scores[j, i] = max(mlp.adam_train(x_train, y_train, 1e-3, 100, beta1s[i], beta2s[j], 1e-8, 32, x_test, y_test)['test_accuracy'])


idx = np.unravel_index(np.argmax(scores), scores.shape)

plt.contourf(beta1s, beta2s, scores)
plt.xlabel(r'$\beta{}_{1}$')
plt.ylabel(r'$\beta{}_{2}$')
plt.title('Adam Test Accuracy')
plt.colorbar()
plt.savefig('figures/adam1.png')

with open('output/adam1.txt', 'w') as f:
    f.write(f'The maximal test accuracy of {scores[idx]*100}% is achieved ' + f'at {(beta1s[idx[1]], beta2s[idx[0]])}.')
