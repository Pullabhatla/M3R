import numpy as np
import matplotlib.pyplot as plt
from Req import LeakyReLUSoftmaxCCE, load_data

(x_train, y_train), (x_test, y_test) = load_data()

scores = np.empty((20, 101))
for i in range(20):
    np.random.seed(1112001)
    mlp = LeakyReLUSoftmaxCCE((28, 28), 10, [16, 16, 16])
    scores[i] = mlp.nesterov_sgd_train(x_train, y_train, 1e-3, 100, 32, x_test, y_test, i/20)['test_accuracy']

epochs = [i for i in range(101)]
momentums = [i/20 for i in range(20)]

idx = np.unravel_index(np.argmax(scores), scores.shape)

plt.contourf(epochs, momentums, scores)
plt.xlabel('Epochs')
plt.ylabel(r'$\beta{}$')
plt.title('Nesterov Test Accuracy')
plt.colorbar()
plt.savefig('figures/nesterov1.png')

with open('output/nesterov1.txt', 'w') as f:
    f.write(f'The maximal test accuracy of {scores[idx]*100}% is achieved ' + f'with a momentum of {idx[0]/20} after {idx[1]} epochs.')
