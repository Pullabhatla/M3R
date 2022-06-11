import numpy as np
import matplotlib.pyplot as plt
from Req import LeakyReLUSoftmaxCCE, load_data

(x_train, y_train), (x_test, y_test) = load_data()

penalty1s = np.arange(0, 0.0001, 0.00001)[:10]
penalty2s = np.arange(0, 0.001, 0.0001)[:10]

scores = np.empty((10, 10))
for i in range(10):
    for j in range(10):
        np.random.seed(1112001)
        mlp = LeakyReLUSoftmaxCCE((28, 28), 10, [16, 16, 16])
        scores[j, i] = max(mlp.elastic_net_train(x_train, y_train, 1e-3, 100, 32, penalty1s[i], penalty2s[j], x_test, y_test)['test_accuracy'])


idx = np.unravel_index(np.argmax(scores), scores.shape)

plt.contourf(penalty1s, penalty2s, scores)
plt.xlabel(r'$\lambda{}_{1}$')
plt.ylabel(r'$\lambda{}_{2}$')
plt.title('Elastic Net Regularisation Test Accuracy')
plt.colorbar()
plt.savefig('figures/elastic_net.png')

with open('output/elastic_net.txt', 'w') as f:
    f.write(f'The maximal test accuracy of {scores[idx]*100}% is achieved ' + f'at {(penalty1s[idx[1]], penalty2s[idx[0]])}.')
