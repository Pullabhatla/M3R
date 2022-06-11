from Req import load_data, LeakyReLUSoftmaxCCE
import matplotlib.pyplot as plt
import numpy as np
from time import time

(x_train, y_train), (x_test, y_test) = load_data()

mlp = LeakyReLUSoftmaxCCE((28,28), 10, [16,16,16])
lines = []

start = time()
history = mlp.elastic_net_train(x_train, y_train, 1e-3, 1000, 32, 2e-5, 1e-4, x_test, y_test)
end = time()

stop = (np.argmax(history['test_accuracy']), max(history['test_accuracy']))

plt.plot(history['train_loss'], label='Train')
plt.plot(history['test_loss'], label='Test')
plt.axvline(stop[0], linestyle='--', color='black', label='Best Test')
plt.title(r'$\eta{}=10^{-3}, \lambda{}_{1}=2 \times{} 10^{-5}, \lambda{}_{2}=10^{-4}$')
plt.ylabel('Cross Entropy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('figures/lnet.png')
plt.clf()

thresh60 = None
for i, a in enumerate(history['test_accuracy']):
    if a > 0.6:
        thresh60 = (i, a)
        break

thresh70 = None
for i, a in enumerate(history['test_accuracy']):
    if a > 0.7:
        thresh70 = (i, a)
        break

thresh80 = None
for i, a in enumerate(history['test_accuracy']):
    if a > 0.8:
        thresh80 = (i, a)
        break

lines.append('------Learning Rate 1e-3------')
lines.append(f'{round(end-start)} seconds elapsed ({round((end-start)/1000, 2)} seconds per epoch)')
if thresh60 is not None:
    lines.append(f'60% Threshold passed after epoch {thresh60[0]} ({round(100*thresh60[1], 3)}%)')
    plt.axvline(thresh60[0], linestyle='--', color='r', label='60% threshold')
else:
    lines.append('60% Threshold not passed')

if thresh70 is not None:
    lines.append(f'70% Threshold passed after epoch {thresh70[0]} ({round(100*thresh70[1], 3)}%)')
    plt.axvline(thresh70[0], linestyle='--', color='g', label='70% threshold')
else:
    lines.append('70% Threshold not passed')

if thresh80 is not None:
    lines.append(f'80% Threshold passed after epoch {thresh80[0]} ({round(100*thresh80[1], 3)}%)')
    plt.axvline(thresh80[0], linestyle='--', color='b', label='80% threshold')
else:
    lines.append('80% Threshold not passed')

lines.append(f'Maximal Test Accuracy is reached after epoch {stop[0]} ({round(100*stop[1], 3)}%)')

plt.plot(history['train_accuracy'], label='Train')
plt.plot(history['test_accuracy'], label='Test')
plt.axvline(stop[0], linestyle='--', color='black', label='Best Test')
plt.title(r'$\eta{}=10^{-3}, \lambda{}_{1}=2 \times{} 10^{-5}, \lambda{}_{2}=10^{-4}$')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('figures/anet.png')
plt.clf()

with open('output/elastic_net1.txt', 'w') as f:
    f.write('\n'.join(lines))
