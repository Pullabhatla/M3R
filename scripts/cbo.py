from Req import load_data, carrillo_cbo
import numpy as np
from time import time
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = load_data()
lines = []
np.random.seed(1112001)

start = time()
X_bar, history = carrillo_cbo(x_train, y_train, 100, 100, 20000, 100, 1, 10, 1, np.sqrt(0.1), (28,28), 10, [16,16,16], x_test, y_test)
end = time()

stop = (np.argmax(history['test_accuracy']), max(history['test_accuracy']))

plt.plot(history['train_loss'], label='Train')
plt.plot(history['test_loss'], label='Test')
plt.axvline(stop[0], linestyle='--', color='black', label='Best Test')
plt.title(r'$N=100,M=100,m=100,\eta{}=1,\lambda{}=1,\sigma{}_{0}=\sqrt{0.1},\beta{}=10$')
plt.ylabel('Cross Entropy')
plt.xlabel('Iteration')
plt.legend()
plt.savefig('figures/lcbo.png')
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

lines.append(f'{round(end-start)} seconds elapsed ({round((end-start)/20000, 2)} seconds per iteration)')
if thresh60 is not None:
    lines.append(f'60% Threshold passed after iteration {thresh60[0]} ({round(100*thresh60[1], 3)}%)')
    plt.axvline(thresh60[0], linestyle='--', color='r', label='60% threshold')
else:
    lines.append('60% Threshold not passed')

if thresh70 is not None:
    lines.append(f'70% Threshold passed after iteration {thresh70[0]} ({round(100*thresh70[1], 3)}%)')
    plt.axvline(thresh70[0], linestyle='--', color='g', label='70% threshold')
else:
    lines.append('70% Threshold not passed')

if thresh80 is not None:
    lines.append(f'80% Threshold passed after iteration {thresh80[0]} ({round(100*thresh80[1], 3)}%)')
    plt.axvline(thresh80[0], linestyle='--', color='b', label='80% threshold')
else:
    lines.append('80% Threshold not passed')

lines.append(f'Maximal Test Accuracy is reached after iteration {stop[0]} ({round(100*stop[1], 3)}%)')

plt.plot(history['train_accuracy'], label='Train')
plt.plot(history['test_accuracy'], label='Test')
plt.axvline(stop[0], linestyle='--', color='black', label='Best Test')
plt.title(r'$N=100,M=100,m=100,\eta{}=1,\lambda{}=1,\sigma{}_{0}=\sqrt{0.1},\beta{}=10$')
plt.ylabel('Accuracy')
plt.xlabel('Iteration')
plt.legend()
plt.savefig('figures/acbo.png')
plt.clf()

with open('output/cbo.txt', 'w') as f:
    f.write('\n'.join(lines))
