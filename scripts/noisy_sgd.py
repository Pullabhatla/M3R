import numpy as np
import matplotlib.pyplot as plt
from Req import LeakyReLUSoftmaxCCE, load_data
from time import time

np.random.seed(1112001)

(x_train, y_train), (x_test, y_test) = load_data()

lines = []
noise = []
stoch = []

for trial in range(1, 21):
    lines.append('-------')
    lines.append(f'Trial {trial}')
    lines.append('-------')

    mlp1 = LeakyReLUSoftmaxCCE((28, 28), 10, [50 for _ in range(20)])

    start = time()
    history1 = mlp1.noisy_sgd_train(x_train, y_train, 1e-4, 200, 32, 0.55, x_test, y_test)
    end = time()

    stop = (np.argmax(history1['test_accuracy']), max(history1['test_accuracy']))
    noise.append(stop[1])

    thresh60 = None
    for i, a in enumerate(history1['test_accuracy']):
        if a > 0.6:
            thresh60 = (i, a)
            break

    thresh70 = None
    for i, a in enumerate(history1['test_accuracy']):
        if a > 0.7:
            thresh70 = (i, a)
            break

    thresh80 = None
    for i, a in enumerate(history1['test_accuracy']):
        if a > 0.8:
            thresh80 = (i, a)
            break

    lines.append('------NSGD------')
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

    mlp2 = LeakyReLUSoftmaxCCE((28, 28), 10, [50 for _ in range(20)])

    start = time()
    history2 = mlp2.sgd_train(x_train, y_train, 1e-4, 200, 32, x_test, y_test)
    end = time()

    stop = (np.argmax(history2['test_accuracy']), max(history2['test_accuracy']))
    stoch.append(stop[1])
    thresh60 = None
    for i, a in enumerate(history2['test_accuracy']):
        if a > 0.6:
            thresh60 = (i, a)
            break

    thresh70 = None
    for i, a in enumerate(history2['test_accuracy']):
        if a > 0.7:
            thresh70 = (i, a)
            break

    thresh80 = None
    for i, a in enumerate(history2['test_accuracy']):
        if a > 0.8:
            thresh80 = (i, a)
            break

    lines.append('-----SGD------')
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

noise_avg = np.mean(noise)
stoch_avg = np.mean(stoch)
noise_max = np.max(noise)
stoch_max = np.max(stoch)
noise_se = np.std(noise)/np.sqrt(20)
stoch_se = np.std(stoch)/np.sqrt(20)

lines.append('\nSUMMARY STATISTICS')
lines.append(f'No noise: Maximum {stoch_max} with average {stoch_avg} (+\- {stoch_se})')
lines.append(f'With noise: Maximum {noise_max} with average {noise_avg} (+\- {noise_se})')
my_dict = {'No Noise':stoch, 'Noise':noise}
fig, ax = plt.subplots()
ax.boxplot(my_dict.values())
ax.set_xticklabels(my_dict.keys())
plt.ylabel('Test Accuracy')
plt.title('Effect of Noise')
plt.savefig('figures/noisy_sgd.png')

with open('output/noisy_sgd.txt', 'w') as f:
    f.write('\n'.join(lines))
