import numpy as np
import matplotlib.pyplot as plt
from Req import LeakyReLUSoftmaxCCE, load_data
from time import time

np.random.seed(1112001)

(x_train, y_train), (x_test, y_test) = load_data()

lines = []

for trial in range(1, 4):
    lines.append('-------')
    lines.append(f'Trial {trial}')
    lines.append('-------')

    mlp1 = LeakyReLUSoftmaxCCE((28, 28), 10, [16, 16, 16])

    start = time()
    history1 = mlp1.nesterov_sgd_train(x_train, y_train, 1e-2, 1000, 32, x_test, y_test, 0.9)
    end = time()

    stop = (np.argmax(history1['test_accuracy']), max(history1['test_accuracy']))

    plt.plot(history1['train_loss'], label='Train')
    plt.plot(history1['test_loss'], label='Test')
    plt.axvline(stop[0], linestyle='--', color='black', label='Best Test')
    plt.title(r'$\eta{}=10^{-2}$')
    plt.ylabel('Cross Entropy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('figures/' + str(trial) + 'lnsgde-2.png')
    plt.clf()

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

    lines.append('------Learning Rate 1e-2------')
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

    plt.plot(history1['train_accuracy'], label='Train')
    plt.plot(history1['test_accuracy'], label='Test')
    plt.axvline(stop[0], linestyle='--', color='black', label='Best Test')
    plt.title(r'$\eta{}=10^{-2}$')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('figures/' + str(trial) + 'ansgde-2.png')
    plt.clf()

    mlp2 = LeakyReLUSoftmaxCCE((28, 28), 10, [16, 16, 16])

    start = time()
    history2 = mlp2.nesterov_sgd_train(x_train, y_train, 1e-3, 1000, 32, x_test, y_test, 0.9)
    end = time()

    stop = (np.argmax(history2['test_accuracy']), max(history2['test_accuracy']))

    plt.plot(history2['train_loss'], label='Train')
    plt.plot(history2['test_loss'], label='Test')
    plt.axvline(stop[0], linestyle='--', color='black', label='Best Test')
    plt.title(r'$\eta{}=10^{-3}$')
    plt.ylabel('Cross Entropy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('figures/' + str(trial) + 'lnsgde-3.png')
    plt.clf()

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

    lines.append('-----Learning Rate 1e-3------')
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

    plt.plot(history2['train_accuracy'], label='Train')
    plt.plot(history2['test_accuracy'], label='Test')
    plt.axvline(stop[0], linestyle='--', color='black', label='Best Test')
    plt.title(r'$\eta{}=10^{-3}$')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('figures/' + str(trial) + 'ansgde-3.png')
    plt.clf()

    mlp3 = LeakyReLUSoftmaxCCE((28, 28), 10, [16, 16, 16])

    start = time()
    history3 = mlp3.nesterov_sgd_train(x_train, y_train, 1e-5, 1000, 32, x_test, y_test, 0.9)
    end = time()

    stop = (np.argmax(history3['test_accuracy']), max(history3['test_accuracy']))

    plt.plot(history3['train_loss'], label='Train')
    plt.plot(history3['test_loss'], label='Test')
    plt.axvline(stop[0], linestyle='--', color='black', label='Best Test')
    plt.title(r'$\eta{}=10^{-5}$')
    plt.ylabel('Cross Entropy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('figures/' + str(trial) + 'lnsgde-5.png')
    plt.clf()

    thresh60 = None
    for i, a in enumerate(history3['test_accuracy']):
        if a > 0.6:
            thresh60 = (i, a)
            break

    thresh70 = None
    for i, a in enumerate(history3['test_accuracy']):
        if a > 0.7:
            thresh70 = (i, a)
            break

    thresh80 = None
    for i, a in enumerate(history3['test_accuracy']):
        if a > 0.8:
            thresh80 = (i, a)
            break

    lines.append('-----Learning Rate 1e-5------')
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

    plt.plot(history3['train_accuracy'], label='Train')
    plt.plot(history3['test_accuracy'], label='Test')
    plt.axvline(stop[0], linestyle='--', color='black', label='Best Test')
    plt.title(r'$\eta{}=10^{-5}$')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('figures/' + str(trial) + 'ansgde-5.png')
    plt.clf()

with open('output/nesterov.txt', 'w') as f:
    f.write('\n'.join(lines))
