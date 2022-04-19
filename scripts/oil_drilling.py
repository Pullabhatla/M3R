import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

a = 0.3
b = 0.3
r = 0.5

n = 50

redsx = []
redsy = []
bluesx = []
bluesy = []

X_train = np.zeros((2, n))
y_train = np.zeros(n)

np.random.seed(1112001)
for i in range(n):
    point = np.random.uniform(0, 1, 2)
    if np.linalg.norm(point-np.array([a, b])) < r:
        redsx.append(point[0])
        redsy.append(point[1])
        X_train[0, i] = point[0]
        X_train[1, i] = point[1]
        y_train[i] = 1

    else:
        bluesx.append(point[0])
        bluesy.append(point[1])
        X_train[0, i] = point[0]
        X_train[1, i] = point[1]
        y_train[i] = -1

theta = np.linspace(0, 2*np.pi, 100)
x = r*np.cos(theta) + a
y = r*np.sin(theta) + b
plt.fill_between(x, y, color='black', edgecolor='none', alpha=0.2)
plt.scatter(redsx, redsy, color='none', edgecolors='red')
plt.scatter(bluesx, bluesy, color='blue', marker='x')
plt.plot(x, y, 'k--')
plt.axis('square')
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig('figures/oil_wells.png')
plt.clf()

X_aug = np.vstack([X_train, np.ones(X_train.shape[1])])
X_aug
a = y_train@np.linalg.pinv(X_aug)
a

x = np.linspace(0, 1, 100)
y = (-a[0]*x-a[2])/a[1]
plt.scatter(redsx, redsy, color='none', edgecolors='red')
plt.scatter(bluesx, bluesy, color='blue', marker='x')
plt.fill_between(x, y, color='black', edgecolor='none', alpha=0.2)
plt.plot(x, y, 'k--')
plt.axis('square')
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig('figures/linear_oil.png')
plt.clf()


dummy = []
for y in y_train:
    if y == 1:
        dummy.append([1, 0])
    else:
        dummy.append([0, 1])

y_train = dummy


def sigmoid(X):
    return 1/(1+np.exp(X))


def predict(theta):
    W1 = np.array(theta[:4]).reshape(2, 2)
    b1 = np.array(theta[4:6])
    W2 = np.array(theta[6:12]).reshape(2, 3)
    b2 = np.array(theta[12:15])
    W3 = np.array(theta[15:21]).reshape(3, 2)
    b3 = np.array(theta[21:23])
    W4 = np.array(theta[23:27]).reshape(2, 2)
    b4 = np.array(theta[27:29])

    A = X_train.T

    A = sigmoid(A@W1 + b1)
    A = sigmoid(A@W2 + b2)
    A = sigmoid(A@W3 + b3)
    A = sigmoid(A@W4 + b4)

    return A


def res(theta):
    return (predict(theta) - y_train).flatten()


theta0 = list(np.random.randn(29))
theta_trained = least_squares(res, theta0).x

n = 1000
x_range = np.linspace(0, 1, n)
y_range = np.linspace(0, 1, n)


def landscape(theta):
    W1 = np.array(theta[:4]).reshape(2, 2)
    b1 = np.array(theta[4:6])
    W2 = np.array(theta[6:12]).reshape(2, 3)
    b2 = np.array(theta[12:15])
    W3 = np.array(theta[15:21]).reshape(3, 2)
    b3 = np.array(theta[21:23])
    W4 = np.array(theta[23:27]).reshape(2, 2)
    b4 = np.array(theta[27:29])

    X, Y = np.meshgrid(x_range, y_range)

    A = np.array([[x, y] for x, y in zip(X.flatten(), Y.flatten())])

    A = sigmoid(A@W1 + b1)
    A = sigmoid(A@W2 + b2)
    A = sigmoid(A@W3 + b3)
    A = sigmoid(A@W4 + b4)

    return np.sign(A[:, 0] - A[:, 1]).reshape(n, n)


plt.contourf(x_range, y_range, -landscape(theta_trained), cmap='gray', alpha=0.2)
plt.contour(x_range, y_range, landscape(theta_trained), linestyles='dashed', colors='black')
plt.scatter(redsx, redsy, color='none', edgecolors='red')
plt.scatter(bluesx, bluesy, color='blue', marker='x')
plt.axis('square')
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig('figures/sigmoid_oil.png')
plt.clf()
