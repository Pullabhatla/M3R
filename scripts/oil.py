import matplotlib.pyplot as plt
import numpy as np

a = 0.5
b = 0.5
r = 0.4

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

theta =  np.linspace(0, 2*np.pi, 100)
x = r*np.cos(theta) + a
y = r*np.sin(theta) + b
plt.fill_between(x, y, color='black', edgecolor='none', alpha=0.2)
plt.scatter(redsx, redsy, color='none', edgecolors='red')
plt.scatter(bluesx, bluesy, color='blue', marker='x')
plt.plot(x, y, 'k--')
plt.axis('square')
plt.xticks([0,1])
plt.yticks([0, 1])
plt.xlim([0,1])
plt.ylim([0,1])
plt.savefig('figures/oil_wells.png')

X_aug = np.vstack([X_train, np.ones(X_train.shape[1])])
X_aug
a = y_train@np.linalg.pinv(X_aug)
a

x = np.linspace(0,1,100)
y = (-a[0]*x-a[2])/a[1]
plt.scatter(redsx, redsy, color='none', edgecolors='red')
plt.scatter(bluesx, bluesy, color='blue', marker='x')
plt.fill_between(x, y, color='black', edgecolor='none', alpha=0.2)
plt.plot(x, y, 'k--')
plt.axis('square')
plt.xticks([0,1])
plt.yticks([0, 1])
plt.xlim([0,1])
plt.ylim([0,1])
plt.savefig('figures/linear_oil.png')
