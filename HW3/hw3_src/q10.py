import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# parameters
N = 32
EXPERIMENTS = 1126

# loading data
x, y = load_svmlight_file('cpusmall_scale')  
x = x.toarray()

x = np.hstack((np.ones((x.shape[0], 1)), x))    # adding x0 = 1

Ein = []
Eout = []

for i in range(EXPERIMENTS):

    # randomly take N samples for training
    train_indices = np.random.choice(len(x), N, replace=False)
    x_train, y_train = x[train_indices], y[train_indices]

    # use the rest for testing
    test_indices = np.setdiff1d(np.arange(len(x)), train_indices)
    x_test, y_test = x[test_indices], y[test_indices]

    w_lin = np.linalg.pinv(x_train.T @ x_train) @ x_train.T @ y_train

    y_train_pred = x_train @ w_lin
    y_test_pred = x_test @ w_lin

    Ein.append(mean_squared_error(y_train, y_train_pred))
    Eout.append(mean_squared_error(y_test, y_test_pred))

plt.scatter(Ein, Eout, alpha=0.5)
plt.xlabel('Ein (Training Error)')
plt.ylabel('Eout (Test Error)')
plt.title('Scatter Plot of Ein vs. Eout for N = 32')
plt.show()


