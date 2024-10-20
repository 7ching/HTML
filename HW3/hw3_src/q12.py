import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# parameters
N = list(range(25, 2025, 25))
EXPERIMENTS = 16

# loading data
x, y = load_svmlight_file('cpusmall_scale')  
x = x.toarray()
x = x[:, :2]

x = np.hstack((np.ones((x.shape[0], 1)), x))    # adding x0 = 1

Ein_means = []
Eout_means = []

for n in N:
    Ein = []
    Eout = []

    for i in range(EXPERIMENTS):

        # randomly take N samples for training
        train_indices = np.random.choice(len(x), n, replace=False)
        x_train, y_train = x[train_indices], y[train_indices]

        # use the rest for testing
        test_indices = np.setdiff1d(np.arange(len(x)), train_indices)
        x_test, y_test = x[test_indices], y[test_indices]

        w_lin = np.linalg.pinv(x_train.T @ x_train) @ x_train.T @ y_train

        y_train_pred = x_train @ w_lin
        y_test_pred = x_test @ w_lin

        Ein.append(mean_squared_error(y_train, y_train_pred))
        Eout.append(mean_squared_error(y_test, y_test_pred))

    Ein_means.append(np.mean(Ein))
    Eout_means.append(np.mean(Eout))

plt.plot(N, Ein_means, label='Average Ein(N)', marker='o')
plt.plot(N, Eout_means, label='Average Eout(N)', marker='o')
plt.xlabel('N (Number of Training Samples)')
plt.ylabel('Error')
plt.title('Learning Curves: Average Ein(N) and Average Eout(N)')
plt.legend()
plt.grid(True)
plt.show()


