import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# parameters
N = 64
EXPERIMENTS = 1126
ITERATION = 100000
lr = 0.01
Q = 3

# loading data
x, y = load_svmlight_file('cpusmall_scale')  
x = x.toarray()

# Polynomial transformation
def polynomial(x, Q):
    x_poly = [np.ones(x.shape[0])]  # start with 1
    for q in range(1, Q + 1):
        for j in range(x.shape[1]):
            x_poly.extend([x[:, j] ** q ])
    return np.column_stack(x_poly)

x_poly = polynomial(x, Q)

Ein_gain = []

for i in range(EXPERIMENTS):

    # randomly take N samples for training
    train_indices = np.random.choice(len(x), N, replace=False)
    x_train, y_train = x[train_indices], y[train_indices]
    x_poly_train = x_poly[train_indices]

    # use the rest for testing
    test_indices = np.setdiff1d(np.arange(len(x)), train_indices)
    x_test, y_test = x[test_indices], y[test_indices]
    x_poly_test = x_poly[test_indices]

    # without transformation
    w_lin = np.linalg.pinv(x_train.T @ x_train) @ x_train.T @ y_train
    Ein_lin = mean_squared_error(y_train, x_train @ w_lin)

    # with transformation
    w_poly = np.linalg.pinv(x_poly_train.T @ x_poly_train) @ x_poly_train.T @ y_train
    Ein_poly = mean_squared_error(y_train, x_poly_train @ w_poly)

    Ein_gain.append(Ein_lin - Ein_poly)

# Plot histogram of Ein improvements
plt.hist(Ein_gain, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Ein Gain (Ein_lin - Ein_poly)')
plt.ylabel('Frequency')
plt.title('Histogram of Ein Gain for Polynomial Transform')
plt.show()

avg_gain = np.mean(Ein_gain)
print(f"Average Ein Gain: {avg_gain}")


