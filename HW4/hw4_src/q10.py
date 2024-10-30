import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# parameters
N = 64
EXPERIMENTS = 1126
ITERATION = 100000
lr = 0.01


# loading data
x, y = load_svmlight_file('cpusmall_scale')  
x = x.toarray()

x = np.hstack((np.ones((x.shape[0], 1)), x))    # adding x0 = 1

Ein = []
Eout = []
Ein_sgd = []
Eout_sgd = []

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

    # SGD 
    w_sgd = np.zeros(x_train.shape[1])
    Ein_sgd_200 = []
    Eout_sgd_200 = []

    for t in range(ITERATION):
        idx = np.random.randint(0, N)  # Randomly select index
        gradient = -2 * (y_train[idx] - x_train[idx] @ w_sgd) * x_train[idx]
        w_sgd -= lr * gradient

        if (t + 1) % 200 == 0:
            Ein_sgd_200.append(mean_squared_error(y_train, x_train @ w_sgd))
            Eout_sgd_200.append(mean_squared_error(y_test, x_test @ w_sgd))

    Ein_sgd.append(Ein_sgd_200)
    Eout_sgd.append(Eout_sgd_200)

# Calculate average Ein and Eout for linear regression
avg_Ein = np.mean(Ein)
avg_Eout = np.mean(Eout)

# Calculate average Ein and Eout for SGD 
avg_Ein_sgd = np.mean(Ein_sgd, axis=0)
avg_Eout_sgd = np.mean(Eout_sgd, axis=0)


plt.plot(range(200, ITERATION + 1, 200), avg_Ein_sgd, label="Average Ein (SGD)")
plt.plot(range(200, ITERATION + 1, 200), avg_Eout_sgd, label="Average Eout (SGD)")
plt.axhline(y=avg_Ein, color='r', linestyle='--', label="Avg Ein (Linear)")
plt.axhline(y=avg_Eout, color='b', linestyle='--', label="Avg Eout (Linear)")
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Average Ein and Eout over SGD Iterations')
plt.legend()
plt.show()


