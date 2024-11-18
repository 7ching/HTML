from liblinear.liblinearutil import *
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm

# parameters
EXPERIMENTS = 1126  
lambdas = [-2, -1, 0, 1, 2, 3]
K_FOLDS = 3

# loading data
train_x, train_y = load_svmlight_file('mnist.scale')
test_x, test_y = load_svmlight_file('mnist.scale.t')

train_x = train_x.astype(np.float64)
test_x = test_x.astype(np.float64)

# filter class 2 & 6
def filter_data(x, y, class1=2, class2=6):
    mask = (y == class1) | (y == class2)
    x_filtered = x[mask]
    y_filtered = np.where(y[mask] == class1, 1, -1)  # class 2 : 1，class 6 : -1
    return x_filtered, y_filtered

train_x, train_y = filter_data(train_x, train_y)
test_x, test_y = filter_data(test_x, test_y)

Eout_histogram = []

for i in tqdm(range(EXPERIMENTS), desc="Experiments"):
    # 3-fold cross validation
    kf = KFold(n_splits=K_FOLDS, shuffle=True)
    best_lambda = None
    best_Ecv = float('inf')

    for log_lambda in lambdas:
        current_lambda = 10 ** log_lambda
        c = 1 / (2 * current_lambda)
        fold_errors = []

        # split current fold (train & valid set)
        for train_idx, val_idx in kf.split(train_x):
            sub_train_x, val_x = train_x[train_idx], train_x[val_idx]
            sub_train_y, val_y = train_y[train_idx], train_y[val_idx]

            # training
            model = train(sub_train_y, sub_train_x, f'-s 6 -c {c} -B 1 -q')
            # validation
            _, Eval_acc, _ = predict(val_y, val_x, model, '-q')
            eval_error = 1 - (Eval_acc[0] / 100)  
            fold_errors.append(eval_error)

        # mean Ecv of current lambda
        avg_Ecv = np.mean(fold_errors)
        if avg_Ecv < best_Ecv or (avg_Ecv == best_Ecv and log_lambda > np.log10(best_lambda)):
            best_lambda = current_lambda
            best_Ecv = avg_Ecv

    print(f"Experiment {i+1}: ")
    print("best lambda = ",best_lambda)
    print("best Ecv = ",best_Ecv)
    # training on the whole training set
    c_best = 1 / (2 * best_lambda)
    model = train(train_y, train_x, f'-s 6 -c {c_best} -B 1 -q')

    # testing Eout
    _, Eout_acc, _ = predict(test_y, test_x, model, '-q')
    eout_error = 1 - (Eout_acc[0] / 100)
    Eout_histogram.append(eout_error)

    print("Eout = ",eout_error)

plt.figure(figsize=(8, 5))
plt.hist(Eout_histogram, bins=50, edgecolor='black')
plt.title(f'Eout Distribution over {EXPERIMENTS} Experiments (3-Fold CV)')
plt.xlabel('Eout')
plt.ylabel('Frequency')
plt.show()