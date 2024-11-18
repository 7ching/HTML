from liblinear.liblinearutil import *
import numpy as np
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
from tqdm import tqdm

# parameters
EXPERIMENTS = 1126
lambdas = [-2, -1, 0, 1, 2, 3]

# loading data
train_x, train_y = load_svmlight_file('mnist.scale')
test_x, test_y= load_svmlight_file('mnist.scale.t')

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

# finding best lambda
Ein_histogram = []
Eout_histogram = []
non_zero_histogram = []

for i in tqdm(range(EXPERIMENTS), desc="Experiments"):
    best_lambda = None
    best_Ein = float('inf')

    for log_lambda in lambdas:
        current_lambda = 10 ** log_lambda
        c = 1 / (2 * current_lambda)

        model = train(train_y, train_x, f'-s 6 -c {c} -B 1 -q') 

        _, Ein_acc, _ = predict(train_y, train_x, model, '-q')
        ein_error = 1 - (Ein_acc[0] / 100)  # 0/1 error

        if ein_error < best_Ein or (ein_error == best_Ein and log_lambda > np.log10(best_lambda)):
            best_lambda = current_lambda
            best_Ein = ein_error
        
    print(f"Experiment {i+1}: best lambda = ",best_lambda)
    print("best Ein = ",best_Ein)

    c_best = 1 / (2 * best_lambda)
    model = train(train_y, train_x, f'-s 6 -c {c_best} -B 1 -q')

    _, Eout_acc, _ = predict(test_y, test_x, model, '-q')
    eout_error = 1 - (Eout_acc[0] / 100)  
    Eout_histogram.append(eout_error)

    w = np.array(model.get_decfun()[0])
    non_zero_count = np.count_nonzero(w)
    non_zero_histogram.append(non_zero_count)

    print("Eout = ",eout_error)
    print("non_zero_count = ",non_zero_count)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(Eout_histogram, bins=50, edgecolor='black')
plt.title(f'Eout Distribution')
plt.xlabel('Eout')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(non_zero_histogram, bins=30, edgecolor='black')
plt.title('Number of Non-Zero Components')
plt.xlabel('Non-Zero Count')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()    