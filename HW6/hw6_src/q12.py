from libsvm.svmutil import *
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# Parameters
C_values = 1
gamma_values = [0.01, 0.1, 1, 10, 100]
REAPEAT = 128
VALID_SIZE = 200

# Load data
train_y, train_x = svm_read_problem('mnist.scale')

# Filter classes 3 and 7
filtered_train_x = []
filtered_train_y = []
for i in range(len(train_y)):
    if train_y[i] == 3:
        filtered_train_x.append(train_x[i])
        filtered_train_y.append(1)  # class 3 : +1
    elif train_y[i] == 7:
        filtered_train_x.append(train_x[i])
        filtered_train_y.append(-1)  # class 7 : -1

gamma_selection_counts = {gamma: 0 for gamma in gamma_values}

for times in tqdm(range(REAPEAT), desc="Validation Trials"):
    indices = list(range(len(filtered_train_y)))
    random.shuffle(indices)
    val_indices = indices[:VALID_SIZE]
    train_indices = indices[VALID_SIZE:]

    val_x = [filtered_train_x[i] for i in val_indices]
    val_y = [filtered_train_y[i] for i in val_indices]
    train_x = [filtered_train_x[i] for i in train_indices]
    train_y = [filtered_train_y[i] for i in train_indices]

    # evaluation
    best_gamma = None
    min_error = float('inf')

    for gamma in gamma_values:
        model = svm_train(train_y, train_x, f'-t 2 -c {C_values} -g {gamma} -q')
        p_labels, p_acc, p_vals = svm_predict(val_y, val_x, model, options="-q")

        error = sum([1 if p_labels[i] != val_y[i] else 0 for i in range(len(val_y))])   # 0/1 error
        
        if error < min_error or (error == min_error and (best_gamma is None or gamma < best_gamma)):
            min_error = error
            best_gamma = gamma

    gamma_selection_counts[best_gamma] += 1

# Plot results
plt.bar(gamma_selection_counts.keys(), gamma_selection_counts.values())
plt.xlabel("Gamma")
plt.ylabel("Selection Frequency")
plt.title("Gamma Selection Frequency Over 128 Trials")
plt.xscale('log')  # Use log scale for gamma
plt.show()

# Print selection counts
print("Gamma Selection Counts:")
for gamma, count in gamma_selection_counts.items():
    print(f"Gamma: {gamma}, Count: {count}")