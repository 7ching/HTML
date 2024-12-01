from libsvm.svmutil import *
import numpy as np
from tqdm import tqdm

# Parameters
C_values = [0.1, 1, 10]
gamma_values = [0.1, 1, 10]

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

# Calculate margin for each (C, gamma) pair
results = []
max_margin = 0
best_C, best_gamma = None, None

total_iterations = len(C_values) * len(gamma_values)
with tqdm(total=total_iterations, desc="Training SVM Models") as pbar:
    for C in C_values:
        for gamma in gamma_values:
            # Train SVM with RBF kernel
            model = svm_train(filtered_train_y, filtered_train_x, f'-t 2 -c {C} -g {gamma} -q')
            
            # Get margin (1 / ||w||)
            sv_coef = model.get_sv_coef()
            sv_indices = model.get_sv_indices()
            rho = model.rho[0]
            w_norm = np.sqrt(sum([coef[0] ** 2 for coef in sv_coef]))
            margin = 1 / w_norm
            
            # Store results
            results.append((C, gamma, margin))
            if margin > max_margin:
                max_margin = margin
                best_C, best_gamma = C, gamma
            
            # Update progress bar
            pbar.update(1)


print(f"{'C':<10}{'gamma':<10}{'1/||w|| (margin)':<20}")
for C, gamma, margin in results:
    print(f"{C:<10}{gamma:<10}{margin:<20.4f}")

print(f"\nBest (C, gamma) combination: C = {best_C}, gamma = {best_gamma}, with max margin = {max_margin:.4f}")