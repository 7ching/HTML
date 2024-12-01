import libsvm
from libsvm.svmutil import *
import numpy as np
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
from tqdm import tqdm

# parameters
C_values = [0.1, 1, 10]
Q_values = [2, 3, 4]

# loading data
train_y, train_x = svm_read_problem('mnist.scale')

# filter class 3 & 7
filtered_train_x = []
filtered_train_y = []
for i in range(len(train_y)):
    if train_y[i] == 3:
        filtered_train_x.append(train_x[i])
        filtered_train_y.append(1)  # class 3 : 1
    elif train_y[i] == 7:
        filtered_train_x.append(train_x[i])
        filtered_train_y.append(-1)  # class 7 : -1

# training SVM models
minC = 0.1
minQ = 2
minSV = float('inf')
result = []

for C in C_values:
    for Q in Q_values:
        
        model = svm_train(filtered_train_y, filtered_train_x, f'-t 1 -c {C} -d {Q} -q -r 1 -g 1')
        sv = model.get_nr_sv()

        result.append((C, Q, sv))
        if sv < minSV:
            minSV = sv
            minC = C
            minQ = Q

print(f"{'C':<10}{'Q':<10}{'support vectors':<20}")
for C, Q, sv in result:
    print(f"{C:<10}{Q:<10}{sv:<20}")

print(f"\nBest (C, Q) combination: C = {minC}, Q = {minQ}, Minimum number of support vectors = {minSV}")