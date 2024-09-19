import numpy as np
import random
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt

# parameters ##
experiment = 1000
sample = 200
#max_update = 5000
update_counts = []      # save every experiment update counts

# loading datas
def load_data(path, sample):
    x, y = load_svmlight_file(path)     # loading LIBSVM form
    x = x[:sample].toarray()        # to numpy array
    y = y[:sample]
    return x, y

x, y = load_data('rcv1_train.binary', sample)

# adding bias x0 = 1 to every xn
x =  np.hstack([np.ones((x.shape[0], 1)), x])

# PLA
def pla(x,y):
    w = np.zeros(x[0].shape)       # initialize w = 0
    updates = 0
    sample = x.shape[0]
    max_5N = 5 * sample
    no_error = 0

    while no_error < max_5N:
        i = np.random.choice(sample)   # with replacement
        y_pred = np.sign(np.dot(w, x[i]))
        while y_pred != y[i]:    # sign different
            w = w + y[i] * x[i]
            updates += 1
            y_pred = np.sign(np.dot(w, x[i]))
            no_error = 0
        no_error += 1
    return w, updates

for i in range(experiment):
    i, updates = pla(x,y)
    update_counts.append(updates)


# plotting updates
plt.hist(update_counts, bins=30, edgecolor='black')
plt.xlabel('Number of Updates')
plt.ylabel('Frequency')
plt.title('PLA Updates Distribution with Repeated Correction')
plt.show()

# finding median number of updates
median = np.median(update_counts)
print(f"Median number of updates: {median}")
