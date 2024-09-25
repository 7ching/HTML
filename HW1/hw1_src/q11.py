import numpy as np
import random
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt

# parameters
experiment = 1000
sample = 200
#max_update = 5000
update_counts = []      # save every experiment update counts
all_w_norm = []         # save every experiment w norms
min_update = np.inf     # save minimum updates


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
    w_norm = [] 

    while True:
        i = np.random.choice(sample)   # with replacement
        y_pred = -1 if np.sign(np.dot(w, x[i])) < 0 else 1
        if y_pred != y[i]:    # sign different
            w = w + y[i] * x[i]
            updates += 1
            no_error = 0
            w_norm.append(np.linalg.norm(w))
        else:
            no_error += 1
        if no_error >= max_5N:
            break
    return w, updates, w_norm

for i in range(experiment):
    i, updates, w_norm = pla(x,y)
    update_counts.append(updates)
    all_w_norm.append(w_norm)
    min_update = min(min_update, updates)

all_w_norm = [norms[:min_update] for norms in all_w_norm]

'''
# plotting updates
plt.hist(update_counts, bins=30, edgecolor='black')
plt.xlabel('Number of Updates')
plt.ylabel('Frequency')
plt.title('PLA Updates Distribution with 5N Stopping Criterion')
plt.show()
'''
# plotting weight norms
plt.figure(figsize=(10, 6))
for norms in all_w_norm:
    plt.plot(range(1, min_update + 1), norms, alpha=0.1, color='blue')  

plt.xlabel('t (updates)')
plt.ylabel('||w_t|| (norm of the weight vector)')
plt.title('Weight Norm ||w_t|| over Time in 1000 PLA Experiments')
plt.show()