import numpy as np
import matplotlib.pyplot as plt

# generate datas
def generate_data(N,p):
    x = np.sort(np.random.uniform(-1, 1, N))
    y = np.sign(x)

    noise = np.random.rand(N) < p 
    y[noise] *= -1
    return x, y

# calculate Ein
def calculate_Ein(x, y, s, theta):
    pred = s * np.sign(x-theta)
    return np.mean(pred != y)

# decision stump algorithm
def decision_stump(x, y):
    N = len(x)

    best_theta = None
    best_s = None
    min_Ein = float('inf')

    for i in range(N-1):
        theta = (x[i] + x[i+1]) / 2     # middle point
        for s in [-1, 1]:
            Ein = calculate_Ein(x, y, s, theta)
            if Ein < min_Ein:
                min_Ein = Ein
                best_theta = theta
                best_s = s
    
    return min_Ein, best_s, best_theta

# calculate Eout
def calculate_Eout(s, theta, p):
    v = s * (0.5 - p)
    u = 0.5 - v
    return u + v * np.abs(theta)

# main code
Ein_list = []
Eout_list = []

TIMES = 2000    # 2000 times
N = 12          # data size: 12
P = 0.15        # p: 15%

for i in range(TIMES):   
    x, y = generate_data(N, P)    
    min_Ein, best_s, best_theta = decision_stump(x, y)
    Eout = calculate_Eout(best_s, best_theta, P)

    Ein_list.append(min_Ein)
    Eout_list.append(Eout)

plt.scatter(Ein_list, Eout_list, alpha=0.5)
plt.xlabel("Ein(g)")
plt.ylabel("Eout(g)")
plt.title("Ein(g) vs Eout(g)")
plt.show()

median_diff = np.median(np.array(Eout_list) - np.array(Ein_list))
print(f"Median of Eout(g) - Ein(g): {median_diff}")