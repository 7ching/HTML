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

# random hypothesis
def random_hypothesis():
    s = np.random.choice([-1, 1])
    theta = np.random.uniform(-1, 1)
    return s, theta

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
    s, theta = random_hypothesis()
    Ein = calculate_Ein(x, y, s, theta)
    Eout = calculate_Eout(s, theta, P)

    Ein_list.append(Ein)
    Eout_list.append(Eout)

plt.scatter(Ein_list, Eout_list, alpha=0.5)
plt.xlabel("Ein(g_RND)")
plt.ylabel("Eout(g_RND)")
plt.title("Ein(g_RND) vs Eout(g_RND)")
plt.show()

median_diff = np.median(np.array(Eout_list) - np.array(Ein_list))
print(f"Median of Eout(g_RND) - Ein(g_RND): {median_diff}")