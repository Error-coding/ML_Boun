import numpy as np
import matplotlib.pyplot as pl

data = np.load('./pla_data/data_small.npy')
targets = np.load('./pla_data/label_small.npy')
scounter = 19
ccounter = 0
index = scounter
max = len(data)
counter = 0
w = np.array([0.0,0.0,0.0])

# perceptron algorithm
while counter < len(targets):
    if (targets[index] * np.dot(w, data[index]) <= 0):
        ccounter += 1
        w += data[index] * targets[index]
        counter = 0
    else:
        counter += 1
    index = (index + 1) % len(targets)


# number of iterations
print(ccounter)
# final w
print(w)
