from ucimlrepo import fetch_ucirepo 
import math
import numpy as np
import random
  
#regularization param
reg = 10

def sigmoid(x):
    return (1/(1 + math.exp(-x)))

def stochastic_desc_step(w, x, t):
    return (t * x * (1/(1 + math.exp(t * np.dot(x, w)))))

def stochastic_desc_step_reg(w, x, t):
    return (t * x * (1/(1 + math.exp(t * np.dot(x, w))))) - (1 / reg) * w

# fetch dataset 
rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 
  
# data (as pandas dataframes) 
X = rice_cammeo_and_osmancik.data.features 
y = rice_cammeo_and_osmancik.data.targets 

# variable information 
cols = list(X.columns.values)
colsTarget = list(y.columns.values)
length = len(cols)

total = len(X[cols[0]])
subsize = int(total / 5)

sample_means = []
sample_vars= []

    


# identify mean and variance of samples
for i in range(len(cols)):
    name = cols[i]
    sum = 0
    for e in range(total):
            sum += X[name][e]

    sample_means.append(sum / total)


for i in range(len(cols)):
    name = cols[i]
    sum = 0
    for e in range(total):
        sum += (X[name][e] - sample_means[i]) * (X[name][e] - sample_means[i])

    sample_vars.append(sum / total)

# Z-score normalization: transformed data has 0 mean and 1 variance in all columns
for i in range(len(cols)):
    X[cols[i]] = X[cols[i]].apply(lambda x: (x - sample_means[i]) / math.sqrt(sample_vars[i]))

    targets = np.zeros((total,))
data = X.to_numpy()


#Transform target data 
for i in range(total):
    if y[colsTarget[0]][i] == 'Cammeo':
        targets[i] = -1
    else:
        targets[i] = 1






trainaccsum = 0
testaccsum = 0
trainaccsumGD = 0
testaccsumGD = 0
for d in range(5):
    

    indecestrain = []
    indecestest = []
    curr = 0
    for i in range(4):
        if not i == d:
            for a in np.arange(curr, curr + subsize):
                indecestrain.append(a)
        else:
            for a in np.arange(curr, curr + subsize):
                indecestest.append(a)
        curr += subsize
    
    if d == 4:
        for a in np.arange(curr, total):
                indecestest.append(a)
    else:
        for a in np.arange(curr, total):
                indecestrain.append(a)


    # SGD
    w = np.zeros((length, ))
    epochs = 5
    decay = 0
    for e in range(epochs):
        np.random.shuffle(indecestrain)
        for a in indecestrain:
            learningrate = 1 / (1 + 0.005 * decay)
            w += learningrate * stochastic_desc_step(w, data[a, :], targets[a])
            decay += 1

    errorsum = 0
    for i in indecestrain:
        errorsum +=  abs(targets[i] - (2 * (-0.5 + sigmoid(np.dot(w, data[i, :])))))

    #print("SGD train: " + str(errorsum/len(indecestrain)))
    trainaccsum += errorsum/len(indecestrain)

    errorsum = 0
    for i in indecestest:
        errorsum +=  abs(targets[i] - (2 * (-0.5 + sigmoid(np.dot(w, data[i, :])))))

    testaccsum += errorsum/len(indecestest)

    # GD
    w = np.zeros((length, ))
    steps = 500
    for e in range(steps):
        delta = np.zeros((length,))
        for a in indecestrain:
            delta += stochastic_desc_step_reg(w, data[a, :], targets[a])
        delta /= len(indecestrain)
        w += delta

    errorsum = 0
    for i in indecestrain:
        errorsum +=  abs(targets[i] - (2 * (-0.5 + sigmoid(np.dot(w, data[i, :])))))

    trainaccsumGD += errorsum/len(indecestrain)

    errorsum = 0
    for i in indecestest:
        errorsum +=  abs(targets[i] - (2 * (-0.5 + sigmoid(np.dot(w, data[i, :])))))

    testaccsumGD += errorsum/len(indecestest)


print("SGD train avg: " + str(trainaccsum / 5))
print("SGD test avg: " + str(testaccsum / 5))

print("GD train avg: " + str(trainaccsumGD / 5))
print("GD test avg: " + str(testaccsumGD / 5))
