import math
from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd

# returns the value of the normal distribution at point x
def normal_dist(mean, variance, x):
    return 1/math.sqrt(2 * math.pi * variance) * math.exp(-(1/(2 * variance)) * (x - mean) * (x - mean))



# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 
  
  
# variable information 
cols = list(X.columns.values)
colsTarget = list(y.columns.values)
length = len(cols)

total = len(X[cols[0]])
maxTrain = int(total * 0.8)
minTest = maxTrain + 1


numM = 0
numB = 0
for i in range(maxTrain + 1):
    if y[colsTarget[0]][i] == 'M':
        numM += 1
    else:
        numB += 1

sample_meansM = []
sample_varsM = []
sample_meansB = []
sample_varsB = []

#compute means for malignant and benign samples
for i in range(len(cols)):
    name = cols[i]
    sumM = 0
    sumB = 0
    for e in range(maxTrain + 1):
        if y[colsTarget[0]][e] == 'M':
            sumM += X[name][e]
        else:
            sumB += X[name][e]
    sample_meansM.append(sumM / numM)
    sample_meansB.append(sumB / numB)

#compute variance for malignant and benign samples
for i in range(len(cols)):
    name = cols[i]
    sumM = 0
    sumB = 0
    for e in range(maxTrain + 1):
        if y[colsTarget[0]][e] == 'M':
            sumM += (X[name][e] - sample_meansM[i]) * (X[name][e] - sample_meansM[i])
        else:
            sumB += (X[name][e] - sample_meansB[i]) * (X[name][e] - sample_meansB[i])
    sample_varsM.append(sumM / numM)
    sample_varsB.append(sumB / numB)

right = 0
res = 'D'
#computes training accuracy
for i in range(maxTrain + 1):
    resM = numM/total
    resB = numB/total
    for j in range(len(cols)):
        resM *= normal_dist(sample_meansM[j], sample_varsM[j], float(X[cols[j]][i]))
        resB *= normal_dist(sample_meansB[j], sample_varsB[j], float(X[cols[j]][i]))
    if resM > resB:
        res = 'M'
    else:
        res = 'B'
    if res == y[colsTarget[0]][i]:
        right += 1

print("Train: " + str(right/(maxTrain + 1)))

right = 0
res = 'D'
#computes test accuracy
for i in range(maxTrain + 2, total):
    resM = numM/total
    resB = numB/total
    for j in range(len(cols)):
        resM *= normal_dist(sample_meansM[j], sample_varsM[j], float(X[cols[j]][i]))
        resB *= normal_dist(sample_meansB[j], sample_varsB[j], float(X[cols[j]][i]))
    if resM > resB:
        res = 'M'
    else:
        res = 'B'
    if res == y[colsTarget[0]][i]:
        right += 1

print("Test: " + str(right/(total - maxTrain)))
print(len(cols))
