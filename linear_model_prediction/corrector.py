import pandas as pd
import numpy as np
import datetime
import json
from numpy import loadtxt
from numpy import savetxt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
x_0 = loadtxt('x_0.csv', delimiter=',')
y_0 = loadtxt('y_0.csv', delimiter=',')
# ubaciti skaliranje

# form learning and validation dataset
n = x_0.shape[0]
perm = np.random.permutation(n)
trainratio = 0.8
testratio = 0.2
trainind = perm[:int(np.ceil(n * trainratio))]
validind = perm[int(np.ceil(n * trainratio)):]

# create model
knn = KNeighborsRegressor()
model = MultiOutputRegressor(knn)
model.fit(x_0[trainind], y_0[trainind])

# create predictions on validation dataset and calculate MAE
y_pred = model.predict(x_0[validind])
valid_error = mean_absolute_error(y_0[validind], y_pred)

# print single example
plt.plot(y_0[validind][0], label='Calculated error')
plt.plot(y_pred[0], label='Predicted error')
plt.grid()
plt.legend()
plt.show()
