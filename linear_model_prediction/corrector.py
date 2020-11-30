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


def calculate_correction(filename_x, filename_y):
    x_0 = loadtxt(filename_x, delimiter=',')
    y_0 = loadtxt(filename_y, delimiter=',')
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
    return validind


def calculate_correction_valid(filename_x, filename_y,validind):
    x_0 = loadtxt(filename_x, delimiter=',')
    y_0 = loadtxt(filename_y, delimiter=',')
    # ubaciti skaliranje

    # form learning and validation dataset
    n = x_0.shape[0]
    perm = np.random.permutation(n)
    trainratio = 0.8
    testratio = 0.2
    trainind = perm[:int(np.ceil(n * trainratio))]
    #validind = perm[int(np.ceil(n * trainratio)):]

    # create model
    knn = KNeighborsRegressor()
    model = MultiOutputRegressor(knn)
    model.fit(x_0[trainind], y_0[trainind])

    # create predictions on validation dataset and calculate MAE
    y_pred = model.predict(x_0[validind])
    valid_error = mean_absolute_error(y_0[validind], y_pred)
    print(validind)
    # print single example
    plt.plot(y_0[validind][0], label='Calculated error')
    plt.plot(y_pred[0], label='Predicted error')
    plt.grid()
    plt.legend()
    plt.show()
    return validind


validind_all = calculate_correction('all_x.csv','all_y.csv')
validind_0 = calculate_correction('x_0.csv','y_0.csv')
validind =list(set(validind_0)&set(validind_all))
x = loadtxt('all_timestamp.csv', delimiter=',',dtype='datetime64')
x = np.copy(x[validind, ])
#x = x[(x.astype("M8[ms]")).hour==0]
all_timestamp = x.astype("M8[ms]").tolist()
all_0_time =[]
y = loadtxt('x_0_timestamp.csv', delimiter=',',dtype='datetime64')
for i in all_timestamp:
    if i.hour == 0:
        np.append(all_0_time,i)
        y = np.where(y == i)
all_0_time = np.array(all_0_time)


validind_all = calculate_correction_valid('all_x.csv','all_y.csv',validind)
validind_0 = calculate_correction_valid('x_0.csv','y_0.csv',validind)