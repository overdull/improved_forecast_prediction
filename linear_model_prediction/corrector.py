import pandas as pd
import numpy as np
import datetime
import json
from numpy import loadtxt
from numpy import savetxt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt


def calculate_correction(filename_x_train, filename_y_train, filename_x_test, filename_y_test,test_n, hour):
    x_0_train = loadtxt(filename_x_train, delimiter=',')
    y_0_train = loadtxt(filename_y_train, delimiter=',')
    x_0_test = loadtxt(filename_x_test, delimiter=',')
    y_0_test = loadtxt(filename_y_test, delimiter=',')
    # ubaciti skaliranje
    y_0_test = y_0_test
    # form learning and validation dataset
    n = x_0_train.shape[0]
    perm = np.random.permutation(n)
    trainratio = 1
    testratio = 0.2
    trainind = perm[:int(np.ceil(n * trainratio))]
    #validind = perm[int(np.ceil(n * trainratio)):]
    # m = x_0_test.shape[0]
    # perm = np.arange(0,m,1)

    #testind = perm
    # create model
    #knn = KNeighborsRegressor()

    #model = MultiOutputRegressor(knn)
    model = RandomForestRegressor()
    model.fit(x_0_train[trainind], y_0_train[trainind])
    #ubaciti valid set istraziti

    # create predictions on validation dataset and calculate MAE
    y_pred = model.predict(x_0_test)
    #y_pred = model.predict(x_0_test[testind])
    test_error = mean_absolute_error(y_0_test, y_pred)
    a = y_0_test[test_n]
    # print single example
    # hour + 4
    dim = np.arange(hour+4, 4+ 24+hour, 1)

    #plt.xticks(dim)
    plt.plot(dim, a, label='Calculated error' + filename_x_test)
    plt.plot(dim,y_pred[test_n], label='Predicted error' + filename_x_test)
    plt.xticks(dim)
    plt.grid()
    plt.legend()
    plt.show()
    return test_error


# def calculate_correction_valid(filename_x, filename_y, validind):
#     x_0_train = loadtxt('x_0_train.csv', delimiter=',')
#     y_0_train = loadtxt('y_0_train.csv', delimiter=',')
#     x_0_test = loadtxt('x_0_test.csv', delimiter=',')
#     y_0_test = loadtxt('y_0_test.csv', delimiter=',')
#     # ubaciti skaliranje
#
#     # form learning and validation dataset
#     n = x_0_train.shape[0]
#     perm = np.random.permutation(n)
#     trainratio = 0.8
#     testratio = 0.2
#     trainind = perm[:int(np.ceil(n * trainratio))]
#     validind = perm[int(np.ceil(n * trainratio)):]
#
#     # create model
#     knn = KNeighborsRegressor()
#     model = MultiOutputRegressor(knn)
#     model.fit(x_0_train[trainind], y_0_train[trainind])
#
#     # create predictions on validation dataset and calculate MAE
#     y_pred = model.predict(x_0_train[validind])
#     valid_error = mean_absolute_error(y_0_train[validind], y_pred)
#
#     # print single example
#     plt.plot(y_0_train[validind][0], label='Calculated error')
#     plt.plot(y_pred[0], label='Predicted error')
#     plt.grid()
#     plt.legend()
#     plt.show()

hour = 0
index = 2
temp = 4
print(calculate_correction('x_' + str(hour) + '_train.csv', 'y_' + str(hour) + '_train.csv', 'x_' + str(hour) + '_test.csv', 'y_' + str(hour) + '_test.csv',index,hour=hour))
print(calculate_correction('x_all_train.csv', 'y_all_train.csv', 'x_all_test.csv', 'y_all_test.csv',int(index*(temp+(hour/6))),hour=hour))

#dodati pomicanje po satima
