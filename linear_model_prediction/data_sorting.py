import pandas as pd
import numpy as np
import datetime
import json
from numpy import asarray
from numpy import savetxt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

def save_data_for_models(hour,error_all):
    # example for 24 hours ahead predictions at 00:00 (ne zaboravi iskoristiti sve kombinacije povijesnih podataka)
    if hour == 'all':
        x_all = error_all[err_col[0:4]]
        x_train = x_all[x_all.index < datetime.datetime(year=2019, month=7, day=24)]
        y_all = error_all[err_col[4:28]]
        y_train = y_all[y_all.index < datetime.datetime(year=2019, month=7, day=24)]
        y_test = y_all[y_all.index >= datetime.datetime(year=2019, month=7, day=24)]
        x_test = x_all[x_all.index >= datetime.datetime(year=2019, month=7, day=24)]
    else:
        error = error_all[error_all.index.hour == datetime.datetime(year=2000, month=1, day=1, hour=hour).hour]
        error_train = error[error.index < datetime.datetime(year=2019, month=7, day=24)]
        error_test = error[error.index >= datetime.datetime(year=2019, month=7, day=24)]

        # x_all = error[err_col[0:4]]
        # all_timestamp = error.index
        # y_all = error[err_col[4:28]]
        x_train = error_train[err_col[0:4]].values[[*range(0, error_train.shape[0])]]
        y_train = error_train[err_col[4:28]].values[[*range(0, error_train.shape[0])]]
        x_test = error_test[err_col[0:4]].values[[*range(0, error_test.shape[0])]]
        y_test = error_test[err_col[4:28]].values[[*range(0, error_test.shape[0])]]
        #
        # x_0_timestamp = error_test.index
        #
        #
        # savetxt('all_x.csv', x_all, delimiter=',')
        # savetxt('all_y.csv', y_all, delimiter=',')
        # savetxt('all_timestamp.csv', all_timestamp, delimiter=',', fmt='%s')
        # savetxt('x_0_timestamp.csv', x_0_timestamp, delimiter=',', fmt='%s')

    savetxt('x_' + str(hour) + '_train.csv', x_train, delimiter=',')
    savetxt('y_' + str(hour) + '_train.csv', y_train, delimiter=',')
    savetxt('x_' + str(hour) + '_test.csv', x_test, delimiter=',')
    savetxt('y_' + str(hour) + '_test.csv', y_test, delimiter=',')

# load data
weather = pd.read_csv('forecast_data/weather_measurements.csv', index_col='weather_station_measurement_timestamp',
                      parse_dates=True, date_parser=pd.to_datetime)
forecast = pd.read_csv('forecast_data/weather_forecasts.csv', index_col='weather_prediction_timestamp',
                       parse_dates=True, date_parser=pd.to_datetime)

# get temperature measurements
temp = weather[['weather_station_measurement_outdoor_temperature_south']]
temp.index.name = 'timestamp'
temp.columns = ['temperature']
temp = temp.resample('1H', label='left').mean()
temp = temp[temp.index >= datetime.datetime(year=2018, month=4, day=18, hour=0)]

# get temperature forecasts
temp_for = forecast[['weather_prediction_start_timestamp', 'weather_prediction_temperature_at_2m']]
temp_for = temp_for[temp_for.index >= datetime.datetime(year=2018, month=4, day=18, hour=0)]
temp_for['weather_prediction_start_timestamp'] = pd.to_datetime(temp_for['weather_prediction_start_timestamp'].values)

# create dataset
shift = [*range(0, 24 + 9 + 1)]
temp_col = ['temperature(t)']
for_col = ['forecast(t)']
err_col = ['error(t)']
for i in shift[1:]:
    temp_col.append('temperature(t+' + str(i) + ')')
    for_col.append('forecast(t+' + str(i) + ')')
    err_col.append('error(t+' + str(i) + ')')
dataset = pd.DataFrame(index=temp.index[[ind in [0, 6, 12, 18] for ind in temp.index.hour]],
                       columns=temp_col + for_col)
# fill dataset with measured and forecasted temperature values
for ind in dataset.index:
    if any(temp_for['weather_prediction_start_timestamp'] == ind):
        aux_for = temp_for[temp_for['weather_prediction_start_timestamp'] == ind][
            'weather_prediction_temperature_at_2m']
        aux_for = json.loads(aux_for[0])

        aux_temp = temp.loc[
            (temp.index >= ind) & (temp.index <= ind + datetime.timedelta(hours=33)), 'temperature'].values
        # skip if there are some missing temperatures
        if aux_temp.shape[0] == 34:
            dataset.loc[ind, temp_col] = aux_temp
            dataset.loc[ind, for_col] = aux_for[0:34]

        # Sporije ali mislim da dobro radi
        # for i in shift:
        #     if i == 0:
        #         dataset.loc[ind,'temperature(t)'] = temp.loc[ind,'temperature']
        #         dataset.loc[ind,'forecast(t)'] = aux_for[i]
        #     else:
        #         dataset.loc[ind,'temperature(t+' + str(i) + ')'] = temp.loc[ind+datetime.timedelta(hours=i),'temperature']
        #         dataset.loc[ind,'forecast(t+' + str(i) + ')'] = aux_for[i]

# drop rows with any NaN values
dataset = dataset.dropna(how='any')

# error calculation
error = pd.DataFrame(index=dataset.index, columns=err_col, data=[])
for ind in dataset.index:
    error.loc[ind, err_col] = dataset.loc[ind, temp_col].values - dataset.loc[ind, for_col].values

save_data_for_models(0,error)
save_data_for_models(6,error)
save_data_for_models(12,error)
save_data_for_models(18,error)
save_data_for_models('all',error)
