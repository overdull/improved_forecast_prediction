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

# example for 24 hours ahead predictions at 00:00 (ne zaboravi iskoristiti sve kombinacije povijesnih podataka)
error_0 = error[error.index.hour == datetime.datetime(year=2000, month=1, day=1, hour=0).hour]
error_6 = error[error.index.hour == datetime.datetime(year=2000, month=1,day=1, hour=6).hour]
error_12 = error[error.index.hour == datetime.datetime(year=2000, month=1,day=1, hour=12).hour]
error_18 = error[error.index.hour == datetime.datetime(year=2000, month=1,day=1, hour=18).hour]
x_all = error[err_col[0:4]]
all_timestamp = error.index
y_all = error[err_col[4:28]]
x_0 = error_0[err_col[0:4]].values[[*range(0, error_0.shape[0])]]
x_0_timestamp = error_0.index
y_0 = error_0[err_col[4:28]].values[[*range(0, error_0.shape[0])]]
x_6 = error_6[err_col[0:4]].values[[*range(0, error_6.shape[0])]]
y_6 = error_6[err_col[4:28]].values[[*range(0, error_6.shape[0])]]
x_12 = error_12[err_col[0:4]].values[[*range(0, error_12.shape[0])]]
y_12 = error_12[err_col[4:28]].values[[*range(0, error_12.shape[0])]]
x_18 = error_18[err_col[0:4]].values[[*range(0, error_18.shape[0])]]
y_18 = error_18[err_col[4:28]].values[[*range(0, error_18.shape[0])]]



savetxt('all_x.csv',x_all, delimiter=',')
savetxt('all_y.csv',y_all, delimiter=',')
savetxt('all_timestamp.csv',all_timestamp, delimiter=',',fmt='%s')
savetxt('x_0_timestamp.csv',x_0_timestamp, delimiter=',',fmt='%s')


savetxt('x_0.csv', x_0, delimiter=',')
savetxt('x_6.csv', x_6, delimiter=',')
savetxt('x_12.csv', x_12, delimiter=',')
savetxt('x_18.csv', x_18, delimiter=',')
savetxt('y_0.csv', y_0, delimiter=',')
savetxt('y_6.csv', y_6, delimiter=',')
savetxt('y_12.csv', y_12, delimiter=',')
savetxt('y_18.csv', y_18, delimiter=',')