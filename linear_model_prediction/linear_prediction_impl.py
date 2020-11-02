import pandas as pd

# weather directly measured at FER building
# measurements are taken within one minute interval
weather = pd.read_csv('forecast_data/weather_measurements.csv', index_col='weather_station_measurement_timestamp',
                      parse_dates=True, date_parser=pd.to_datetime)

# weather forecast predicted for FER building
forecast = pd.read_csv('forecast_data/weather_forecasts.csv', index_col='weather_prediction_timestamp',
                       parse_dates=True, date_parser=pd.to_datetime)
