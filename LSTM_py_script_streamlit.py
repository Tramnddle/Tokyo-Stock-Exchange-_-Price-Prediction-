
import subprocess

# Install required packages from requirements.txt
subprocess.call("pip install -r requirements.txt", shell=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from st_files_connection import FilesConnection
import tensorflow as tf
from keras.models import load_model
import gcsfs
import os

secrets = st.secrets["connections_gcs"]
secret_value = os.environ.get('connections_gcs')

# Create a GCS connection
conn = st.experimental_connection('gcs', type=FilesConnection)

# Read a file from the cloud storage
df = conn.read("gs://tokyostockexchange/stock_prices.csv", input_format="csv")

# Open the df file
st.dataframe(df)

# Read the CSV file with Dask
stock_list = conn.read("gs://tokyostockexchange/stock_list.csv", input_format="csv")
st.dataframe(stock_list)

# Create a dropdown menu
Securities_List = st.selectbox('Securities reference list: ', list(stock_list[['SecuritiesCode', 'Name']].itertuples(index=False, name=None)))

# Convert the Date column
df['Date'] = df['Date'].astype('M8[ns]') 


st.title('Price Forecast/Tokyo Stock Exchange JPX (2017-01-04 to 2021-12-03)')
user_inputs = st.text_input('Enter Stock Codes (comma-separated)', '6201')  # Example input

st.set_option('deprecation.showPyplotGlobalUse', False)

data = df[df["SecuritiesCode"]==int(user_inputs)]

data.index=data.pop('Date')

plt.figure(figsize=(16, 8))

data['Close'].plot(label=user_inputs)

st.subheader('Close Price')

plt.title('Close Price')
plt.legend()
st.pyplot()


import datetime
def str_to_datetime(s):
  return datetime.datetime.strptime(str(s), '%Y-%m-%d')


def df_to_windowed_df(dataframe, first_date_str, last_date_str, n, display_dataframe=True):
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)

    target_date = first_date

    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n + 1)

        if len(df_subset) != n + 1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

        if last_time:
            break

        target_date = next_date

        if target_date == last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        ret_df[f'Target-{n - i}'] = X[:, i]

    ret_df['Target'] = Y


def windowed_df_to_date_X_y(windowed_dataframe):

    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]

    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)


dates, X, y = windowed_df_to_date_X_y(df_to_windowed_df(data, '2020-12-03', '2021-12-03', n=5, display_dataframe=False))


from sklearn.preprocessing import MinMaxScaler

X_flat = X.reshape(X.shape[0], -1)  # This will flatten the timesteps while keeping the samples intact

    
# Now apply the MinMaxScaler
min_max_scaler = MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X_flat)
y_scaled = min_max_scaler.fit_transform(y.reshape(-1, 1))
# reshape it back
X_scaled_reshaped = X_scaled.reshape(X.shape[0], X.shape[1], -1)

# Calculate the length of the training data by taking 80% of the total length of the 'date' array
q_80 = int(len(dates) * .8)

dates_train, X_train, y_train = dates[:q_80], X_scaled_reshaped[:q_80], y_scaled[:q_80]

dates_test, X_test, y_test = dates[q_80:], X_scaled_reshaped[q_80:], y_scaled[q_80:]

from google.cloud import storage

# Initialize Google Cloud Storage client
client = storage.Client()

# Define your Google Cloud Storage bucket name and model file path
bucket_name = 'lstm_model_stockexchange'
model_blob_name = 'LSTM_stockprediction_model.h5'  # Path within the bucket
# Get the bucket
bucket = client.get_bucket(bucket_name)

# Download the model file from Google Cloud Storage
blob = bucket.blob(model_blob_name)
local_model_file = 'LSTM_stockprediction_model.h5'
blob.download_to_filename(local_model_file)

# Load the model
model = load_model(local_model_file)


# Plot prediction
train_predictions = model.predict(X_train).flatten()
test_predictions = model.predict(X_test).flatten()
# Convert the scaled values back into real data
y_train_original = min_max_scaler.inverse_transform(y_train)
train_predictions_original = min_max_scaler.inverse_transform(train_predictions.reshape(-1, 1))
y_test_original = min_max_scaler.inverse_transform(y_test)
test_predictions_original = min_max_scaler.inverse_transform(test_predictions.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(dates_train, train_predictions_original)
plt.plot(dates_train, y_train_original)
plt.plot(dates_test, test_predictions_original)
plt.plot(dates_test, y_test_original)
plt.legend(['Training Predictions', 'Training Observations', 'Testing Predictions', 'Testing Observations'])

st.subheader('Prediction model performance in the past')
plt.title(f'Price prediction model of {user_inputs} in the past')
st.pyplot()

# Loading predict dataset
Predict = conn.read("gs://tokyostockexchange/stock_prices_predict.csv", input_format="csv")
Predict.loc[:, "Date"] = pd.to_datetime(Predict.loc[:, "Date"], format="%Y-%m-%d")
Predict.index = Predict.pop('Date')
Pr_selectedstock = Predict[Predict["SecuritiesCode"] == int(user_inputs)]
data = data.drop('Target', axis=1)
Pr_selectedstock = pd.concat([data, Pr_selectedstock], axis=0)

dates, X, y = windowed_df_to_date_X_y(df_to_windowed_df(Pr_selectedstock,
                                          '2021-12-06',
                                          '2021-12-07',
                                          n=5))

X_flat = X.reshape(X.shape[0], -1)  # This will flatten the timesteps while keeping the samples intact

# Now apply the MinMaxScaler
X_scaled = min_max_scaler.fit_transform(X_flat)
y_scaled = min_max_scaler.fit_transform(y.reshape(-1, 1))
# reshape it back
X_predict = X_scaled.reshape(X.shape[0], X.shape[1], -1)

Predictions_selectedstock = model.predict(X_predict).flatten()
O_Pred_selected_stock = min_max_scaler.inverse_transform(Predictions_selectedstock.reshape(-1, 1))

dates_string = ' and '.join([str(date) for date in dates])
predictions_string = ' and '.join([str(prediction[0]) for prediction in O_Pred_selected_stock])

# Display the text only when needed
if st.button("Show Predictions"):
            st.write(f'The predicted price of {user_inputs} on {dates_string} are {predictions_string}')        







