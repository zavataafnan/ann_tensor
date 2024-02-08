import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import mplfinance as mpf
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# provide your file name instead of 'data.csv'
data = pd.read_csv('XAUUSD_Daily_now.csv', sep="\t")

# data object is pandas dataframe
# Remove unwanted characters like '<' and '>' from column names
data.columns = data.columns.str.replace('<', '').str.replace('>', '')

data = data.dropna()
# extract columns into a dictionary of lists (array)
output_data = {
    'open': data['OPEN'].tolist(),
    'close': data['CLOSE'].tolist(),
    'high': data['HIGH'].tolist(),
    'low': data['LOW'].tolist(),
}



# Assuming output_data is a dictionary as previously defined
# {"open": [...], "close": [...], "high": [...], "low": [...]}

# Convert the lists to numpy arrays
open = np.array(output_data['open'])
close = np.array(output_data['close'])
high = np.array(output_data['high'])
low = np.array(output_data['low'])



# Verify that all the arrays have the same length
assert len(open) == len(close) == len(high) == len(low)

data = []
data_y = []
for i in range(len(open) - 4):
    # Form a 16-dimensional array and append it to the data list
    # Each element of the list is a numpy array containing 4 successive candles (open, close, high, and low)
    data.append(np.array([open[i], close[i], high[i], low[i],
                          open[i+1], close[i+1], high[i+1], low[i+1],
                          open[i+2], close[i+2], high[i+2], low[i+2],
                          open[i+3], close[i+3], high[i+3], low[i+3]]))

    data_y.append(np.array([open[i+4], close[i+4], high[i+4], low[i+4]]))

# Convert the list into a numpy array
data = np.array(data)
data_y = np.array(data_y)

np.savetxt("output.csv", data_y, delimiter=",")
np.savetxt("input.csv", data, delimiter=",")

print(data.shape)  # Prints: (num_samples, 16)

# y = data[:, -4:]  # Let's assume the last candle of each data sample is what we want to predict

1
x_train, x_test, y_train, y_test = train_test_split(data, data_y, test_size=0.2, random_state=42)

print(f"Train Input Shape: {x_train.shape}")
print(f"Train Output Shape: {y_train.shape}")
print(f"Test Input Shape: {x_test.shape}")
print(f"Test Output Shape: {y_test.shape}")



# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(16,), activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=150, batch_size=10)
loss = model.evaluate(x_test, y_test)
print('Test loss:', loss)

predictions = model.predict(x_test)
predictions = predictions.tolist()

open_pred, close_pred, high_pred, low_pred = [], [], [], []
for pred in predictions:
    open_pred.append(pred[0])
    close_pred.append(pred[1])
    high_pred.append(pred[2])
    low_pred.append(pred[3])

# Combine the lists into a dictionary
output_pred = {
    'open': open_pred,
    'close': close_pred,
    'high': high_pred,
    'low': low_pred,
}

print(output_pred)





# Convert lists to DataFrame
df_pred = pd.DataFrame(output_pred)

# Plot Predicted Data
mpf.plot(df_pred, type='candle', title='Predicted Prices', style='yahoo')

# Convert x_test to open, close, high, low
open_test, close_test, high_test, low_test = [], [], [], []
for i in range(len(x_test)):
    open_test.append(x_test[i][0])
    close_test.append(x_test[i][1])
    high_test.append(x_test[i][2])
    low_test.append(x_test[i][3])

# Format x_test to a Dictionary
output_test = {
    'open': open_test,
    'close': close_test,
    'high': high_test,
    'low': low_test,
}

# Convert lists to DataFrame
df_test = pd.DataFrame(output_test)

# Plot Test Data
mpf.plot(df_test, type='candle', title='Test Prices', style='yahoo')

# fit the keras model on the dataset
# model.fit(x_train, y_train, epochs=150, batch_size=10)
