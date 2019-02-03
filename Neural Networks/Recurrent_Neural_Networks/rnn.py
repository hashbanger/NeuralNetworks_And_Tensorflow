#Implementing Recurrent Neural Networks

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Getting the data
train_set = pd.read_csv('Google_Stock_Price_Train.csv')
train_set = train_set.iloc[:,1:2].values #Keeping it as matrix

#Scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train_set = sc.fit_transform(train_set)

#Shifting and Setting up the X_train and Y_train
X_train = train_set[0:1257]
y_train = train_set[1:1258]

#Reshaping
X_train = np.reshape(X_train, (1257,1,1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#Initialising the RNN
regressor = Sequential()

#adding the LSTM units
regressor.add(LSTM(units = 4, input_shape = (None, 1), activation = 'sigmoid'))

#adding the output layer
regressor.add(Dense(units = 1)) 

#compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

 #fitting the RNN
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)     

#Getting the real data of 2017
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values

inputs = real_stock_price
inputs = sc.transform(inputs)
inputs  = np.reshape(inputs, (20,1,1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visulaising the stock price
plt.plot(predicted_stock_price, 'r',label = 'Predicted PStock Price')
plt.plot(real_stock_price, 'b', label = 'Real Stock Price')



