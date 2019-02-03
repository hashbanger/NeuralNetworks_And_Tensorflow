#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Getting the data
data = pd.read_csv('Credit_Card_Applications.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

#Scaling the features
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))
X = sc.fit_transform(X)

#Imprting the Minisom Class
from minisom import MiniSom
ms = MiniSom(10,10, input_len= 15, sigma= 1.0, learning_rate= 0.5 )
ms.random_weights_init(X)
ms.train_random(X, num_iteration= 100)

#Visualising
plt.bone()
plt.pcolor(ms.distance_map().T)
plt.colorbar()
markers = ['o','s']
colors = ['r','g']
for i, x in enumerate(X):
    w = ms.winner(x)
    plt.plot(w[0]+ 0.5, w[1] + 0.5,
             marker = markers[y[i]],
             markeredgecolor = colors[y[i]],
             markerfacecolor = 'None',
             markersize = 10,
             markeredgewidth = 2)
plt.show()    

#finding the frauds
mappings = ms.win_map(X)
frauds = np.concatenate((mappings[(6,5)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)

#Creating the matrx of features
customers = data.iloc[:, 1:]

#Creating the dependent variable
is_fraud = np.zeros(len(data))

for i in range(len(data)):
    if data.iloc[i, 0] in frauds:
        is_fraud[i] = 1
        
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
customers = sc.fit_transform(customers)     

#Making the ANN
from keras.layers import Dense
from keras.models import Sequential

classifier = Sequential()
#Adding the input and hidden layer
classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15)) 
#Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#Comiling the Ann
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Fitting the ann
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 5)

y_pred = classifier.predict(customers)
y_pred = np.concatenate((data.iloc[:,0:1].values, y_pred), axis =1)
y_pred = y_pred[y_pred[:, 1].argsort()] 

