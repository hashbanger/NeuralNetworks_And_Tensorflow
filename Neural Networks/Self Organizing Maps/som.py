#Creating a Self Organising Map

import pandas as pd
import numpy as np


#Getting the data
data = pd.read_csv('Credit_Card_Applications.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

#Scaling the values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X,y)

#Importing the downloaded class
from minisom import MiniSom
ms = MiniSom(x =10,y= 10, sigma = 1.0, input_len = 15, learning_rate = 0.5)
ms.random_weights_init(X)
ms.train_random(data = X, num_iteration = 100)

#Visualising
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(ms.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r','g']
for i, x in enumerate(X):
    w = ms.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5, 
         markers[y[i]], 
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

#Finding the frauds
mapping = ms.win_map(X)
frauds = np.concatenate((mapping[(8,1)], mapping[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)

