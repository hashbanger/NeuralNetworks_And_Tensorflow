#Importing the libraries
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers import MaxPool2D, Conv2D
from keras.datasets import mnist

batch_size = 50
epochs = 10
num_classes = 10

#image diomensions
img_x, img_y = 28, 28

#loading and splitting the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Reshaping the Data into 4D Tensor
#Since image is greyscale, channels = 1
X_train = X_train.reshape(X_train.shape[0], img_x, img_y,  1)
X_test = X_test.reshape(X_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1) 
#Declaring the input shape is only required of the first layer – Keras is 
#good enough to work out the size of the tensors flowing through 
#the model from there.

#Converting into appropriate datatype
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

#Normalizing the inputs , the pixel values from 0-255
#Standar Scaler won't work on dimensions above 2
X_train /= 255
X_test /= 255

print('X_train shape: ', X_train.shape)
print('Train Examples: ', len(X_train))
print('Test Examples: ', len(X_test))

#One hot encoding returned sparse scipy matrix
'''
#One hot encoding the y labels
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
'''
#Converting the code
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Creating the CNN
#With the general approach of Using two conv and pooling layers
model = Sequential()
model.add(Conv2D( filters = 32, kernel_size = (5,5), strides = (1,1),
                 activation= 'relu', input_shape = input_shape))
model.add(MaxPool2D(pool_size= (2,2), strides= (2,2)))
model.add(Conv2D(filters = 64, kernel_size= (5,5), strides= (1,1), 
                 activation= 'relu'))
model.add(MaxPool2D(pool_size= (2,2), strides= (2,2)))
model.add(Flatten())
model.add(Dense(1000, activation = 'relu'))
model.add(Dense(10, activation= 'softmax'))

#Compiling the Network
model.compile(optimizer= 'adam',
              loss = keras.losses.categorical_crossentropy,
              metrics = ['accuracy'])

#The Callback super class that the code above inherits from has a number of methods
#that can be overridden in our callback definition such as on_train_begin, 
#on_epoch_end, on_batch_begin and on_batch_end.

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs = {}):
        self.acc = []
        
    def on_epoch_end(self, batch, logs = {}):
        self.acc.append(logs.get('acc'))
        
#we can extract the variable we want from the logs, which is a dictionary 
#that holds, as a default, the loss and accuracy during training. 
#We then instantiate this callback        
history = AccuracyHistory()

#Fitting the model
model.fit(X_train, y_train, batch_size= batch_size, epochs= epochs,
          verbose= 1, validation_data= (X_test, y_test),
          callbacks = [history])

#Evaluating
score = model.evaluate(X_test, y_test, verbose= 0)  
print('Test loss', score[0])
print('Test accuracy', score[1])

#Visualising the results
plt.plot(range(1,11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()







