# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 00:01:25 2020

@author: Ivano Dibenedetto mat. 654678
"""
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import pickle

# leggo il trainingset (immagini per il training)
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

# leggo il testset (immagini per il test)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

# normalizza i colori nel range di [0,1]
X = np.array(X/255.0)
y = np.array(y)


model = Sequential()

"""
filtri ottimali
32,64,128
"""

# architettura
model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:], activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  

model.add(Dense(256))
model.add(Dense(1, activation="sigmoid"))


model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X, y, batch_size=32, epochs=10, validation_split=0.2)

model.save("modello_rete_neurale.model")

#ACCURACY
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


#LOSS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
