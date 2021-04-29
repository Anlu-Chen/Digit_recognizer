import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rand
import tensorflow as tf
print(tf.__version__)
import time as time
import math as math
from tensorflow.keras.callbacks import LearningRateScheduler

data = np.loadtxt('train.csv',delimiter=',',skiprows=1)
print(data.shape)

data_images = data[0:42000, 1:785]
data_number = data[0:42000, 0]
print(data_images.shape, data_number.shape)

numeros = list(range(0,42000))
rand.shuffle(numeros)
print(type(numeros))
print(len(numeros)-len(set(numeros)))  #para comprobar si se ha repetido algun numero, si sale 0, no se ha repetido
numeros_random = np.array(numeros) #pasar la lista a vector


train_data= data_images[numeros_random[0:33600],:]
train_number= data_number[numeros_random[0:33600]]
test_data= data_images[numeros_random[33600:42000],:]
test_number=data_number[numeros_random[33600:42000]]
print(train_data.shape,
      train_number.shape,
      test_data.shape,
      test_number.shape)

train_images0 = np.reshape(train_data,(33600,28,28,1))
train_images = train_images0/255.0
test_images0 = np.reshape(test_data,(8400,28,28,1))
test_images = test_images0/255.0

#plt.imshow(np.asarray(test_images[0]),cmap='gray')
#plt.colorbar()
#plt.show()
print(train_images[0].shape)

# Entrenamiento de red CNN.
tic=time.time()

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=(3,3),input_shape=train_images[0].shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, kernel_size=(3,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),  
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(2048,kernel_initializer='random_normal',bias_initializer='zeros', use_bias=True), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Dense(1024, use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Dense(512, use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Dense(256, use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Dense(128, use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Dense(128, use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Dense(128, use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Dense(10, activation='softmax', use_bias=True)])
############################################################################
# Compilamos el modelo
# Definimos como va a ser el ratio de aprendizaje exponencial.

opt=tf.keras.optimizers.SGD(momentum=0.9, nesterov=True, name='SGD')
model.compile(
    optimizer=opt,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']) 

initial_learning_rate = 0.1
def lr_exp_decay(epoch, learning_rate):
    k = 0.3
    return initial_learning_rate * math.exp(-k*epoch)

# Ajustamos el modelo.
training_history = model.fit(
        train_images, 
        train_number, 
        batch_size=20,
        epochs=10, 
        validation_data=(test_images, test_number),
        callbacks=[LearningRateScheduler(lr_exp_decay, verbose=1)]
        )

############################################################################
# Pintamos el modelo
model.summary() 
pd.DataFrame(training_history.history).plot(figsize=(12,7))
plt.grid(True)
plt.show()

toc=time.time()
print((toc-tic)/60,'\n minutos ha durado')