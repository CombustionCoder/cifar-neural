# Author: Jukka Hirvonen
# This script classifies CIFAR-10 images using a neural net.
# Tested on Google Colab.

import pickle
import time

start_time = time.time() # for calculating runtime
print("Start time: " + time.ctime(start_time))
print("-----")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def data_open():
    datadict = unpickle('/content/drive/MyDrive/c10/data_batch_1')
    datadict2 = unpickle('/content/drive/MyDrive/c10/data_batch_2')
    datadict3 = unpickle('/content/drive/MyDrive/c10/data_batch_3')
    datadict4 = unpickle('/content/drive/MyDrive/c10/data_batch_4')
    datadict5 = unpickle('/content/drive/MyDrive/c10/data_batch_5')
    testdict = unpickle('/content/drive/MyDrive/c10/test_batch')
    
    X = datadict["data"]
    Y = datadict["labels"]
    X2 = datadict2["data"]
    Y2 = datadict2["labels"]
    X3 = datadict3["data"]
    Y3 = datadict3["labels"]
    X4 = datadict4["data"]
    Y4 = datadict4["labels"]
    X5 = datadict5["data"]
    Y5 = datadict5["labels"]
    X = np.concatenate((X,X2,X3,X4,X5), axis=0)
    Y = np.concatenate((Y,Y2,Y3,Y4,Y5), axis=0)
    
    Tx = testdict["data"]
    Ty = testdict["labels"]
    return(X,Y,Tx,Ty)


def class_ac(pred,gt):
    score = 0
    for i in range(pred.shape[0]):
        if pred[i] == gt[i]:
            score += 1
    acc = score / pred.shape[0]
    print("Training samples " + str(sampleSize) + ", learning rate " + str(learning_rate) + ", epochs " + str(epochs) + ", batch size " + str(batch_size))
    print("Correctly classified " + str(score) + " out of " + str(pred.shape[0]) + " test samples = " + str(round((acc*100),2)) + "%")

def one_hot(yy):
    b = np.zeros((yy.size, yy.max() + 1)).astype('uint8')
    b[np.arange(yy.size), yy] = 1
    return(b)

# execution segment below ---------------------

X,Y,Tx,Ty=data_open()

# sample size limiter for testing 1-50000
sampleSize = 50000
y = Y[:sampleSize] # training labels
tt=Tx[:sampleSize,:] # test samples
ty=Ty[:sampleSize] # test labels
xx=X[:sampleSize,:] # training samples

y1h=one_hot(y)
t1h=one_hot(np.array(ty))

model = Sequential()

# normalize
layer = tf.keras.layers.Normalization(axis=-1)
layer.adapt(xx)
model.add(layer)

# hidden
model.add(Dense(100, input_dim=3072, activation='sigmoid'))

# output
model.add(Dense(10, activation='sigmoid'))

epochs=4800
batch_size=1000
learning_rate=0.06

# compile and optimize
opt=tf.keras.optimizers.SGD(learning_rate=learning_rate)
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

# fit
with tf.device('/device:GPU:0'):
  tr_hist = model.fit(xx, y1h, batch_size=batch_size, epochs=epochs, verbose=0,
                      shuffle=True, use_multiprocessing=True, validation_data=(tt,t1h))

plt.plot(tr_hist.history['accuracy'])
plt.plot(tr_hist.history['val_accuracy'], color='red')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# check result
processed_data = np.array(model(tt))
result=np.argmax(processed_data, axis=1)
class_ac(result,Ty)
model.summary()

# lets see how long it took
print("-----")
print("End time: " + time.ctime(time.time()))
print("Runtime: " + str(time.time()-start_time) +" seconds")