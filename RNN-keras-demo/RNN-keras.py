__author__ = 'Fan'
__license__ = 'SYSU'

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense


# data prepare
mnist = input_data.read_data_sets('dataset/', one_hot=True)
x_train = np.array(mnist.train.images[0:1000]).reshape([-1, 28, 28])
y_train = np.array(mnist.train.labels[0:1000])
x_test = np.array(mnist.test.images[0:100]).reshape([-1, 28, 28])
y_test = np.array(mnist.test.labels[0:100])

model = Sequential()

model.add(SimpleRNN(
    units=50,
    batch_input_shape=(None, 28, 28),
    unroll=True
))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Training
print('Training ------------')
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Testing
print('\nTesting ------------')
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
