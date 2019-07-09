__date__ = '2019.7.8'
__author__ = 'Fan'
__license__ = 'SYSU'

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Activation, MaxPool2D, Flatten
from keras.optimizers import SGD


# data prepare
mnist = input_data.read_data_sets('dataset/', one_hot=True)
x_train = np.array(mnist.train.images[0:100]).reshape([-1, 1, 28, 28])
y_train = np.array(mnist.train.labels[0:100])
x_test = np.array(mnist.test.images[0:10]).reshape([-1, 1, 28, 28])
y_test = np.array(mnist.test.labels[0:10])

# 定义model
model = Sequential()

# Convolution2D, 二维卷积层
# output shape = (32, 28, 28)
model.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    data_format='channels_first',
))
model.add(Activation('relu'))

# MaxPool2D, 二维最大池化层
# output shape = (32, 14, 14)
model.add(MaxPool2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first',
))

# output shape = (64, 14, 14)
model.add(Convolution2D(
    filters=64,
    kernel_size=5,
    strides=1,
    padding='same',
    data_format='channels_first',
))
model.add(Activation('relu'))

# output shape = (64, 7, 7)
model.add(MaxPool2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first',
))

# Flatten, 三维数据压平
# output shape = (3136,)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

# 定义SGD优化器
sgd = SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Training
print('Training ------------')
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Testing
print('\nTesting ------------')
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
